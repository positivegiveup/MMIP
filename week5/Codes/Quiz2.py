"""
Quiz 2b（新版 Quiz 2）：UPMC Food-101 多模態（食物影像 + 網頁食譜文字）101 類分類
與舊版 Quiz2.py（CrisisMMD）並存，輸出檔名用 Q2b_*.png / q2b_*.pt，不會覆蓋舊結果。

實驗設計
  1) Backbone 網格（皆凍結、先抽特徵並快取）：
       影像：ResNet-50（ImageNet 監督式）、CLIP ViT-B/16 影像編碼器（圖文對比式）
       文字：BERT-base（mean pooling, 128 tokens）、CLIP 文字編碼器（77 tokens）
       -> 4 組單模態基準 + 4 組「concat 融合」
  2) 融合策略比較（用網格中最佳的組合）：concat / late（各模態 logits 平均）/ gated（學習式門控）
  3) 分析：逐類別增益、模態消融（推論時遮掉某一模態）、最常被混淆的類別
每組 3 個 seed（mean ± std），early stopping 以 val Macro-F1 為準。

資料來源：UPMC Food-101（Wang et al., 2015；約 9 萬筆，train/test，101 類，每筆 = 一張影像 + 一段網頁文字）。
  - 若 Data/upmc_food101/ 已有資料就直接使用（可手動放）；
  - 否則嘗試用 kagglehub 下載 KAGGLE_ID（此 ID 為依記憶設定、尚未驗證；若失敗會印出手動下載說明）。
  - 資料夾結構/文字檔格式不確定，所以載入時會「自動偵測」（影像資料夾、csv/tsv/json/txt 文字），並印出偵測結果。
"""
import os
import re
import tarfile
import textwrap
import time
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

import Quiz1 as q1

device = q1.device
KAGGLE_ID = "gianmarco96/upmcfood101"      # 未驗證；若下載失敗請手動放到 Data/upmc_food101/
IMG_EXT = {".jpg", ".jpeg", ".png"}
CLIP_NAME = "openai/clip-vit-base-patch16"
IMG_BB = {"resnet50": 2048, "clip_img": 512}
TXT_BB = {"bert": 768, "clip_txt": 512}
CLIP_MEAN, CLIP_STD = [0.4815, 0.4578, 0.4082], [0.2686, 0.2613, 0.2758]
TF = {"resnet50": T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(q1.MEAN, q1.STD)]),
      "clip_img": T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224),
                             T.ToTensor(), T.Normalize(CLIP_MEAN, CLIP_STD)])}


# DATA-BEGIN ------------------------------------------------------------
# ----------------------------------------------------------------------
# 資料：下載、自動偵測結構、組成 DataFrame
# ----------------------------------------------------------------------
def _has_images(root):
    return root.exists() and any(p.suffix.lower() in IMG_EXT for p in root.rglob("*"))


def _unpack(root):
    for p in list(root.rglob("*.zip")) + list(root.rglob("*.tar.gz")) + list(root.rglob("*.tgz")) + list(root.rglob("*.tar")):
        flag = Path(str(p) + ".done")
        if flag.exists():
            continue
        print("extracting", p.name)
        if p.suffix == ".zip":
            zipfile.ZipFile(p).extractall(p.parent)
        else:
            with tarfile.open(p) as t:
                try:
                    t.extractall(p.parent, filter="data")
                except TypeError:
                    t.extractall(p.parent)
        flag.touch()


def download_upmc(paths, root=None):
    """回傳資料根目錄；本機沒有就嘗試 kagglehub 下載。"""
    root = Path(root) if root else paths["data"] / "upmc_food101"
    _unpack(root) if root.exists() else None
    if _has_images(root):
        return root
    try:
        os.environ.setdefault("KAGGLEHUB_CACHE", str(paths["data"] / "kagglehub"))
        import kagglehub
        got = Path(kagglehub.dataset_download(KAGGLE_ID))
        _unpack(got)
        if _has_images(got):
            return got
    except Exception as e:
        print("kagglehub 下載失敗：", str(e)[:200])
    raise RuntimeError(
        "找不到 UPMC Food-101。請手動下載（官方說明頁 https://visiir.isir.upmc.fr/explore 或 Kaggle 上的 UPMC Food-101），"
        f"解壓縮到 {root}（需含影像與文字檔），或修改 KAGGLE_ID / 呼叫 prepare_upmc(paths, root=...)。")


def _read_table(p):
    if p.suffix.lower() in (".json", ".jsonl"):
        try:
            return pd.read_json(p, lines=True)
        except ValueError:
            return pd.read_json(p)
    df = pd.read_csv(p, header=None, sep=None, engine="python", dtype=str, on_bad_lines="skip", keep_default_na=False)
    first = [str(x).strip().lower() for x in df.iloc[0]]
    if any(x in {"image", "img", "image_name", "filename", "file", "text", "label", "class", "title", "id", "path"}
           for x in first):
        df.columns = first
        df = df.iloc[1:].reset_index(drop=True)
    return df


def _roles(df, stems, classes):
    """不依賴欄位名稱：用內容判斷哪欄是影像檔名 / 類別 / 文字。"""
    stat = {}
    for c in df.columns:
        v = df[c].astype(str).head(2000)
        stat[c] = (v.map(lambda s: Path(s.strip()).stem in stems).mean(),
                   v.map(lambda s: s.strip() in classes).mean(), v.str.len().mean())
    img = max(stat, key=lambda c: stat[c][0])
    img = img if stat[img][0] >= 0.5 else None
    others = [c for c in stat if c != img]
    lab = max(others, key=lambda c: stat[c][1], default=None)
    lab = lab if lab is not None and stat[lab][1] >= 0.8 else None
    rest = [c for c in others if c != lab]
    txt = max(rest, key=lambda c: stat[c][2], default=None)
    return img, lab, txt


def _split_hint(s):
    s = str(s).lower()
    for k, v in (("train", "train"), ("test", "test"), ("val", "val"), ("dev", "val")):
        if k in s:
            return v
    return None


def load_records(root, min_class_imgs=100):
    """掃描影像與文字，回傳 DataFrame(stem,img,label,split,text) 與類別清單。"""
    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT]
    index = {}
    for p in imgs:
        index.setdefault(p.stem, p)
    skip = {"train", "test", "val", "dev", "images", "image", "img", "imgs"}
    cnt = pd.Series([p.parent.name for p in imgs if p.parent.name.lower() not in skip]).value_counts()
    classes = sorted(cnt[cnt >= min_class_imgs].index)
    print(f"images={len(imgs)} | class folders={len(classes)}")
    cset, stems = set(classes), set(index)

    texts = {}          # stem -> (text, split_hint, label_from_table)
    tables = [p for p in root.rglob("*") if p.suffix.lower() in (".csv", ".tsv", ".json", ".jsonl")
              and p.parent.name.lower() not in cset and "kaggle" not in p.name.lower()]
    for p in tables:
        try:
            df = _read_table(p)
            ic, lc, tc = _roles(df, stems, cset)
        except Exception as e:
            print("skip table", p.name, str(e)[:80])
            continue
        if ic is None or tc is None:
            continue
        print(f"table {p.name}: rows={len(df)} img_col={ic} label_col={lc} text_col={tc}")
        for s, t, l in zip(df[ic].astype(str), df[tc].astype(str), df[lc].astype(str) if lc else [None] * len(df)):
            texts.setdefault(Path(s.strip()).stem, (t, _split_hint(p.stem), l))
    if not texts:       # 備案：每筆一個 .txt，檔名與影像同名
        for p in root.rglob("*.txt"):
            if p.stem in stems:
                texts[p.stem] = (p.read_text(errors="ignore"), _split_hint(str(p)), None)
        print("text files found:", len(texts))
    if not texts:
        raise RuntimeError("找不到文字資料（csv/tsv/json 或與影像同名的 .txt），請檢查資料夾結構。")

    rows = []
    for stem, (t, sh, lab) in texts.items():
        p = index.get(stem)
        if p is None:
            continue
        label = p.parent.name if p.parent.name in cset else (lab if lab in cset else None)
        if label is None:
            continue
        rows.append({"stem": stem, "img": str(p), "label": label,
                     "split": _split_hint(str(p.relative_to(root))) or sh,
                     "text": re.sub(r"\s+", " ", t).strip()[:2000]})
    return pd.DataFrame(rows), classes
# DATA-END --------------------------------------------------------------


def prepare_upmc(paths, root=None, max_per_class_train=None, max_per_class_test=None, val_frac=0.1, seed=42):
    """下載/偵測 -> 組 DataFrame -> train/val/test（val 由 train 分層切 10%）。可用 max_per_class_* 縮小資料量加速。"""
    root = download_upmc(paths, root)
    df, classes = load_records(root)
    rng = np.random.RandomState(seed)
    if df.split.isna().any():       # 沒有官方切分資訊 -> 隨機 25% 當 test
        m = df.split.isna()
        df.loc[m, "split"] = np.where(rng.rand(m.sum()) < 0.25, "test", "train")
    df["y"] = df.label.map({c: i for i, c in enumerate(classes)}).astype(int)
    tr, te = df[df.split.isin(["train", "val"])], df[df.split == "test"]
    # 不用 groupby.apply：新版 pandas 會把分組欄位（y）從回傳的 DataFrame 中移除
    per_class = lambda d, fn: pd.concat([fn(g) for _, g in d.groupby("y")])
    va = per_class(tr, lambda g: g.sample(frac=val_frac, random_state=seed))
    tr = tr.drop(va.index)
    cap = lambda d, k: d if not k else per_class(d, lambda g: g.head(k))
    splits = {"train": cap(tr, max_per_class_train), "val": va, "test": cap(te, max_per_class_test)}
    splits = {k: v.sample(frac=1, random_state=seed).reset_index(drop=True) for k, v in splits.items()}
    assert all("y" in v.columns and v.y.notna().all() for v in splits.values()), "label -> y 對應失敗"
    print({k: len(v) for k, v in splits.items()}, "| empty text:",
          {k: f"{(v.text.str.len() == 0).mean():.1%}" for k, v in splits.items()})
    return {"paths": paths, "root": root, "splits": splits, "classes": classes, "feats": {}, "bb": {},
            "models": {}, "hist": {}, "pred": {}, "rows": {}, "tag": f"{max_per_class_train}_{max_per_class_test}"}


def plot_data_overview(ctx):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    cnt = ctx["splits"]["train"].label.value_counts()
    axes[0].bar(range(len(cnt)), cnt.values)
    axes[0].set(title="Train images per class (sorted)", xlabel="class rank", ylabel="# items")
    for s, d in ctx["splits"].items():
        axes[1].hist(np.clip(d.text.str.split().str.len(), 0, 500), bins=50, alpha=0.5, label=s)
    axes[1].set(title="Text length (words, clipped at 500)", xlabel="# words", ylabel="# items")
    axes[1].legend()
    d = ctx["splits"]["train"]
    axes[2].axis("off")
    ex = d.sample(3, random_state=0)
    axes[2].text(0, 1, "Text examples (train)\n\n" + "\n\n".join(f"[{r.label}] " + textwrap.fill(r.text[:150], 55)
                                                               for r in ex.itertuples()), va="top", fontsize=8)
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_data_overview.png")


# ----------------------------------------------------------------------
# Backbone 與特徵
# ----------------------------------------------------------------------
def _clip(ctx):
    if "clip" not in ctx["bb"]:
        from transformers import CLIPModel, CLIPTokenizer
        ctx["bb"]["clip"] = (CLIPModel.from_pretrained(CLIP_NAME).to(device).eval(), CLIPTokenizer.from_pretrained(CLIP_NAME))
    return ctx["bb"]["clip"]


def _tensor(o):
    return o if torch.is_tensor(o) else o.pooler_output


def _img_forward(ctx, name, x):
    if name == "resnet50":
        if name not in ctx["bb"]:
            m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            m.fc = nn.Identity()
            ctx["bb"][name] = m.to(device).eval()
        return ctx["bb"][name](x)
    return _tensor(_clip(ctx)[0].get_image_features(pixel_values=x))


class _ImgDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            return self.tf(Image.open(self.paths[i]).convert("RGB"))
        except Exception:
            return torch.zeros(3, 224, 224)


@torch.no_grad()
def _img_feats(ctx, name, paths, bs=128):
    out = []
    for x in DataLoader(_ImgDS(list(paths), TF[name]), bs, num_workers=4):
        with torch.autocast(**q1._AMP):
            out.append(_img_forward(ctx, name, x.to(device)).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def _txt_feats(ctx, name, texts, bs=256):
    out = []
    texts = [t if t else "none" for t in texts]
    if name == "bert":
        if name not in ctx["bb"]:
            from transformers import AutoModel, AutoTokenizer
            ctx["bb"][name] = (AutoTokenizer.from_pretrained("bert-base-uncased"),
                               AutoModel.from_pretrained("bert-base-uncased").to(device).eval())
        tok, m = ctx["bb"][name]
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            with torch.autocast(**q1._AMP):
                h = m(**b).last_hidden_state
            mask = b["attention_mask"].unsqueeze(-1).float()
            out.append(((h.float() * mask).sum(1) / mask.sum(1)).cpu())
    else:
        model, tok = _clip(ctx)
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
            with torch.autocast(**q1._AMP):
                out.append(_tensor(model.get_text_features(**b)).float().cpu())
    return torch.cat(out)


def extract_all(ctx):
    """對 4 個 backbone 抽特徵並快取到 Data/upmc_food101/feats_*.pt（資料量改變會自動重算）。"""
    for name in list(IMG_BB) + list(TXT_BB):
        f = ctx["paths"]["data"] / "upmc_food101" / f"feats_{name}_{ctx['tag']}.pt"
        f.parent.mkdir(parents=True, exist_ok=True)
        if f.exists():
            d = torch.load(f)
            if all(len(d[s]) == len(df) for s, df in ctx["splits"].items()):
                ctx["feats"][name] = d
                print(f"{name}: loaded from cache {f.name} {tuple(d['train'].shape)}")
                continue
        t0, d = time.time(), {}
        for s, df in ctx["splits"].items():
            d[s] = _img_feats(ctx, name, df.img) if name in IMG_BB else _txt_feats(ctx, name, df.text.tolist())
        torch.save(d, f)
        ctx["feats"][name] = d
        print(f"{name}: {tuple(d['train'].shape)} in {time.time() - t0:.0f}s")


# ----------------------------------------------------------------------
# 融合模型與訓練
# ----------------------------------------------------------------------
class Net(nn.Module):
    """d_img / d_txt = 0 代表不使用該模態；mode：concat / late / gated（僅雙模態時有差異）。"""

    def __init__(self, d_img, d_txt, n_cls, mode="concat", hid=512, drop=0.4):
        super().__init__()
        mk = lambda d: nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop))
        self.bi, self.bt = (mk(d_img) if d_img else None), (mk(d_txt) if d_txt else None)
        self.mode = mode if (d_img and d_txt) else "concat"
        k = int(bool(d_img)) + int(bool(d_txt))
        mlp = lambda d: nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Dropout(drop), nn.Linear(hid, n_cls))
        if self.mode == "concat":
            self.head = mlp(hid * k)
        elif self.mode == "late":
            self.hi, self.ht = nn.Linear(hid, n_cls), nn.Linear(hid, n_cls)
        else:
            self.gate, self.head = nn.Linear(hid * 2, hid), mlp(hid)

    def forward(self, xi, xt):
        zi = self.bi(xi) if self.bi is not None else None
        zt = self.bt(xt) if self.bt is not None else None
        if self.mode == "late":
            return (self.hi(zi) + self.ht(zt)) / 2
        if self.mode == "gated":
            g = torch.sigmoid(self.gate(torch.cat([zi, zt], -1)))
            return self.head(g * zi + (1 - g) * zt)
        return self.head(torch.cat([z for z in (zi, zt) if z is not None], -1))


def _data(ctx, img, txt, split):
    n = len(ctx["splits"][split])
    xi = ctx["feats"][img][split] if img else torch.zeros(n, 0)
    xt = ctx["feats"][txt][split] if txt else torch.zeros(n, 0)
    return xi.to(device), xt.to(device), torch.tensor(ctx["splits"][split].y.values).to(device)


def _eval(model, xi, xt, y):
    model.eval()
    with torch.no_grad():
        lg = model(xi, xt)
    yt, yp = y.cpu().numpy(), lg.argmax(-1).cpu().numpy()
    top5 = lg.topk(5, -1).indices.eq(y[:, None]).any(1).float().mean().item()
    return {"loss": F.cross_entropy(lg, y).item(), "acc": accuracy_score(yt, yp),
            "f1": f1_score(yt, yp, average="macro"), "top5": top5, "yt": yt, "yp": yp}


def _train_one(ctx, img, txt, mode, seed, epochs=40, patience=5, bs=256, lr=1e-3):
    q1.set_seed(seed)
    tr, va, te = (_data(ctx, img, txt, s) for s in ("train", "val", "test"))
    model = Net(tr[0].size(1), tr[1].size(1), len(ctx["classes"]), mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    es, hist = q1.EarlyStopping(patience, "max"), {"train_loss": [], "val_loss": [], "val_f1": []}
    for ep in range(epochs):
        model.train()
        perm, tl = torch.randperm(len(tr[2]), device=device), 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(model(tr[0][b], tr[1][b]), tr[2][b], label_smoothing=0.1)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(b)
        v = _eval(model, *va)
        hist["train_loss"].append(tl / len(perm)); hist["val_loss"].append(v["loss"]); hist["val_f1"].append(v["f1"])
        if es.step(v["f1"], model):
            break
    es.restore(model)
    hist["best_epoch"] = es.best_epoch
    return model, hist, _eval(model, *te)


def exp_name(img, txt, mode="concat"):
    if img and txt:
        return f"{img}+{txt}" + ("" if mode == "concat" else f"|{mode}")
    return f"img:{img}" if img else f"txt:{txt}"


def _run(ctx, cfgs, seeds):
    for img, txt, mode in cfgs:
        name, res = exp_name(img, txt, mode), []
        for sd in seeds:
            model, hist, r = _train_one(ctx, img, txt, mode, sd)
            res.append(r)
            if sd == seeds[0]:
                ctx["models"][name], ctx["hist"][name], ctx["pred"][name] = model, hist, (r["yt"], r["yp"])
                safe = re.sub(r"[^a-z0-9_]+", "_", name.lower())
                torch.save({"state_dict": model.state_dict(), "img": img, "txt": txt, "mode": mode,
                            "classes": ctx["classes"]}, ctx["paths"]["outputs"] / f"q2b_{safe}.pt")
        row = {"Type": "fusion" if img and txt else "single-modal", "Fusion": mode if img and txt else "-"}
        for k, kk in (("acc", "Accuracy"), ("f1", "Macro-F1"), ("top5", "Top5-Acc")):
            row[kk] = np.mean([x[k] for x in res]); row[kk + "_std"] = np.std([x[k] for x in res])
        ctx["rows"][name] = row
        print(name, {k: round(v, 4) for k, v in row.items() if isinstance(v, float)})


def results_df(ctx):
    df = pd.DataFrame(ctx["rows"]).T
    for c in df.columns:
        if c not in ("Type", "Fusion"):
            df[c] = df[c].astype(float).round(4)
    df.to_csv(ctx["paths"]["outputs"] / "q2b_results.csv")
    return df


def run_backbone_grid(ctx, seeds=(0, 1, 2)):
    """4 組單模態 + 2x2 concat 融合。"""
    cfgs = [(i, None, "concat") for i in IMG_BB] + [(None, t, "concat") for t in TXT_BB] + \
           [(i, t, "concat") for i in IMG_BB for t in TXT_BB]
    _run(ctx, cfgs, seeds)
    return results_df(ctx)


def best_pair(ctx):
    d = pd.DataFrame(ctx["rows"]).T
    d = d[(d.Type == "fusion") & (d.Fusion == "concat")]
    return d["Macro-F1"].astype(float).idxmax().split("|")[0]


def run_fusion_study(ctx, pair=None, seeds=(0, 1, 2)):
    """用網格中最佳的 (影像, 文字) 組合，比較 late / gated 與 concat。"""
    pair = pair or best_pair(ctx)
    img, txt = pair.split("+")
    print("fusion study on", pair)
    _run(ctx, [(img, txt, "late"), (img, txt, "gated")], seeds)
    return results_df(ctx)


# ----------------------------------------------------------------------
# 圖表與分析
# ----------------------------------------------------------------------
def plot_training_curves(ctx):
    names = [n for n in ctx["hist"] if "+" in n]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for n in names:
        h = ctx["hist"][n]
        l, = axes[0].plot(h["train_loss"], label=f"{n} train")
        axes[0].plot(h["val_loss"], "--", color=l.get_color())
        axes[1].plot(h["val_f1"], color=l.get_color(), label=f"{n} (best ep {h['best_epoch']})")
    axes[0].set(xlabel="epoch", ylabel="CE loss (label smoothing 0.1)", title="Fusion models: train (solid) / val (dashed)")
    axes[1].set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1 (early-stopping metric)")
    axes[0].legend(fontsize=6); axes[1].legend(fontsize=6)
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_training_curves.png")


def plot_comparison(ctx, df):
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    pal = {"single-modal": "#9ecae1", "concat": "#4C72B0", "late": "#DD8452", "gated": "#55A868"}
    cols = [pal["single-modal"] if t == "single-modal" else pal[f] for t, f in zip(df.Type, df.Fusion)]
    for ax, m in zip(axes, ["Accuracy", "Macro-F1", "Top5-Acc"]):
        ax.bar(df.index, df[m], yerr=df[m + "_std"], color=cols, capsize=3)
        ax.set_title(f"{m} (mean ± std over seeds)")
        ax.set_ylim(max(0, df[m].min() - 0.1), min(1.0, df[m].max() + 0.04))
        ax.tick_params(axis="x", rotation=50)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in pal.values()]
    axes[0].legend(handles, pal.keys(), fontsize=7)
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_model_comparison.png")


def _parts(name):
    base = name.split("|")[0]
    return tuple(base.split("+")) if "+" in base else (None, None)


def plot_class_analysis(ctx, fusion=None):
    """逐類別：影像單模態 vs 文字單模態的準確率，以及融合帶來的增益。"""
    fusion = fusion or best_pair(ctx)
    img, txt = _parts(fusion)
    acc = {}
    for key, n in (("image", f"img:{img}"), ("text", f"txt:{txt}"), ("fusion", fusion)):
        yt, yp = ctx["pred"][n]
        acc[key] = np.array([(yp[yt == c] == c).mean() if (yt == c).any() else np.nan for c in range(len(ctx["classes"]))])
    df = pd.DataFrame(acc, index=ctx["classes"])
    df["gain"] = df.fusion - df[["image", "text"]].max(axis=1)
    df.round(4).to_csv(ctx["paths"]["outputs"] / "q2b_per_class_acc.csv")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = axes[0].scatter(df.image, df.text, c=df.fusion, cmap="viridis", s=25)
    axes[0].plot([0, 1], [0, 1], "k:", lw=0.7)
    for n in df.fusion.nsmallest(5).index:
        axes[0].annotate(n, (df.image[n], df.text[n]), fontsize=7)
    axes[0].set(xlabel=f"image-only accuracy ({img})", ylabel=f"text-only accuracy ({txt})",
                title="Per-class accuracy: image vs text (color = fusion)")
    plt.colorbar(sc, ax=axes[0])
    g = pd.concat([df.gain.nlargest(10), df.gain.nsmallest(5)]).sort_values()
    axes[1].barh(g.index, g.values, color=["#C44E52" if v < 0 else "#55A868" for v in g.values])
    axes[1].set(title="Fusion gain over best unimodal (top 10 / bottom 5 classes)", xlabel="accuracy gain")
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_class_analysis.png")
    return df


def modality_ablation(ctx, fusion=None):
    """推論時把某一模態的輸入置零，觀察融合模型依賴哪個模態。"""
    fusion = fusion or best_pair(ctx)
    img, txt = _parts(fusion)
    model = ctx["models"][fusion]
    xi, xt, y = _data(ctx, img, txt, "test")
    res = {"both": _eval(model, xi, xt, y), "image masked": _eval(model, torch.zeros_like(xi), xt, y),
           "text masked": _eval(model, xi, torch.zeros_like(xt), y)}
    df = pd.DataFrame({k: {"Accuracy": v["acc"], "Macro-F1": v["f1"]} for k, v in res.items()}).T.round(4)
    ax = df.plot.bar(rot=0, figsize=(6, 4))
    ax.set(title=f"Modality ablation at inference ({fusion})", ylim=(0, 1))
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_modality_ablation.png")
    return df


def plot_top_confusions(ctx, name=None, k=12):
    name = name or best_pair(ctx)
    yt, yp = ctx["pred"][name]
    n = len(ctx["classes"])
    cm = np.zeros((n, n), int)
    np.add.at(cm, (yt, yp), 1)
    np.fill_diagonal(cm, 0)
    idx = np.dstack(np.unravel_index(np.argsort(cm.ravel())[::-1][:k], cm.shape))[0]
    labels = [f"{ctx['classes'][a]} -> {ctx['classes'][b]}" for a, b in idx][::-1]
    plt.figure(figsize=(8, 4.5))
    plt.barh(labels, [cm[a, b] for a, b in idx][::-1], color="#C44E52")
    plt.xlabel("# test items"); plt.title(f"Most frequent confusions ({name})")
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_top_confusions.png")


# ----------------------------------------------------------------------
# 端到端推論：原始影像 + 原始文字 -> 類別
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(ctx, name, pil, text, topk=3):
    img, txt = _parts(name) if "+" in name else ((name[4:], None) if name.startswith("img:") else (None, name[4:]))
    if img:
        x = TF[img](pil.convert("RGB")).unsqueeze(0).to(device)
        with torch.autocast(**q1._AMP):
            xi = _img_forward(ctx, img, x).float()
    else:
        xi = torch.zeros(1, 0, device=device)
    xt = _txt_feats(ctx, txt, [text]).to(device) if txt else torch.zeros(1, 0, device=device)
    p = torch.softmax(ctx["models"][name].eval()(xi, xt), -1)[0]
    v, i = p.topk(topk)
    return [(ctx["classes"][j], float(a)) for a, j in zip(v, i)]


def plot_predictions(ctx, fusion=None, n=6, seed=5):
    fusion = fusion or best_pair(ctx)
    img, txt = _parts(fusion)
    te = ctx["splits"]["test"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, i in zip(axes.ravel(), np.random.RandomState(seed).choice(len(te), n, replace=False)):
        r = te.iloc[i]
        pil = Image.open(r.img).convert("RGB")
        f, im, tx = (predict(ctx, fusion, pil, r.text)[0], predict(ctx, f"img:{img}", pil, r.text)[0],
                     predict(ctx, f"txt:{txt}", pil, r.text)[0])
        ax.imshow(pil); ax.axis("off")
        ok = f[0] == r.label
        ax.set_title(f"[{'OK' if ok else 'X'}] true: {r.label}\nfusion: {f[0]} ({f[1]:.2f})\n"
                     f"image-only: {im[0]} | text-only: {tx[0]}\n" + textwrap.fill(r.text[:140], 62),
                     fontsize=8, loc="left", color="green" if ok else "red")
    plt.suptitle(f"End-to-end multimodal inference ({fusion})")
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q2b_predictions.png")
