"""
Quiz 1：Oxford Buildings (Oxford5k) 影像檢索 —— CNN (ResNet-50) vs ViT (ViT-B/16)
本檔同時提供 Quiz2 / Quiz3 共用工具：set_seed、get_paths、download、EarlyStopping、save_fig。

評估協定（自訂，避免訓練/測試洩漏）：
  - 取 11 個地標的 good+ok 影像作為「有標註影像」，每個地標依 50% / 15% / 35% 切成 train / val / test
  - 另外抽 400 / 100 張 distractor（無標註影像）作為「其他」類別，加入 train / val，並自檢索庫移除
  - 檢索庫 gallery = test 有標註影像 + 其餘 distractor；query = test 有標註影像（以影像本身查詢，排除自己）
  - 指標：mAP、Recall@1、Recall@5
圖片輸出 -> Outputs/Q1_*.png ；模型輸出 -> Outputs/q1_*.pt
"""
import copy
import random
import re
import tarfile
import time
import urllib.request
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_AMP = dict(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda"))
OX_URL = "https://www.robots.ox.ac.uk/~vgg/data/oxbuildings/"


# ----------------------------------------------------------------------
# 共用工具（Quiz2 / Quiz3 也會 import）
# ----------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_paths(root=None):
    """root 預設為目前工作目錄；若在 Codes/ 底下執行則自動往上一層。"""
    root = Path(root or Path.cwd()).resolve()
    if root.name == "Codes":
        root = root.parent
    paths = {"root": root, "codes": root / "Codes", "data": root / "Data", "outputs": root / "Outputs"}
    for k in ("data", "outputs"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def download(url, dst):
    """串流下載並顯示進度；檔案已存在則略過。"""
    dst = Path(dst)
    if dst.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(req) as r, open(tmp, "wb") as f:
        total, got, last = int(r.headers.get("Content-Length", 0)), 0, -1
        while True:
            buf = r.read(1 << 20)
            if not buf:
                break
            f.write(buf)
            got += len(buf)
            if total and int(got * 10 / total) != last:
                last = int(got * 10 / total)
                print(f"  {last * 10}%")
    tmp.rename(dst)
    return dst


def untar(src, dst):
    with tarfile.open(src) as t:
        try:
            t.extractall(dst, filter="data")
        except TypeError:       # 舊版 Python 沒有 filter 參數
            t.extractall(dst)


class EarlyStopping:
    """mode='min' 看 loss，mode='max' 看 F1/accuracy。step() 回傳 True 代表該停止。"""

    def __init__(self, patience=4, mode="min", min_delta=1e-4):
        self.patience, self.mode, self.min_delta = patience, mode, min_delta
        self.best = np.inf if mode == "min" else -np.inf
        self.bad, self.state, self.best_epoch, self.epoch = 0, None, 0, 0

    def step(self, value, model):
        self.epoch += 1
        better = value < self.best - self.min_delta if self.mode == "min" else value > self.best + self.min_delta
        if better:
            self.best, self.bad, self.best_epoch = value, 0, self.epoch
            self.state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.bad += 1
        return self.bad >= self.patience

    def restore(self, model):
        if self.state is not None:
            model.load_state_dict(self.state)


def save_fig(paths, name):
    out = paths["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("[saved]", out)


# ----------------------------------------------------------------------
# 資料：下載 Oxford5k、解析 ground truth、切分
# ----------------------------------------------------------------------
def prepare_oxford(paths, seed=42, n_dis_train=400, n_dis_val=100):
    root = paths["data"] / "oxford5k"
    img_dir, gt_dir = root / "images", root / "gt"
    if not any(img_dir.glob("*.jpg")):
        f = download(OX_URL + "oxbuild_images-v1.tgz", root / "oxbuild_images-v1.tgz")
        img_dir.mkdir(parents=True, exist_ok=True)
        untar(f, img_dir)
        f.unlink()
    if not any(gt_dir.glob("*.txt")):
        f = download(OX_URL + "gt_files_170407.tgz", root / "gt_files_170407.tgz")
        gt_dir.mkdir(parents=True, exist_ok=True)
        untar(f, gt_dir)
        f.unlink()

    all_names = sorted(p.stem for p in img_dir.glob("*.jpg"))
    exist = set(all_names)
    pat = re.compile(r"^(.*)_(\d+)_(good|ok|junk|query)\.txt$")
    pos, junk = {}, set()
    for f in sorted(gt_dir.rglob("*.txt")):
        m = pat.match(f.name)
        if not m or m.group(3) == "query":
            continue
        names = [l.split()[0] for l in f.read_text().splitlines() if l.strip()]
        if m.group(3) == "junk":
            junk.update(names)
        else:
            pos.setdefault(m.group(1), set()).update(n for n in names if n in exist)
    landmarks = sorted(pos)
    labeled = {}
    for li, lm in enumerate(landmarks):
        for n in sorted(pos[lm]):
            labeled.setdefault(n, li)
    distractors = [n for n in all_names if n not in labeled and n not in junk]

    rng = random.Random(seed)
    train, val, test = [], [], []
    for li in range(len(landmarks)):
        ns = [n for n, l in labeled.items() if l == li]
        rng.shuffle(ns)
        a, b = int(0.5 * len(ns)), int(0.65 * len(ns))
        train += [(n, li) for n in ns[:a]]
        val += [(n, li) for n in ns[a:b]]
        test += [(n, li) for n in ns[b:]]
    rng.shuffle(distractors)
    other = len(landmarks)
    train += [(n, other) for n in distractors[:n_dis_train]]
    val += [(n, other) for n in distractors[n_dis_train:n_dis_train + n_dis_val]]
    rest = distractors[n_dis_train + n_dis_val:]

    ctx = {"paths": paths, "img_dir": img_dir, "landmarks": landmarks, "n_cls": other + 1,
           "train": train, "val": val,
           "test_names": [n for n, _ in test], "test_labels": [l for _, l in test],
           "gal_names": [n for n, _ in test] + rest,
           "gal_labels": np.array([l for _, l in test] + [-1] * len(rest)),
           "models": {}, "zs": {}, "emb": {}, "hist": {}}
    print(f"landmarks={len(landmarks)} | train={len(train)} val={len(val)} queries={len(test)} "
          f"gallery={len(ctx['gal_names'])} (distractors in gallery={len(rest)})")
    return ctx


def plot_dataset_overview(ctx):
    lm = ctx["landmarks"] + ["other (distractor)"]
    cnt = {k: np.zeros(len(lm), int) for k in ("train", "val")}
    for k in cnt:
        for _, l in ctx[k]:
            cnt[k][l] += 1
    cnt["test"] = np.bincount(ctx["test_labels"], minlength=len(lm))
    fig, ax = plt.subplots(figsize=(10, 4))
    bottom = np.zeros(len(lm))
    for k, c in zip(("train", "val", "test"), ("#4C72B0", "#DD8452", "#55A868")):
        ax.bar(lm, cnt[k], bottom=bottom, label=k, color=c)
        bottom += cnt[k]
    ax.set_ylabel("# images")
    ax.set_title("Oxford5k labeled images per landmark (train / val / test)")
    plt.xticks(rotation=40, ha="right")
    ax.legend()
    plt.tight_layout()
    save_fig(ctx["paths"], "Q1_dataset_overview.png")


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
train_tf = T.Compose([T.RandomResizedCrop(224, scale=(0.5, 1.0)), T.RandomHorizontalFlip(),
                      T.ColorJitter(0.2, 0.2, 0.2), T.ToTensor(), T.Normalize(MEAN, STD)])
eval_tf = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize(MEAN, STD)])


class _DS(Dataset):
    def __init__(self, d, items, tf):
        self.d, self.items, self.tf = d, items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        n, l = self.items[i]
        return self.tf(Image.open(self.d / f"{n}.jpg").convert("RGB")), l


class Embedder(nn.Module):
    """backbone -> (可選) 線性投影 -> L2 normalize；訓練時用 cosine classifier（11 地標 + other）。"""

    def __init__(self, kind, n_cls, dim=256):
        super().__init__()
        self.kind = kind
        if kind == "cnn":
            self.backbone, feat = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2), 2048
            self.backbone.fc = nn.Identity()
        else:
            self.backbone, feat = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1), 768
            self.backbone.heads = nn.Identity()
        self.proj = nn.Linear(feat, dim)
        self.cls = nn.Parameter(torch.randn(n_cls, dim) * 0.02)
        self.params_m = sum(p.numel() for p in self.backbone.parameters()) / 1e6

    def embed(self, x, tuned=True):
        f = self.backbone(x)
        return F.normalize(self.proj(f) if tuned else f, dim=-1)

    def forward(self, x):
        return 20.0 * self.embed(x) @ F.normalize(self.cls, dim=-1).T

    def train(self, mode=True):
        super().train(mode)
        if self.kind == "cnn":      # 小 batch，凍結 BN 統計量
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self


def _set_trainable(model):
    for p in model.backbone.parameters():
        p.requires_grad = False
    if model.kind == "cnn":
        mods = [model.backbone.layer4]
    else:
        mods = list(model.backbone.encoder.layers)[-2:] + [model.backbone.encoder.ln]
    for m in mods:
        for p in m.parameters():
            p.requires_grad = True


def train_model(ctx, kind, epochs=20, patience=4, bs=32, lr_bb=5e-5, lr_head=1e-3):
    """微調最後階段 + 投影 + cosine 分類頭，early stopping（val loss），存成 q1_{kind}.pt。"""
    set_seed(42)
    model = Embedder(kind, ctx["n_cls"]).to(device)
    _set_trainable(model)
    dl_tr = DataLoader(_DS(ctx["img_dir"], ctx["train"], train_tf), bs, shuffle=True, num_workers=2, drop_last=True)
    dl_va = DataLoader(_DS(ctx["img_dir"], ctx["val"], eval_tf), 64, num_workers=2)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    head = list(model.proj.parameters()) + [model.cls]
    opt = torch.optim.AdamW([{"params": bb, "lr": lr_bb}, {"params": head, "lr": lr_head}], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit, es = nn.CrossEntropyLoss(), EarlyStopping(patience, "min")
    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            with torch.autocast(**_AMP):
                loss = crit(model(x).float(), y)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tl += loss.item() * len(y)
            n += len(y)
        model.eval()
        vl, correct, m = 0.0, 0, 0
        with torch.no_grad():
            for x, y in dl_va:
                x, y = x.to(device), y.to(device)
                with torch.autocast(**_AMP):
                    lg = model(x).float()
                vl += crit(lg, y).item() * len(y)
                correct += (lg.argmax(-1) == y).sum().item()
                m += len(y)
        hist["train_loss"].append(tl / n)
        hist["val_loss"].append(vl / m)
        hist["val_acc"].append(correct / m)
        print(f"[{kind}] ep{ep + 1} train_loss={tl / n:.4f} val_loss={vl / m:.4f} val_acc={correct / m:.3f}")
        if es.step(vl / m, model):
            print("early stop, best epoch =", es.best_epoch)
            break
    es.restore(model)
    hist["best_epoch"], hist["train_time_s"] = es.best_epoch, time.time() - t0
    out = ctx["paths"]["outputs"] / f"q1_{kind}.pt"
    torch.save({"state_dict": model.state_dict(), "kind": kind, "landmarks": ctx["landmarks"]}, out)
    print("[saved]", out)
    ctx["models"][kind], ctx["hist"][kind] = model, hist
    return hist


# ----------------------------------------------------------------------
# 檢索與評估
# ----------------------------------------------------------------------
@torch.no_grad()
def embed_names(ctx, model, names, tuned, bs=64):
    model.eval()
    dl = DataLoader(_DS(ctx["img_dir"], [(n, 0) for n in names], eval_tf), bs, num_workers=2)
    out = []
    for x, _ in dl:
        with torch.autocast(**_AMP):
            out.append(model.embed(x.to(device), tuned).float().cpu())
    return torch.cat(out)


def _rank(G, qi):
    """回傳 query 對 gallery 的排序（排除自己）。"""
    s = G @ G[qi]
    s[qi] = -1e9
    return torch.argsort(s, descending=True)[:-1].numpy()


def retrieval_metrics(ctx, G):
    gl, nq = ctx["gal_labels"], len(ctx["test_names"])
    aps, r1, r5 = [], [], []
    for qi in range(nq):
        rel = gl[_rank(G, qi)] == gl[qi]
        hits = np.cumsum(rel)
        aps.append((hits[rel] / (np.nonzero(rel)[0] + 1)).mean())
        r1.append(rel[:1].any())
        r5.append(rel[:5].any())
    return np.array(aps), np.mean(r1), np.mean(r5)


def evaluate_all(ctx):
    """CNN / ViT × (zero-shot ImageNet 特徵, 微調後) 共四組，輸出 mAP / R@1 / R@5。"""
    rows, per_lm = [], {}
    for kind in ("cnn", "vit"):
        for tuned in (False, True):
            if tuned and kind not in ctx["models"]:
                continue
            if not tuned and kind not in ctx["zs"]:
                ctx["zs"][kind] = Embedder(kind, ctx["n_cls"]).to(device)
            model = ctx["models"][kind] if tuned else ctx["zs"][kind]
            t0 = time.time()
            G = embed_names(ctx, model, ctx["gal_names"], tuned)
            ctx["emb"][(kind, tuned)] = G
            aps, r1, r5 = retrieval_metrics(ctx, G)
            name = f"{'ResNet-50' if kind == 'cnn' else 'ViT-B/16'} ({'fine-tuned' if tuned else 'zero-shot'})"
            rows.append({"Model": name, "mAP": aps.mean(), "Recall@1": r1, "Recall@5": r5,
                         "Params_M": model.params_m, "Embed_s_per_img": (time.time() - t0) / len(G)})
            lab = np.array(ctx["test_labels"])
            per_lm[name] = [aps[lab == i].mean() for i in range(len(ctx["landmarks"]))]
            print(rows[-1])
    df = pd.DataFrame(rows).set_index("Model").round(4)
    df.to_csv(ctx["paths"]["outputs"] / "q1_results.csv")
    ctx["per_lm"] = pd.DataFrame(per_lm, index=ctx["landmarks"])
    ctx["per_lm"].round(4).to_csv(ctx["paths"]["outputs"] / "q1_per_landmark_map.csv")
    return df


# ----------------------------------------------------------------------
# 圖表
# ----------------------------------------------------------------------
def plot_training_curves(ctx):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for kind, label in (("cnn", "ResNet-50"), ("vit", "ViT-B/16")):
        if kind in ctx["hist"]:
            h = ctx["hist"][kind]
            axes[0].plot(h["train_loss"], label=f"{label} train")
            axes[0].plot(h["val_loss"], "--", label=f"{label} val")
            axes[1].plot(h["val_acc"], label=label)
            axes[0].axvline(h["best_epoch"] - 1, color="gray", lw=0.5, ls=":")
    axes[0].set(xlabel="epoch", ylabel="CE loss", title="Fine-tuning loss (dotted = best epoch)")
    axes[1].set(xlabel="epoch", ylabel="accuracy", title="Validation accuracy (12-way)")
    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    save_fig(ctx["paths"], "Q1_training_curves.png")


def plot_results(ctx, df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    for ax, c in zip(axes[:2], ["mAP", "Recall@1"]):
        df[c].plot.bar(ax=ax, rot=20, color=["#9ecae1", "#4C72B0", "#fdae6b", "#DD8452"][:len(df)])
        ax.set_title(c)
        ax.set_ylim(0, 1)
        for i, v in enumerate(df[c]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ctx["per_lm"].plot.bar(ax=axes[2], width=0.85)
    axes[2].set(title="Per-landmark AP", ylabel="AP")
    axes[2].legend(fontsize=6)
    axes[2].tick_params(axis="x", rotation=60)
    plt.tight_layout()
    save_fig(ctx["paths"], "Q1_method_comparison.png")


def plot_retrieval_examples(ctx, n_q=3, k=5, seed=1):
    """每個模型（微調後）顯示 n_q 個 query 與 top-k；綠框=同地標、紅框=錯誤。"""
    qs = np.random.RandomState(seed).choice(len(ctx["test_names"]), n_q, replace=False)
    combos = [(kd, lb) for kd, lb in (("cnn", "ResNet-50"), ("vit", "ViT-B/16")) if (kd, True) in ctx["emb"]]
    fig, axes = plt.subplots(n_q * len(combos), k + 1, figsize=(2.3 * (k + 1), 2.3 * n_q * len(combos)))
    gl = ctx["gal_labels"]
    r = 0
    for kind, label in combos:
        G = ctx["emb"][(kind, True)]
        for qi in qs:
            order = _rank(G, qi)[:k]
            ax = axes[r, 0]
            ax.imshow(Image.open(ctx["img_dir"] / f"{ctx['gal_names'][qi]}.jpg"))
            ax.set_title(f"{label} query\n{ctx['landmarks'][gl[qi]]}", fontsize=7)
            for s in ax.spines.values():
                s.set_edgecolor("black"); s.set_linewidth(3)
            for j, gi in enumerate(order):
                ax = axes[r, j + 1]
                ax.imshow(Image.open(ctx["img_dir"] / f"{ctx['gal_names'][gi]}.jpg"))
                ok = gl[gi] == gl[qi]
                ax.set_title(f"#{j + 1} {'correct' if ok else 'wrong'}", fontsize=7)
                for s in ax.spines.values():
                    s.set_edgecolor("green" if ok else "red"); s.set_linewidth(3)
            r += 1
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    save_fig(ctx["paths"], "Q1_retrieval_examples.png")


@torch.no_grad()
def search(ctx, kind, image_path, k=5):
    """單張新影像推論：回傳 gallery 中最相似的 k 張檔名與相似度（使用微調後模型）。"""
    model = ctx["models"][kind].eval()
    x = eval_tf(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    q = model.embed(x, True).float().cpu()[0]
    s = ctx["emb"][(kind, True)] @ q
    v, i = s.topk(k)
    return [(ctx["gal_names"][j], float(a)) for a, j in zip(v, i)]
