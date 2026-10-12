"""
Quiz 4：Image Captioning（Flickr8k）
方法 A：ResNet-50 特徵 + LSTM 解碼器（自訓練）
方法 B：BLIP（預訓練）
評估：BLEU、BERTScore、延遲；Gemini 只負責「判斷描述是否符合影像」
圖片輸出 -> Outputs/Q4_*.png ；模型輸出 -> Outputs/q4_*.pt
"""
import collections
import copy
import json
import random
import re
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

import Quiz1 as q1

device = "cuda" if torch.cuda.is_available() else "cpu"
PAD, START, END, UNK = 0, 1, 2, 3
MAX_CAP = 30
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
feat_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
_AMP = dict(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda"))


def cap_tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def norm(s):
    return " ".join(cap_tok(s))


def _sync():
    if device == "cuda":
        torch.cuda.synchronize()


def _save(ctx, name):
    out = ctx["paths"]["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("[saved]", out)


# ----------------------------------------------------------------------
# 資料 & 特徵
# ----------------------------------------------------------------------
def load_context(paths):
    fl = paths["data"] / "flickr8k"
    with open(fl / "captions.json", encoding="utf-8") as f:
        data = json.load(f)
    return {"paths": paths, "fl": fl, "data": data}


class _ImgDS(Dataset):
    def __init__(self, fl, items):
        self.fl, self.items = fl, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return feat_tf(Image.open(self.fl / self.items[i]["file"]).convert("RGB"))


def extract_features(ctx):
    """用 ImageNet 預訓練 ResNet-50 抽 2048 維特徵並快取。"""
    cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    cnn.fc = nn.Identity()
    ctx["cnn"] = cnn.to(device).eval()
    f = ctx["fl"] / "feats_resnet50.pt"
    if f.exists():
        ctx["feats"] = torch.load(f)
        return
    feats = {}
    with torch.no_grad():
        for split, items in ctx["data"].items():
            out = []
            for x in DataLoader(_ImgDS(ctx["fl"], items), 64, num_workers=2):
                with torch.autocast(**_AMP):
                    out.append(ctx["cnn"](x.to(device)).float().cpu())
            feats[split] = torch.cat(out)
            print(split, feats[split].shape)
    torch.save(feats, f)
    ctx["feats"] = feats


# ----------------------------------------------------------------------
# 方法 A：CNN + LSTM
# ----------------------------------------------------------------------
def build_vocab(ctx, min_freq=5):
    cnt = collections.Counter(w for it in ctx["data"]["train"] for c in it["captions"]
                              for w in cap_tok(c))
    ctx["itos"] = ["<pad>", "<start>", "<end>", "<unk>"] + [w for w, c in cnt.items() if c >= min_freq]
    ctx["stoi"] = {w: i for i, w in enumerate(ctx["itos"])}
    print("caption vocab:", len(ctx["itos"]))


def _enc(ctx, s):
    ids = [ctx["stoi"].get(w, UNK) for w in cap_tok(s)][:MAX_CAP]
    ids = [START] + ids + [END]
    return ids + [PAD] * (MAX_CAP + 2 - len(ids))


def _pairs(ctx, split):
    idx, caps = [], []
    for i, it in enumerate(ctx["data"][split]):
        for c in it["captions"]:
            idx.append(i); caps.append(_enc(ctx, c))
    return torch.tensor(idx), torch.tensor(caps)


class CapDecoder(nn.Module):
    """圖像特徵 -> 初始化 LSTM 狀態，且每個時間步都串接投影後的圖像向量。"""

    def __init__(self, vocab, feat=2048, emb=128, hid=256, drop=0.5):
        super().__init__()
        self.img_proj = nn.Sequential(nn.Linear(feat, emb), nn.ReLU(), nn.Dropout(drop))
        self.init_h, self.init_c = nn.Linear(feat, hid), nn.Linear(feat, hid)
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb * 2, hid, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.out = nn.Linear(hid, vocab)

    def init_state(self, f):
        return torch.tanh(self.init_h(f)).unsqueeze(0), torch.tanh(self.init_c(f)).unsqueeze(0)

    def forward(self, f, caps):
        h, c = self.init_state(f)
        img = self.img_proj(f).unsqueeze(1).expand(-1, caps.size(1), -1)
        o, _ = self.lstm(torch.cat([self.drop(self.emb(caps)), img], -1), (h, c))
        return self.out(self.drop(o))


@torch.no_grad()
def generate(ctx, f, max_len=MAX_CAP):
    """貪婪解碼，f: (B, 2048)。"""
    model = ctx["dec"].eval()
    h, c = model.init_state(f)
    img = model.img_proj(f)
    tok = torch.full((f.size(0),), START, device=f.device)
    outs = []
    for _ in range(max_len):
        o, (h, c) = model.lstm(torch.cat([model.emb(tok), img], -1).unsqueeze(1), (h, c))
        tok = model.out(o.squeeze(1)).argmax(-1)
        outs.append(tok)
    caps = []
    for seq in torch.stack(outs, 1).cpu().tolist():
        words = []
        for t in seq:
            if t == END:
                break
            words.append(ctx["itos"][t])
        caps.append(" ".join(words))
    return caps


def _run_epoch(model, feats, pairs, opt=None, bs=128):
    idx_all, caps_all = pairs
    perm = torch.randperm(len(idx_all)) if opt else torch.arange(len(idx_all))
    model.train(opt is not None)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            f, caps = feats[idx_all[b]].to(device), caps_all[b].to(device)
            logits = model(f, caps[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), caps[:, 1:].reshape(-1))
            if opt:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += loss.item() * len(b); n += len(b)
    return total / n


def train_decoder(ctx, epochs=25, patience=3, lr=1e-3):
    """訓練方法 A 解碼器，early stopping，存成 q4_cnn_lstm.pt。"""
    q1.set_seed(42)
    tr_pairs, va_pairs = _pairs(ctx, "train"), _pairs(ctx, "val")
    dec = CapDecoder(len(ctx["itos"])).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=lr)
    hist, best, best_state, bad, t0 = {"tr": [], "va": []}, 1e9, None, 0, time.time()
    for ep in range(epochs):
        trl = _run_epoch(dec, ctx["feats"]["train"], tr_pairs, opt)
        val = _run_epoch(dec, ctx["feats"]["val"], va_pairs)
        hist["tr"].append(trl); hist["va"].append(val)
        print(f"ep{ep+1} train_loss={trl:.4f} val_loss={val:.4f}")
        if val < best:
            best, best_state, bad = val, copy.deepcopy(dec.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                print("early stop"); break
    dec.load_state_dict(best_state)
    ctx["dec"], ctx["dec_train_time"] = dec, time.time() - t0
    out = ctx["paths"]["outputs"] / "q4_cnn_lstm.pt"
    torch.save({"state_dict": dec.state_dict(), "vocab": ctx["itos"]}, out)
    print("[saved]", out)

    plt.figure(figsize=(6, 4))
    plt.plot(hist["tr"], label="train"); plt.plot(hist["va"], label="val")
    plt.xlabel("epoch"); plt.ylabel("CE loss"); plt.title("Captioning decoder training"); plt.legend()
    _save(ctx, "Q4_training_curve.png")
    return hist


@torch.no_grad()
def caption_A(ctx, pil):
    x = feat_tf(pil).unsqueeze(0).to(device)
    with torch.autocast(**_AMP):
        f = ctx["cnn"](x).float()
    return generate(ctx, f)[0]


# ----------------------------------------------------------------------
# 方法 B：BLIP
# ----------------------------------------------------------------------
def load_blip(ctx, name="Salesforce/blip-image-captioning-base"):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    ctx["blip_proc"] = BlipProcessor.from_pretrained(name)
    ctx["blip"] = BlipForConditionalGeneration.from_pretrained(name).to(device).eval()


@torch.no_grad()
def caption_B(ctx, pil):
    inp = ctx["blip_proc"](images=pil, return_tensors="pt").to(device)
    out = ctx["blip"].generate(**inp, max_new_tokens=30, num_beams=3)
    return ctx["blip_proc"].decode(out[0], skip_special_tokens=True)


# ----------------------------------------------------------------------
# Gemini：僅用於評判（描述是否符合影像）
# ----------------------------------------------------------------------
def setup_gemini(ctx, api_key, model="gemini-2.5-flash", sleep=4):
    ctx["has_gemini"] = bool(api_key)
    if not api_key:
        print("沒有 GEMINI_API_KEY，將略過 Gemini 相關步驟。")
        return
    from google import genai
    ctx["gclient"], ctx["gmodel"], ctx["gsleep"] = genai.Client(api_key=api_key), model, sleep


def gemini_call(ctx, contents, json_mode=False, retries=5):
    from google.genai import types
    cfg = types.GenerateContentConfig(response_mime_type="application/json") if json_mode else None
    for k in range(retries):
        try:
            r = ctx["gclient"].models.generate_content(model=ctx["gmodel"], contents=contents, config=cfg)
            time.sleep(ctx["gsleep"])   # 避免超過速率限制
            return r.text
        except Exception as e:
            print("Gemini error, retrying:", str(e)[:200])
            time.sleep(2 ** k * 2)
    return None


def gemini_judge(ctx, pil, caption):
    prompt = ("You are an image-caption judge. Decide whether the caption accurately describes the image.\n"
              f'Caption: "{caption}"\n'
              'Reply with JSON only: {"match": true or false, "score": integer 1-5, "reason": "one short sentence"}')
    txt = gemini_call(ctx, [pil, prompt], json_mode=True)
    try:
        d = json.loads(txt)
        return bool(d["match"]), int(d["score"]), str(d.get("reason", ""))
    except Exception:
        return None, None, ""


# ----------------------------------------------------------------------
# 評估流程
# ----------------------------------------------------------------------
def build_eval_set(ctx, n_eval=100, seed=42):
    items = random.Random(seed).sample(ctx["data"]["test"], n_eval)
    ctx["eval_items"] = items
    ctx["eval_imgs"] = [Image.open(ctx["fl"] / it["file"]).convert("RGB") for it in items]
    ctx["refs"] = [it["captions"] for it in items]


def run_method(fn, imgs):
    """batch size = 1，計時 PIL 圖片 -> 文字（端到端）。先 warm-up 一次。"""
    caps, lats = [], []
    fn(imgs[0])
    for im in imgs:
        _sync(); t = time.perf_counter()
        caps.append(fn(im))
        _sync(); lats.append(time.perf_counter() - t)
    return caps, lats


def generate_all(ctx, only=None):
    """方法 A、B 產生描述並計時。only=["B_BLIP"] 可只重跑指定方法。"""
    methods = {"A_CNN+LSTM": lambda im: caption_A(ctx, im), "B_BLIP": lambda im: caption_B(ctx, im)}
    if only:
        methods = {k: v for k, v in methods.items() if k in only}
    ctx.setdefault("caps", {}); ctx.setdefault("lat", {})
    for name, fn in methods.items():
        print("running", name)
        ctx["caps"][name], ctx["lat"][name] = run_method(fn, ctx["eval_imgs"])
        n_empty = sum(1 for c in ctx["caps"][name] if not c.strip())
        print(f"  avg latency {np.mean(ctx['lat'][name]):.3f}s | empty captions: {n_empty} "
              f"| e.g. {ctx['caps'][name][0]}")
    _save_captions(ctx)


def _save_captions(ctx):
    pd.DataFrame({"file": [it["file"] for it in ctx["eval_items"]], **ctx["caps"]}).to_csv(
        ctx["paths"]["outputs"] / "q4_generated_captions.csv", index=False)


def text_metrics(ctx, rescale=False):
    """
    BLEU-4（sacrebleu，多參考）與 BERTScore F1（多參考取最高）。
    空描述視為生成失敗：以佔位字串 "empty" 計分（會被自然扣分），並於 Empty_captions 欄位記錄張數。
    rescale=True 時 BERTScore 以 baseline 重新縮放，差異較容易看出（數值不再接近 0.9x）。
    """
    import sacrebleu
    from bert_score import score as bert_score_fn
    refs_n = [[norm(c) for c in r] for r in ctx["refs"]]
    n_ref = min(len(r) for r in refs_n)
    streams = [[r[k] for r in refs_n] for k in range(n_ref)]
    rows = {}
    for name, caps in ctx["caps"].items():
        n_empty = sum(1 for c in caps if not norm(c))
        hyps = [norm(c) or "empty" for c in caps]   # 空字串會讓 bert_score 出錯
        bleu = sacrebleu.corpus_bleu(hyps, streams, lowercase=True).score
        _, _, F1 = bert_score_fn(hyps, refs_n, lang="en", device=device, batch_size=32,
                                 verbose=False, rescale_with_baseline=rescale)
        rows[name] = {"BLEU-4": bleu, "BERTScore_F1": F1.mean().item(),
                      "Latency_s": float(np.mean(ctx["lat"][name])), "Empty_captions": n_empty}
        print(name, rows[name])
    return pd.DataFrame(rows).T


def judge_all(ctx, n_judge=50):
    rows = []
    for name, caps in ctx["caps"].items():
        for i in range(n_judge):
            if not caps[i].strip():
                m, s, r = None, None, "empty caption (skipped)"
            else:
                m, s, r = gemini_judge(ctx, ctx["eval_imgs"][i], caps[i])
            rows.append({"method": name, "idx": i, "caption": caps[i], "match": m, "score": s, "reason": r})
        print("judged", name)
    df = pd.DataFrame(rows)
    df.to_csv(ctx["paths"]["outputs"] / "q4_gemini_judgement.csv", index=False)
    ctx["judge_df"] = df
    return df.dropna(subset=["match"]).groupby("method").agg(
        Gemini_match_rate=("match", "mean"), Gemini_mean_score=("score", "mean"))


def summarize_and_plot(ctx, metric_df, n_judge=50):
    metric_df = metric_df.round(4)
    metric_df.to_csv(ctx["paths"]["outputs"] / "q4_results.csv")

    cols = [c for c in ["BLEU-4", "BERTScore_F1", "Gemini_match_rate", "Latency_s"] if c in metric_df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 3.8))
    for ax, c in zip(np.atleast_1d(axes), cols):
        metric_df[c].plot.bar(ax=ax, rot=20, color=["#4C72B0", "#55A868", "#C44E52"][:len(metric_df)])
        ax.set_title(c)
        if c == "Latency_s":
            ax.set_yscale("log"); ax.set_ylabel("sec / image (log)")
    plt.tight_layout()
    _save(ctx, "Q4_method_comparison.png")

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    jd = ctx.get("judge_df")
    for ax, i in zip(axes.ravel(), range(6)):
        ax.imshow(ctx["eval_imgs"][i]); ax.axis("off")
        lines = []
        for name, caps in ctx["caps"].items():
            mark = ""
            if jd is not None and i < n_judge:
                row = jd[(jd.method == name) & (jd.idx == i)].iloc[0]
                mark = "" if pd.isna(row["match"]) else (" [OK]" if bool(row["match"]) else " [X]")
            lines.append(textwrap.fill(f"{name[0]}: {caps[i]}{mark}", 55))
        ax.set_title("\n".join(lines), fontsize=8, loc="left")
    plt.tight_layout()
    _save(ctx, "Q4_caption_examples.png")
    return metric_df
