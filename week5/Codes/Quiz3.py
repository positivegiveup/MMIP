"""
Quiz 3：CLIP Image Captioning（Flickr8k，Hugging Face: jxie/flickr8k）
比較系統（皆使用快取特徵 + 解碼器，early stopping）：
  W4_ResNet50+LSTM  ：Week 4 方法 A（ResNet-50 + LSTM，貪婪解碼）；直接讀取 Week 4 存的 q4_cnn_lstm.pt（不重訓）-> 基準
  CLIP+LSTM         ：僅把 Encoder 換成 CLIP，解碼器架構相同（貪婪解碼；多一層輸入 LayerNorm）   -> 隔離「Encoder」的效果
  CLIP+GPT2         ：ClipCap 風格，CLIP 向量 -> MLP 前綴 -> DistilGPT-2（beam=3） -> 同時換 Encoder 與 Decoder
  W4_BLIP（可選）   ：Week 4 方法 B 預訓練模型，作為上限參考
評估：BLEU-4（sacrebleu，多參考）、BERTScore F1（原始 + rescaled）、端到端延遲（batch=1）
註：本次新訓練的 CLIP+LSTM 解碼器多了一層輸入 LayerNorm（CLIP 特徵量級不同）；Week 4 載入的模型維持原結構（無 LayerNorm）。
圖片輸出 -> Outputs/Q3_*.png ；模型輸出 -> Outputs/q3_*.pt
"""
import collections
import re
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import ResNet50_Weights, resnet50

import Quiz1 as q1

device = q1.device
PAD, START, END, UNK = 0, 1, 2, 3
MAX_CAP, GPT_LEN = 30, 40
CLIP_NAME, GPT_NAME = "openai/clip-vit-base-patch16", "distilgpt2"
feat_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(q1.MEAN, q1.STD)])


def cap_tok(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def norm(s):
    return " ".join(cap_tok(s))


def _sync():
    if device == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------
# 資料與特徵
# ----------------------------------------------------------------------
def prepare_flickr(paths, hf_repo="jxie/flickr8k"):
    """若本機沒有資料就從 Hugging Face 下載（快取在 Data/flickr8k_hf）。"""
    from datasets import load_dataset
    ds = load_dataset(hf_repo, cache_dir=str(paths["data"] / "flickr8k_hf"))
    key = {"train": "train", "val": "validation" if "validation" in ds else "val", "test": "test"}
    ctx = {"paths": paths, "ds": {s: ds[k] for s, k in key.items()}, "caps": {}, "feats": {},
           "models": {}, "hist": {}, "out": {}, "lat": {}}
    for s, d in ctx["ds"].items():
        cols = [c for c in d.column_names if c.startswith("caption")]
        if cols:
            ctx["caps"][s] = [list(x) for x in zip(*[d[c] for c in cols])]
        elif "captions" in d.column_names:
            ctx["caps"][s] = [list(x) for x in d["captions"]]
        else:
            raise KeyError(f"no caption column in {d.column_names}")
    print({s: len(d) for s, d in ctx["ds"].items()}, "| refs/img:", len(ctx["caps"]["train"][0]))
    return ctx


def load_encoders(ctx):
    from transformers import CLIPImageProcessor, CLIPModel
    cnn = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    cnn.fc = nn.Identity()
    ctx["cnn"] = cnn.to(device).eval()
    ctx["clip"] = CLIPModel.from_pretrained(CLIP_NAME).to(device).eval()
    ctx["clip_proc"] = CLIPImageProcessor.from_pretrained(CLIP_NAME)


@torch.no_grad()
def feat_resnet(ctx, pils):
    x = torch.stack([feat_tf(p) for p in pils]).to(device)
    with torch.autocast(**q1._AMP):
        return ctx["cnn"](x).float()


@torch.no_grad()
def feat_clip(ctx, pils):
    px = ctx["clip_proc"](images=list(pils), return_tensors="pt")["pixel_values"].to(device)
    with torch.autocast(**q1._AMP):
        o = ctx["clip"].get_image_features(pixel_values=px)
    o = o if torch.is_tensor(o) else o.pooler_output
    return F.normalize(o.float(), dim=-1) * 10.0     # 縮放到與 ResNet 特徵相近的量級


def extract_features(ctx, bs=64):
    """ResNet-50 (2048) 與 CLIP (512) 特徵，一次讀圖、兩個 encoder 都抽，快取到 Data/flickr8k/。"""
    load_encoders(ctx)
    f = ctx["paths"]["data"] / "flickr8k" / "feats_resnet50_clip.pt"
    f.parent.mkdir(parents=True, exist_ok=True)
    if f.exists():
        ctx["feats"] = torch.load(f)
        return
    feats = {"resnet50": {}, "clip": {}}
    for s, d in ctx["ds"].items():
        r, c = [], []
        for i in range(0, len(d), bs):
            pils = [im.convert("RGB") for im in d[i:i + bs]["image"]]
            r.append(feat_resnet(ctx, pils).cpu()); c.append(feat_clip(ctx, pils).cpu())
        feats["resnet50"][s], feats["clip"][s] = torch.cat(r), torch.cat(c)
        print(s, feats["resnet50"][s].shape, feats["clip"][s].shape)
    torch.save(feats, f)
    ctx["feats"] = feats


# ----------------------------------------------------------------------
# 解碼器 1：LSTM（Week 4 方法 A 的解碼器）
# ----------------------------------------------------------------------
def build_vocab(ctx, min_freq=5):
    cnt = collections.Counter(w for refs in ctx["caps"]["train"] for c in refs for w in cap_tok(c))
    ctx["itos"] = ["<pad>", "<start>", "<end>", "<unk>"] + [w for w, c in cnt.items() if c >= min_freq]
    ctx["stoi"] = {w: i for i, w in enumerate(ctx["itos"])}
    print("caption vocab:", len(ctx["itos"]))


def _pairs(ctx, split):
    idx, caps = [], []
    for i, refs in enumerate(ctx["caps"][split]):
        for c in refs:
            ids = [START] + [ctx["stoi"].get(w, UNK) for w in cap_tok(c)][:MAX_CAP] + [END]
            idx.append(i); caps.append(ids + [PAD] * (MAX_CAP + 2 - len(ids)))
    return torch.tensor(idx), torch.tensor(caps)


class CapDecoder(nn.Module):
    def __init__(self, vocab, feat, emb=128, hid=256, drop=0.5, use_norm=True):
        super().__init__()
        # use_norm=False 時與 Week 4 的 CapDecoder 完全相同（state_dict 鍵值一致，可直接載入 q4_cnn_lstm.pt）
        self.norm = nn.LayerNorm(feat) if use_norm else nn.Identity()
        self.img_proj = nn.Sequential(nn.Linear(feat, emb), nn.ReLU(), nn.Dropout(drop))
        self.init_h, self.init_c = nn.Linear(feat, hid), nn.Linear(feat, hid)
        self.emb = nn.Embedding(vocab, emb, padding_idx=PAD)
        self.lstm = nn.LSTM(emb * 2, hid, batch_first=True)
        self.drop, self.out = nn.Dropout(drop), nn.Linear(hid, vocab)

    def init_state(self, f):
        f = self.norm(f)
        return torch.tanh(self.init_h(f)).unsqueeze(0), torch.tanh(self.init_c(f)).unsqueeze(0)

    def forward(self, f, caps):
        h, c = self.init_state(f)
        img = self.img_proj(self.norm(f)).unsqueeze(1).expand(-1, caps.size(1), -1)
        o, _ = self.lstm(torch.cat([self.drop(self.emb(caps)), img], -1), (h, c))
        return self.out(self.drop(o))

    @torch.no_grad()
    def greedy(self, f, max_len=MAX_CAP):
        self.eval()
        h, c = self.init_state(f)
        img = self.img_proj(self.norm(f))
        tok = torch.full((f.size(0),), START, device=f.device)
        outs = []
        for _ in range(max_len):
            o, (h, c) = self.lstm(torch.cat([self.emb(tok), img], -1).unsqueeze(1), (h, c))
            tok = self.out(o.squeeze(1)).argmax(-1)
            outs.append(tok)
        return torch.stack(outs, 1).cpu().tolist()


def _lstm_epoch(model, feats, pairs, opt=None, bs=128):
    idx_all, caps_all = pairs
    perm = torch.randperm(len(idx_all)) if opt else torch.arange(len(idx_all))
    model.train(opt is not None)
    crit, total, n = nn.CrossEntropyLoss(ignore_index=PAD), 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            f, caps = feats[idx_all[b]].to(device), caps_all[b].to(device)
            lg = model(f, caps[:, :-1])
            loss = crit(lg.reshape(-1, lg.size(-1)), caps[:, 1:].reshape(-1))
            if opt:
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            total += loss.item() * len(b); n += len(b)
    return total / n


def train_lstm(ctx, name, feat_key, tag, epochs=30, patience=3, lr=1e-3):
    """name：系統名稱；feat_key：'resnet50' 或 'clip'；tag：存檔名 q3_{tag}.pt。"""
    q1.set_seed(42)
    feats = ctx["feats"][feat_key]
    trp, vap = _pairs(ctx, "train"), _pairs(ctx, "val")
    dec = CapDecoder(len(ctx["itos"]), feats["train"].size(1)).to(device)
    opt, es, hist, t0 = torch.optim.Adam(dec.parameters(), lr=lr), q1.EarlyStopping(patience, "min"), \
        {"train": [], "val": []}, time.time()
    for ep in range(epochs):
        tl, vl = _lstm_epoch(dec, feats["train"], trp, opt), _lstm_epoch(dec, feats["val"], vap)
        hist["train"].append(tl); hist["val"].append(vl)
        print(f"[{name}] ep{ep + 1} train={tl:.4f} val={vl:.4f}")
        if es.step(vl, dec):
            print("early stop, best epoch =", es.best_epoch); break
    es.restore(dec)
    hist["best_epoch"], hist["time_s"] = es.best_epoch, time.time() - t0
    torch.save({"state_dict": dec.state_dict(), "vocab": ctx["itos"]}, ctx["paths"]["outputs"] / f"q3_{tag}.pt")
    ctx["models"][name] = {"kind": "lstm", "model": dec, "feat": feat_key}
    ctx["hist"][name] = hist
    return hist


def load_week4(ctx, path=None, name="W4_ResNet50+LSTM"):
    """讀取 Week 4 訓練好的 ResNet-50 + LSTM（q4_cnn_lstm.pt：{'state_dict', 'vocab'}），不重新訓練。
    預設路徑 Outputs/q4_cnn_lstm.pt；Week 4 的檔案若在別處，請複製過來或傳入 path。
    使用 Week 4 自己的詞彙表解碼；輸入特徵與 Week 4 相同（ResNet-50 IMAGENET1K_V2、Resize256/CenterCrop224、2048 維）。"""
    from pathlib import Path
    path = Path(path) if path else ctx["paths"]["outputs"] / "q4_cnn_lstm.pt"
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}；請把 Week 4 的 q4_cnn_lstm.pt 複製到 Outputs/，或用 path= 指定。")
    ck = torch.load(path, map_location=device)
    dec = CapDecoder(len(ck["vocab"]), 2048, use_norm=False).to(device)
    dec.load_state_dict(ck["state_dict"])
    dec.eval()
    ctx["models"][name] = {"kind": "lstm", "model": dec, "feat": "resnet50", "itos": ck["vocab"]}
    print(f"loaded {path.name} | vocab={len(ck['vocab'])} | params={sum(p.numel() for p in dec.parameters()) / 1e6:.2f}M")


# ----------------------------------------------------------------------
# 解碼器 2：ClipCap（CLIP 向量 -> 前綴 -> DistilGPT-2）
# ----------------------------------------------------------------------
class ClipCap(nn.Module):
    def __init__(self, feat, prefix_len=10):
        super().__init__()
        from transformers import GPT2LMHeadModel
        self.gpt, self.pl = GPT2LMHeadModel.from_pretrained(GPT_NAME), prefix_len
        d = self.gpt.config.n_embd
        self.norm = nn.LayerNorm(feat)
        self.mapper = nn.Sequential(nn.Linear(feat, d * prefix_len // 2), nn.Tanh(),
                                    nn.Linear(d * prefix_len // 2, d * prefix_len))

    def prefix(self, f):
        return self.mapper(self.norm(f)).view(f.size(0), self.pl, -1)

    def forward(self, f, ids, mask):
        x = torch.cat([self.prefix(f), self.gpt.transformer.wte(ids)], 1)
        att = torch.cat([torch.ones(ids.size(0), self.pl, device=ids.device, dtype=mask.dtype), mask], 1)
        lg = self.gpt(inputs_embeds=x, attention_mask=att).logits[:, self.pl - 1:-1]
        return lg

    @torch.no_grad()
    def generate(self, f, tok, num_beams=3, max_new=MAX_CAP):
        self.eval()
        pre = self.prefix(f)
        out = self.gpt.generate(inputs_embeds=pre, attention_mask=torch.ones(pre.shape[:2], dtype=torch.long, device=pre.device),
                                max_new_tokens=max_new, num_beams=num_beams, no_repeat_ngram_size=3,
                                early_stopping=True, pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)
        return [t.strip() for t in tok.batch_decode(out, skip_special_tokens=True)]


def _gpt_pairs(ctx, split):
    from transformers import AutoTokenizer
    if "gtok" not in ctx:
        ctx["gtok"] = AutoTokenizer.from_pretrained(GPT_NAME)
        ctx["gtok"].pad_token = ctx["gtok"].eos_token
    eos = ctx["gtok"].eos_token_id
    idx, ids, masks = [], [], []
    for i, refs in enumerate(ctx["caps"][split]):
        for c in refs:
            t = ctx["gtok"](c.strip().lower())["input_ids"][:GPT_LEN - 1] + [eos]
            idx.append(i); ids.append(t + [eos] * (GPT_LEN - len(t))); masks.append([1] * len(t) + [0] * (GPT_LEN - len(t)))
    return torch.tensor(idx), torch.tensor(ids), torch.tensor(masks)


def _gpt_epoch(model, feats, data, opt=None, scaler=None, bs=64):
    idx, ids, masks = data
    perm = torch.randperm(len(idx)) if opt else torch.arange(len(idx))
    model.train(opt is not None)
    total, n = 0.0, 0
    with torch.set_grad_enabled(opt is not None):
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            f, y, m = feats[idx[b]].to(device), ids[b].to(device), masks[b].to(device)
            with torch.autocast(**q1._AMP):
                lg = model(f, y, m)
            loss = F.cross_entropy(lg.float().reshape(-1, lg.size(-1)), y.masked_fill(m == 0, -100).reshape(-1))
            if opt:
                opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update()
            total += loss.item() * len(b); n += len(b)
    return total / n


def train_clipcap(ctx, name="CLIP+GPT2", tag="clip_gpt2", epochs=10, patience=2, lr_gpt=5e-5, lr_map=1e-3):
    q1.set_seed(42)
    feats = ctx["feats"]["clip"]
    trd, vad = _gpt_pairs(ctx, "train"), _gpt_pairs(ctx, "val")
    model = ClipCap(feats["train"].size(1)).to(device)
    opt = torch.optim.AdamW([{"params": model.gpt.parameters(), "lr": lr_gpt},
                             {"params": list(model.mapper.parameters()) + list(model.norm.parameters()), "lr": lr_map}],
                            weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    es, hist, t0 = q1.EarlyStopping(patience, "min"), {"train": [], "val": []}, time.time()
    for ep in range(epochs):
        tl, vl = _gpt_epoch(model, feats["train"], trd, opt, scaler), _gpt_epoch(model, feats["val"], vad)
        hist["train"].append(tl); hist["val"].append(vl)
        print(f"[{name}] ep{ep + 1} train={tl:.4f} val={vl:.4f}")
        if es.step(vl, model):
            print("early stop, best epoch =", es.best_epoch); break
    es.restore(model)
    hist["best_epoch"], hist["time_s"] = es.best_epoch, time.time() - t0
    torch.save({"state_dict": model.state_dict()}, ctx["paths"]["outputs"] / f"q3_{tag}.pt")
    ctx["models"][name] = {"kind": "clipcap", "model": model, "feat": "clip"}
    ctx["hist"][name] = hist
    return hist


# ----------------------------------------------------------------------
# 推論與評估
# ----------------------------------------------------------------------
@torch.no_grad()
def caption_from_feats(ctx, name, f):
    s = ctx["models"][name]
    if s["kind"] == "lstm":
        itos, outs = s.get("itos", ctx["itos"]), []
        for seq in s["model"].greedy(f):
            words = []
            for t in seq:
                if t == END:
                    break
                words.append(itos[t])
            outs.append(" ".join(words))
        return outs
    return s["model"].generate(f, ctx["gtok"])


def caption_image(ctx, name, pil):
    """單張 PIL -> 文字（端到端：前處理 + Encoder + Decoder）。"""
    pil = pil.convert("RGB")
    key = ctx["models"][name]["feat"]
    f = feat_clip(ctx, [pil]) if key == "clip" else feat_resnet(ctx, [pil])
    return caption_from_feats(ctx, name, f)[0]


def generate_test(ctx, bs=100):
    """用快取特徵在整個 test split 產生描述。"""
    for name, s in ctx["models"].items():
        ft, caps = ctx["feats"][s["feat"]]["test"], []
        for i in range(0, len(ft), bs):
            caps += caption_from_feats(ctx, name, ft[i:i + bs].to(device))
        ctx["out"][name] = caps
        print(name, "| e.g.:", caps[0])


def measure_latency(ctx, n=50):
    pils = [ctx["ds"]["test"][i]["image"].convert("RGB") for i in range(n)]
    for name in ctx["models"]:
        caption_image(ctx, name, pils[0])       # warm-up
        lats = []
        for p in pils:
            _sync(); t = time.perf_counter()
            caption_image(ctx, name, p)
            _sync(); lats.append(time.perf_counter() - t)
        ctx["lat"][name] = float(np.mean(lats))


def add_blip(ctx, name="Salesforce/blip-image-captioning-base", bs=32, n_lat=50):
    """Week 4 方法 B（可選）：BLIP 作為預訓練上限參考。"""
    from transformers import BlipForConditionalGeneration, BlipProcessor
    proc = BlipProcessor.from_pretrained(name)
    blip = BlipForConditionalGeneration.from_pretrained(name).to(device).eval()

    @torch.no_grad()
    def cap(pils):
        inp = proc(images=pils, return_tensors="pt").to(device)
        return [proc.decode(o, skip_special_tokens=True) for o in blip.generate(**inp, max_new_tokens=30, num_beams=3)]

    d, outs = ctx["ds"]["test"], []
    for i in range(0, len(d), bs):
        outs += cap([im.convert("RGB") for im in d[i:i + bs]["image"]])
    ctx["out"]["W4_BLIP"] = outs
    pils = [d[i]["image"].convert("RGB") for i in range(n_lat)]
    cap(pils[:1])
    lats = []
    for p in pils:
        _sync(); t = time.perf_counter(); cap([p]); _sync(); lats.append(time.perf_counter() - t)
    ctx["lat"]["W4_BLIP"] = float(np.mean(lats))
    print("W4_BLIP | e.g.:", outs[0])


def text_metrics(ctx):
    """BLEU-4（sacrebleu 多參考）、BERTScore F1（多參考取最高；raw 與 rescaled 各一）、延遲。"""
    import sacrebleu
    from bert_score import score as bert_fn
    refs = [[norm(c) for c in r] for r in ctx["caps"]["test"]]
    n_ref = min(len(r) for r in refs)
    streams = [[r[k] for r in refs] for k in range(n_ref)]
    rows = {}
    for name, caps in ctx["out"].items():
        hyps = [norm(c) or "empty" for c in caps]
        bleu = sacrebleu.corpus_bleu(hyps, streams, lowercase=True)
        kw = dict(lang="en", device=device, batch_size=32, verbose=False)
        f_raw = bert_fn(hyps, refs, **kw)[2].mean().item()
        f_res = bert_fn(hyps, refs, rescale_with_baseline=True, **kw)[2].mean().item()
        rows[name] = {"BLEU-4": bleu.score, "BERTScore_F1": f_raw, "BERTScore_F1_rescaled": f_res,
                      "Avg_len": np.mean([len(c.split()) for c in caps]),
                      "Latency_s": ctx["lat"].get(name, np.nan),
                      "Empty_captions": sum(1 for c in caps if not norm(c))}
        print(name, {k: round(v, 4) for k, v in rows[name].items()})
    df = pd.DataFrame(rows).T.round(4)
    df.to_csv(ctx["paths"]["outputs"] / "q3_results.csv")
    pd.DataFrame({name: caps for name, caps in ctx["out"].items()}).to_csv(
        ctx["paths"]["outputs"] / "q3_generated_captions.csv", index=False)
    return df


# ----------------------------------------------------------------------
# 圖表
# ----------------------------------------------------------------------
def plot_training_curves(ctx):
    plt.figure(figsize=(7, 4.2))
    for name, h in ctx["hist"].items():
        l, = plt.plot(h["train"], label=f"{name} train")
        plt.plot(h["val"], "--", color=l.get_color(), label=f"{name} val")
        plt.scatter([h["best_epoch"] - 1], [h["val"][h["best_epoch"] - 1]], color=l.get_color(), marker="*", s=80)
    plt.xlabel("epoch"); plt.ylabel("token-level CE loss")
    plt.title("Captioning decoders: training / validation loss (* = best epoch)")
    plt.legend(fontsize=7)
    q1.save_fig(ctx["paths"], "Q3_training_curves.png")


def plot_comparison(ctx, df):
    cols = [c for c in ["BLEU-4", "BERTScore_F1", "BERTScore_F1_rescaled", "Latency_s"] if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4.2 * len(cols), 4))
    palette = ["#9ecae1", "#4C72B0", "#55A868", "#C44E52"]
    for ax, c in zip(axes, cols):
        df[c].plot.bar(ax=ax, rot=25, color=palette[:len(df)])
        ax.set_title(c)
        for i, v in enumerate(df[c]):
            ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
        if c == "Latency_s":
            ax.set_yscale("log"); ax.set_ylabel("sec / image (log)")
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q3_method_comparison.png")


def plot_examples(ctx, n=6, seed=7):
    idx = np.random.RandomState(seed).choice(len(ctx["caps"]["test"]), n, replace=False)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(ctx["ds"]["test"][int(i)]["image"].convert("RGB")); ax.axis("off")
        lines = [textwrap.fill("REF: " + ctx["caps"]["test"][i][0], 60)]
        lines += [textwrap.fill(f"{k}: {v[i]}", 60) for k, v in ctx["out"].items()]
        ax.set_title("\n".join(lines), fontsize=8, loc="left")
    plt.tight_layout()
    q1.save_fig(ctx["paths"], "Q3_caption_examples.png")
