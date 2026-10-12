"""
Quiz 2：Sentiment Analysis（Yelp Review Full, 3 類）— RNN vs LSTM
- 使用 early stopping（以驗證集 loss 為準，還原最佳權重）
圖片輸出 -> Outputs/Q2_*.png ；模型輸出 -> Outputs/q2_*.pt
"""
import collections
import copy
import json
import re
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, TensorDataset

import Quiz1 as q1

device = "cuda" if torch.cuda.is_available() else "cpu"
NC = 3
CLASS_NAMES = ["negative", "neutral", "positive"]


# ----------------------------------------------------------------------
# 資料前處理
# ----------------------------------------------------------------------
def tokenize(text: str):
    return re.findall(r"[a-z0-9']+|[!?]", text.lower())


def prepare_data(paths, train_df, test_df, vocab_size=20000, val_ratio=0.1):
    """切 train/val、建詞表(只用訓練集)、轉成 ID 序列。回傳 ctx 字典。"""
    tr_df, va_df = train_test_split(train_df, test_size=val_ratio,
                                    stratify=train_df["label"], random_state=42)
    te_df = test_df.copy()

    counter = collections.Counter(t for x in tr_df["text"] for t in tokenize(x))
    itos = ["<PAD>", "<UNK>"] + [w for w, _ in counter.most_common(vocab_size - 2)]
    stoi = {w: i for i, w in enumerate(itos)}

    def to_ids(texts):
        return [[stoi.get(t, 1) for t in tokenize(x)] for x in texts]

    tr_df = tr_df.assign(ids=to_ids(tr_df["text"]))
    va_df = va_df.assign(ids=to_ids(va_df["text"]))
    te_df = te_df.assign(ids=to_ids(te_df["text"]))
    with open(paths["outputs"] / "q2_vocab.json", "w") as f:
        json.dump(itos, f)
    print(f"train={len(tr_df)} val={len(va_df)} test={len(te_df)} vocab={len(itos)}")
    return {"paths": paths, "tr_df": tr_df, "va_df": va_df, "te_df": te_df,
            "itos": itos, "stoi": stoi, "to_ids": to_ids}


def build_tensors(df, L):
    ids = [x[:L] for x in df["ids"]]
    lens = torch.tensor([max(len(x), 1) for x in ids])
    X = torch.zeros(len(ids), L, dtype=torch.long)
    for i, x in enumerate(ids):
        if x:
            X[i, :len(x)] = torch.tensor(x)
    y = torch.tensor(df["label"].values, dtype=torch.long)
    return X, lens, y


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class SeqClassifier(nn.Module):
    """Embedding -> RNN/LSTM -> 最後一個「真實」時間步的隱藏狀態 -> Linear(NC)"""

    def __init__(self, kind, vocab, nc=NC, emb=128, hid=128, drop=0.3):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        cell = nn.RNN if kind == "rnn" else nn.LSTM
        self.rnn = cell(emb, hid, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(hid, nc)

    def forward(self, x, lens):
        e = self.drop(self.emb(x))
        packed = nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True,
                                                   enforce_sorted=False)
        _, h = self.rnn(packed)
        h = h[0] if isinstance(h, tuple) else h
        return self.fc(self.drop(h[-1]))


@torch.no_grad()
def evaluate(model, loader):
    """回傳 (平均 loss, 機率 (N,NC), 標籤 (N,))。"""
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total, probs, ys = 0.0, [], []
    for x, l, y in loader:
        x, l, y = x.to(device), l.to(device), y.to(device)
        out = model(x, l)
        total += crit(out, y).item()
        probs.append(torch.softmax(out, 1).cpu())
        ys.append(y.cpu())
    probs, ys = torch.cat(probs).numpy(), torch.cat(ys).numpy()
    return total / len(ys), probs, ys


def run_experiment(ctx, kind, L=256, max_epochs=30, patience=3, bs=128, lr=1e-3, verbose=True):
    """
    訓練一個模型（early stopping：val loss 連續 patience 個 epoch 沒進步就停止，並還原最佳權重）。
    回傳 (model, history, test_probs, test_labels, train_time)。
    """
    q1.set_seed(42)
    dl_tr = DataLoader(TensorDataset(*build_tensors(ctx["tr_df"], L)), bs, shuffle=True)
    dl_va = DataLoader(TensorDataset(*build_tensors(ctx["va_df"], L)), 256)
    dl_te = DataLoader(TensorDataset(*build_tensors(ctx["te_df"], L)), 256)

    model = SeqClassifier(kind, len(ctx["itos"])).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
    best_loss, best_state, bad, t0 = 1e9, None, 0, time.time()

    for ep in range(max_epochs):
        model.train()
        tl, tc, n = 0.0, 0, 0
        for x, l, y in dl_tr:
            x, l, y = x.to(device), l.to(device), y.to(device)
            opt.zero_grad()
            out = model(x, l)
            loss = crit(out, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(y)
            tc += (out.argmax(1) == y).sum().item()
            n += len(y)
        vl, vp, vy = evaluate(model, dl_va)
        va_acc = float((vp.argmax(1) == vy).mean())
        hist["tr_loss"].append(tl / n); hist["tr_acc"].append(tc / n)
        hist["va_loss"].append(vl); hist["va_acc"].append(va_acc)
        if verbose:
            print(f"[{kind} L={L}] ep{ep+1} tr_loss={tl/n:.4f} tr_acc={tc/n:.4f} "
                  f"va_loss={vl:.4f} va_acc={va_acc:.4f}")
        if vl < best_loss - 1e-4:
            best_loss, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
            if bad >= patience:
                if verbose:
                    print(f"[{kind}] early stop at epoch {ep+1} (best epoch {hist['best_epoch']})")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    _, tp, ty = evaluate(model, dl_te)
    return model, hist, tp, ty, train_time


def save_model(ctx, model, kind, L):
    out = ctx["paths"]["outputs"] / f"q2_{kind}.pt"
    torch.save({"state_dict": model.state_dict(), "max_len": L, "kind": kind,
                "classes": CLASS_NAMES}, out)
    print("[saved]", out)


# ----------------------------------------------------------------------
# 推論
# ----------------------------------------------------------------------
@torch.no_grad()
def predict_texts(ctx, model, texts, L=256):
    """回傳 (N,3) 機率，順序為 negative / neutral / positive。"""
    model.eval()
    df = pd.DataFrame({"ids": ctx["to_ids"](texts), "label": [0] * len(texts)})
    x, l, _ = build_tensors(df, L)
    return torch.softmax(model(x.to(device), l.to(device)), 1).cpu().numpy()


def save_test_predictions(ctx, test_y, rnn_p, lstm_p):
    res = pd.DataFrame({
        "text": ctx["te_df"]["text"].str[:120].values,
        "true": [CLASS_NAMES[i] for i in test_y],
        "rnn_pred": [CLASS_NAMES[i] for i in rnn_p.argmax(1)], "rnn_conf": rnn_p.max(1),
        "lstm_pred": [CLASS_NAMES[i] for i in lstm_p.argmax(1)], "lstm_conf": lstm_p.max(1)})
    out = ctx["paths"]["outputs"] / "q2_test_predictions.csv"
    res.to_csv(out, index=False)
    print("[saved]", out)
    return res


# ----------------------------------------------------------------------
# 評估與繪圖
# ----------------------------------------------------------------------
def compute_metrics(y, p):
    pred = p.argmax(1)
    return {"Accuracy": accuracy_score(y, pred),
            "Macro_F1": f1_score(y, pred, average="macro"),
            "Macro_AUC": roc_auc_score(y, p, multi_class="ovr", average="macro"),
            # 嚴重錯誤：把 negative 判成 positive，或相反
            "Severe_error": float((np.abs(pred - y) == 2).mean())}


def _save(ctx, name):
    out = ctx["paths"]["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("[saved]", out)


def compare(ctx, rnn_res, lstm_res):
    """rnn_res / lstm_res = run_experiment 的回傳值。輸出比較表與圖。"""
    (rnn_m, rnn_h, rnn_p, y, rnn_t), (lstm_m, lstm_h, lstm_p, _, lstm_t) = rnn_res, lstm_res
    metrics = pd.DataFrame({"RNN": compute_metrics(y, rnn_p), "LSTM": compute_metrics(y, lstm_p)}).T
    metrics["Params"] = [sum(p.numel() for p in m.parameters()) for m in (rnn_m, lstm_m)]
    metrics["Epochs_run"] = [len(rnn_h["tr_loss"]), len(lstm_h["tr_loss"])]
    metrics["Best_epoch"] = [rnn_h["best_epoch"], lstm_h["best_epoch"]]
    metrics["Train_time_s"] = [rnn_t, lstm_t]
    metrics.round(4).to_csv(ctx["paths"]["outputs"] / "q2_metrics.csv")

    # 訓練曲線
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, h in (("RNN", rnn_h), ("LSTM", lstm_h)):
        ep = range(1, len(h["tr_loss"]) + 1)
        l1, = axes[0].plot(ep, h["tr_loss"], "--", label=f"{name} train")
        axes[0].plot(ep, h["va_loss"], color=l1.get_color(), label=f"{name} val")
        axes[0].axvline(h["best_epoch"], color=l1.get_color(), ls=":", lw=0.8)
        l2, = axes[1].plot(ep, h["tr_acc"], "--", label=f"{name} train")
        axes[1].plot(ep, h["va_acc"], color=l2.get_color(), label=f"{name} val")
    axes[0].set_title("Loss (dotted = best epoch)"); axes[1].set_title("Accuracy")
    for a in axes:
        a.set_xlabel("epoch"); a.legend()
    _save(ctx, "Q2_training_curves.png")

    # 混淆矩陣（列正規化）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (name, p) in zip(axes, (("RNN", rnn_p), ("LSTM", lstm_p))):
        cm = confusion_matrix(y, p.argmax(1), normalize="true")
        ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
            ax=ax, colorbar=False, values_format=".2f", xticks_rotation=30)
        ax.set_title(f"{name} confusion matrix (row-normalized)")
    _save(ctx, "Q2_confusion_matrix.png")

    # ROC（one-vs-rest）
    Y = label_binarize(y, classes=range(NC))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (name, p) in zip(axes, (("RNN", rnn_p), ("LSTM", lstm_p))):
        for c in range(NC):
            fpr, tpr, _ = roc_curve(Y[:, c], p[:, c])
            ax.plot(fpr, tpr, label=f"{CLASS_NAMES[c]} (AUC={roc_auc_score(Y[:, c], p[:, c]):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f"{name} ROC (Macro-AUC={metrics.loc[name, 'Macro_AUC']:.3f})")
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=7)
    _save(ctx, "Q2_roc_curve.png")

    # 整體指標
    metrics[["Accuracy", "Macro_F1", "Macro_AUC"]].plot.bar(figsize=(7, 4), rot=0, ylim=(0, 1.0))
    plt.title("RNN vs LSTM (Yelp test)")
    _save(ctx, "Q2_metrics_bar.png")

    # 每類 F1
    pc = pd.DataFrame({n: f1_score(y, p.argmax(1), average=None) for n, p in (("RNN", rnn_p), ("LSTM", lstm_p))},
                      index=CLASS_NAMES)
    pc.plot.bar(figsize=(7, 4), rot=0, ylim=(0, 1.0))
    plt.title("Per-class F1")
    _save(ctx, "Q2_per_class_f1.png")
    return metrics


def length_experiment(ctx, lengths=(64, 128, 256, 400), max_epochs=15, patience=2):
    """不同序列長度下 RNN / LSTM 的測試表現（皆使用 early stopping）。"""
    rows = []
    for L in lengths:
        for kind in ("rnn", "lstm"):
            _, h, p, y, t = run_experiment(ctx, kind, L, max_epochs=max_epochs, patience=patience, verbose=False)
            m = compute_metrics(y, p)
            rows.append({"model": kind.upper(), "max_len": L, "Accuracy": m["Accuracy"],
                         "Macro_F1": m["Macro_F1"], "Macro_AUC": m["Macro_AUC"],
                         "best_epoch": h["best_epoch"], "train_time_s": t})
            print(rows[-1])
    df = pd.DataFrame(rows)
    df.round(4).to_csv(ctx["paths"]["outputs"] / "q2_length_experiment.csv", index=False)
    plt.figure(figsize=(6, 4))
    for kind, g in df.groupby("model"):
        plt.plot(g["max_len"], g["Macro_F1"], marker="o", label=kind)
    plt.xlabel("max sequence length"); plt.ylabel("test Macro-F1")
    plt.title("Effect of sequence length"); plt.legend()
    _save(ctx, "Q2_length_experiment.png")
    return df
