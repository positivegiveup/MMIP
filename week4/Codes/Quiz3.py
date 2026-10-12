"""
Quiz 3：Vision Transformer vs ResNet（Intel Image Classification, 6 類）
- 使用 early stopping（val loss）+ ReduceLROnPlateau
圖片輸出 -> Outputs/Q3_*.png ；模型輸出 -> Outputs/q3_*.pt
"""
import copy
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, f1_score, log_loss,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

import Quiz1 as q1

device = "cuda" if torch.cuda.is_available() else "cpu"
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
_AMP = dict(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda"))


class ArrDS(Dataset):
    """記憶體中的 uint8 影像陣列 (N,H,W,3) + 標籤。"""

    def __init__(self, x, y, tf):
        self.x, self.y, self.tf = x, y.astype(np.int64), tf

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.tf(self.x[i]), self.y[i]


def build_context(paths, intel, bs=64, val_ratio=0.1):
    """
    intel = Quiz1.prepare_intel 的回傳值。
    官方 seg_train 切 90/10 成 train/val（分層），官方 seg_test 當測試集。
    影像 150x150 放大到 224。
    """
    train_tf = T.Compose([T.ToPILImage(), T.RandomResizedCrop(224, scale=(0.6, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToPILImage(), T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    idx_tr, idx_va = train_test_split(np.arange(len(intel["y_train"])), test_size=val_ratio,
                                      stratify=intel["y_train"], random_state=42)
    ds_tr = ArrDS(intel["x_train"][idx_tr], intel["y_train"][idx_tr], train_tf)
    ds_va = ArrDS(intel["x_train"][idx_va], intel["y_train"][idx_va], eval_tf)
    ds_te = ArrDS(intel["x_test"], intel["y_test"], eval_tf)
    print(f"train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)}")
    cn = intel["class_names"]
    return {"paths": paths, "class_names": cn, "nc": len(cn), "ds_te": ds_te,
            "dl_tr": DataLoader(ds_tr, bs, shuffle=True, num_workers=2, pin_memory=True),
            "dl_va": DataLoader(ds_va, 128, num_workers=2),
            "dl_te": DataLoader(ds_te, 128, num_workers=2)}


def build_model(name, nc):
    if name == "vit":
        m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, nc)
    elif name == "resnet":
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, nc)
    else:
        raise ValueError(name)
    return m


@torch.no_grad()
def predict_loader(model, loader):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(**_AMP):
            out = model(x.to(device))
        probs.append(torch.softmax(out.float(), 1).cpu())
        ys.append(y)
    return torch.cat(probs).numpy(), torch.cat(ys).numpy()


def macro_auc(y, p):
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


def train(ctx, name, max_epochs=15, patience=3, lr=1e-4):
    """
    微調預訓練模型。early stopping：val loss 連續 patience 個 epoch 沒進步就停止並還原最佳權重；
    val loss 停滯 1 個 epoch 時學習率 x0.3（ReduceLROnPlateau）。存成 q3_{name}.pt。
    """
    q1.set_seed(42)
    nc = ctx["nc"]
    model = build_model(name, nc).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.3, patience=1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit = nn.CrossEntropyLoss()
    hist = {"tr_loss": [], "va_loss": [], "va_acc": [], "va_auc": []}
    best, best_state, bad, t0 = 1e9, None, 0, time.time()

    for ep in range(max_epochs):
        model.train()
        tl, n = 0.0, 0
        for x, y in ctx["dl_tr"]:
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad()
            with torch.autocast(**_AMP):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tl += loss.item() * len(y); n += len(y)
        vp, vy = predict_loader(model, ctx["dl_va"])
        vl = log_loss(vy, vp, labels=list(range(nc)))
        acc, auc = float((vp.argmax(1) == vy).mean()), macro_auc(vy, vp)
        sched.step(vl)
        for k, v in zip(("tr_loss", "va_loss", "va_acc", "va_auc"), (tl / n, vl, acc, auc)):
            hist[k].append(v)
        print(f"[{name}] ep{ep+1} loss={tl/n:.4f} val_loss={vl:.4f} val_acc={acc:.4f} "
              f"val_macroAUC={auc:.4f} lr={opt.param_groups[0]['lr']:.1e}")
        if vl < best - 1e-4:
            best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            hist["best_epoch"] = ep + 1
        else:
            bad += 1
            if bad >= patience:
                print(f"[{name}] early stop at epoch {ep+1} (best epoch {hist['best_epoch']})")
                break
    model.load_state_dict(best_state)
    train_time = time.time() - t0

    out = ctx["paths"]["outputs"] / f"q3_{name}.pt"
    torch.save({"state_dict": model.state_dict(), "classes": ctx["class_names"]}, out)
    print("[saved]", out)
    return model, hist, train_time


def _save(ctx, name):
    out = ctx["paths"]["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("[saved]", out)


def evaluate_and_plot(ctx, vit_res, resnet_res):
    """vit_res / resnet_res = train() 的回傳值 (model, hist, time)。回傳 (指標表, 每類 AUC 表, probs, y)。"""
    names = ["ViT", "ResNet"]
    runs = dict(zip(names, (vit_res, resnet_res)))
    cn, nc = ctx["class_names"], ctx["nc"]
    probs, results = {}, {}
    for name, (model, _, t) in runs.items():
        p, y = predict_loader(model, ctx["dl_te"])
        probs[name] = p
        results[name] = {"Accuracy": (p.argmax(1) == y).mean(),
                         "Macro_F1": f1_score(y, p.argmax(1), average="macro"),
                         "Macro_AUC": macro_auc(y, p),
                         "Params(M)": sum(q.numel() for q in model.parameters()) / 1e6,
                         "Epochs_run": len(runs[name][1]["tr_loss"]),
                         "Best_epoch": runs[name][1]["best_epoch"], "Train_time_s": t}
    res = pd.DataFrame(results).T
    res.round(4).to_csv(ctx["paths"]["outputs"] / "q3_metrics.csv")

    # 訓練曲線
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for name, (_, h, _) in runs.items():
        ep = range(1, len(h["tr_loss"]) + 1)
        axes[0].plot(ep, h["tr_loss"], label=name)
        axes[1].plot(ep, h["va_loss"], label=name)
        axes[2].plot(ep, h["va_acc"], label=name)
        axes[3].plot(ep, h["va_auc"], label=name)
    for a, t in zip(axes, ("train loss", "val loss", "val accuracy", "val Macro-AUC")):
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    _save(ctx, "Q3_training_curves.png")

    # 每類 AUC
    Y = label_binarize(y, classes=range(nc))
    per_cls = pd.DataFrame({n: roc_auc_score(Y, p, average=None) for n, p in probs.items()}, index=cn)
    per_cls.loc["MACRO"] = [res.loc[n, "Macro_AUC"] for n in per_cls.columns]
    per_cls.round(4).to_csv(ctx["paths"]["outputs"] / "q3_per_class_auc.csv")
    per_cls.plot.bar(figsize=(9, 4), rot=30, ylim=(per_cls.values.min() - 0.01, 1.0))
    plt.title("Per-class AUC & Macro-AUC (Intel test)")
    _save(ctx, "Q3_auc_comparison.png")

    # ROC
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, (name, p) in zip(axes, probs.items()):
        for c in range(nc):
            fpr, tpr, _ = roc_curve(Y[:, c], p[:, c])
            ax.plot(fpr, tpr, label=cn[c])
        ax.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax.set_title(f'{name} ROC (Macro-AUC={res.loc[name, "Macro_AUC"]:.4f})')
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=6)
    _save(ctx, "Q3_roc_curves.png")

    # 混淆矩陣
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (name, p) in zip(axes, probs.items()):
        ConfusionMatrixDisplay(confusion_matrix(y, p.argmax(1)), display_labels=cn).plot(
            ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(f"{name} confusion matrix")
    _save(ctx, "Q3_confusion_matrix.png")
    return res, per_cls, probs, y


def show_predictions(ctx, probs, n=10, seed=0):
    """隨機取 n 張測試圖，顯示 ViT / ResNet 預測。"""
    ds, cn = ctx["ds_te"], ctx["class_names"]
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    cols = 5
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(14, 3.2 * np.ceil(n / cols)))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(ds.x[i]); ax.axis("off")
        ax.set_title(f"GT: {cn[ds.y[i]]}\nViT: {cn[probs['ViT'][i].argmax()]}\n"
                     f"ResNet: {cn[probs['ResNet'][i].argmax()]}", fontsize=8)
    plt.tight_layout()
    _save(ctx, "Q3_prediction_examples.png")
