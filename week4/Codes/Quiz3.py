"""
Quiz 3：Vision Transformer vs ResNet（CUB-200-2011，200 類鳥類細粒度分類）
- 兩個模型皆為 ImageNet 預訓練、完整微調；early stopping（val loss）+ ReduceLROnPlateau
- 分類頭使用較大的學習率（head_lr_mult），因為 200 類的新分類頭要從隨機初始化學起
圖片輸出 -> Outputs/Q3_*.png ；模型輸出 -> Outputs/q3_*.pt
"""
import copy
import textwrap
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as T
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve
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


def build_context(paths, cub, bs=64, val_ratio=0.1):
    """
    cub = Quiz1.prepare_cub 的回傳值（影像已 resize 為 256x256）。
    官方訓練集切 90/10（分層）為 train/val，官方測試集為測試。
    訓練：RandomResizedCrop(224) + 水平翻轉；驗證/測試：CenterCrop(224)。
    """
    train_tf = T.Compose([T.ToPILImage(), T.RandomResizedCrop(224, scale=(0.5, 1.0)),
                          T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    eval_tf = T.Compose([T.ToPILImage(), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    idx_tr, idx_va = train_test_split(np.arange(len(cub["y_train"])), test_size=val_ratio,
                                      stratify=cub["y_train"], random_state=42)
    ds_tr = ArrDS(cub["x_train"][idx_tr], cub["y_train"][idx_tr], train_tf)
    ds_va = ArrDS(cub["x_train"][idx_va], cub["y_train"][idx_va], eval_tf)
    ds_te = ArrDS(cub["x_test"], cub["y_test"], eval_tf)
    print(f"train={len(ds_tr)} val={len(ds_va)} test={len(ds_te)} classes={len(cub['class_names'])}")
    cn = cub["class_names"]
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


# ----------------------------------------------------------------------
# 指標（200 類，部分類別在小的驗證集可能沒有樣本，故自行實作 OvR AUC）
# ----------------------------------------------------------------------
def per_class_auc(y, p):
    """每個類別的 one-vs-rest AUC；該類別沒有正樣本或負樣本時為 NaN。"""
    nc = p.shape[1]
    Y = label_binarize(y, classes=range(nc))
    out = np.full(nc, np.nan)
    for c in range(nc):
        if 0 < Y[:, c].sum() < len(y):
            out[c] = roc_auc_score(Y[:, c], p[:, c])
    return out


def macro_auc(y, p):
    return float(np.nanmean(per_class_auc(y, p)))


def topk_acc(y, p, k=5):
    top = np.argsort(-p, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def nll(y, p):
    return float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean())


# ----------------------------------------------------------------------
# 訓練
# ----------------------------------------------------------------------
def train(ctx, name, max_epochs=30, patience=5, lr=1e-4, head_lr_mult=10):
    """
    微調預訓練模型。
    - 骨幹學習率 lr、分類頭學習率 lr * head_lr_mult（兩個模型使用相同倍率）
    - early stopping：val loss 連續 patience 個 epoch 沒進步就停止並還原最佳權重
    - ReduceLROnPlateau：val loss 停滯 1 個 epoch 即 x0.3
    存成 q3_{name}.pt。
    """
    q1.set_seed(42)
    nc = ctx["nc"]
    model = build_model(name, nc).to(device)
    head, backbone = [], []
    for n, p in model.named_parameters():
        (head if n.startswith(("heads.", "fc.")) else backbone).append(p)
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr},
                             {"params": head, "lr": lr * head_lr_mult}], weight_decay=0.05)
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
        vl, acc, auc = nll(vy, vp), float((vp.argmax(1) == vy).mean()), macro_auc(vy, vp)
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


# ----------------------------------------------------------------------
# 評估與繪圖
# ----------------------------------------------------------------------
def _save(ctx, name):
    out = ctx["paths"]["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print("[saved]", out)


def macro_roc(y, p, n_grid=501):
    """巨觀平均 ROC：各類別 ROC 在共同 FPR 網格上內插後取平均。"""
    Y = label_binarize(y, classes=range(p.shape[1]))
    grid, tprs = np.linspace(0, 1, n_grid), []
    for c in range(p.shape[1]):
        if 0 < Y[:, c].sum() < len(y):
            fpr, tpr, _ = roc_curve(Y[:, c], p[:, c])
            tprs.append(np.interp(grid, fpr, tpr))
    return grid, np.mean(tprs, axis=0)


def evaluate_and_plot(ctx, vit_res, resnet_res, top_n_confusions=10, n_worst=15):
    """
    vit_res / resnet_res = train() 的回傳值 (model, hist, time)。
    回傳 (指標表, 每類別表, probs, y)。每類別表欄位：AUC_/Acc_ + 模型名稱。
    """
    names = ["ViT", "ResNet"]
    runs = dict(zip(names, (vit_res, resnet_res)))
    cn, nc, out_dir = ctx["class_names"], ctx["nc"], ctx["paths"]["outputs"]

    probs, results = {}, {}
    for name, (model, hist, t) in runs.items():
        p, y = predict_loader(model, ctx["dl_te"])
        probs[name] = p
        results[name] = {"Accuracy": float((p.argmax(1) == y).mean()), "Top5_Acc": topk_acc(y, p, 5),
                         "Macro_F1": f1_score(y, p.argmax(1), average="macro"),
                         "Macro_AUC": macro_auc(y, p),
                         "Params(M)": sum(q.numel() for q in model.parameters()) / 1e6,
                         "Epochs_run": len(hist["tr_loss"]), "Best_epoch": hist["best_epoch"],
                         "Train_time_s": t}
    res = pd.DataFrame(results).T
    res.round(4).to_csv(out_dir / "q3_metrics.csv")

    # 訓練曲線
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.8))
    for name, (_, h, _) in runs.items():
        ep = range(1, len(h["tr_loss"]) + 1)
        for ax, k in zip(axes, ("tr_loss", "va_loss", "va_acc", "va_auc")):
            ax.plot(ep, h[k], label=name)
    for a, t in zip(axes, ("train loss", "val loss", "val accuracy", "val Macro-AUC")):
        a.set_title(t); a.set_xlabel("epoch"); a.legend()
    _save(ctx, "Q3_training_curves.png")

    # 每類別 AUC / Accuracy
    per_cls = pd.DataFrame(index=cn)
    for name, p in probs.items():
        per_cls[f"AUC_{name}"] = per_class_auc(y, p)
        cm = confusion_matrix(y, p.argmax(1), labels=range(nc))
        per_cls[f"Acc_{name}"] = np.diag(cm) / np.maximum(cm.sum(1), 1)
    per_cls.round(4).to_csv(out_dir / "q3_per_class.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    lo = np.nanmin(per_cls[["AUC_ViT", "AUC_ResNet"]].values)
    bins = np.linspace(lo, 1.0, 40)
    for name in names:
        axes[0].hist(per_cls[f"AUC_{name}"].dropna(), bins=bins, alpha=0.6, label=name)
    axes[0].set_title("Per-class AUC distribution (200 classes)")
    axes[0].set_xlabel("AUC"); axes[0].set_ylabel("number of classes"); axes[0].legend()
    axes[1].scatter(per_cls["AUC_ResNet"], per_cls["AUC_ViT"], s=10, alpha=0.6)
    axes[1].plot([lo, 1], [lo, 1], "k--", lw=0.8)
    axes[1].set_xlabel("ResNet per-class AUC"); axes[1].set_ylabel("ViT per-class AUC")
    axes[1].set_title("Per-class AUC: ViT vs ResNet (above line = ViT better)")
    _save(ctx, "Q3_auc_comparison.png")

    # 巨觀平均 ROC（完整 + 低 FPR 放大）
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for name, p in probs.items():
        g, t_ = macro_roc(y, p)
        for ax in axes:
            ax.plot(g, t_, label=f'{name} (Macro-AUC={res.loc[name, "Macro_AUC"]:.4f})')
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    axes[0].set_title("Macro-average ROC")
    axes[1].set_xlim(0, 0.05); axes[1].set_ylim(0.8, 1.0); axes[1].set_title("Zoom: FPR <= 0.05")
    for ax in axes:
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(loc="lower right")
    _save(ctx, "Q3_roc_curves.png")

    # 錯誤熱圖（200x200 的混淆矩陣去掉對角線）與最常被混淆的類別對
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    conf_rows = []
    for ax, (name, p) in zip(axes, probs.items()):
        cm = confusion_matrix(y, p.argmax(1), labels=range(nc))
        off = cm.copy(); np.fill_diagonal(off, 0)
        im = ax.imshow(off, cmap="Reds", vmin=0, vmax=max(off.max(), 1))
        ax.set_title(f"{name}: misclassifications (diagonal removed)")
        ax.set_xlabel("predicted class index"); ax.set_ylabel("true class index")
        fig.colorbar(im, ax=ax, fraction=0.046)
        for k in np.argsort(-off, axis=None)[:top_n_confusions]:
            i, j = divmod(int(k), nc)
            if off[i, j] > 0:
                conf_rows.append({"model": name, "true": cn[i], "pred": cn[j], "count": int(off[i, j])})
    _save(ctx, "Q3_confusion_matrix.png")
    pd.DataFrame(conf_rows).to_csv(out_dir / "q3_top_confusions.csv", index=False)

    # 最難的類別（兩模型平均 Accuracy 最低）
    worst = per_cls.assign(m=per_cls[["Acc_ViT", "Acc_ResNet"]].mean(1)).sort_values("m").head(n_worst)
    fig, ax = plt.subplots(figsize=(8, 0.38 * n_worst + 1.5))
    yy = np.arange(len(worst))
    ax.barh(yy - 0.2, worst["Acc_ViT"], 0.4, label="ViT")
    ax.barh(yy + 0.2, worst["Acc_ResNet"], 0.4, label="ResNet")
    ax.set_yticks(yy); ax.set_yticklabels([c[:30] for c in worst.index], fontsize=8)
    ax.invert_yaxis(); ax.set_xlabel("test accuracy"); ax.set_xlim(0, 1)
    ax.set_title(f"{n_worst} hardest classes"); ax.legend()
    _save(ctx, "Q3_worst_classes.png")
    return res, per_cls, probs, y


def show_predictions(ctx, probs, n=10, seed=0):
    """隨機取 n 張測試圖，顯示 ViT / ResNet 預測；任一模型答錯時標題為紅色。"""
    ds, cn = ctx["ds_te"], ctx["class_names"]
    short = lambda s: textwrap.shorten(s, 22, placeholder="..")
    idx = np.random.RandomState(seed).choice(len(ds), n, replace=False)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3.6 * rows))
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        ax.imshow(ds.x[i]); ax.axis("off")
        gt, pv, pr = ds.y[i], probs["ViT"][i].argmax(), probs["ResNet"][i].argmax()
        ax.set_title(f"GT: {short(cn[gt])}\nViT: {short(cn[pv])}\nResNet: {short(cn[pr])}",
                     fontsize=8, color="black" if (pv == gt and pr == gt) else "red")
    plt.tight_layout()
    _save(ctx, "Q3_prediction_examples.png")
