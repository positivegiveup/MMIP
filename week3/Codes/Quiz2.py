"""
Quiz2.py - Plain CNN vs. Transfer Learning（經典 CNN Backbone）
包含：模型、訓練、Top-1/Top-5、測試集預測、ROC/Macro-AUC、參數量比較、超參數實驗
"""
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import Quiz1 as q1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================================================
# 1. 模型
# ==========================================================================
class PlainCNN(nn.Module):
    """VGG 風格：每個 stage = [Conv-BN-ReLU] x2 + MaxPool，最後 GAP + Dropout + FC。"""

    def __init__(self, num_classes=37, widths=(32, 64, 128, 256), dropout=0.3, use_bn=True):
        super().__init__()
        layers, in_ch = [], 3
        for w in widths:
            for _ in range(2):
                layers.append(nn.Conv2d(in_ch, w, 3, padding=1, bias=not use_bn))
                if use_bn:
                    layers.append(nn.BatchNorm2d(w))
                layers.append(nn.ReLU(inplace=True))
                in_ch = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(in_ch, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


def build_backbone(name="resnet18", num_classes=37, freeze=False, dropout=0.0):
    """載入 ImageNet 預訓練權重並替換分類頭。freeze=True -> 只訓練分類頭 (feature extraction)。"""
    m = getattr(models, name)(weights="DEFAULT")
    if name.startswith("resnet"):
        in_f = m.fc.in_features
        head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, num_classes))
        m.fc = head
    elif name in ("mobilenet_v2", "efficientnet_b0"):
        in_f = m.classifier[-1].in_features
        m.classifier = nn.Sequential(nn.Dropout(max(dropout, 0.2)), nn.Linear(in_f, num_classes))
    else:
        raise ValueError(f"unsupported backbone: {name}")
    if freeze:
        for p in m.parameters():
            p.requires_grad = False
        for p in (m.fc if name.startswith("resnet") else m.classifier).parameters():
            p.requires_grad = True
    return m


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ==========================================================================
# 2. 訓練 / 評估
# ==========================================================================
def topk_correct(logits, y, ks=(1, 5)):
    top = logits.topk(max(ks), dim=1).indices
    hit = top.eq(y.unsqueeze(1))
    return [hit[:, :k].any(1).sum().item() for k in ks]


@torch.no_grad()
def evaluate(model, loader, criterion=None, device=DEVICE):
    """回傳 dict：loss, top1, top5, probs(N,C), labels(N,)"""
    model.eval()
    probs, labels, loss_sum, c1, c5, n = [], [], 0.0, 0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        if criterion is not None:
            loss_sum += criterion(out, y).item() * len(y)
        a, b = topk_correct(out, y)
        c1, c5, n = c1 + a, c5 + b, n + len(y)
        probs.append(out.softmax(1).cpu())
        labels.append(y.cpu())
    return dict(loss=loss_sum / n, top1=c1 / n, top5=c5 / n,
                probs=torch.cat(probs).numpy(), labels=torch.cat(labels).numpy())


def make_optimizer(model, name="adamw", lr=1e-3, weight_decay=1e-4):
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, optimizer="adamw",
                weight_decay=1e-4, label_smoothing=0.0, device=DEVICE, verbose=True):
    """以 val Top-1 最佳的 epoch 還原權重。回傳 (model, history DataFrame, 訓練秒數)。"""
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    opt = make_optimizer(model, optimizer, lr, weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * len(train_loader), pct_start=0.2)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best_acc, best_state, hist = -1, None, []
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tl, tc, tn = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item() * len(y)
            tc += out.argmax(1).eq(y).sum().item()
            tn += len(y)
        v = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)
        hist.append(dict(epoch=ep, train_loss=tl / tn, train_top1=tc / tn,
                         val_loss=v["loss"], val_top1=v["top1"], val_top5=v["top5"]))
        if v["top1"] > best_acc:
            best_acc, best_state = v["top1"], copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[{ep:02d}/{epochs}] train_loss {tl/tn:.3f} acc {tc/tn:.3f} | "
                  f"val_loss {v['loss']:.3f} top1 {v['top1']:.3f} top5 {v['top5']:.3f}")
    model.load_state_dict(best_state)
    return model, pd.DataFrame(hist), time.time() - t0


def plot_history(hist, title="", tag="model"):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
    ax[0].plot(hist["epoch"], hist["train_loss"], label="train")
    ax[0].plot(hist["epoch"], hist["val_loss"], label="val")
    ax[0].set_title(f"{title} Loss"); ax[0].legend()
    ax[1].plot(hist["epoch"], hist["train_top1"], label="train top1")
    ax[1].plot(hist["epoch"], hist["val_top1"], label="val top1")
    ax[1].plot(hist["epoch"], hist["val_top5"], label="val top5")
    ax[1].set_title(f"{title} Accuracy"); ax[1].legend()
    plt.tight_layout(); q1.save_fig(f"Q2_{tag}_history")


# ==========================================================================
# 3. 測試集預測與整理
# ==========================================================================
def predict_testset(model, test_loader, test_df, class_names, device=DEVICE):
    """回傳 (預測結果 DataFrame, metrics dict)。test_loader 不可 shuffle。"""
    model.to(device)
    r = evaluate(model, test_loader, None, device)
    probs, labels = r["probs"], r["labels"]
    top5 = np.argsort(-probs, axis=1)[:, :5]
    names = np.array(class_names)
    df = pd.DataFrame({
        "file": test_df["name"].values,
        "true": names[labels],
        "pred_top1": names[top5[:, 0]],
        "confidence": probs[np.arange(len(probs)), top5[:, 0]],
        "pred_top5": [", ".join(names[row]) for row in top5],
        "correct_top1": top5[:, 0] == labels,
        "correct_top5": (top5 == labels[:, None]).any(1),
    })
    metrics = dict(top1=r["top1"], top5=r["top5"], probs=probs, labels=labels)
    print(f"Test Top-1: {r['top1']:.4f} | Top-5: {r['top5']:.4f}")
    return df, metrics


def per_class_accuracy(pred_df):
    g = pred_df.groupby("true")[["correct_top1", "correct_top5"]].mean()
    return g.sort_values("correct_top1")


def _load_rgb(path):
    from PIL import Image
    return Image.open(path).convert("RGB")


def show_predictions(pred_df, test_df, mode="random", n=12, ncols=4, title="Model", tag="model",
                     seed=q1.SEED):
    """
    顯示測試集影像與預測結果。標題：T=真實類別、P=Top-1 預測(信心)，綠色=答對、紅色=答錯。
    mode: 'random' 隨機 | 'wrong' 信心最高的錯誤 | 'correct' 信心最高的正確 | 'low_conf' 信心最低
    pred_df 由 predict_testset 產生，與 test_df 同順序。
    """
    if mode == "random":
        idx = pred_df.sample(n, random_state=seed).index
    elif mode == "wrong":
        idx = pred_df[~pred_df.correct_top1].sort_values("confidence", ascending=False).head(n).index
    elif mode == "correct":
        idx = pred_df[pred_df.correct_top1].sort_values("confidence", ascending=False).head(n).index
    else:
        idx = pred_df.sort_values("confidence").head(n).index
    nrows = int(np.ceil(len(idx) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), idx):
        r = pred_df.loc[i]
        ax.imshow(_load_rgb(test_df.loc[i, "path"]))
        ax.set_title(f"T: {r['true']}\nP: {r['pred_top1']} ({r['confidence']:.2f})", fontsize=8,
                     color="green" if r["correct_top1"] else "red")
    plt.suptitle(f"{title} - test predictions ({mode})")
    plt.tight_layout(); q1.save_fig(f"Q2_{tag}_predictions_{mode}")


def show_top5(pred_df, probs, test_df, class_names, idxs, title="Model", tag="model"):
    """單張影像的 Top-5 機率長條圖。probs: predict_testset 回傳 metrics['probs']。"""
    fig, axes = plt.subplots(len(idxs), 2, figsize=(9, 3 * len(idxs)),
                             gridspec_kw={"width_ratios": [1, 1.6]})
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        axes[r, 0].imshow(_load_rgb(test_df.loc[i, "path"])); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"true: {pred_df.loc[i, 'true']}", fontsize=9)
        top = np.argsort(-probs[i])[:5][::-1]
        colors = ["tab:green" if class_names[k] == pred_df.loc[i, "true"] else "tab:gray" for k in top]
        axes[r, 1].barh([class_names[k] for k in top], probs[i][top], color=colors)
        axes[r, 1].set_xlim(0, 1); axes[r, 1].set_title("Top-5 probability (green = true class)", fontsize=9)
    plt.suptitle(f"{title} - Top-5"); plt.tight_layout(); q1.save_fig(f"Q2_{tag}_top5")


def compare_predictions(pred_a, pred_b, test_df, names=("Plain CNN", "ResNet-18"), n=8, seed=q1.SEED,
                        tag="comparison"):
    """同一批隨機測試影像，並排顯示兩個模型的預測。"""
    idx = pred_a.sample(n, random_state=seed).index
    fig, axes = plt.subplots(2, n // 2, figsize=(3.2 * (n // 2), 7.2))
    for ax, i in zip(axes.ravel(), idx):
        ax.imshow(_load_rgb(test_df.loc[i, "path"])); ax.axis("off")
        a, b = pred_a.loc[i], pred_b.loc[i]
        ax.set_title(f"True: {a['true']}\n{names[0]}: {a['pred_top1']} {'✓' if a['correct_top1'] else '✗'}"
                     f"\n{names[1]}: {b['pred_top1']} {'✓' if b['correct_top1'] else '✗'}", fontsize=8)
    plt.tight_layout(); q1.save_fig(f"Q2_prediction_{tag}")


def top_confusions(pred_df, k=10):
    """最常被混淆的 (真實 -> 預測) 品種對。"""
    wrong = pred_df[~pred_df.correct_top1]
    return (wrong.groupby(["true", "pred_top1"]).size().sort_values(ascending=False)
            .head(k).rename("count").reset_index())


# ==========================================================================
# 4. ROC / Macro-AUC
# ==========================================================================
def plot_roc(labels, probs, class_names, title="ROC", show_each=True, tag="model"):
    """One-vs-Rest ROC；Macro-AUC = 各類別 AUC 平均；另畫 macro-average 曲線。"""
    n = len(class_names)
    y = label_binarize(labels, classes=list(range(n)))
    fpr, tpr, aucs = {}, {}, {}
    for i in range(n):
        fpr[i], tpr[i], _ = roc_curve(y[:, i], probs[:, i])
        aucs[i] = auc(fpr[i], tpr[i])
    grid = np.linspace(0, 1, 500)
    macro_tpr = np.mean([np.interp(grid, fpr[i], tpr[i]) for i in range(n)], axis=0)
    macro_auc = float(np.mean(list(aucs.values())))

    plt.figure(figsize=(7, 6))
    if show_each:
        for i in range(n):
            plt.plot(fpr[i], tpr[i], lw=0.7, alpha=0.35)
    plt.plot(grid, macro_tpr, "k-", lw=2.5, label=f"Macro-average (AUC={macro_auc:.4f})")
    plt.plot([0, 1], [0, 1], "r--", lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"{title} - per-class ROC (37 classes)"); plt.legend(loc="lower right")
    plt.tight_layout(); q1.save_fig(f"Q2_{tag}_roc")
    worst = sorted(aucs.items(), key=lambda kv: kv[1])[:5]
    print("AUC 最低的 5 個類別：", [(class_names[i], round(a, 4)) for i, a in worst])
    return macro_auc, {class_names[i]: a for i, a in aucs.items()}


# ==========================================================================
# 5. 模型比較（Accuracy vs. 參數量）
# ==========================================================================
def compare_models(rows):
    """rows: list of dict(name, params, top1, top5, macro_auc, train_sec)"""
    df = pd.DataFrame(rows)
    df["params_M"] = df["params"] / 1e6
    print(df[["name", "params_M", "top1", "top5", "macro_auc", "train_sec"]].round(4).to_string(index=False))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    x = np.arange(len(df)); w = 0.35
    ax[0].bar(x - w / 2, df["top1"], w, label="Top-1")
    ax[0].bar(x + w / 2, df["top5"], w, label="Top-5")
    ax[0].set_xticks(x); ax[0].set_xticklabels(df["name"], rotation=15)
    ax[0].set_title("Test accuracy"); ax[0].legend()
    ax[1].scatter(df["params_M"], df["top1"], s=90)
    for _, r in df.iterrows():
        ax[1].annotate(r["name"], (r["params_M"], r["top1"]), fontsize=8)
    ax[1].set_xlabel("Parameters (M)"); ax[1].set_ylabel("Top-1")
    ax[1].set_title("Accuracy vs. #Params")
    plt.tight_layout(); q1.save_fig("Q2_model_comparison")
    return df


# ==========================================================================
# 5.5 完整 K-Fold 交叉驗證
# ==========================================================================
def run_kfold_cv(tag, build_model, make_fold_loaders, n_splits=5, epochs=10, lr=1e-3,
                 optimizer="adamw", reuse=None, label_smoothing=0.0):
    """
    對 trainval 的每一折：重新建模 -> 訓練 -> 在該折的驗證集評估 Top-1 / Top-5。
    只使用 trainval，不碰官方 test。
    make_fold_loaders(fold) -> (train_loader, val_loader)
    reuse: {fold: (model, hist, sec)} 已訓練好的折（例如 Quiz2 主模型是 fold 0）直接沿用，只做評估。
    回傳 DataFrame（每折成績 + mean/std 兩列）。
    """
    reuse = reuse or {}
    rows = []
    for k in range(n_splits):
        tr, va = make_fold_loaders(k)
        if k in reuse:
            print(f"\n===== [{tag}] fold {k + 1}/{n_splits} (沿用已訓練模型) =====")
            model, _, sec = reuse[k]
        else:
            print(f"\n===== [{tag}] fold {k + 1}/{n_splits} (重新訓練) =====")
            q1.set_seed(q1.SEED + k)
            model = build_model()
            model, _, sec = train_model(model, tr, va, epochs=epochs, lr=lr, optimizer=optimizer,
                                        label_smoothing=label_smoothing, verbose=False)
        r = evaluate(model, va)
        rows.append(dict(fold=k, val_top1=r["top1"], val_top5=r["top5"], sec=sec))
        print(f"  val top1 {r['top1']:.4f} | top5 {r['top5']:.4f}")
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(rows)
    summ = pd.DataFrame([
        dict(fold="mean", val_top1=df.val_top1.mean(), val_top5=df.val_top5.mean(), sec=df.sec.mean()),
        dict(fold="std", val_top1=df.val_top1.std(ddof=1), val_top5=df.val_top5.std(ddof=1), sec=df.sec.std(ddof=1))])
    out = pd.concat([df, summ], ignore_index=True)
    print(out.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(n_splits)
    ax.bar(x - 0.2, df.val_top1, 0.4, label="Top-1")
    ax.bar(x + 0.2, df.val_top5, 0.4, label="Top-5")
    ax.axhline(df.val_top1.mean(), color="C0", ls="--", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"fold{k}" for k in x]); ax.set_ylim(0, 1)
    ax.set_title(f"{tag}: {n_splits}-fold CV  Top-1 {df.val_top1.mean():.3f}±{df.val_top1.std(ddof=1):.3f}")
    ax.legend(loc="lower right")
    plt.tight_layout(); q1.save_fig(f"Q2_{tag}_kfold")
    out.to_csv(q1.OUT_DIR / f"q2_{tag}_kfold.csv", index=False)
    return out


# ==========================================================================
# 6. 超參數實驗
# ==========================================================================
PLAIN_GRID = [  # 一次只改一個因素（one-factor-at-a-time），其餘為 base
    dict(tag="base",         lr=2e-3, widths=(32, 64, 128, 256), dropout=0.3, augment="none",   optimizer="adamw", label_smoothing=0.0),
    dict(tag="lr=5e-4",      lr=5e-4),
    dict(tag="lr=5e-3",      lr=5e-3),
    dict(tag="wide(64..512)", widths=(64, 128, 256, 512)),
    dict(tag="narrow(16..128)", widths=(16, 32, 64, 128)),
    dict(tag="dropout=0.0",  dropout=0.0),
    dict(tag="dropout=0.5",  dropout=0.5),
    dict(tag="aug=basic",    augment="basic"),
    dict(tag="aug=strong",   augment="strong"),
    dict(tag="opt=sgd",      optimizer="sgd", lr=2e-2),
    dict(tag="label_smooth=0.1", label_smoothing=0.1),
]

BACKBONE_GRID = [
    dict(tag="base(resnet18,finetune)", name="resnet18", freeze=False, lr=1e-3, optimizer="adamw", augment="none", label_smoothing=0.0),
    dict(tag="freeze(only head)",  freeze=True, lr=3e-3),
    dict(tag="lr=1e-4",            lr=1e-4),
    dict(tag="lr=3e-3",            lr=3e-3),
    dict(tag="opt=sgd",            optimizer="sgd", lr=1e-2),
    dict(tag="aug=basic",          augment="basic"),
    dict(tag="aug=strong",         augment="strong"),
    dict(tag="label_smooth=0.1",   label_smoothing=0.1),
    dict(tag="mobilenet_v2",       name="mobilenet_v2"),
    dict(tag="resnet50",           name="resnet50"),
]


def run_experiments(kind, grid, make_loaders_fn, epochs=8, num_classes=37, img_size=224):
    """
    kind: 'plain' | 'backbone'
    make_loaders_fn(augment=..., img_size=...) -> (train_loader, val_loader)
    每組設定只在驗證集評估（不碰 test），以 best val Top-1/Top-5 為準。
    """
    base = dict(grid[0])
    results = []
    for i, cfg in enumerate(grid):
        c = {**base, **cfg} if i else dict(base)
        print(f"\n===== [{kind}] {c['tag']} =====")
        tr, va = make_loaders_fn(augment=c["augment"], img_size=img_size)
        if kind == "plain":
            model = PlainCNN(num_classes, c["widths"], c["dropout"])
        else:
            model = build_backbone(c["name"], num_classes, c["freeze"])
        total, trainable = count_params(model)
        model, hist, sec = train_model(
            model, tr, va, epochs=epochs, lr=c["lr"], optimizer=c["optimizer"],
            label_smoothing=c["label_smoothing"], verbose=False)
        best = hist.loc[hist["val_top1"].idxmax()]
        results.append(dict(tag=c["tag"], params_M=total / 1e6, trainable_M=trainable / 1e6,
                            val_top1=best["val_top1"], val_top5=best["val_top5"],
                            train_top1=best["train_top1"], sec=sec))
        print(f"  val top1 {best['val_top1']:.4f} top5 {best['val_top5']:.4f} ({sec:.0f}s)")
        del model
        torch.cuda.empty_cache()
    df = pd.DataFrame(results)
    print(df.round(4).to_string(index=False))
    ax = df.plot.barh(x="tag", y=["val_top1", "val_top5"], figsize=(8, 0.5 * len(df) + 2))
    ax.invert_yaxis(); ax.set_xlim(0, 1); plt.title(f"{kind} hyper-parameter experiments")
    plt.tight_layout(); q1.save_fig(f"Q2_{kind}_hparam_experiments")
    df.to_csv(q1.OUT_DIR / f"q2_{kind}_hparam_experiments.csv", index=False)
    return df
