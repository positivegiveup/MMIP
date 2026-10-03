"""
Quiz3.py - Data Augmentation、Kernel 視覺化、XAI（Grad-CAM / Occlusion）
依賴 Quiz1.py（資料）與 Quiz2.py（模型、訓練、評估）
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

import Quiz1 as q1
import Quiz2 as q2

DEVICE = q2.DEVICE
_NORM = T.Normalize(q1.IMAGENET_MEAN, q1.IMAGENET_STD)

# ==========================================================================
# 1. Data Augmentation 策略
# ==========================================================================
AUG_RATIONALE = """
【擴增策略說明】（皆只作用於訓練集；驗證/測試集只做 Resize + Normalize）
- none         ：基準組，無擴增。
- geometric    ：RandomResizedCrop(scale 0.6~1) + 水平翻轉 + 旋轉±15°。
                 模擬拍攝距離、構圖、角度差異，讓模型不依賴寵物在畫面中的固定位置。
                 不用垂直翻轉（寵物幾乎不會倒立）；crop 不縮太小，避免裁掉品種關鍵特徵（耳型、臉部）。
- photometric  ：ColorJitter(亮度/對比/飽和 0.3, 色相僅 0.03) + 輕微模糊。
                 模擬光線與相機差異；色相刻意只動一點，因為毛色本身是品種的重要線索。
- full         ：geometric + photometric + RandomErasing(p=0.25)。
                 RandomErasing 隨機遮蔽局部，迫使模型使用多個部位判斷，而非依賴單一特徵（降低過擬合）。
- trivialaug   ：geometric 的 crop/flip + TrivialAugmentWide（每張圖隨機一種操作），自動化擴增的對照組。
預期：Plain CNN 從零訓練、資料量小（約 3k 張）→ 過擬合嚴重，擴增效益大；
      預訓練 Backbone 已有強特徵，擴增主要縮小 train/val 差距，增益較小但仍可提升泛化。
"""


def build_train_transform(policy, img_size):
    resize = T.Resize((img_size, img_size))
    crop = T.RandomResizedCrop(img_size, scale=(0.6, 1.0))
    flip = T.RandomHorizontalFlip()
    rot = T.RandomRotation(15)
    color = T.ColorJitter(0.3, 0.3, 0.3, 0.03)
    blur = T.RandomApply([T.GaussianBlur(3, (0.1, 1.5))], p=0.2)
    tail = [T.ToTensor(), _NORM]
    if policy == "none":
        return T.Compose([resize] + tail)
    if policy == "geometric":
        return T.Compose([crop, flip, rot] + tail)
    if policy == "photometric":
        return T.Compose([resize, color, blur] + tail)
    if policy == "full":
        return T.Compose([crop, flip, rot, color, blur] + tail + [T.RandomErasing(p=0.25)])
    if policy == "trivialaug":
        return T.Compose([crop, flip, T.TrivialAugmentWide()] + tail)
    raise ValueError(policy)


def show_aug_examples(df, policies, img_size=224, idx=0, n=4):
    """同一張圖在各策略下抽樣 n 次，直觀看擴增效果。"""
    img = Image.open(df.iloc[idx]["path"]).convert("RGB")
    inv = T.Normalize([-m / s for m, s in zip(q1.IMAGENET_MEAN, q1.IMAGENET_STD)],
                      [1 / s for s in q1.IMAGENET_STD])
    fig, axes = plt.subplots(len(policies), n, figsize=(2.3 * n, 2.3 * len(policies)))
    for r, p in enumerate(policies):
        tf = build_train_transform(p, img_size)
        for c in range(n):
            axes[r, c].imshow(inv(tf(img)).clamp(0, 1).permute(1, 2, 0))
            axes[r, c].axis("off")
        axes[r, 0].set_title(p, loc="left", fontsize=9)
    plt.tight_layout(); q1.save_fig("Q3_aug_examples")


def make_policy_loaders(trainval_df, test_df, splits, fold, policy, img_size=224,
                        batch_size=64, num_workers=4):
    tr_idx, va_idx = splits[fold]
    eval_tf = q1.get_transforms(img_size, "none")[1]
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    tr_df, va_df = trainval_df.iloc[tr_idx], trainval_df.iloc[va_idx]
    return dict(
        train=DataLoader(q1.PetDataset(tr_df, build_train_transform(policy, img_size)),
                         batch_size, shuffle=True, drop_last=True, **kw),
        train_clean=DataLoader(q1.PetDataset(tr_df, eval_tf), batch_size, **kw),  # 量測真實訓練準確率
        val=DataLoader(q1.PetDataset(va_df, eval_tf), batch_size, **kw),
        test=DataLoader(q1.PetDataset(test_df, eval_tf), batch_size, **kw))


def run_aug_comparison(label, build_model, trainval_df, test_df, splits, fold=0, policies=("none", "geometric", "full"),
                       img_size=224, epochs=10, lr=1e-3, optimizer="adamw", batch_size=64, num_workers=4,
                       pretrained=None):
    """
    每個 policy 使用相同超參數訓練 -> 只有擴增不同（控制變因）。
    pretrained: {policy: (model, hist, sec)}，已在 Quiz2 訓練好的模型（通常是 'none' 基準）直接沿用、不重訓。
                其餘（加了擴增的）必須重新訓練，因為擴增是訓練階段的設定。
    回傳 (summary DataFrame, {policy: model}, {policy: history})
    """
    pretrained = pretrained or {}
    rows, models_, hists = [], {}, {}
    for p in policies:
        L = make_policy_loaders(trainval_df, test_df, splits, fold, p, img_size, batch_size, num_workers)
        if p in pretrained:
            print(f"\n===== {label} | augmentation = {p} (沿用 Quiz2 已訓練模型) =====")
            model, hist, sec = pretrained[p]
        else:
            print(f"\n===== {label} | augmentation = {p} (重新訓練) =====")
            q1.set_seed()
            model = build_model()
            model, hist, sec = q2.train_model(model, L["train"], L["val"], epochs=epochs, lr=lr,
                                              optimizer=optimizer, verbose=False)
            q1.save_model(model, f"q3_{label.lower()}_{p}")
        tr = q2.evaluate(model, L["train_clean"])
        va = q2.evaluate(model, L["val"])
        te = q2.evaluate(model, L["test"])
        rows.append(dict(model=label, aug=p, train_top1=tr["top1"], val_top1=va["top1"], val_top5=va["top5"],
                         test_top1=te["top1"], test_top5=te["top5"],
                         gap=tr["top1"] - va["top1"], sec=sec))
        print(f"  train {tr['top1']:.3f} | val {va['top1']:.3f} | test top1 {te['top1']:.3f} top5 {te['top5']:.3f}")
        models_[p], hists[p] = model, hist
    df = pd.DataFrame(rows)
    print(df.round(4).to_string(index=False))
    df.to_csv(q1.OUT_DIR / f"q3_{label.lower()}_aug_comparison.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
    for p, h in hists.items():
        ax[0].plot(h["epoch"], h["train_loss"], label=p)
        ax[1].plot(h["epoch"], h["val_loss"], label=p)
    ax[0].set_title(f"{label} train loss (augmented data)"); ax[1].set_title(f"{label} val loss")
    ax[0].legend(); ax[1].legend()
    x = np.arange(len(df)); w = 0.4
    ax[2].bar(x - w / 2, df["test_top1"], w, label="Test Top-1")
    ax[2].bar(x + w / 2, df["gap"], w, label="Train-Val gap")
    ax[2].set_xticks(x); ax[2].set_xticklabels(df["aug"]); ax[2].legend()
    ax[2].set_title("Accuracy & overfitting gap")
    plt.tight_layout(); q1.save_fig(f"Q3_{label.lower()}_aug_comparison")
    return df, models_, hists


# ==========================================================================
# 2. Kernel 視覺化（第一層卷積）
# ==========================================================================
def first_conv(model):
    return next(m for m in model.modules() if isinstance(m, nn.Conv2d))


def analyze_kernels(model, pad=32):
    """
    對第一層每個 kernel 計算：
      edge_score  ：去除直流後的亮度成分佔總能量比例（越高越像邊緣/條紋偵測器）
      orient_sel  ：2D FFT 峰值能量占比（越高方向性越明確）
      grad_dir    ：亮度變化方向（0~180°；邊緣線方向 = 此角度 + 90°）
      freq        ：空間頻率（cycles/pixel，越高越細緻紋理）
      color_score ：RGB 直流成分的色彩差異占比（越高越像顏色/色彩對立偵測器）
      color_desc  ：偏好的顏色
    """
    W = first_conv(model).weight.detach().cpu().numpy()  # (O,3,k,k)
    rows = []
    for i, w in enumerate(W):
        tot = np.linalg.norm(w) + 1e-8
        gray = w.mean(0)
        g0 = gray - gray.mean()
        edge_score = np.linalg.norm(g0) * np.sqrt(3) / tot
        mag = np.abs(np.fft.fftshift(np.fft.fft2(g0, s=(pad, pad))))
        mag[pad // 2, pad // 2] = 0
        py, px = np.unravel_index(mag.argmax(), mag.shape)
        fy, fx = (py - pad // 2) / pad, (px - pad // 2) / pad
        orient_sel = mag.max() / (mag.sum() + 1e-8)
        grad_dir = (np.degrees(np.arctan2(fy, fx))) % 180
        freq = float(np.hypot(fx, fy))
        dc = w.sum((1, 2))
        color_score = np.linalg.norm(dc - dc.mean()) / (w.shape[-1] * tot)  # 歸一化到 0~1
        rel = dc - dc.mean()
        names = ["R", "G", "B"]
        pos, neg = names[int(rel.argmax())], names[int(rel.argmin())]
        rows.append(dict(idx=i, edge_score=edge_score, orient_sel=orient_sel, grad_dir=grad_dir,
                         freq=freq, color_score=color_score, color_desc=f"{pos}+ / {neg}-"))
    return pd.DataFrame(rows)


def _norm01(a):
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def describe_kernel(r):
    parts = []
    if r["color_score"] > 0.5 and r["edge_score"] < 0.8:
        parts.append(f"色彩/色塊偵測器（偏好 {r['color_desc']}），對毛色、背景色調敏感")
    if r["edge_score"] >= 0.6:
        line = (r["grad_dir"] + 90) % 180
        shape = "細緻紋理" if r["freq"] > 0.25 else "較粗的邊緣/輪廓"
        parts.append(f"方向性{shape}偵測器：對約 {line:.0f}° 的邊緣線反應最強（亮度變化方向 {r['grad_dir']:.0f}°）")
    return "；".join(parts) or "混合型（同時含色彩與邊緣成分）"


def visualize_two_kernels(model, img_tensor, idxs=None, title="Model", tag="model"):
    """
    img_tensor: (3,H,W) 已 normalize。idxs=None 時自動選：
      kernel A = edge_score*orient_sel 最高（邊緣/紋理型）, kernel B = color_score 最高（色彩型）
    畫出 [原圖 | kernel A | A 的 feature map | kernel B | B 的 feature map]。
    """
    stats = analyze_kernels(model)
    if idxs is None:
        a = int(stats.assign(s=stats.edge_score * stats.orient_sel).sort_values("s").idx.iloc[-1])
        b = int(stats[stats.idx != a].sort_values("color_score").idx.iloc[-1])
        idxs = [a, b]
    conv = first_conv(model).eval().to(DEVICE)
    with torch.no_grad():
        fm = conv(img_tensor.unsqueeze(0).to(DEVICE))[0].cpu().numpy()
    W = conv.weight.detach().cpu().numpy()
    inv = T.Normalize([-m / s for m, s in zip(q1.IMAGENET_MEAN, q1.IMAGENET_STD)],
                      [1 / s for s in q1.IMAGENET_STD])

    fig, ax = plt.subplots(1, 5, figsize=(16, 3.4))
    ax[0].imshow(inv(img_tensor).clamp(0, 1).permute(1, 2, 0)); ax[0].set_title("Input")
    for j, k in enumerate(idxs):
        ax[1 + 2 * j].imshow(_norm01(W[k]).transpose(1, 2, 0), interpolation="nearest")
        ax[1 + 2 * j].set_title(f"Kernel #{k} ({W.shape[2]}x{W.shape[3]})")
        ax[2 + 2 * j].imshow(fm[k], cmap="gray")
        ax[2 + 2 * j].set_title(f"Feature map #{k}")
    for a_ in ax:
        a_.axis("off")
    plt.suptitle(f"{title} - first conv layer"); plt.tight_layout(); q1.save_fig(f"Q3_{tag}_two_kernels")
    for k in idxs:
        r = stats.iloc[k]
        print(f"Kernel #{k}: edge={r.edge_score:.2f}, orient_sel={r.orient_sel:.2f}, freq={r.freq:.2f}, "
              f"color={r.color_score:.2f} ({r.color_desc})\n  -> {describe_kernel(r)}")
    return stats, idxs


def show_all_kernels(model, title="Model", ncols=16, tag="model"):
    W = first_conv(model).weight.detach().cpu().numpy()
    n = len(W); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.9, nrows * 0.9))
    for i, a in enumerate(np.atleast_1d(axes).ravel()):
        a.axis("off")
        if i < n:
            a.imshow(_norm01(W[i]).transpose(1, 2, 0), interpolation="nearest")
    plt.suptitle(f"{title} - all first-layer kernels"); plt.tight_layout(); q1.save_fig(f"Q3_{tag}_all_kernels")


# ==========================================================================
# 3. XAI：Grad-CAM 與 Occlusion Sensitivity
# ==========================================================================
def _disable_inplace(model):
    for m in model.modules():
        if isinstance(m, (nn.ReLU, nn.ReLU6)):
            m.inplace = False


def last_conv_layer(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)][-1]


def gradcam(model, x, target=None, layer=None):
    """x: (3,H,W)。回傳 (cam[H,W] in 0~1, pred_idx, prob_pred, target_used)"""
    model.eval().to(DEVICE)
    _disable_inplace(model)
    layer = layer or last_conv_layer(model)
    store = {}
    def fwd_hook(m, i, o):
        store["a"] = o
        o.register_hook(lambda g: store.__setitem__("g", g))

    h = layer.register_forward_hook(fwd_hook)
    try:
        with torch.enable_grad():
            xin = x.unsqueeze(0).to(DEVICE).requires_grad_(True)  # 確保凍結層下仍可取得梯度
            out = model(xin)
            probs = out.softmax(1)[0].detach()
            pred = int(out.argmax(1))
            t = pred if target is None else target
            model.zero_grad()
            out[0, t].backward()
    finally:
        h.remove()
    a, g = store["a"][0].detach(), store["g"][0].detach()
    w = g.mean((1, 2), keepdim=True)                      # 通道重要性 = 梯度的空間平均
    cam = F.relu((w * a).sum(0))
    cam = F.interpolate(cam[None, None], size=x.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.cpu().numpy(), pred, float(probs[pred]), t


@torch.no_grad()
def occlusion_map(model, x, target, patch=32, stride=16, batch=64):
    """以灰色方塊（normalize 後 = 0）滑動遮蔽，機率下降越多代表該區域越重要。"""
    model.eval().to(DEVICE)
    _, H, W = x.shape
    base = model(x.unsqueeze(0).to(DEVICE)).softmax(1)[0, target].item()
    ys = list(range(0, H - patch + 1, stride)); xs = list(range(0, W - patch + 1, stride))
    boxes = [(y, x0) for y in ys for x0 in xs]
    heat = torch.zeros(H, W); cnt = torch.zeros(H, W)
    for s in range(0, len(boxes), batch):
        chunk = boxes[s:s + batch]
        xb = x.unsqueeze(0).repeat(len(chunk), 1, 1, 1).clone()
        for k, (y, x0) in enumerate(chunk):
            xb[k, :, y:y + patch, x0:x0 + patch] = 0.0
        p = model(xb.to(DEVICE)).softmax(1)[:, target].cpu()
        for k, (y, x0) in enumerate(chunk):
            heat[y:y + patch, x0:x0 + patch] += base - p[k]
            cnt[y:y + patch, x0:x0 + patch] += 1
    heat = (heat / cnt.clamp(min=1)).clamp(min=0).numpy()
    return heat / (heat.max() + 1e-8)


def _overlay(ax, img, cam, title):
    ax.imshow(img); ax.imshow(cam, cmap="jet", alpha=0.45); ax.set_title(title, fontsize=8); ax.axis("off")


def explain_samples(model, df, class_names, idxs, img_size=224, title="Model", occlusion=True, tag="model"):
    """每個樣本顯示：原圖(真實/預測) | Grad-CAM | Occlusion"""
    eval_tf = q1.get_transforms(img_size, "none")[1]
    inv = T.Normalize([-m / s for m, s in zip(q1.IMAGENET_MEAN, q1.IMAGENET_STD)],
                      [1 / s for s in q1.IMAGENET_STD])
    ncol = 3 if occlusion else 2
    fig, axes = plt.subplots(len(idxs), ncol, figsize=(3.2 * ncol, 3.2 * len(idxs)))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        row = df.iloc[i]
        x = eval_tf(Image.open(row["path"]).convert("RGB"))
        img = inv(x).clamp(0, 1).permute(1, 2, 0).numpy()
        cam, pred, prob, _ = gradcam(model, x)
        ok = "✓" if pred == row["label"] else "✗"
        axes[r, 0].imshow(img); axes[r, 0].axis("off")
        axes[r, 0].set_title(f"true: {row['class_name']}\npred: {class_names[pred]} ({prob:.2f}) {ok}", fontsize=8)
        _overlay(axes[r, 1], img, cam, "Grad-CAM")
        if occlusion:
            _overlay(axes[r, 2], img, occlusion_map(model, x, pred), "Occlusion")
    plt.suptitle(f"{title} - XAI"); plt.tight_layout(); q1.save_fig(f"Q3_{tag}_xai")


# ---- 以 trimap（官方前景/背景標註）量化「模型到底看哪裡」 ----
def find_trimap_dir(root=None):
    root = Path(root) if root else q1.DATA_DIR
    return next(iter(root.rglob("trimaps")), None)


def xai_quantify(model, df, class_names, trimap_dir, n=150, img_size=224, seed=q1.SEED):
    """
    trimap：1=寵物, 2=背景, 3=邊界。fg = trimap != 2。
      fg_energy ：Grad-CAM 能量落在寵物（含邊界）的比例
      fg_area   ：寵物占畫面比例（隨機注意力的期望值）
      gain      ：fg_energy / fg_area（>1 表示比隨機更集中在寵物身上）
      hit       ：Grad-CAM 最高點是否落在寵物上（pointing game）
    """
    if trimap_dir is None:
        print("找不到 trimaps 目錄，略過前景比例量化（仍可使用 Grad-CAM 視覺化）。")
        return pd.DataFrame(), pd.DataFrame()
    eval_tf = q1.get_transforms(img_size, "none")[1]
    sub = df.sample(min(n, len(df)), random_state=seed)
    rows = []
    for _, r in sub.iterrows():
        tm = Path(trimap_dir) / f"{r['name']}.png"
        if not tm.exists():
            continue
        fg = np.array(Image.open(tm).resize((img_size, img_size), Image.NEAREST)) != 2
        x = eval_tf(Image.open(r["path"]).convert("RGB"))
        cam, pred, prob, _ = gradcam(model, x)
        e = cam[fg].sum() / (cam.sum() + 1e-8)
        rows.append(dict(name=r["name"], correct=pred == r["label"], fg_energy=e, fg_area=fg.mean(),
                         gain=e / (fg.mean() + 1e-8),
                         hit=bool(fg[np.unravel_index(cam.argmax(), cam.shape)])))
    out = pd.DataFrame(rows)
    summ = out.groupby("correct")[["fg_energy", "fg_area", "gain", "hit"]].mean()
    summ["n"] = out.groupby("correct").size()
    summ.loc["all"] = [out.fg_energy.mean(), out.fg_area.mean(), out.gain.mean(), out.hit.mean(), len(out)]
    print(summ.round(3))
    return out, summ


def pick_cases(pred_df, k=3, seed=q1.SEED):
    """從測試預測表挑 k 個答對、k 個答錯（高信心錯誤優先）的索引。"""
    good = pred_df[pred_df.correct_top1].sample(k, random_state=seed).index.tolist()
    bad = pred_df[~pred_df.correct_top1].sort_values("confidence", ascending=False).head(k).index.tolist()
    return good, bad
