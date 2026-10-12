"""
Quiz 1：資料集準備
- Yelp Review Full           -> Quiz 2 (Sentiment Analysis, 3 類)
- Intel Image Classification -> Quiz 3 (Vision Transformer / ResNet, 6 類)
- Flickr8k        -> Quiz 4 (Image Captioning)

所有圖片輸出皆存到 Outputs/，命名 Q1_{}.png
"""
import json
import random
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42


# ----------------------------------------------------------------------
# 共用工具
# ----------------------------------------------------------------------
def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_paths(root) -> dict:
    """root 底下建立 Codes / Data / Outputs，回傳路徑字典。"""
    root = Path(root)
    paths = {
        "root": root,
        "codes": root / "Codes",
        "data": root / "Data",
        "outputs": root / "Outputs",
    }
    for key in ("codes", "data", "outputs"):
        paths[key].mkdir(parents=True, exist_ok=True)
    return paths


def _save_fig(paths, name):
    out = paths["outputs"] / name
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[saved] {out}")


# ----------------------------------------------------------------------
# 1) Yelp Review Full -> 3 類（negative / neutral / positive）
# ----------------------------------------------------------------------
YELP_CLASSES = ["negative", "neutral", "positive"]
_STAR2LABEL = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}


def _clean_yelp(text: str) -> str:
    text = text.replace("\\n", " ").replace('\\"', '"')
    return re.sub(r"\s+", " ", text).strip()


def prepare_yelp(paths, n_per_class_train: int = 30000, n_per_class_test: int = 10000):
    """
    下載 Yelp/yelp_review_full 並轉成 3 類：
        1~2 星 -> negative(0)，3 星 -> neutral(1)，4~5 星 -> positive(2)
    為了讓三類數量相同：negative / positive 由兩個星等各取一半，neutral 取 3 星。
    訓練集每類 n_per_class_train 筆，測試集每類 n_per_class_test 筆（皆已平衡）。
    """
    root = paths["data"] / "yelp"
    root.mkdir(parents=True, exist_ok=True)
    f_train, f_test = root / "train_3class.csv", root / "test_3class.csv"

    if not (f_train.exists() and f_test.exists()):
        from datasets import load_dataset
        ds = load_dataset("Yelp/yelp_review_full")

        def build(split, n):
            df = pd.DataFrame({"text": ds[split]["text"], "stars": np.array(ds[split]["label"]) + 1})
            take = {1: n // 2, 2: n // 2, 3: n, 4: n // 2, 5: n // 2}
            df = pd.concat([df[df.stars == st].sample(n=k, random_state=SEED) for st, k in take.items()])
            df["label"] = df["stars"].map(_STAR2LABEL)
            df["text"] = df["text"].map(_clean_yelp)
            return df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        build("train", n_per_class_train).to_csv(f_train, index=False)
        build("test", n_per_class_test).to_csv(f_test, index=False)

    train_df, test_df = pd.read_csv(f_train), pd.read_csv(f_test)
    for df in (train_df, test_df):
        df["n_words"] = df["text"].str.split().str.len()
    print(f"Yelp(3-class) train={len(train_df)}, test={len(test_df)}")
    print(train_df["n_words"].describe().round(1).to_string())

    fig, ax = plt.subplots(figsize=(6, 3.5))
    counts = pd.DataFrame({"train": train_df["label"].value_counts().sort_index(),
                           "test": test_df["label"].value_counts().sort_index()})
    counts.index = YELP_CLASSES
    counts.plot.bar(ax=ax, rot=0)
    ax.set_title("Yelp 3-class label distribution")
    ax.set_ylabel("count")
    _save_fig(paths, "Q1_yelp_label_dist.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(train_df["n_words"].clip(upper=800), bins=60)
    for L in (128, 256, 400):
        ax.axvline(L, color="red", ls="--", lw=0.8)
    ax.set_title("Yelp review length (words, clipped at 800)")
    ax.set_xlabel("words")
    _save_fig(paths, "Q1_yelp_length_hist.png")

    return train_df.drop(columns="n_words"), test_df.drop(columns="n_words")


# ----------------------------------------------------------------------
# 2) Intel Image Classification（6 類場景）
# ----------------------------------------------------------------------
def _find_class_root(base, name):
    """在 kaggle 下載目錄中找出 seg_train / seg_test（資料夾有巢狀結構）。"""
    import os
    for r, dirs, _ in os.walk(base):
        if name in dirs and (Path(r) / name / "buildings").is_dir():
            return Path(r) / name
    raise FileNotFoundError(f"找不到 {name}，請確認資料集已完整下載")


def _load_folder(root, class_names, size):
    from PIL import Image
    xs, ys = [], []
    for ci, c in enumerate(class_names):
        for f in sorted((root / c).iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                xs.append(np.asarray(Image.open(f).convert("RGB").resize((size, size))))
                ys.append(ci)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def prepare_intel(paths, size: int = 150):
    """
    以 kagglehub 下載 puneet6060/intel-image-classification，全部讀進記憶體並快取成
    Data/intel/intel_{size}.npz（之後不必再下載、也不用逐檔讀 Drive）。
    若匿名下載失敗：到 Kaggle 建立 API token，於 Colab 密鑰設定 KAGGLE_USERNAME / KAGGLE_KEY。
    回傳 dict: x_train, y_train（官方 seg_train）、x_test, y_test（官方 seg_test）、class_names。
    seg_pred 沒有標籤，不使用。
    """
    root = paths["data"] / "intel"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"intel_{size}.npz"

    if not cache.exists():
        import kagglehub
        base = Path(kagglehub.dataset_download("puneet6060/intel-image-classification"))
        tr_root, te_root = _find_class_root(base, "seg_train"), _find_class_root(base, "seg_test")
        class_names = sorted(d.name for d in tr_root.iterdir() if d.is_dir())
        x_tr, y_tr = _load_folder(tr_root, class_names, size)
        x_te, y_te = _load_folder(te_root, class_names, size)
        np.savez(cache, x_train=x_tr, y_train=y_tr, x_test=x_te, y_test=y_te,
                 class_names=np.array(class_names))

    d = np.load(cache, allow_pickle=False)
    out = {k: d[k] for k in ("x_train", "y_train", "x_test", "y_test")}
    out["class_names"] = [str(c) for c in d["class_names"]]
    print({k: v.shape for k, v in out.items() if k != "class_names"}, out["class_names"])

    rng = np.random.RandomState(SEED)
    cn = out["class_names"]
    fig, axes = plt.subplots(len(cn), 4, figsize=(6, 1.6 * len(cn)))
    for c, name in enumerate(cn):
        idx = rng.choice(np.where(out["y_train"] == c)[0], 4, replace=False)
        for j, i in enumerate(idx):
            axes[c, j].imshow(out["x_train"][i]); axes[c, j].axis("off")
        axes[c, 0].set_title(name, fontsize=8, loc="left")
    plt.tight_layout()
    _save_fig(paths, "Q1_intel_samples.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    pd.DataFrame({"train": np.bincount(out["y_train"], minlength=len(cn)),
                  "test": np.bincount(out["y_test"], minlength=len(cn))}, index=cn).plot.bar(ax=ax, rot=30)
    ax.set_title("Intel Image class distribution")
    ax.set_ylabel("count")
    _save_fig(paths, "Q1_intel_label_dist.png")
    return out


# ----------------------------------------------------------------------
# 3) Flickr8k
# ----------------------------------------------------------------------
def prepare_flickr8k(paths, hf_repo: str = "jxie/flickr8k"):
    """
    從 HuggingFace 下載 Flickr8k（train 6000 / val 1000 / test 1000），
    圖片存到 Data/flickr8k/images/，描述存成 captions.json：
        {"train": [{"file": "images/train_0000.jpg", "captions": [...5句...]}, ...], ...}
    """
    root = paths["data"] / "flickr8k"
    (root / "images").mkdir(parents=True, exist_ok=True)
    cap_json = root / "captions.json"

    if not cap_json.exists():
        from datasets import load_dataset
        ds = load_dataset(hf_repo)
        out = {}
        for split in ds:
            name = "val" if split.startswith("val") else split
            items = []
            cap_keys = None
            for i, row in enumerate(ds[split]):
                if cap_keys is None:
                    cap_keys = sorted(k for k in row if k.startswith("caption"))
                    print(f"[{split}] caption columns: {cap_keys}")
                fn = f"images/{name}_{i:04d}.jpg"
                row["image"].convert("RGB").save(root / fn, quality=95)
                caps = [str(row[k]).strip() for k in cap_keys if row[k]]
                items.append({"file": fn, "captions": caps})
            out[name] = items
        with open(cap_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False)

    with open(cap_json, encoding="utf-8") as f:
        data = json.load(f)
    print({k: len(v) for k, v in data.items()})

    # 樣本圖
    from PIL import Image
    rng = random.Random(SEED)
    picks = rng.sample(data["train"], 6)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, it in zip(axes.ravel(), picks):
        ax.imshow(Image.open(root / it["file"]))
        ax.set_title(it["captions"][0][:60], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    _save_fig(paths, "Q1_flickr8k_samples.png")

    # 描述長度分佈
    lens = [len(c.split()) for it in data["train"] for c in it["captions"]]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(lens, bins=range(1, 40))
    ax.set_title("Flickr8k caption length (train)")
    ax.set_xlabel("words")
    _save_fig(paths, "Q1_flickr8k_caption_length.png")

    return data
