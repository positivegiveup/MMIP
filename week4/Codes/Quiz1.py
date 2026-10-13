"""
Quiz 1：資料集準備
- Yelp Review Full           -> Quiz 2 (Sentiment Analysis, 3 類)
- CUB-200-2011              -> Quiz 3 (Vision Transformer / ResNet, 200 類)
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
# 2) CUB-200-2011（200 種鳥類，細粒度分類）
# ----------------------------------------------------------------------
def _find_file(base, name):
    import os
    for r, _, files in os.walk(base):
        if name in files:
            return Path(r) / name
    return None


def _find_class_dir(base):
    """找出含有 >=200 個 'NNN.Name' 類別資料夾的目錄（即 images/）。"""
    import os
    for r, dirs, _ in os.walk(base):
        if sum(bool(re.match(r"^\d{3}\.", d)) for d in dirs) >= 200:
            return Path(r)
    return None


def _clean_cub_name(name: str) -> str:
    return re.sub(r"^\d+\.", "", name).replace("_", " ")


def _read_cub_index(base):
    """
    解析 CUB 目錄，回傳 (圖片路徑 list, 標籤 0-based, is_train bool, 類別名稱 list, 切分來源)。
    優先使用官方檔案 images.txt / image_class_labels.txt / train_test_split.txt / classes.txt；
    若找不到，退回「資料夾結構 + 每類 50% 分層切分」。
    """
    f_img, f_lab = _find_file(base, "images.txt"), _find_file(base, "image_class_labels.txt")
    f_spl, f_cls = _find_file(base, "train_test_split.txt"), _find_file(base, "classes.txt")

    img_root = None
    if f_img is not None and (f_img.parent / "images").is_dir():
        img_root = f_img.parent / "images"
    else:
        img_root = _find_class_dir(base)
    if img_root is None:
        raise FileNotFoundError("找不到 CUB 的 images/ 資料夾，請確認資料集已完整下載")

    if all(f is not None for f in (f_img, f_lab, f_spl, f_cls)):
        rows = [l.split() for l in f_img.read_text().splitlines() if l.strip()]
        lab = dict(l.split() for l in f_lab.read_text().splitlines() if l.strip())
        spl = dict(l.split() for l in f_spl.read_text().splitlines() if l.strip())
        cls = [l.split(maxsplit=1)[1] for l in f_cls.read_text().splitlines() if l.strip()]
        paths_list = [img_root / rel for _, rel in rows]
        y = np.array([int(lab[i]) - 1 for i, _ in rows])
        is_train = np.array([spl[i] == "1" for i, _ in rows])
        return paths_list, y, is_train, [_clean_cub_name(c) for c in cls], "official"

    # 備援：由資料夾結構建立標籤，自行做分層切分
    from sklearn.model_selection import train_test_split
    cls_dirs = sorted(d for d in img_root.iterdir() if d.is_dir())
    paths_list, y = [], []
    for ci, d in enumerate(cls_dirs):
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                paths_list.append(f); y.append(ci)
    y = np.array(y)
    idx_tr, _ = train_test_split(np.arange(len(y)), train_size=0.5, stratify=y, random_state=SEED)
    is_train = np.zeros(len(y), dtype=bool)
    is_train[idx_tr] = True
    print("[warn] 找不到官方切分檔，改用每類 50% 的分層切分")
    return paths_list, y, is_train, [_clean_cub_name(d.name) for d in cls_dirs], "folder"


def _build_cub_arrays(base, size):
    from PIL import Image
    paths_list, y, is_train, class_names, source = _read_cub_index(base)
    x = np.empty((len(paths_list), size, size, 3), dtype=np.uint8)
    for k, p in enumerate(paths_list):
        x[k] = np.asarray(Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR))
        if (k + 1) % 2000 == 0:
            print(f"  loaded {k + 1}/{len(paths_list)}")
    return {"x_train": x[is_train], "y_train": y[is_train].astype(np.int64),
            "x_test": x[~is_train], "y_test": y[~is_train].astype(np.int64),
            "class_names": class_names, "source": source}


def prepare_cub(paths, size: int = 256):
    """
    以 kagglehub 下載 wenewone/cub2002011，全部讀進記憶體（resize 成 size x size）並快取成
    Data/cub/cub_{size}.npz。官方切分：約 5,994 張訓練、5,794 張測試，共 200 類。
    若匿名下載失敗：於 Kaggle 建立 API token，設定 KAGGLE_USERNAME / KAGGLE_KEY 環境變數。
    回傳 dict: x_train, y_train, x_test, y_test, class_names, source。
    """
    root = paths["data"] / "cub"
    root.mkdir(parents=True, exist_ok=True)
    cache = root / f"cub_{size}.npz"

    if not cache.exists():
        import kagglehub
        base = Path(kagglehub.dataset_download("wenewone/cub2002011"))
        out = _build_cub_arrays(base, size)
        np.savez(cache, x_train=out["x_train"], y_train=out["y_train"], x_test=out["x_test"],
                 y_test=out["y_test"], class_names=np.array(out["class_names"]),
                 source=np.array(out["source"]))

    d = np.load(cache, allow_pickle=False)
    out = {k: d[k] for k in ("x_train", "y_train", "x_test", "y_test")}
    out["class_names"] = [str(c) for c in d["class_names"]]
    out["source"] = str(d["source"])
    nc = len(out["class_names"])
    print({k: v.shape for k, v in out.items() if k.startswith(("x_", "y_"))}, f"classes={nc}",
          f"split={out['source']}")

    rng = np.random.RandomState(SEED)
    cls_pick = rng.choice(nc, min(24, nc), replace=False)
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))
    for ax, c in zip(axes.ravel(), cls_pick):
        i = rng.choice(np.where(out["y_train"] == c)[0])
        ax.imshow(out["x_train"][i]); ax.axis("off")
        ax.set_title(out["class_names"][c][:26], fontsize=7)
    plt.tight_layout()
    _save_fig(paths, "Q1_cub_samples.png")

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    n_tr, n_te = np.bincount(out["y_train"], minlength=nc), np.bincount(out["y_test"], minlength=nc)
    ax.hist(n_tr, bins=range(0, int(max(n_tr.max(), n_te.max())) + 3), alpha=0.6, label="train")
    ax.hist(n_te, bins=range(0, int(max(n_tr.max(), n_te.max())) + 3), alpha=0.6, label="test")
    ax.set_title(f"CUB-200-2011: images per class ({nc} classes)")
    ax.set_xlabel("images in a class"); ax.set_ylabel("number of classes"); ax.legend()
    _save_fig(paths, "Q1_cub_label_dist.png")
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
