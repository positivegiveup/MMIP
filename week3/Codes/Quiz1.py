"""
Quiz1.py - Oxford-IIIT Pet 資料集：問題定義、資料切分、K-Fold、DataLoader
專案結構：
    main.ipynb
    Codes/Quiz1.py, Codes/Quiz2.py
    Data/            <- kagglehub 會把資料集下載到這裡
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
DATA_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "Outputs"            # 所有輸出：圖片、模型、預測結果
FIG_DIR, MODEL_DIR = OUT_DIR / "figures", OUT_DIR / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
# 必須在 import kagglehub 之前設定，資料才會下載到 ./Data
os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR))

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from sklearn.model_selection import StratifiedKFold

SEED = 42
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

PROBLEM_DEFINITION = """
【問題定義】
- 任務：影像分類（Image Classification），單標籤多類別（37 類）。
- 輸入：一張 RGB 寵物影像（大小不一、姿勢/背景/光線多變）。
- 輸出：37 個品種的機率分布，取最高者為預測品種（Top-1），並可評估 Top-5。
- 資料類別：12 種貓（如 Abyssinian、Bengal、Persian...）+ 25 種狗
  （如 Beagle、Pug、Shiba Inu...），每類約 200 張，總計約 7,390 張。
- 困難點：細粒度分類（fine-grained），品種間外觀高度相似
  （例如 Staffordshire Bull Terrier vs American Pit Bull Terrier）。
- 資料切分：先分出 trainval 與 test（有官方 trainval/test.txt 就用官方，約各 3,680 / 3,669 張；
  找不到則 50/50 分層切分），trainval 再做 Stratified K-Fold 切成 train/val，
  test 僅用於最終評估，避免資料洩漏。
- 評估指標：Top-1 / Top-5 Accuracy、各類別 ROC 與 Macro-AUC、參數量。
"""


def save_fig(name, dpi=150):
    """儲存目前圖片到 Outputs/figures/{name}.png（命名規則 Q1_xxx / Q2_xxx / Q3_xxx）並顯示。"""
    path = FIG_DIR / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print("saved figure:", path)
    plt.show()


def save_model(model, name):
    """儲存權重到 Outputs/models/{name}.pt（命名規則 q1_xxx / q2_xxx / q3_xxx）。"""
    path = MODEL_DIR / f"{name}.pt"
    torch.save(model.state_dict(), path)
    print("saved model:", path)
    return path


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# 1. 下載與讀取 metadata
# --------------------------------------------------------------------------
def download_dataset():
    import kagglehub
    path = kagglehub.dataset_download("tanlikesmath/the-oxfordiiit-pet-dataset")
    print("Path to dataset files:", path)
    return Path(path)


def _find_file(root, name):
    return next(iter(root.rglob(name)), None)


def _read_names(txt_path):
    """官方切分檔每行第一欄是影像名稱（'#' 開頭為註解）。"""
    return [l.split()[0] for l in open(txt_path, encoding="utf-8")
            if l.strip() and not l.startswith("#")]


def build_dataframes(root=None, seed=SEED):
    """
    回傳 (trainval_df, test_df, class_names)。
    - 類別、貓/狗皆由檔名推得（貓品種檔名首字母大寫，狗為小寫），不依賴 list.txt。
    - 若找得到官方 trainval.txt / test.txt 就使用；否則改用 50/50 分層切分（與官方比例相近）。
    """
    root = Path(root) if root else download_dataset()
    jpgs = {p.stem: p for p in root.rglob("*.jpg")}
    if not jpgs:
        tree = [str(p.relative_to(root)) for p in sorted(root.rglob("*"))[:40]]
        raise FileNotFoundError(f"在 {root} 找不到 .jpg 影像，目錄內容(前40)：\n" + "\n".join(tree))

    def to_df(names):
        names = [n for n in names if n in jpgs]
        df = pd.DataFrame({"name": names})
        df["path"] = df["name"].map(lambda n: str(jpgs[n]))
        df["class_name"] = df["name"].str.rsplit("_", n=1).str[0]
        df["species"] = df["name"].str[0].map(lambda c: "cat" if c.isupper() else "dog")
        return df

    tv_txt, te_txt = _find_file(root, "trainval.txt"), _find_file(root, "test.txt")
    if tv_txt is not None and te_txt is not None:
        print("使用官方切分：", tv_txt.parent)
        trainval_df, test_df = to_df(_read_names(tv_txt)), to_df(_read_names(te_txt))
    else:
        print("找不到官方 trainval/test.txt，改用 50/50 分層切分（seed 固定）。")
        from sklearn.model_selection import train_test_split
        all_df = to_df(sorted(jpgs))
        trainval_df, test_df = train_test_split(
            all_df, test_size=0.5, stratify=all_df["class_name"], random_state=seed)

    class_names = sorted(set(trainval_df["class_name"]) | set(test_df["class_name"]))
    cid = {c: i for i, c in enumerate(class_names)}
    for df in (trainval_df, test_df):
        df["label"] = df["class_name"].map(cid)
    trainval_df = trainval_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return trainval_df, test_df, class_names


# --------------------------------------------------------------------------
# 2. 資料探索
# --------------------------------------------------------------------------
def summarize(trainval_df, test_df, class_names):
    print(PROBLEM_DEFINITION)
    print(f"類別數：{len(class_names)}  | trainval：{len(trainval_df)}  | test：{len(test_df)}")
    print("貓/狗(trainval)：", trainval_df["species"].value_counts().to_dict())
    fig, ax = plt.subplots(figsize=(14, 4))
    trainval_df["class_name"].value_counts().sort_index().plot.bar(ax=ax)
    ax.set_title("Class distribution (trainval)")
    plt.tight_layout()
    save_fig("Q1_class_distribution")


def show_samples(df, n=12, seed=SEED):
    s = df.sample(n, random_state=seed)
    fig, axes = plt.subplots(2, n // 2, figsize=(2.2 * n // 2, 5))
    for ax, (_, r) in zip(axes.ravel(), s.iterrows()):
        ax.imshow(Image.open(r["path"]).convert("RGB"))
        ax.set_title(r["class_name"], fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    save_fig("Q1_samples")


# --------------------------------------------------------------------------
# 3. 切分：Stratified K-Fold
# --------------------------------------------------------------------------
def get_kfold_splits(df, n_splits=5, seed=SEED):
    """回傳 [(train_idx, val_idx), ...]，各 fold 類別比例一致。"""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(df["path"], df["label"]))


# --------------------------------------------------------------------------
# 4. Dataset / Transform / DataLoader
# --------------------------------------------------------------------------
def get_transforms(img_size=224, augment="basic"):
    """augment: 'none' | 'basic' | 'strong'"""
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    eval_tf = T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])
    if augment == "none":
        train_tf = eval_tf
    elif augment == "basic":
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(), T.ToTensor(), norm])
    else:  # strong
        train_tf = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
            T.RandomHorizontalFlip(), T.RandomRotation(15),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05), T.ToTensor(), norm,
            T.RandomErasing(p=0.25)])
    return train_tf, eval_tf


class PetDataset(Dataset):
    def __init__(self, df, transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")  # 少數圖為灰階/RGBA
        if self.transform:
            img = self.transform(img)
        return img, self.labels[i]


def make_loaders(trainval_df, test_df, splits, fold=0, img_size=224,
                 batch_size=64, augment="basic", num_workers=4):
    tr_idx, va_idx = splits[fold]
    train_tf, eval_tf = get_transforms(img_size, augment)
    kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0)
    train_ds = PetDataset(trainval_df.iloc[tr_idx], train_tf)
    val_ds = PetDataset(trainval_df.iloc[va_idx], eval_tf)
    test_ds = PetDataset(test_df, eval_tf)
    return (DataLoader(train_ds, batch_size, shuffle=True, drop_last=True, **kw),
            DataLoader(val_ds, batch_size, shuffle=False, **kw),
            DataLoader(test_ds, batch_size, shuffle=False, **kw))
