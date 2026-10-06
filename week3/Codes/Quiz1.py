"""
Quiz1.py - Oxford-IIIT Pet 資料集：問題定義、資料切分、K-Fold、DataLoader
專案結構：
    main.ipynb
    Codes/Quiz1.py, Codes/Quiz2.py, Codes/Quiz3.py
    Data/            <- 官方資料集會下載到 Data/oxford-iiit-pet/（images + annotations）

資料來源：Oxford VGG 官方網站（https://www.robots.ox.ac.uk/~vgg/data/pets/），
由 torchvision.datasets.OxfordIIITPet 負責下載與解壓。
與 Kaggle 版不同，官方版附有 annotations/（trimaps、xmls、trainval.txt、test.txt），
因此 (1) 可以直接使用官方切分，(2) Quiz3 的 XAI 才有 trimap 可做前景量化。
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "Data"
DATA_DIR.mkdir(exist_ok=True)
OXFORD_DIR = DATA_DIR / "oxford-iiit-pet"   # torchvision 固定解壓到這個資料夾名稱
OUT_DIR = ROOT / "Outputs"            # 所有輸出：圖片、模型、預測結果
FIG_DIR, MODEL_DIR = OUT_DIR / "figures", OUT_DIR / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
  （如 Beagle、Pug、Shiba Inu...），每類約 200 張，官方標註共 7,349 張。
- 資料來源：Oxford VGG 官方網站（images.tar.gz + annotations.tar.gz）。
  annotations 內含 trimap（前景/背景/邊界的像素級標註），供 Quiz3 的 XAI 量化模型注意力是否落在寵物身上。
- 困難點：細粒度分類（fine-grained），品種間外觀高度相似
  （例如 Staffordshire Bull Terrier vs American Pit Bull Terrier）。
- 資料切分：直接使用官方 trainval.txt（3,680 張）與 test.txt（3,669 張）；
  trainval 再做 Stratified K-Fold 切成 train/val，test 僅用於最終評估，避免資料洩漏。
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
    """
    從官方網站下載 images.tar.gz 與 annotations.tar.gz 並解壓到 Data/oxford-iiit-pet/。
    已下載過則 torchvision 會直接略過（不會重複下載）。回傳資料集根目錄。
    """
    from torchvision.datasets import OxfordIIITPet
    OxfordIIITPet(root=str(DATA_DIR), split="trainval", target_types="category", download=True)
    print("Dataset dir:", OXFORD_DIR)
    return OXFORD_DIR


def _read_names(txt_path):
    """官方切分檔每行第一欄是影像名稱（'#' 開頭為註解）。"""
    with open(txt_path, encoding="utf-8") as f:
        return [l.split()[0] for l in f if l.strip() and not l.startswith("#")]


def build_dataframes(root=None):
    """
    回傳 (trainval_df, test_df, class_names)。
    DataFrame 欄位：name, path, trimap_path, class_name, species, label
    - 使用官方 annotations/trainval.txt 與 test.txt 切分。
    - 類別、貓/狗皆由檔名推得（貓品種檔名首字母大寫，狗為小寫），不依賴 list.txt。
    - trimap_path 為 annotations/trimaps/{name}.png；Quiz2 不使用此欄位，多一欄不影響既有流程。
    """
    root = Path(root) if root else download_dataset()
    img_dir, ann_dir = root / "images", root / "annotations"
    tv_txt, te_txt, trimap_dir = ann_dir / "trainval.txt", ann_dir / "test.txt", ann_dir / "trimaps"
    for p in (img_dir, tv_txt, te_txt, trimap_dir):
        if not p.exists():
            raise FileNotFoundError(
                f"找不到 {p}。請確認 {root} 是官方解壓後的 oxford-iiit-pet 資料夾，"
                f"或刪除 {root} 後重新執行以重新下載。")

    # 只看 images/ 底層，並排除 macOS 壓縮檔可能夾帶的 ._xxx 垃圾檔
    jpgs = {p.stem: p for p in img_dir.glob("*.jpg") if not p.name.startswith("._")}

    def to_df(names, split):
        kept = [n for n in names if n in jpgs]
        if len(kept) < len(names):
            print(f"[{split}] 警告：{len(names) - len(kept)} 個名稱在 images/ 找不到對應 jpg，已略過。")
        df = pd.DataFrame({"name": kept})
        df["path"] = df["name"].map(lambda n: str(jpgs[n]))
        df["trimap_path"] = df["name"].map(lambda n: str(trimap_dir / f"{n}.png"))
        df["class_name"] = df["name"].str.rsplit("_", n=1).str[0]
        df["species"] = df["name"].str[0].map(lambda c: "cat" if c.isupper() else "dog")
        n_miss = int((~df["trimap_path"].map(lambda p: Path(p).exists())).sum())
        if n_miss:
            print(f"[{split}] 警告：{n_miss} 張影像沒有對應的 trimap。")
        return df

    trainval_df = to_df(_read_names(tv_txt), "trainval")
    test_df = to_df(_read_names(te_txt), "test")
    print(f"使用官方切分：{ann_dir}  (trainval={len(trainval_df)}, test={len(test_df)})")

    class_names = sorted(set(trainval_df["class_name"]) | set(test_df["class_name"]))
    cid = {c: i for i, c in enumerate(class_names)}
    for df in (trainval_df, test_df):
        df["label"] = df["class_name"].map(cid)
    return trainval_df.reset_index(drop=True), test_df.reset_index(drop=True), class_names


# --------------------------------------------------------------------------
# 2. 資料探索
# --------------------------------------------------------------------------
def summarize(trainval_df, test_df, class_names):
    print(PROBLEM_DEFINITION)
    print(f"類別數：{len(class_names)}  | trainval：{len(trainval_df)}  | test：{len(test_df)}")
    print("貓/狗(trainval)：", trainval_df["species"].value_counts().to_dict())
    print("貓/狗(test)：", test_df["species"].value_counts().to_dict())
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
# 2.5 Trimap（官方像素級標註）：1 = 寵物, 2 = 背景, 3 = 邊界/未分類
# --------------------------------------------------------------------------
def load_trimap(path, img_size=224):
    """
    讀取 trimap 並縮放到 (img_size, img_size)，回傳 uint8 陣列 (H, W)，值為 1/2/3。
    必須用 NEAREST：任何內插都會產生不存在的 1.5、2.5 之類的假標籤。
    尺寸需與 get_transforms 的 eval transform（Resize((img_size, img_size))）一致，才能與 Grad-CAM 對齊。
    """
    return np.array(Image.open(path).resize((img_size, img_size), Image.NEAREST))


def trimap_to_fg(trimap, include_border=True):
    """trimap -> 布林前景遮罩。include_border=True 時前景 = (trimap != 2)，與 Quiz3.xai_quantify 一致；
    False 時只取寵物本體 (trimap == 1)。"""
    return (trimap != 2) if include_border else (trimap == 1)


def show_trimap_samples(df, n=4, seed=SEED, img_size=224):
    """每列：原圖 | trimap | 前景遮罩（含邊界）疊圖，用來確認影像與 trimap 對得上。"""
    s = df.sample(n, random_state=seed)
    fig, axes = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
    for r, (_, row) in enumerate(s.iterrows()):
        img = Image.open(row["path"]).convert("RGB").resize((img_size, img_size))
        tm = load_trimap(row["trimap_path"], img_size)
        axes[r, 0].imshow(img); axes[r, 0].set_title(row["class_name"], fontsize=8)
        axes[r, 1].imshow(tm, cmap="gray", vmin=1, vmax=3); axes[r, 1].set_title("trimap (1=pet,2=bg,3=edge)", fontsize=7)
        axes[r, 2].imshow(img); axes[r, 2].imshow(trimap_to_fg(tm), cmap="Greens", alpha=0.45)
        axes[r, 2].set_title(f"foreground {trimap_to_fg(tm).mean():.0%}", fontsize=8)
        for a in axes[r]:
            a.axis("off")
    plt.tight_layout()
    save_fig("Q1_trimap_samples")


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


class PetTrimapDataset(Dataset):
    """
    供 XAI 使用：回傳 (影像 tensor, label, trimap tensor[H,W] long，值 1/2/3)。
    只使用 eval transform（Resize + Normalize，無隨機擴增），影像與 trimap 以相同尺寸對齊。
    PetDataset 不變，因此 Quiz2 的訓練流程完全不受影響。
    """
    def __init__(self, df, img_size=224):
        self.paths = df["path"].tolist()
        self.trimaps = df["trimap_path"].tolist()
        self.labels = df["label"].tolist()
        self.img_size = img_size
        self.tf = get_transforms(img_size, "none")[1]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = self.tf(Image.open(self.paths[i]).convert("RGB"))
        tm = torch.from_numpy(load_trimap(self.trimaps[i], self.img_size)).long()
        return img, self.labels[i], tm


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
