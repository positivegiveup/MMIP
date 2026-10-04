# Week 3：CNN 影像分類（Oxford-IIIT Pet）

本週作業包含三個 Quiz，皆整合於 [`main.ipynb`](main.ipynb)（可直接於 Google Colab 執行），程式碼模組化放在 `Codes/` 中。

| Quiz  | 主題                                   | 資料集                                                                                         | 模型 / 方法                                        |
| ----- | -------------------------------------- | ---------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| Quiz1 | 問題定義、資料切分、K-Fold             | [Oxford-IIIT Pet](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)     | Stratified 5-Fold                                  |
| Quiz2 | Plain CNN vs. Transfer Learning、ROC、超參數實驗 | 同 Quiz1                                                                              | 自製 Plain CNN、ResNet-18（ImageNet 預訓練）       |
| Quiz3 | 資料擴增、Kernel 視覺化、XAI           | 同 Quiz1                                                                                       | 擴增前後比較、Grad-CAM、Occlusion Sensitivity      |

## 目錄

- [環境需求](#環境需求)
- [專案結構](#專案結構)
- [Quiz1 資料集與切分](#quiz1-資料集與切分)
- [Quiz2 訓練 CNN 影像分類模型](#quiz2-訓練-cnn-影像分類模型)
- [Quiz3 提升泛化能力與模型解釋](#quiz3-提升泛化能力與模型解釋)
- [整體結論與限制](#整體結論與限制)

## 環境需求

- Python 3.13（Google Colab，GPU：CUDA）
- numpy、pandas、matplotlib、scikit-learn、pillow
- torch、torchvision
- kagglehub

```
pip install -r requirements.txt
```

資料集透過 `kagglehub` 自動下載到 `Data/`，無須手動準備。ResNet-18 / MobileNetV2 / ResNet-50 的 ImageNet 預訓練權重由 torchvision 自動下載。

## 專案結構

```
week3/
├── main.ipynb          # 三個 Quiz 的完整程式、輸出與說明
├── README.md
├── requirements.txt
├── Codes/
│   ├── Quiz1.py        # 資料下載、切分、K-Fold、Dataset / DataLoader、存圖與存模型
│   ├── Quiz2.py        # Plain CNN、Backbone、訓練、評估、ROC、K-Fold CV、超參數實驗
│   └── Quiz3.py        # 資料擴增、Kernel 視覺化、Grad-CAM / Occlusion
├── Data/               # kagglehub 下載的資料集
└── Outputs/
    ├── figures/        # Q{1,2,3}_*.png（本 README 使用的結果圖）
    ├── models/         # q{2,3}_*.pt（模型權重）
    └── *.csv           # 測試集預測結果、比較表
```

圖片以 `Q1_`、`Q2_`、`Q3_` 開頭命名，模型權重以 `q2_`、`q3_` 開頭命名。

---

## Quiz1 資料集與切分

**目標**：說明資料來源與內容，建立訓練 / 驗證 / 測試資料，並定義影像分類問題。

### 資料來源與內容

- 資料集：[Oxford-IIIT Pet Dataset](https://www.kaggle.com/datasets/tanlikesmath/the-oxfordiiit-pet-dataset)（Kaggle 版本，約 1.48 GB）。
- 共 **37 個貓狗品種**（**12 種貓、25 種狗**），每個品種約 200 張，影像大小、姿勢、背景與光線皆不一。
- 實際掃描到的 `.jpg` 為 7,390 張（trainval 3,695 + test 3,695）。官方標註的數量為 7,349 張，推測 Kaggle 版本多出的影像未包含在官方切分檔中，本作業以實際讀到的數量為準。
- 貓 / 狗與品種標籤皆由檔名推得（例如 `Abyssinian_1.jpg`、`american_bulldog_100.jpg`）。

[![類別分布](Outputs/figures/Q1_class_distribution.png)](Outputs/figures/Q1_class_distribution.png)

[![資料樣本](Outputs/figures/Q1_samples.png)](Outputs/figures/Q1_samples.png)

### 問題定義

| 項目       | 內容                                                                                       |
| ---------- | ------------------------------------------------------------------------------------------ |
| 任務       | 影像分類（單標籤、多類別，37 類）                                                          |
| 輸入       | 一張 RGB 寵物影像                                                                          |
| 輸出       | 37 個品種的機率分布；最高者為 Top-1 預測，並另外評估 Top-5                                 |
| 困難點     | **細粒度分類**：品種間外觀非常相似（例如 Staffordshire Bull Terrier 與 American Pit Bull Terrier、British Shorthair 與 Russian Blue） |
| 評估指標   | Top-1 / Top-5 Accuracy、各類別 ROC 與 Macro-AUC、參數量、訓練時間                           |

### 資料切分

- 找不到官方 `trainval.txt` / `test.txt`，因此改用**固定種子的 50/50 分層切分**（依品種 stratify），得到 trainval（3,695 張：狗 2,495、貓 1,200）與 test（3,695 張）。**因此成績不能直接與官方 benchmark 比較。**
- trainval 再以 **Stratified 5-Fold** 切成 train / val，每折皆為 **train 2,956 / val 739**，各品種比例一致。
- **test 只在最後評估時使用**，不參與訓練、挑選模型或調整超參數。
- 前處理：影像轉 RGB，Resize 成方形，以 ImageNet 平均 / 標準差正規化（Plain CNN 輸入 128×128，Backbone 輸入 224×224）。

| 折 | Train | Val |
| -- | ----- | --- |
| Fold 0 ~ 4 | 2,956 | 739 |

> 主要實驗（Quiz2 主模型、Quiz3）固定使用 **Fold 0** 的 train / val；完整 5-Fold 交叉驗證見 [Quiz2-4](#quiz2-4-完整-5-fold-交叉驗證)。

---

## Quiz2 訓練 CNN 影像分類模型

### Quiz2-1 Plain CNN（自行設計，無資料擴增）

**架構**：VGG 風格，共 4 個 stage，通道數 32 → 64 → 128 → 256；每個 stage 為 2 × (Conv3×3 → BatchNorm → ReLU) + MaxPool；最後 GAP → Dropout(0.3) → FC(37)。**參數量 1,182,725（1.18M）**。

| 超參數        | 設定                          |
| ------------- | ----------------------------- |
| 輸入大小      | 128 × 128                     |
| Optimizer     | AdamW（lr = 2e-3，weight decay 1e-4） |
| 學習率排程    | OneCycle                      |
| Batch size    | 64                            |
| Epochs        | 30（以 val Top-1 最佳 epoch 還原權重） |
| 資料擴增      | 無                            |

[![Plain CNN 訓練曲線](Outputs/figures/Q2_plain_history.png)](Outputs/figures/Q2_plain_history.png)

**結果（Test Dataset，3,695 張）**：**Top-1 = 52.75%、Top-5 = 86.58%**。

- 明顯過擬合：訓練準確率 87.7%，驗證約 53%。前 20 個 epoch 驗證曲線震盪大（例如第 10 epoch 為 23.4%、第 11 epoch 降到 13.3%），後期隨學習率下降才穩定。
- 預測結果存於 `Outputs/q2_pred_plain.csv`（欄位：`file`、`true`、`pred_top1`、`confidence`、`pred_top5`、`correct_top1`、`correct_top5`）。
- 最難分辨的品種（Top-1）：

| 品種                       | Top-1 | Top-5 |
| -------------------------- | ----- | ----- |
| american_pit_bull_terrier  | 0.200 | 0.810 |
| boxer                      | 0.300 | 0.850 |
| beagle                     | 0.330 | 0.930 |
| chihuahua                  | 0.340 | 0.800 |
| english_cocker_spaniel     | 0.370 | 0.900 |
| staffordshire_bull_terrier | 0.379 | 0.611 |

### Quiz2-2 經典 CNN Backbone：ResNet-18（Transfer Learning，無資料擴增）

**做法**：載入 ImageNet 預訓練的 ResNet-18，把 `fc` 換成 Dropout + Linear(37)，**全網路微調**。**參數量 11,195,493（11.2M）**。

| 超參數        | 設定                          |
| ------------- | ----------------------------- |
| 輸入大小      | 224 × 224                     |
| Optimizer     | AdamW（lr = 1e-3，weight decay 1e-4） |
| 學習率排程    | OneCycle                      |
| Batch size    | 64                            |
| Epochs        | 10（以 val Top-1 最佳 epoch 還原權重） |
| 資料擴增      | 無                            |

[![ResNet-18 訓練曲線](Outputs/figures/Q2_resnet18_history.png)](Outputs/figures/Q2_resnet18_history.png)

**結果（Test Dataset）**：**Top-1 = 85.58%、Top-5 = 98.35%**。

- 預訓練讓第 1 個 epoch 的驗證 Top-1 就達 76.6%。第 2 ~ 4 epoch 驗證暫時下降（最低 36.9%），推測是 OneCycle 在第 2 個 epoch 到達最大學習率 1e-3，對預訓練權重偏大，之後隨學習率下降而恢復（最佳為第 8 epoch，val Top-1 = 84.6%）。
- 第 7 epoch 起訓練準確率已達 100%，仍有過擬合（訓練 100% vs. 驗證約 85%）。
- 預測結果存於 `Outputs/q2_pred_resnet18.csv`。

**測試集預測範例**（前 8 筆）：

| 檔名                  | 真實類別          | Plain CNN 預測（信心）        | ResNet-18 預測（信心）        |
| --------------------- | ----------------- | ----------------------------- | ----------------------------- |
| wheaten_terrier_107   | wheaten_terrier   | wheaten_terrier (0.63) ✓      | wheaten_terrier (1.00) ✓      |
| Russian_Blue_24       | Russian_Blue      | Russian_Blue (0.53) ✓         | Russian_Blue (1.00) ✓         |
| pug_81                | pug               | havanese (0.28) ✗             | pug (0.91) ✓                  |
| boxer_144             | boxer             | basset_hound (0.77) ✗         | boxer (0.95) ✓                |
| British_Shorthair_154 | British_Shorthair | Russian_Blue (0.70) ✗         | British_Shorthair (0.99) ✓    |
| pug_109               | pug               | pug (0.79) ✓                  | pug (1.00) ✓                  |
| leonberger_127        | leonberger        | leonberger (0.87) ✓           | leonberger (0.99) ✓           |
| Siamese_154           | Siamese           | Siamese (0.28) ✓              | Siamese (0.39) ✓              |

### Quiz2-3 ROC Curve、Macro-AUC 與參數量比較

**做法**：對 37 個類別各畫一條 One-vs-Rest ROC 曲線；**Macro-AUC = 各類別 AUC 的平均**；另畫 macro-average 曲線。

| Plain CNN | ResNet-18 |
| --------- | --------- |
| [![Plain ROC](Outputs/figures/Q2_plain_roc.png)](Outputs/figures/Q2_plain_roc.png) | [![ResNet ROC](Outputs/figures/Q2_resnet18_roc.png)](Outputs/figures/Q2_resnet18_roc.png) |

AUC 最低的類別：

| 模型      | AUC 最低的 5 個類別（AUC）                                                                                          |
| --------- | ------------------------------------------------------------------------------------------------------------------- |
| Plain CNN | american_pit_bull_terrier (0.868)、staffordshire_bull_terrier (0.871)、chihuahua (0.893)、boxer (0.900)、shiba_inu (0.933) |
| ResNet-18 | american_pit_bull_terrier (0.978)、staffordshire_bull_terrier (0.980)、Ragdoll (0.983)、american_bulldog (0.984)、boxer (0.986) |

**Accuracy 與參數量比較（Test Dataset）**：

| 模型           | 參數量  | Top-1  | Top-5  | Macro-AUC | 訓練時間 |
| -------------- | ------- | ------ | ------ | --------- | -------- |
| Plain CNN      | 1.18M   | 52.75% | 86.58% | 0.9551    | 398 s    |
| ResNet-18 (TL) | 11.20M  | 85.58% | 98.35% | 0.9946    | 177 s    |

[![模型比較](Outputs/figures/Q2_model_comparison.png)](Outputs/figures/Q2_model_comparison.png)

- ResNet-18 的參數量約為 Plain CNN 的 **9.5 倍**，Top-1 高 **32.8 個百分點**，Macro-AUC 高約 0.04，遷移學習以較多的參數換到很大的準確率提升。
- 兩個模型 AUC 最低的類別幾乎相同（pit bull、staffordshire、boxer），表示這些品種本身就難分，並非單一模型的問題。
- Macro-AUC 比 Top-1 高很多，因為 AUC 衡量機率排序能力：即使 argmax 預測錯誤，正確類別的分數仍可能排在前面（Top-5 遠高於 Top-1 也是同樣原因）。
- Plain CNN 較小但訓練時間較長，因為它訓練 30 個 epoch（ResNet 為 10）；Colab 僅 2 個 CPU，推測資料載入也是瓶頸。

### Quiz2-4 完整 5-Fold 交叉驗證

**做法**：對 trainval 的 5 折各自重新建模、訓練，並在該折驗證集評估；fold 0 沿用上面已訓練好的模型。只使用 trainval，不碰 test。目的是確認 fold 0 的成績不是靠運氣切到特定資料。

| 模型      | Fold 0 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | **Top-1 平均 ± 標準差** | Top-5 平均 ± 標準差 |
| --------- | ------ | ------ | ------ | ------ | ------ | ----------------------- | ------------------- |
| Plain CNN | 53.59% | 54.13% | 56.29% | 50.20% | 53.99% | **53.64% ± 2.19%**      | 85.76% ± 0.67%      |
| ResNet-18 | 84.57% | 84.98% | 87.82% | 85.66% | 85.93% | **85.79% ± 1.25%**      | 98.43% ± 0.25%      |

| Plain CNN | ResNet-18 |
| --------- | --------- |
| [![Plain K-Fold](Outputs/figures/Q2_plain_kfold.png)](Outputs/figures/Q2_plain_kfold.png) | [![ResNet K-Fold](Outputs/figures/Q2_resnet18_kfold.png)](Outputs/figures/Q2_resnet18_kfold.png) |

- 兩個模型的差距（約 32 個百分點）遠大於各自的標準差（2.2 與 1.3 個百分點），**ResNet-18 優於 Plain CNN 的結論穩定**。
- Fold 0 的成績與 5 折平均接近（Plain 53.6% vs. 53.6%；ResNet 84.6% vs. 85.8%，低約 1.2 個百分點、在 1 個標準差內），代表前面只用 fold 0 的結果沒有明顯偏離。
- ResNet-18 的 5-Fold 平均（85.8%）與獨立 test 的 Top-1（85.6%）一致。

### Quiz2 進階：超參數實驗

> 以下每一組設定都是**重新初始化模型並重新訓練**（超參數須在訓練前決定，無法套用在已訓練模型上），所以**不使用**上面 Quiz2-1 / 2-2 訓練好的模型。
> 為節省時間，每組只訓練較少的 epoch，且只用 Fold 0 的驗證集比較，不碰 test。各組之間可互相比較，但**不能與 30 / 10 epochs 的主模型直接比較**。
> 驗證集僅 739 張（標準誤約 ±1.6 個百分點），且每組只訓練一次，**3 個百分點內的差距不宜視為顯著**。

#### Plain CNN 實驗設計

採 **one-factor-at-a-time**：以 base 為基準，每次只改一個因素。

- **Base**：widths (32, 64, 128, 256)、dropout 0.3、AdamW、lr 2e-3、無擴增、無 label smoothing，**8 epochs**，輸入 128 × 128。
- **變動因素**：學習率（5e-4 / 5e-3）、網路寬度（narrow 16 ~ 128 / wide 64 ~ 512）、Dropout（0.0 / 0.5）、資料擴增（basic / strong）、Optimizer（SGD，lr 2e-2）、Label smoothing（0.1）。

| 設定                 | 參數量 | val Top-1 | val Top-5 | train Top-1 | 與 base 差異 |
| -------------------- | ------ | --------- | --------- | ----------- | ------------ |
| base                 | 1.18M  | 24.90%    | 58.19%    | 23.95%      | —            |
| lr = 5e-4            | 1.18M  | 23.41%    | 58.32%    | 27.58%      | −1.5         |
| lr = 5e-3            | 1.18M  | 20.57%    | 52.50%    | 22.21%      | −4.3         |
| wide (64 ~ 512)      | 4.71M  | 24.76%    | 61.16%    | 28.23%      | −0.1         |
| narrow (16 ~ 128)    | 0.30M  | 19.35%    | 54.40%    | 20.21%      | −5.5         |
| dropout = 0.0        | 1.18M  | **26.93%** | 62.38%   | 30.71%      | +2.0（最佳） |
| dropout = 0.5        | 1.18M  | 22.87%    | 57.51%    | 22.55%      | −2.0         |
| aug = basic          | 1.18M  | 25.44%    | 62.25%    | 25.99%      | +0.5         |
| aug = strong         | 1.18M  | 16.64%    | 48.04%    | 15.35%      | −8.3（最差） |
| opt = SGD            | 1.18M  | 25.30%    | 61.03%    | 29.11%      | +0.4         |
| label smoothing 0.1  | 1.18M  | 25.71%    | 60.62%    | 27.89%      | +0.8         |

[![Plain 超參數實驗](Outputs/figures/Q2_plain_hparam_experiments.png)](Outputs/figures/Q2_plain_hparam_experiments.png)

**分析**

- 8 epochs 時所有設定的訓練準確率只有 15% ~ 31%，**模型仍在欠擬合**，因此「增加正則化」（高 dropout、強擴增）反而傷害表現，去掉 dropout 略有幫助。
- 明確有害的只有：強擴增（−8.3）、網路過窄（−5.5）、學習率過大（−4.3）。其他設定的差異都在雜訊範圍內。加寬網路（4 倍參數）沒有帶來收益。
- 此表是短訓練下的比較，**不代表 30 epochs 完整訓練後的排序**（完整訓練時過擬合才是主要問題，見 Quiz3）。

#### ResNet-18 Backbone 實驗設計

- **Base**：ResNet-18 全網路微調、AdamW、lr 1e-3、無擴增、無 label smoothing，**5 epochs**，輸入 224 × 224。
- **變動因素**：凍結 backbone（只訓練分類頭，lr 3e-3）、學習率（1e-4 / 3e-3）、Optimizer（SGD，lr 1e-2）、資料擴增（basic / strong）、Label smoothing（0.1）、換 Backbone（MobileNetV2 / ResNet-50）。

| 設定                          | 參數量 | 可訓練參數 | val Top-1 | val Top-5 | train Top-1 |
| ----------------------------- | ------ | ---------- | --------- | --------- | ----------- |
| base（ResNet-18，AdamW 1e-3） | 11.20M | 11.20M     | 86.47%    | 98.92%    | 99.97%      |
| 凍結（只訓練分類頭）          | 11.20M | 0.019M     | 89.31%    | 98.92%    | 93.44%      |
| lr = 1e-4                     | 11.20M | 11.20M     | 89.85%    | 99.46%    | 99.01%      |
| lr = 3e-3                     | 11.20M | 11.20M     | 77.81%    | 96.89%    | 98.95%      |
| opt = SGD                     | 11.20M | 11.20M     | **90.80%** | 99.32%   | 99.93%      |
| aug = basic                   | 11.20M | 11.20M     | 86.47%    | 98.92%    | 98.78%      |
| aug = strong                  | 11.20M | 11.20M     | 84.44%    | 98.65%    | 91.58%      |
| label smoothing 0.1           | 11.20M | 11.20M     | 87.55%    | 98.78%    | 100.00%     |
| MobileNetV2                   | 2.27M  | 2.27M      | 90.66%    | **99.73%** | 99.32%     |
| ResNet-50                     | 23.58M | 23.58M     | 88.36%    | 99.32%    | 99.69%      |

[![Backbone 超參數實驗](Outputs/figures/Q2_backbone_hparam_experiments.png)](Outputs/figures/Q2_backbone_hparam_experiments.png)

**分析**

- **學習率是最敏感的因素**：lr 太大（3e-3）會明顯破壞預訓練特徵（−8.7）；較小的 lr（1e-4）或 SGD 都比 base 好 3 ~ 4 個百分點。連只訓練分類頭的凍結版本（89.3%）都勝過 base，顯示 base 的 AdamW lr = 1e-3 對微調偏激進。
- **MobileNetV2 的性價比最高**：參數量只有 ResNet-18 的約 1/5、ResNet-50 的約 1/10，準確率與最佳設定相當（90.7%）；ResNet-50 參數最多但沒有更好。
- 5 epochs 內強擴增略差，推測是訓練時間短、尚未收斂。
- 啟示：Quiz2 主模型的 lr = 1e-3 並非最佳，改用 SGD 或 lr = 1e-4 預期可再提升（**本次未重新訓練驗證**）。

---

## Quiz3 提升泛化能力與模型解釋

### Quiz3-1 資料擴增前後比較

**實驗設計**

- **擴增前**：直接沿用 Quiz2 訓練好的 Plain CNN / ResNet-18（皆為無擴增）。
- **擴增後**：相同的超參數與 epoch（Plain 30 / ResNet 10）、相同種子，**只改擴增策略並重新訓練**（擴增發生在訓練階段，無法套在已訓練模型上）。
- 擴增只作用於訓練集；驗證 / 測試集只做 Resize + Normalize。「train」準確率是在**未擴增**的訓練影像上評估，train − val gap 用來衡量過擬合。

**擴增策略**

| 策略        | 內容                                                                                   | 設計理由                                                                                   |
| ----------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| none        | 無擴增（基準）                                                                         | 對照組                                                                                     |
| geometric   | RandomResizedCrop（scale 0.6 ~ 1）+ 水平翻轉 + 旋轉 ±15°                              | 模擬拍攝距離、構圖與角度差異；**不用垂直翻轉**（寵物幾乎不會倒立）；crop 不縮太小，避免裁掉耳型、臉部等品種特徵 |
| photometric | ColorJitter（亮度 / 對比 / 飽和 0.3，色相僅 0.03）+ 輕微模糊                           | 模擬光線與相機差異；**色相刻意只動一點**，因為毛色是品種的重要線索                          |
| full        | geometric + photometric + RandomErasing（p = 0.25）                                    | RandomErasing 隨機遮蔽局部，迫使模型使用多個部位判斷，降低過擬合                            |
| trivialaug  | crop / flip + TrivialAugmentWide                                                       | 自動化擴增對照組（已實作於程式中，本次實驗未訓練）                                          |

[![擴增範例](Outputs/figures/Q3_aug_examples.png)](Outputs/figures/Q3_aug_examples.png)

**結果**

| 模型      | 擴增      | train（無擴增圖） | val Top-1 | test Top-1 | test Top-5 | train − val gap | 訓練時間 |
| --------- | --------- | ----------------- | --------- | ---------- | ---------- | --------------- | -------- |
| Plain CNN | none      | 93.9%             | 53.6%     | 52.7%      | 86.6%      | 0.404           | 398 s    |
| Plain CNN | geometric | 65.6%             | 53.2%     | 50.5%      | 85.0%      | 0.124           | 435 s    |
| Plain CNN | full      | 54.5%             | 43.7%     | 44.7%      | 81.9%      | 0.108           | 612 s    |
| ResNet-18 | none      | 100%              | 84.6%     | 85.6%      | 98.3%      | 0.154           | 177 s    |
| ResNet-18 | geometric | 99.8%             | 86.5%     | 86.1%      | 98.2%      | 0.133           | 195 s    |
| ResNet-18 | full      | 99.5%             | 86.7%     | **87.1%**  | **98.9%**  | 0.127           | 350 s    |

| Plain CNN | ResNet-18 |
| --------- | --------- |
| [![Plain 擴增比較](Outputs/figures/Q3_plaincnn_aug_comparison.png)](Outputs/figures/Q3_plaincnn_aug_comparison.png) | [![ResNet 擴增比較](Outputs/figures/Q3_resnet18_aug_comparison.png)](Outputs/figures/Q3_resnet18_aug_comparison.png) |

**分析**

- **ResNet-18**：擴增使 train − val gap 由 0.154 降至 0.127，val 與 test Top-1 都小幅上升（full：test +1.5 個百分點，Top-5 +0.5）。增幅不大但方向一致，符合預期：預訓練特徵已很強，擴增主要改善泛化。
- **Plain CNN**：擴增確實**大幅縮小過擬合**（gap 0.404 → 0.108 ~ 0.124），但**準確率反而下降**（52.7% → 50.5% / 44.7%），與原先「從零訓練的小模型受益最大」的預期**不符**。未擴增圖上的訓練準確率只剩 65.6% / 54.5%，代表模型在固定的 30 epochs 內尚未擬合擴增後的資料（欠擬合）。較合理的推測是需要更多 epochs 才能發揮擴增效果，**本次未驗證**。
- **成本**：full 策略使訓練時間約增為 2 倍（影像轉換在 CPU 上進行，Colab 僅 2 核心）。

### Quiz3-2 Kernel 視覺化

**做法**：取 Quiz3 擴增（驗證 Top-1 最佳者為 `full`）後的 ResNet-18，視覺化**第一層 7×7 卷積**。先畫出全部 64 個 kernel，再挑兩個：一個邊緣 / 紋理型、一個色彩型，並顯示它們對同一張輸入圖產生的 feature map。

[![第一層全部 Kernel](Outputs/figures/Q3_resnet18_q3_all_kernels.png)](Outputs/figures/Q3_resnet18_q3_all_kernels.png)

[![兩個 Kernel 與 Feature Map](Outputs/figures/Q3_resnet18_q3_two_kernels.png)](Outputs/figures/Q3_resnet18_q3_two_kernels.png)

| Kernel | 外觀                           | 推測用途                                                                                                                       |
| ------ | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| #26    | 上半部偏暗、下半部偏亮         | **水平邊緣（亮度漸層）偵測器**。feature map 突顯水平方向的邊緣（地面橫向紋理、狗頭頂輪廓），垂直方向結構被抑制，用於萃取輪廓與紋理的方向資訊。 |
| #13    | 整體平滑、橙黃色（R 高、B 低） | **低頻色彩 / 色塊偵測器**。沒有邊緣結構，feature map 呈大面積響應：背景較亮、狗被陰影遮住的一側最暗，推測與毛色、光線等整塊區域的顏色有關。 |

- 全部 64 個 kernel 可看出兩大類：**不同方向的邊緣 / 條紋**，以及**色彩對比 / 色塊**（紅綠、藍橙等），這是 CNN 第一層常見的濾波器。後續層再把它們組合成耳朵、眼睛、毛紋等高階特徵。
- 這些 kernel 主要沿用 ImageNet 預訓練的通用濾波器；本次未與預訓練原始權重比對變化量。
- 用途判讀以 kernel 圖與 feature map 為準；程式中的數值指標（`edge_score`、`orient_sel`）為輔助、屬相對數值。

### Quiz3-3 XAI：Grad-CAM 與 Occlusion Sensitivity

**做法**（模型：Quiz3 擴增 `full` 的 ResNet-18）

- **Grad-CAM**：取最後一層卷積的特徵圖，以目標類別分數對該層的梯度做通道加權，得到注意力熱圖並放大回輸入大小（7×7 放大，空間解析度較粗）。
- **Occlusion Sensitivity**：用 32×32 灰色方塊（stride 16）滑動遮蔽，記錄預測機率下降多少，下降越多代表該區域越重要。此法不依賴梯度，可與 Grad-CAM 交叉驗證。
- 由測試集預測結果中挑選 3 個答對案例與 3 個**高信心錯誤**案例。

**答對的案例**

[![答對案例 XAI](Outputs/figures/Q3_resnet18_q3_correct_xai.png)](Outputs/figures/Q3_resnet18_q3_correct_xai.png)

Maine Coon、British Shorthair、Beagle：Grad-CAM 的高亮區集中在**臉部與頭頸**，Occlusion 的高敏感方塊也落在眼睛、鼻口與臉部周圍；背景（沙發、地毯、牆面）幾乎沒有貢獻。兩種方法互相印證，表示模型主要依**動物本體的臉部特徵**判斷，而非背景。

**高信心錯誤案例**

[![錯誤案例 XAI](Outputs/figures/Q3_resnet18_q3_errors_xai.png)](Outputs/figures/Q3_resnet18_q3_errors_xai.png)

| 真實類別      | 預測（信心）            | 模型關注的區域與推測原因                                                           |
| ------------- | ----------------------- | ---------------------------------------------------------------------------------- |
| Maine_Coon    | Bengal (1.00)           | 貓在畫面中很小，注意力落在身體上的虎斑花紋，花紋與 Bengal 相似                     |
| Abyssinian    | Bengal (1.00)           | 注意力在臉與眼睛（下緣略延伸到綠色布料），圖中有浮水印與強烈綠色背景；臉部特徵與 Bengal 接近 |
| Russian_Blue  | British_Shorthair (0.99) | 注意力在臉部與眼睛，但兩品種皆為灰藍短毛、圓臉，外觀極為相似                       |

- 這些錯誤多半**不是「看錯位置」，而是品種外觀本身相近**；模型卻給出接近 1.0 的信心，顯示有**過度自信**的問題（可考慮 label smoothing、溫度校正等）。

**限制**：(1) 找不到 trimaps 目錄，因此「Grad-CAM 落在前景的比例」量化被略過，以上只是少量樣本的定性觀察，不能推論整體；(2) Grad-CAM 空間解析度較粗；(3) 擴增前後注意力區域的差異本次未量化比較。

---

## 整體結論與限制

### 結論

1. **模型比較**：遷移學習的 ResNet-18（11.2M）Test Top-1 85.6%、Top-5 98.4%、Macro-AUC 0.995，遠勝自行設計的 Plain CNN（1.18M，52.7% / 86.6% / 0.955）；5-Fold CV 顯示此差距穩定（85.8% ± 1.3% vs. 53.6% ± 2.2%）。兩者最難分的品種相同（pit bull、staffordshire、boxer 等），屬資料本身的細粒度困難。
2. **超參數**：Plain CNN 在短訓練下處於欠擬合，各設定差異多不顯著，只有「強擴增」「過窄」「lr 過大」明顯有害；ResNet-18 對學習率最敏感，**較小 lr 或 SGD 優於 AdamW 1e-3**；MobileNetV2 以約 1/5 的參數達到最佳等級準確率，性價比最高。
3. **資料擴增**：對 ResNet-18 有小幅而一致的好處（Test Top-1 85.6% → 87.1%，gap 縮小）；對 Plain CNN 雖大幅降低過擬合，但在 30 epochs 內造成欠擬合、準確率下降，需更長訓練才能判斷其真實效益。
4. **可解釋性**：第一層 kernel 可分為方向邊緣與色彩色塊兩類；Grad-CAM / Occlusion 顯示模型聚焦在臉部；錯誤多來自外觀相近品種且信心過高。

### 限制與後續

- 自行 50/50 切分而非官方切分，成績不能直接與文獻比較。
- 超參數與擴增實驗皆為單折單次，小差距可能只是隨機波動。
- 未能取得 trimaps，XAI 缺少前景集中度的量化。
- 後續可做：以 lr = 1e-4 或 SGD 重新訓練 ResNet-18；Plain CNN + 擴增訓練更多 epochs；補上 trimap 量化並比較擴增前後的注意力區域。
