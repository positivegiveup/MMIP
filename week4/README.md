# Week 4 — NLP & Vision（Sentiment Analysis / ViT vs ResNet / Image Captioning）

## 專案簡介

| 題目 | 主題 | 重點 |
| ----- | --------- | -------------------------------------------------------------------------------- |
| Quiz1 | 資料集準備 | Yelp Review Full（3 類情感）、CUB-200-2011（200 類鳥類）、Flickr8k（影像描述），並輸出分佈與樣本圖 |
| Quiz2 | Sentiment Analysis | 以相同設定訓練 RNN 與 LSTM，比較 Accuracy、Macro-F1、Macro-AUC 與嚴重錯誤率；加做不同序列長度的對照實驗 |
| Quiz3 | Vision Transformer | 在 CUB-200-2011 細粒度分類上微調 ViT-B/16 與 ResNet-50，以 Macro-AUC、Top-1/Top-5 accuracy 比較，並分析每類別表現與最常混淆的鳥種 |
| Quiz4 | Image Captioning | 自訓練 CNN+LSTM 與預訓練 BLIP 比較 BLEU-4、BERTScore、延遲，並用 Gemini 判斷描述是否符合影像 |

> 完整的設定表、逐段結果分析與結論請見 `main.ipynb`（notebook 內含所有程式碼、圖片輸出與分析文字，本檔案僅作專案導覽與環境設置用）。

## 專案結構

```
week4
├── main.ipynb               # 含完整分析報告
├── requirements.txt
├── .env                     # 自行建立，放 GEMINI_API_KEY（不要上傳）
├── Codes/
│   ├── Quiz1.py             # 資料集準備（Yelp / CUB-200-2011 / Flickr8k）
│   ├── Quiz2.py             # RNN vs LSTM（Yelp 3 類）
│   ├── Quiz3.py             # ViT vs ResNet（CUB-200-2011，200 類）
│   └── Quiz4.py             # Image Captioning + Gemini 評判
├── Data/                    # 第一次執行 Quiz1 時自動下載與快取
│   ├── yelp/                # train_3class.csv、test_3class.csv
│   ├── cub/                 # cub_256.npz
│   └── flickr8k/            # images/、captions.json、feats_resnet50.pt
└── Outputs/                 # 圖片 Q{n}_*.png、模型 q{n}_*.pt、結果 q{n}_*.csv
```

## Environment

OS: Linux（NVIDIA GPU，使用 CUDA）
Python: 3.11

> 實驗在 Linux 叢集的 GPU 環境執行。Notebook 的專案根目錄取自 `Path.cwd()`，所以**必須從 `week4` 目錄啟動 Jupyter**。

### Setup

#### 1. Clone the repository

```
git clone <repository-url>
cd MMIP/week4
```

#### 2. Create a virtual environment with Python 3.11

Create the virtual environment:

```
python3.11 -m venv .venv
```

Activate the virtual environment:

```
source .venv/bin/activate
```

Verify the Python version:

```
python --version
```

Expected output:

```
Python 3.11.x
```

#### 3. Install Week 4 dependencies

PyTorch 請先依你的 CUDA 版本到 https://pytorch.org 取得對應安裝指令，再安裝其餘套件：

```
pip install -r requirements.txt
```

#### 4. 設定 API 金鑰與資料來源

- **Gemini（Quiz4 評判用）**：在 `week4/.env` 建立一行

  ```
  GEMINI_API_KEY=你的金鑰
  ```

  Notebook 預設使用的模型名稱在 4 開頭的設定 cell（`GEMINI_MODEL`），若該模型不可用請改成目前可用的 Gemini 模型。
- **CUB-200-2011（Kaggle `wenewone/cub2002011`）**：以 `kagglehub` 下載。若匿名下載失敗，請設定 `KAGGLE_USERNAME` 與 `KAGGLE_KEY` 環境變數。程式會自動尋找官方的 `images.txt`、`train_test_split.txt` 等檔案；找不到時退回資料夾結構並自行做每類 50% 的分層切分（會印出警告）。
- **Yelp / Flickr8k**：由 HuggingFace `datasets` 自動下載（`Yelp/yelp_review_full`、`jxie/flickr8k`），不需金鑰。
- 第一次執行 Quiz1 會下載資料並快取到 `Data/`，Flickr8k 會寫入 8,000 張圖片，需要數分鐘。

### Run Project on Jupyter Notebook

Enter the `week4` directory, then:

```
jupyter notebook
```

Open:

```
week4/main.ipynb
```

> `main.ipynb` 開頭的第一個 code cell 是 Colab 版的 Drive 掛載程式碼（已註解成字串），本機執行不需要。其餘 cell 由上往下執行即可。

### Environment Verification

```
python --version
pip list
python -c "import torch; print(torch.cuda.is_available())"
```

最後一行應印出 `True`（Quiz3、Quiz4 在 CPU 上會非常慢）。

---

## 重點摘要

以下僅列各題的關鍵數據與結論，完整推導與逐段分析請見 `main.ipynb` 對應段落。所有實驗皆為**單一隨機種子（42）**，小幅差距不宜視為統計上顯著。

### Quiz1：資料集準備

| 任務 | 資料集 | 規模 |
| --- | --- | --- |
| Sentiment Analysis | Yelp Review Full → 3 類（1–2 星 negative、3 星 neutral、4–5 星 positive，各類數量相同） | 訓練 90,000、測試 30,000；評論平均 136 字、中位數 101 字 |
| Image Classification | CUB-200-2011，200 種鳥類；沿用官方切分，官方訓練集再切 90/10 為 train/val，官方測試集為測試 | 官方約 5,994 張訓練、5,794 張測試（以執行輸出為準）；影像 resize 為 256×256，訓練裁切 224 |
| Image Captioning | Flickr8k，沿用官方切分，每張 5 句描述 | train 6,000、val 1,000、test 1,000 |

- 輸出：`Q1_yelp_label_dist.png`、`Q1_yelp_length_hist.png`、`Q1_cub_samples.png`、`Q1_cub_label_dist.png`、`Q1_flickr8k_samples.png`、`Q1_flickr8k_caption_length.png`

### Quiz2：RNN vs LSTM（Yelp 3 類）

設定：詞表 20,000、序列長度 256、Embedding 128、單層單向循環層（hidden 128）、Dropout 0.3、Adam lr 1e-3、early stopping（val loss，patience 5，上限 30 epochs）。RNN 與 LSTM 僅差循環單元。

| 模型 | Accuracy | Macro-F1 | Macro-AUC | 嚴重錯誤率 | 參數量 | 最佳 / 實際 epoch |
| --- | --- | --- | --- | --- | --- | --- |
| RNN | 0.6607 | 0.6535 | 0.8309 | 4.28% | 2.59 M | 13 / 18 |
| LSTM | **0.7414** | **0.7423** | **0.8958** | **1.85%** | 2.69 M | 7 / 12 |

- LSTM 在所有指標上都明顯較佳，差距主要在最模糊的 neutral 類別（recall 0.66 對 0.46）；RNN 的 val loss 起伏大，收斂較慢。
- 序列長度實驗（64 / 128 / 256 / 400，patience 2）：兩個模型都隨長度增加而進步，LSTM 在每個長度都領先 8.8–12.6 個百分點，但差距並未隨長度單調擴大，因此**不能**推論「序列越長 RNN 越差」。
- 輸出：

| 檔案 | 說明 |
| --- | --- |
| `Q2_training_curves.png` | RNN / LSTM 的 loss 與 accuracy 曲線 |
| `Q2_confusion_matrix.png` | 列正規化混淆矩陣 |
| `Q2_roc_curve.png` | 三類 OvR ROC |
| `Q2_metrics_bar.png`、`Q2_per_class_f1.png` | 整體指標與每類 F1 |
| `Q2_length_experiment.png` | 序列長度對照 |
| `q2_rnn.pt`、`q2_lstm.pt`、`q2_vocab.json` | 模型權重與詞表 |
| `q2_metrics.csv`、`q2_length_experiment.csv`、`q2_test_predictions.csv` | 結果表與測試集預測 |

### Quiz3：ViT vs ResNet（CUB-200-2011，200 類）

設定：兩者皆使用 ImageNet 預訓練權重並完整微調；AdamW（weight decay 0.05）、ReduceLROnPlateau、混合精度 fp16、batch size 64；骨幹學習率 ViT 3e-5、ResNet 1e-4，分類頭學習率為骨幹的 10 倍；early stopping（val loss，patience 5，上限 30 epochs）。

| 模型 | Top-1 Acc | Top-5 Acc | Macro-F1 | Macro-AUC | 參數量 | 最佳 / 實際 epoch | 訓練時間 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ViT-B/16** | 0.8217 | 0.9563 | 0.8222 | 0.9957 | 85.95 M | 17 / 22 | 365.08 s |
| **ResNet-50** | 0.7970 | 0.9505 | 0.7980 | 0.9964 | 23.92 M | 14 / 19 | 142.70 s |

- ViT 在 Top-1、Top-5 Accuracy 與 Macro-F1 均優於 ResNet，Top-1 高出 2.47 個百分點；ResNet 的 Macro-AUC 則略高 0.0007，顯示 AUC 不一定與最終分類準確率一致。

- ViT 的參數量約為 ResNet 的 3.6 倍，訓練時間約為 2.6 倍。部分鳥種的辨識率偏低，例如 California Gull 的兩模型 Accuracy 均為 0.333，顯示細粒度鳥類分類仍具挑戰性。

- Early stopping 分別選出 ViT 第 17 個 epoch、ResNet 第 14 個 epoch 的最佳檢查點；由於驗證集每類約僅 3 張圖片，驗證指標容易波動，最佳 epoch 的選擇可能受到少數樣本影響。

- 兩者皆使用 ImageNet 預訓練權重，而 ImageNet 本身包含鳥類相關視覺特徵，因此結果具有遷移學習的先天優勢。由於僅使用單一隨機種子，效能差距仍需透過重複實驗確認，不能直接視為統計顯著差異。

| 檔案 | 說明 |
| --- | --- |
| `Q3_training_curves.png` | 訓練 / 驗證 loss、accuracy、Macro-AUC |
| `Q3_auc_comparison.png` | 每類別 AUC 的分佈與 ViT 對 ResNet 散點圖 |
| `Q3_roc_curves.png` | 巨觀平均 ROC（含低 FPR 放大） |
| `Q3_confusion_matrix.png` | 去除對角線的錯誤熱圖 |
| `Q3_worst_classes.png` | 兩模型平均 accuracy 最低的 15 個類別 |
| `Q3_prediction_examples.png` | 隨機 10 張測試圖的預測（任一模型答錯則標題為紅色） |
| `q3_vit.pt`、`q3_resnet.pt` | 模型權重 |
| `q3_metrics.csv`、`q3_per_class.csv`、`q3_top_confusions.csv` | 整體結果、每類別 AUC 與 accuracy、最常混淆的類別對 |

### Quiz4：Image Captioning

兩種方法（batch size = 1 計時圖片 → 文字的端到端時間），共用從測試集隨機抽出的 100 張圖：

- **A：CNN + LSTM（自訓練）**：凍結的 ResNet-50 特徵 + LSTM 解碼器（embedding 128、hidden 256），貪婪解碼，early stopping 於第 21 個 epoch 停止。
- **B：BLIP（預訓練）**：`Salesforce/blip-image-captioning-base`，零樣本推論，beam search（3 beams）。
- **Gemini 只負責評判**（每個方法判斷 10 張，`N_JUDGE = 10`），不產生描述。

| 方法 | BLEU-4 | BERTScore F1 | 平均延遲 | Gemini 符合率 | Gemini 平均分 |
| --- | --- | --- | --- | --- | --- |
| A：CNN+LSTM | 18.52 | 0.9248 | **0.007 s** | 0.1（1/10） | 1.9 |
| B：BLIP | **28.25** | **0.9402** | 0.079 s | **0.8（8/10）** | **4.3** |

- BLIP 在所有品質指標上都較佳；自訓練模型快約 11 倍，但這個比較含解碼方式差異（貪婪對 beam search），不全是模型大小造成。
- BERTScore 未做 baseline 縮放，數值被壓縮在 0.92–0.94，差距比數字顯示的大。
- Gemini 判斷每個方法只有 10 張，符合率各代表 1 張與 8 張，只具方向性；Gemini 的判斷本身也不是人工標準。
- 輸出：

| 檔案 | 說明 |
| --- | --- |
| `Q4_training_curve.png` | 解碼器訓練 / 驗證 loss |
| `Q4_method_comparison.png` | BLEU / BERTScore / Gemini 符合率 / 延遲比較 |
| `Q4_caption_examples.png` | 範例圖與兩種方法的描述（附 Gemini 判斷） |
| `q4_cnn_lstm.pt` | 自訓練解碼器權重與詞表 |
| `q4_generated_captions.csv`、`q4_gemini_judgement.csv`、`q4_results.csv` | 產生的描述、Gemini 判斷明細與結果表 |

## 已知限制

1. 所有實驗皆為單一隨機種子，Quiz3 的小差距與 Quiz2 的長度實驗細節不宜過度解讀。
2. Quiz4 的 BLEU / BERTScore 僅基於 100 張測試圖，Gemini 評判僅基於每方法 10 張。
3. `requirements.txt` 未固定版本；若需完全重現，請以實際環境的 `pip freeze` 取代。
