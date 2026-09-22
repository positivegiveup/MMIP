# -*- coding: utf-8 -*-
"""
Quiz2: Histogram Equalization
- 自行以 NumPy 實作 Histogram Equalization
- 與 OpenCV 的 cv2.equalizeHist() 比較
- 繪製處理前後的灰階直方圖
- 重複執行 N 次取平均，比較執行速度與差異
"""

import os
import time

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 無需顯示視窗，直接存檔
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.NumPy 實作 Histogram Equalization
# ---------------------------------------------------------------------------
def numpy_histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    """
    gray_img : np.ndarray
        單通道灰階影像 (dtype=uint8)
    """
    # 1. 計算灰階直方圖 (0~255)
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))

    # 2. 計算累積分布函數 (CDF)
    cdf = hist.cumsum()

    # 3. 依定義做正規化: s = round((cdf - cdf_min) / (M*N - cdf_min) * 255)
    #    只使用非零的 cdf 部分做 mask，避免除以 0
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    total_pixels = gray_img.size

    cdf_normalized = (cdf_masked - cdf_min) * 255.0 / (total_pixels - cdf_min)
    cdf_normalized = np.ma.filled(cdf_normalized, 0).astype(np.uint8)

    # 4. 用查表 (LUT) 方式將每個像素值映射到新的灰階值
    equalized_img = cdf_normalized[gray_img]

    return equalized_img.astype(np.uint8)

def show_histogram_equalization(image_path: str):
    """
    Quiz2 - 基礎
    顯示 Histogram Equalization 前後的影像與灰階 Histogram
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 使用 NumPy 實作 Histogram Equalization
    equalized = numpy_histogram_equalization(gray)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    axes[0, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Before Histogram Equalization")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(equalized, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("After Histogram Equalization")
    axes[0, 1].axis("off")

    hist_before = compute_hist(gray)
    axes[1, 0].bar(
        np.arange(256),
        hist_before,
        width=1.0
    )
    axes[1, 0].set_title("Before Histogram")
    axes[1, 0].set_xlabel("Gray Level")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].set_xlim(0, 255)

    hist_after = compute_hist(equalized)
    axes[1, 1].bar(
        np.arange(256),
        hist_after,
        width=1.0
    )
    axes[1, 1].set_title("After Histogram")
    axes[1, 1].set_xlabel("Gray Level")
    axes[1, 1].set_ylabel("Pixel Count")
    axes[1, 1].set_xlim(0, 255)

    plt.tight_layout()
    plt.show()

    #return gray, equalized
    return None

# ---------------------------------------------------------------------------
# 2. 使用 OpenCV 內建函式
# ---------------------------------------------------------------------------
def opencv_histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    return cv2.equalizeHist(gray_img)


# ---------------------------------------------------------------------------
# 3. 計算灰階直方圖 (供繪圖使用)
# ---------------------------------------------------------------------------
def compute_hist(gray_img: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(gray_img.flatten(), bins=256, range=(0, 256))
    return hist


# ---------------------------------------------------------------------------
# 4. 主流程
# ---------------------------------------------------------------------------
def run_quiz2(image_path: str, output_dir: str, N: int = 50):
    """
    Parameters
    ----------
    image_path : str
        輸入影像路徑 (例如 "Data/Quiz2.jpg")
    output_dir : str
        輸出資料夾路徑
    N : int
        重複執行次數，取平均計算執行時間

    Returns
    -------
    dict
        內含執行結果統計數值
    """
    os.makedirs(output_dir, exist_ok=True)

    # 讀取影像並轉灰階
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ------------------------------------------------------------------
    # 先各自執行一次，取得輸出影像 (供顯示 / 繪圖 / 存檔用)
    # ------------------------------------------------------------------
    numpy_result = numpy_histogram_equalization(gray)
    opencv_result = opencv_histogram_equalization(gray)

    # ------------------------------------------------------------------
    # 重複執行 N 次，比較平均執行時間
    # ------------------------------------------------------------------
    numpy_times = []
    opencv_times = []

    for _ in range(N):
        t0 = time.perf_counter()
        _ = numpy_histogram_equalization(gray)
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)

        t0 = time.perf_counter()
        _ = opencv_histogram_equalization(gray)
        t1 = time.perf_counter()
        opencv_times.append(t1 - t0)

    avg_numpy_time = float(np.mean(numpy_times))
    avg_opencv_time = float(np.mean(opencv_times))

    # ------------------------------------------------------------------
    # 影像增強效果評估：標準差 (對比度) 與熵 (資訊量)
    # ------------------------------------------------------------------
    def std_and_entropy(im: np.ndarray):
        std_val = float(np.std(im))
        hist = compute_hist(im).astype(np.float64)
        p = hist / hist.sum()
        p_nonzero = p[p > 0]
        entropy = float(-np.sum(p_nonzero * np.log2(p_nonzero)))
        return std_val, entropy

    orig_std, orig_entropy = std_and_entropy(gray)
    numpy_std, numpy_entropy = std_and_entropy(numpy_result)
    opencv_std, opencv_entropy = std_and_entropy(opencv_result)

    # ------------------------------------------------------------------
    # 兩種方法輸出影像的差異
    # ------------------------------------------------------------------
    diff = np.abs(numpy_result.astype(np.int16) - opencv_result.astype(np.int16))
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    # 差異在 <=1 個灰階內視為相同 (四捨五入誤差)
    match_ratio = float(np.mean(diff <= 1))

    # ------------------------------------------------------------------
    # 存檔：處理後影像
    # ------------------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_gray.png"), gray)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_numpy_eq.png"), numpy_result)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_opencv_eq.png"), opencv_result)

    # ------------------------------------------------------------------
    # 繪圖：影像 + 直方圖 (處理前 / NumPy 處理後 / OpenCV 處理後)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 6))

    images = [gray, numpy_result, opencv_result]
    titles = ["Original", "NumPy Equalized", "OpenCV Equalized"]

    for i, (im, title) in enumerate(zip(images, titles)):
        axes[0, i].imshow(im, cmap="gray", vmin=0, vmax=255)
        axes[0, i].set_title(title)
        axes[0, i].axis("off")

        hist = compute_hist(im)
        axes[1, i].bar(np.arange(256), hist, width=1.0, color="steelblue")
        axes[1, i].set_title(f"{title} Histogram")
        axes[1, i].set_xlabel("Gray Level")
        axes[1, i].set_ylabel("Pixel Count")
        axes[1, i].set_xlim([0, 255])

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{base_name}_result.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # 印出比較結果
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Quiz2 - Histogram Equalization ({image_path})")
    print("=" * 60)
    print(f"重複執行次數 N = {N}")
    print(f"[執行速度]")
    print(f"  NumPy  平均執行時間: {avg_numpy_time * 1000:.4f} ms")
    print(f"  OpenCV 平均執行時間: {avg_opencv_time * 1000:.4f} ms")
    print(f"  速度比 (NumPy / OpenCV): {avg_numpy_time / avg_opencv_time:.2f}x")
    print(f"[影像增強效果 (標準差 / 熵)]")
    print(f"  原始影像     : std={orig_std:.2f}, entropy={orig_entropy:.3f}")
    print(f"  NumPy 結果   : std={numpy_std:.2f}, entropy={numpy_entropy:.3f}")
    print(f"  OpenCV 結果  : std={opencv_std:.2f}, entropy={opencv_entropy:.3f}")
    print(f"[兩方法差異]")
    print(f"  最大像素差異  : {max_diff}")
    print(f"  平均像素差異  : {mean_diff:.4f}")
    print(f"  誤差 <=1 比例 : {match_ratio * 100:.2f}%")
    print(f"輸出圖檔: {fig_path}")
    print("=" * 60)

    return {
        "avg_numpy_time_sec": avg_numpy_time,
        "avg_opencv_time_sec": avg_opencv_time,
        "orig_std": orig_std,
        "orig_entropy": orig_entropy,
        "numpy_std": numpy_std,
        "numpy_entropy": numpy_entropy,
        "opencv_std": opencv_std,
        "opencv_entropy": opencv_entropy,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "match_ratio": match_ratio,
        "figure_path": fig_path,
    }

