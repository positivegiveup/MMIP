# -*- coding: utf-8 -*-
"""
Quiz1: RGB 彩色影像轉灰階影像
- 自行以 NumPy 實作 RGB -> Grayscale 轉換
- 與 OpenCV 的 cv2.cvtColor() 比較
- 重複執行 N 次取平均，比較執行速度、轉換結果與兩者差異
"""

import os
import time

import cv2
import numpy as np
#matplotlib.use("Agg")  # 無需顯示視窗，直接存檔
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1. 自行以 NumPy 實作 RGB -> Grayscale
# ---------------------------------------------------------------------------
def numpy_rgb_to_gray(bgr_img: np.ndarray) -> np.ndarray:
    """
    採用與 OpenCV 相同的 ITU-R BT.601 加權公式:
        Gray = 0.299*R + 0.587*G + 0.114*B

    bgr_img : np.ndarray
        3 通道彩色影像 (OpenCV 預設為 BGR 順序, dtype=uint8)
    """
    # 仿 OpenCV 讀進來的影像通道順序是 B, G, R
    b = bgr_img[:, :, 0].astype(np.float64)
    g = bgr_img[:, :, 1].astype(np.float64)
    r = bgr_img[:, :, 2].astype(np.float64)

    gray = 0.299 * r + 0.587 * g + 0.114 * b

    # 四捨五入並限制在 0~255 範圍內
    gray = np.round(gray).clip(0, 255).astype(np.uint8)

    return gray

def show_images(original, gray):
    img_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title("NumPy Grayscale")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 2. 使用 OpenCV 內建函式
# ---------------------------------------------------------------------------
def opencv_rgb_to_gray(bgr_img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 3. 主流程
# ---------------------------------------------------------------------------
def run_quiz1(image_path: str, output_dir: str, N: int = 50):
    """
    image_path : str
        輸入彩色影像路徑 (例如 "Data/Quiz1.jpg")
    output_dir : str
        輸出資料夾路徑
    N : int
        重複執行次數，取平均計算執行時間
    """
    os.makedirs(output_dir, exist_ok=True)

    # 讀取彩色影像 (BGR)
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")

    # ------------------------------------------------------------------
    # 先各自執行一次，取得輸出影像 (供顯示 / 存檔用)
    # ------------------------------------------------------------------
    numpy_result = numpy_rgb_to_gray(img)
    opencv_result = opencv_rgb_to_gray(img)

    # ------------------------------------------------------------------
    # 重複執行 N 次，比較平均執行時間
    # ------------------------------------------------------------------
    numpy_times = []
    opencv_times = []

    for _ in range(N):
        t0 = time.perf_counter()
        _ = numpy_rgb_to_gray(img)
        t1 = time.perf_counter()
        numpy_times.append(t1 - t0)

        t0 = time.perf_counter()
        _ = opencv_rgb_to_gray(img)
        t1 = time.perf_counter()
        opencv_times.append(t1 - t0)

    avg_numpy_time = float(np.mean(numpy_times))
    avg_opencv_time = float(np.mean(opencv_times))

    # ------------------------------------------------------------------
    # 兩種方法輸出影像的差異
    # ------------------------------------------------------------------
    diff = np.abs(numpy_result.astype(np.int16) - opencv_result.astype(np.int16))
    diff = np.abs(
    numpy_result.astype(np.int16) -
    opencv_result.astype(np.int16)
    )

    total_pixels = diff.size
    different_pixels = int(np.count_nonzero(diff))
    different_ratio = float(different_pixels / total_pixels)
    # 最大像素差異
    max_diff = int(diff.max())
    # 平均像素差異
    mean_diff = float(diff.mean())
    # 誤差 <= 1 的比例
    match_ratio = float(np.mean(diff <= 1))

    # ------------------------------------------------------------------
    # 存檔：轉換後影像
    # ------------------------------------------------------------------
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_numpy_gray.png"), numpy_result)
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_opencv_gray.png"), opencv_result)

    diff_vis = (diff.astype(np.uint8))
    cv2.imwrite(os.path.join(output_dir, f"{base_name}_diff.png"), diff_vis * 50)  # 放大方便觀察

    # ------------------------------------------------------------------
    # 繪圖：原圖 + 兩種灰階結果 + 差異圖
    # ------------------------------------------------------------------
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(12, 6))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original (RGB)")
    axes[0].axis("off")

    axes[1].imshow(numpy_result, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("NumPy Grayscale")
    axes[1].axis("off")

    axes[2].imshow(opencv_result, cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("OpenCV Grayscale")
    axes[2].axis("off")

    im3 = axes[3].imshow(diff, cmap="hot", vmin=0, vmax=max(1, max_diff))
    axes[3].set_title(f"|Diff| (max={max_diff})")
    axes[3].axis("off")
    fig.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"{base_name}_result.png")
    plt.savefig(fig_path, dpi=150)
    plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # 印出比較結果
    # ------------------------------------------------------------------
    print("=" * 60)
    print(f"Quiz1 - RGB to Grayscale ({image_path})")
    print("=" * 60)
    print(f"重複執行次數 N = {N}")
    print(f"[執行速度]")
    print(f"  NumPy  平均執行: {avg_numpy_time * 1000:.4f} ms")
    print(f"  OpenCV 平均執行: {avg_opencv_time * 1000:.4f} ms")
    print(f"  速度比 (NumPy / OpenCV): {avg_numpy_time / avg_opencv_time:.2f}x")
    print(f"[轉換結果]")
    print(f"  NumPy  結果 - mean={numpy_result.mean():.2f}, std={numpy_result.std():.2f}")
    print(f"  OpenCV 結果 - mean={opencv_result.mean():.2f}, std={opencv_result.std():.2f}")
    print(f"[兩方法差異]")
    print(f"  不同像素數:總像素數 : {different_pixels}/{total_pixels}")
    print(f"  比例               : {different_ratio * 100:.6f}%")
    print(f"  最大像素差異        : {max_diff}")
    print(f"  平均像素差異       : {mean_diff:.6f}")
    print(f"  誤差 <=1 比例      :{match_ratio * 100:.2f}%")
    print(f"輸出圖檔: {fig_path}")
    print("=" * 60)

    return {
        "avg_numpy_time_sec": avg_numpy_time,
        "avg_opencv_time_sec": avg_opencv_time,
        "numpy_mean": float(numpy_result.mean()),
        "numpy_std": float(numpy_result.std()),
        "opencv_mean": float(opencv_result.mean()),
        "opencv_std": float(opencv_result.std()),
        "different_pixels": different_pixels,
        "different_ratio": different_ratio,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "match_ratio": match_ratio,
        "figure_path": fig_path,
    }
