"""
generate_variations.py
python .\Codes\utils\Quiz3_data_augmentation.py --image .\Data\utils_pic\Quiz3_aug.jpg --out .\Data\Quiz3_aug

以同一張圖片為基礎，自動生成不同拍攝條件下的變化版本，用於測試/擴增資料集
"""

import os

import cv2
import numpy as np


# ----------------------------------------------------------------------
# 1. 視角變化：模擬 3D 相機旋轉 (水平 / 垂直)
# ----------------------------------------------------------------------

def rotate_view(image, theta_x=0.0, theta_y=0.0, theta_z=0.0,
                f=None, scale=0.6,
                border_value=(0, 0, 0)):
    """
    模擬 3D 空間中的平面旋轉。

    theta_x: 垂直旋轉（pitch）
    theta_y: 水平旋轉（yaw）
    theta_z: 平面旋轉（roll）

    scale:
        控制圖片在視野中的大小。
        越小 -> 圖片越小、看起來相機越遠
    """

    h, w = image.shape[:2]

    tx = np.radians(theta_x)
    ty = np.radians(theta_y)
    tz = np.radians(theta_z)

    if f is None:
        f = w * 3.0   # 拉遠視野

    # --------------------------------------------------
    # 1. 將圖片中心移到原點
    # --------------------------------------------------
    A1 = np.array([
        [1, 0, -w / 2],
        [0, 1, -h / 2],
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.float32)

    # --------------------------------------------------
    # 2. 3D 旋轉
    # --------------------------------------------------
    RX = np.array([
        [1, 0, 0, 0],
        [0, np.cos(tx), -np.sin(tx), 0],
        [0, np.sin(tx),  np.cos(tx), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    RY = np.array([
        [ np.cos(ty), 0, np.sin(ty), 0],
        [0, 1, 0, 0],
        [-np.sin(ty), 0, np.cos(ty), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    RZ = np.array([
        [np.cos(tz), -np.sin(tz), 0, 0],
        [np.sin(tz),  np.cos(tz), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    R = RX @ RY @ RZ

    # --------------------------------------------------
    # 3. 將圖片放到相機前方
    # --------------------------------------------------
    T = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, f],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    # --------------------------------------------------
    # 4. 投影到 2D
    # --------------------------------------------------
    A2 = np.array([
        [f * scale, 0, w / 2, 0],
        [0, f * scale, h / 2, 0],
        [0, 0, 1, 0]
    ], dtype=np.float32)

    H = A2 @ T @ R @ A1

    warped = cv2.warpPerspective(
        image,
        H,
        (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )

    return warped


# ----------------------------------------------------------------------
# 2. 光源變化
# ----------------------------------------------------------------------

def apply_directional_light(image, direction="left", intensity=0.5):

    """
    模擬方向性光源。

    direction:
        left / right / top / bottom

    intensity:
        0.0 ~ 1.0
        越大 -> 明暗差異越明顯
    """

    h, w = image.shape[:2]

    if direction in ("left", "right"):

        gradient = np.linspace(-1, 1, w, dtype=np.float32)
        gradient = np.tile(gradient, (h, 1))

        if direction == "right":
            gradient = -gradient

    elif direction in ("top", "bottom"):

        gradient = np.linspace(-1, 1, h, dtype=np.float32)
        gradient = np.tile(
            gradient.reshape(-1, 1),
            (1, w)
        )

        if direction == "bottom":
            gradient = -gradient

    else:
        raise ValueError(
            "direction 必須是 left / right / top / bottom"
        )

    light_map = gradient * 255.0 * intensity

    light_map = cv2.merge([
        light_map,
        light_map,
        light_map
    ])

    lit_image = image.astype(np.float32) + light_map

    lit_image = np.clip(
        lit_image,
        0,
        255
    ).astype(np.uint8)

    return lit_image


# ----------------------------------------------------------------------
# 3. 亮度 / 對比度
# ----------------------------------------------------------------------

def adjust_brightness_contrast(
        image,
        brightness=0,
        contrast=1.0):

    """
    brightness:
        正值 -> 變亮
        負值 -> 變暗

    contrast:
        > 1 -> 提高對比
        < 1 -> 降低對比
    """

    adjusted = (
        image.astype(np.float32) * contrast
        + brightness
    )

    adjusted = np.clip(
        adjusted,
        0,
        255
    ).astype(np.uint8)

    return adjusted


# ----------------------------------------------------------------------
# 4. 解析度變化
# ----------------------------------------------------------------------

def resize_resolution(image, scale):

    """
    模擬不同解析度。

    scale:
        1.0  = 原始解析度
        0.75 = 75%
        0.5  = 50%
        0.25 = 25%

    最後放回原始尺寸，
    模擬低解析度影像再被放大的情況。
    """

    h, w = image.shape[:2]

    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))

    small = cv2.resize(
        image,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA
    )

    restored = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    return restored


# ----------------------------------------------------------------------
# 5. 模糊
# ----------------------------------------------------------------------

def apply_blur(image, kernel_size):

    """
    Gaussian Blur。

    kernel_size:
        3 -> 輕微模糊
        5 -> 中等模糊
        9 -> 明顯模糊
    """

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0
    )


# ----------------------------------------------------------------------
# 6. Gaussian Noise
# ----------------------------------------------------------------------

def add_gaussian_noise(image, sigma):

    """
    加入 Gaussian Noise。

    sigma:
        標準差越大 -> 雜訊越強
    """

    noise = np.random.normal(
        0,
        sigma,
        image.shape
    ).astype(np.float32)

    noisy = image.astype(np.float32) + noise

    noisy = np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)

    return noisy    

# ----------------------------------------------------------------------
# 主流程：批次生成各種變化版本
# ----------------------------------------------------------------------

def generate_variations(
        image_path, output_dir,
        light_directions=("left", "right", "top", "bottom"),
        light_intensity=0.5,
        brightness_levels=(75, 175),
        contrast_levels=(0.2, 0.5, 1.8),
        resolution_levels=[0.25, 0.1, 0.05],
        blur_levels=[9, 15],
        noise_levels={"low": 70, "medium": 80, "high": 90}):
    """
    以同一張圖片生成不同視角 (水平/垂直旋轉)、不同光源方向、不同明亮度的變化版本。

    參數:
        image_path (str): 原始影像路徑
        output_dir (str): 輸出資料夾
        angle_steps (tuple[int]): 視角旋轉角度清單 (度)，會同時套用在水平與垂直方向，正負皆生成
        light_directions (tuple[str]): 要生成的光源方向
        light_intensity (float): 光源強度
        brightness_levels (tuple[int]): 要生成的亮度偏移量清單
        contrast_levels (tuple[float]): 要生成的對比度倍率清單

    回傳:
        list[str]: 所有輸出影像的路徑
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = []

    def _save(img, suffix):
        out_path = os.path.join(output_dir, f"{basename}_{suffix}.jpg")
        cv2.imwrite(out_path, img)
        saved_paths.append(out_path)
        print(f"[generate_variations] 已生成: {out_path}")

    # --- 1. 水平旋轉 ---
    for angle in [30, 45, 60, 70, 80, 85]:
        warped = rotate_view(image, theta_y=angle, theta_z=-15, scale=0.6)
        _save(warped, f"horizontal_{angle}")


    # --- 1. 垂直旋轉 ---
    for angle in [30, 45, 60, 70, 80, 85]:
        warped = rotate_view(image, theta_x=angle, theta_z=15, scale=0.6)
        _save(warped, f"vertical_{angle}")


    # --- 2~7 的測試基準：垂直旋轉 60 度 ---
    vertical_60 = rotate_view(
        image,
        theta_x=60,
        theta_z=15,
        scale=0.6
    )

    '''
    # --- 2. 不同光源方向 ---
    for direction in light_directions:
        lit = apply_directional_light(
            vertical_60,
            direction=direction,
            intensity=light_intensity
        )
        _save(lit, f"light_{direction}")
    ''' 

    # --- 3. 不同明亮度 ---
    for b in brightness_levels:
        bright_img = adjust_brightness_contrast(
            vertical_60,
            brightness=b,
            contrast=1.0
        )
        label = "bright" if b > 0 else "dark"
        _save(bright_img, f"brightness_{abs(b)}")


    # --- 4. 不同對比度 ---
    for c in contrast_levels:
        contrast_img = adjust_brightness_contrast(
            vertical_60,
            brightness=0,
            contrast=c
        )
        label = "highcontrast" if c > 1.0 else "lowcontrast"
        _save(contrast_img, f"contrast_{c}")


    # --- 5. 不同解析度 ---
    for scale in resolution_levels:
        resolution_img = resize_resolution(
            vertical_60,
            scale
        )
        _save(
            resolution_img,
            f"resolution_{int(scale * 100)}percent"
        )


    # --- 6. 不同模糊程度 ---
    for k in blur_levels:
        blur_img = apply_blur(
            vertical_60,
            k
        )
        _save(
            blur_img,
            f"blur_{k}x{k}"
        )


    # --- 7. 不同雜訊程度 ---
    for label, sigma in noise_levels.items():
        noise_img = add_gaussian_noise(
            vertical_60,
            sigma
        )
        _save(
            noise_img,
            f"noise_{label}"
        )

    print(f"[generate_variations] 共生成 {len(saved_paths)} 張變化影像，輸出於: {output_dir}")
    return saved_paths
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="以同一張影像生成不同視角/光源/明亮度的變化版本")
    parser.add_argument("--image", type=str, required=True, help="原始影像路徑")
    parser.add_argument("--out", type=str, default="Data/variations", help="輸出資料夾")
    args = parser.parse_args()

    generate_variations(args.image, args.out)