"""
Quiz4: 選擇兩張具有重疊區域的影像，使用 SIFT 等特徵偵測與匹配方法完成影像拼接
"""

import os
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 預設前處理組合：去雜訊 + CLAHE 對比度增強
DEFAULT_PREPROCESSING = {
    "denoise": True,
    "clahe": True,
    "sharpen": False,
    "gray_equalize": False,
}


def _preprocess_for_detection(image, denoise=False, clahe=False,
                               sharpen=False, gray_equalize=False):
    """
    對影像做前處理，目的是提升 SIFT 特徵偵測與匹配的穩定性。
    僅用於「偵測階段」，不影響最終拼接輸出的原始影像品質。

    參數:
        denoise (bool): 是否做去雜訊 (Non-local Means Denoising)
        clahe (bool): 是否做 CLAHE 局部對比度增強 (在 LAB 色彩空間的 L 通道上做)
        sharpen (bool): 是否做銳化 (Unsharp Masking)，加強邊緣/紋理利於特徵偵測
        gray_equalize (bool): 是否對灰階影像做直方圖等化 (適合整體偏暗/偏亮的影像)

    回傳:
        處理後的彩色影像 (BGR)，尺寸與輸入相同
    """
    processed = image.copy()

    # 1. 去雜訊：雜訊點容易被誤判為虛假的特徵點，先平滑掉高頻雜訊
    if denoise:
        processed = cv2.fastNlMeansDenoisingColored(processed, None, 10, 10, 7, 21)

    # 2. CLAHE：在光照不均、對比度不足的影像中，強化局部細節，
    #    讓角落、紋理處更容易被 SIFT 偵測到穩定的關鍵點
    if clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_op = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe_op.apply(l)
        processed = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

    # 3. 銳化：加強邊緣與紋理對比，對模糊或失焦影像有幫助
    if sharpen:
        blurred = cv2.GaussianBlur(processed, (0, 0), sigmaX=3)
        processed = cv2.addWeighted(processed, 1.5, blurred, -0.5, 0)

    # 4. 灰階直方圖等化：適合整體曝光不足/過曝的影像，會覆蓋掉部分色彩資訊，
    #    因此僅在明確需要時才啟用，且只作用在灰階通道供偵測使用
    if gray_equalize:
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        gray_eq = cv2.equalizeHist(gray)
        processed = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)

    return processed


def _detect_and_match_sift(img1_gray, img2_gray, ratio_thresh=0.75):
    """使用 SIFT 偵測特徵點，並用 KNN + Lowe's ratio test 過濾匹配"""
    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    knn_matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in knn_matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    return kp1, kp2, good_matches


def _compute_homography(kp1, kp2, good_matches, min_match_count=4):
    """利用良好匹配點以 RANSAC 計算 Homography 矩陣"""
    if len(good_matches) < min_match_count:
        raise ValueError(
            f"匹配點數量不足 ({len(good_matches)} < {min_match_count})，無法計算 Homography"
        )

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return H, mask


def _warp_and_stitch(img1, img2, H):
    """將 img1 依 Homography 投影到 img2 座標系統，計算畫布大小並融合重疊區域"""
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    corners_img1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners_img1_transformed = cv2.perspectiveTransform(corners_img1, H)

    corners_img2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)

    all_corners = np.concatenate((corners_img1_transformed, corners_img2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    translation = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ], dtype=np.float64)

    output_size = (x_max - x_min, y_max - y_min)

    warped_img1 = cv2.warpPerspective(img1, translation.dot(H), output_size)
    warped_img2 = cv2.warpPerspective(img2, translation, output_size)

    mask1 = (cv2.cvtColor(warped_img1, cv2.COLOR_BGR2GRAY) > 0)
    mask2 = (cv2.cvtColor(warped_img2, cv2.COLOR_BGR2GRAY) > 0)
    overlap = mask1 & mask2
    only1 = mask1 & (~overlap)
    only2 = mask2 & (~overlap)

    result = np.zeros_like(warped_img1, dtype=np.float32)
    for c in range(3):
        result[..., c][only1] = warped_img1[..., c][only1]
        result[..., c][only2] = warped_img2[..., c][only2]
        result[..., c][overlap] = (
            warped_img1[..., c][overlap].astype(np.float32) * 0.5
            + warped_img2[..., c][overlap].astype(np.float32) * 0.5
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def _stitch_once(img1, img2, ratio_thresh=0.75, preprocessing=None):
    """
    執行一次完整的 SIFT 拼接流程，回傳結果影像與統計資訊。

    preprocessing (dict | None): 前處理參數，傳入 _preprocess_for_detection 的 kwargs，
        例如 {"denoise": True, "clahe": True}。僅影響特徵偵測階段，
        最終 warp/融合仍使用原始的 img1、img2，確保輸出畫質不受前處理影響。
        傳 None 或空 dict 代表不做任何前處理 (等同原本行為)。
    """
    if preprocessing:
        proc1 = _preprocess_for_detection(img1, **preprocessing)
        proc2 = _preprocess_for_detection(img2, **preprocessing)
    else:
        proc1, proc2 = img1, img2

    gray1 = cv2.cvtColor(proc1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(proc2, cv2.COLOR_BGR2GRAY)

    kp1, kp2, good_matches = _detect_and_match_sift(gray1, gray2, ratio_thresh)
    H, mask = _compute_homography(kp1, kp2, good_matches)

    # 融合階段一律使用「原始影像」，前處理只是拿來輔助找特徵點，不影響最終畫質
    result = _warp_and_stitch(img1, img2, H)

    inliers = int(mask.sum()) if mask is not None else 0
    stats = {
        "kp1_count": len(kp1),
        "kp2_count": len(kp2),
        "good_matches": len(good_matches),
        "inliers": inliers,
        "homography": H,
    }
    return result, kp1, kp2, good_matches, stats


def run_quiz4(image_path1, image_path2, output_dir, ratio_thresh=0.75,
              preprocessing=None, save_preprocessed=False, detail=False):
    """
    Quiz4 主流程：讀取兩張重疊影像，使用 SIFT 特徵偵測與匹配完成影像拼接。

    參數:
        image_path1 (str): 第一張影像路徑 (將被投影變形)
        image_path2 (str): 第二張影像路徑 (作為基準座標系)
        output_dir (str): 輸出資料夾
        ratio_thresh (float): Lowe's ratio test 門檻值
        preprocessing (dict | None): 拼接前的前處理設定，用於提升特徵偵測/匹配穩定性。
            可用的 key: "denoise", "clahe", "sharpen", "gray_equalize" (皆為 bool)。
            範例: {"denoise": True, "clahe": True}
            傳 None 表示不做前處理；也可直接傳入 DEFAULT_PREPROCESSING 快速套用建議組合。
        save_preprocessed (bool): 是否額外輸出前處理後的影像，方便比較前後差異

    回傳:
        dict: 包含拼接結果路徑、特徵匹配統計資訊、執行時間等
    """
    img1 = cv2.imread(image_path1)
    if img1 is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path1}")

    img2 = cv2.imread(image_path2)
    if img2 is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path2}")

    os.makedirs(output_dir, exist_ok=True)

    if preprocessing:
        print(f"前處理設定: {preprocessing}")

    # 若需要，額外輸出前處理後的影像供比較 (不影響實際拼接流程)
    if save_preprocessed and preprocessing:
        proc1_preview = _preprocess_for_detection(img1, **preprocessing)
        proc2_preview = _preprocess_for_detection(img2, **preprocessing)
        cv2.imwrite(os.path.join(output_dir, "Quiz4-preprocessed-1.jpg"), proc1_preview)
        cv2.imwrite(os.path.join(output_dir, "Quiz4-preprocessed-2.jpg"), proc2_preview)

    # 執行 SIFT 拼接並計時
    start = time.time()

    result, kp1, kp2, good_matches, stats = _stitch_once(
        img1, img2, ratio_thresh, preprocessing=preprocessing
    )

    elapsed_time = time.time() - start

    # 顯示統計資訊
    if detail:
        print(f"SIFT 特徵點數量: img1={stats['kp1_count']}, img2={stats['kp2_count']}")
        print(f"良好匹配數量: {stats['good_matches']}, RANSAC 內點: {stats['inliers']}")
        print(f"執行時間: {elapsed_time:.4f} 秒")

    # 取得兩張輸入影像的檔名（不含副檔名）
    base_name1 = os.path.splitext(os.path.basename(image_path1))[0]
    base_name2 = os.path.splitext(os.path.basename(image_path2))[0]
    # 取檔名較長者
    base_name = base_name1 if len(base_name1) >= len(base_name2) else base_name2

    # 輸出拼接結果
    result_path = os.path.join(output_dir, f"{base_name}_result.jpg")
    cv2.imwrite(result_path, result)
    #print(f"拼接結果已儲存: {result_path}")

    # 輸出特徵匹配預覽圖
    matches_img = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        good_matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    matches_path = os.path.join(output_dir, f"{base_name}_matches.jpg")
    cv2.imwrite(matches_path, matches_img)
    #print(f"特徵匹配預覽圖已儲存: {matches_path}")

    # plot
    # Plot：左 Matches，右 Result
    plt.figure(figsize=(6, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(matches_img, cv2.COLOR_BGR2RGB))
    plt.title("SIFT Feature Matches")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    return {
        "result_path": result_path,
        "matches_path": matches_path,
        "kp1_count": stats["kp1_count"],
        "kp2_count": stats["kp2_count"],
        "good_matches": stats["good_matches"],
        "inliers": stats["inliers"],
        "homography": stats["homography"],
        "time_sec": elapsed_time,
        "preprocessing": preprocessing,
    }