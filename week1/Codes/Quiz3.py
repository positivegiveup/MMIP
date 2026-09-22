"""
Quiz3: 選擇一張斜拍影像，利用 Perspective Transformation 將影像校正為正視影像。
流程可自動應用於多張不同拍攝條件的影像，讓同一套流程針對多張影像自動完成梯形校正。

流程: Gaussian Blur -> Canny -> 輪廓偵測 -> 找出四邊形輪廓 -> Perspective Transform 校正
"""

import os
import glob

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _order_points(pts):
    """
    將四個角點依照：
    左上、右上、右下、左下
    排列
    """

    pts = np.asarray(pts, dtype=np.float32)

    # 計算四點中心
    center = np.mean(pts, axis=0)

    # 計算每個點相對於中心的角度
    angles = np.arctan2(
        pts[:, 1] - center[1],
        pts[:, 0] - center[0]
    )

    # 依角度排序
    ordered = pts[np.argsort(angles)]

    # 找出最左上的點作為起點
    start = np.argmin(ordered[:, 0] + ordered[:, 1])

    ordered = np.roll(ordered, -start, axis=0)

    # 確保順序為 TL -> TR -> BR -> BL
    if ordered[1][0] < ordered[3][0]:
        ordered[[1, 3]] = ordered[[3, 1]]

    return ordered
'''
def _order_points(pts):
    """
    將四個角點依照 左上, 右上, 右下, 左下 的順序排列
    pts: shape (4, 2)
    """
    rect = np.zeros((4, 2), dtype="float32")

    # 左上角座標和最小，右下角座標和最大
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 右上角座標差(x-y)最小，左下角座標差最大
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect
'''   


def _find_document_contour(image, canny_low=75, canny_high=200,
                            blur_ksize=(5, 5), debug=False):
    """
    使用 Gaussian Blur -> Canny -> 輪廓偵測，找出影像中最大的四邊形輪廓。
    若找不到明確的四邊形，則退回使用最小外接矩形 (minAreaRect) 作為 fallback。

    回傳: 4 個角點 (4, 2) ndarray，若完全找不到輪廓則回傳 None
    """
    orig_h, orig_w = image.shape[:2]

    # 縮小影像以加速邊緣偵測與輪廓計算，最後再把座標還原
    ratio = 500.0 / orig_h if orig_h > 500 else 1.0
    resized = cv2.resize(image, (int(orig_w * ratio), int(orig_h * ratio)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, blur_ksize, 0)
    edged = cv2.Canny(blurred, canny_low, canny_high)

    # 膨脹讓邊緣更連續，避免輪廓斷裂
    edged = cv2.dilate(edged, None, iterations=1)
    edged = cv2.erode(edged, None, iterations=1)

    if debug:
        cv2.imwrite("_debug_blurred.jpg", blurred)
        cv2.imwrite("_debug_edged.jpg", edged)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # TEST contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )
    
    doc_cnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # 找到第一個近似為四邊形且面積夠大的輪廓
        if len(approx) == 4 and cv2.contourArea(approx) > 0.05 * resized.shape[0] * resized.shape[1]:
            doc_cnt = approx.reshape(4, 2)
            break

    # Fallback: 若找不到四邊形，使用最大輪廓的最小外接矩形
    if doc_cnt is None and len(contours) > 0:
        largest = contours[0]
        rect = cv2.minAreaRect(largest)
        box = cv2.boxPoints(rect)
        doc_cnt = box
    

    if doc_cnt is None:
        return None

    # 座標還原回原始影像尺寸
    doc_cnt = doc_cnt.astype("float32") / ratio
    return doc_cnt



def _four_point_transform(image, pts):
    """
    給定影像與四個角點，計算 Perspective Transform 並輸出校正後的正視影像
    """
    rect = _order_points(pts)
    
    '''
    # TEST
    print("原始四點:")
    print(pts)
    print("排序後四點:")
    print(rect)
    '''

    (tl, tr, br, bl) = rect

    # 計算校正後影像的寬度 (取上下邊較長者)
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # 計算校正後影像的高度 (取左右邊較長者)
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (max_width, max_height))

    return warped, M


def _correct_single_image(img_path, output_dir, canny_low=75, canny_high=200, debug=False, show=False):
    """
    對單張影像執行完整的梯形校正流程，並輸出結果與除錯用中間影像
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {img_path}")

    basename = os.path.splitext(os.path.basename(img_path))[0]

    doc_cnt = _find_document_contour(image, canny_low, canny_high, debug=debug)
    if doc_cnt is None:
        raise ValueError(f"無法在影像中偵測到四邊形輪廓: {img_path}")

    warped, M = _four_point_transform(image, doc_cnt)

    '''
    # 畫出偵測到的輪廓，方便檢查校正是否正確
    contour_preview = image.copy()
    cv2.drawContours(contour_preview, [doc_cnt.astype(int)], -1, (0, 255, 0), 3)

    contour_path = os.path.join(output_dir, f"{basename}-contour.jpg")
    result_path = os.path.join(output_dir, f"{basename}-corrected.jpg")

    cv2.imwrite(contour_path, contour_preview)
    cv2.imwrite(result_path, warped)
    '''
    # 畫出偵測到的輪廓，方便檢查校正是否正確
    contour_preview = image.copy()

    # 畫四邊形
    cv2.drawContours(
        contour_preview,
        [doc_cnt.astype(int)],
        -1,
        (0, 255, 0),
        3
    )

    # 將四個點排序成：TL TR BR BL
    ordered_pts = _order_points(doc_cnt)

    labels = ["TL", "TR", "BR", "BL"]

    for i, (x, y) in enumerate(ordered_pts.astype(int)):

        # 畫角點
        cv2.circle(
            contour_preview,
            (x, y),
            8,
            (0, 0, 255),
            -1
        )

        # 標示角點名稱
        cv2.putText(
            contour_preview,
            labels[i],
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 0, 0),
            2
        )

    contour_path = os.path.join(
        output_dir,
        f"{basename}-contour.jpg"
    )

    result_path = os.path.join(
        output_dir,
        f"{basename}-corrected.jpg"
    )

    cv2.imwrite(contour_path, contour_preview)
    cv2.imwrite(result_path, warped)

    # 輸出圖像
    if show: 
        contour_rgb = cv2.cvtColor(contour_preview, cv2.COLOR_BGR2RGB)
        warped_rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 2, figsize=(4, 3))

        axes[0].imshow(contour_rgb)
        axes[0].set_title("Detected Contour")
        axes[0].axis("off")

        axes[1].imshow(warped_rgb)
        axes[1].set_title("Perspective Corrected")
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
        plt.close(fig)

    return {
        "input_path": img_path,
        "contour_preview_path": contour_path,
        "result_path": result_path,
        "corners": doc_cnt.tolist(),
        "homography": M,
        "output_size": (warped.shape[1], warped.shape[0]),
    }


def run_quiz3(image_path, output_dir, canny_low=75, canny_high=200, debug=False, show=False):
    """
    Quiz3 主流程：選擇一張(或多張)斜拍影像，利用 Perspective Transformation 校正為正視影像。

    此函式會自動判斷 image_path 是「單一影像檔案」還是「資料夾」：
        - 若為單一檔案: 只處理該張影像
        - 若為資料夾: 自動走訪資料夾內所有影像，套用同一套流程逐一校正

    參數:
        image_path (str): 單張影像路徑，或包含多張影像的資料夾路徑
        output_dir (str): 輸出資料夾
        canny_low, canny_high (int): Canny 邊緣偵測的雙門檻值
        debug (bool): 是否輸出中間過程影像 (blur, edge)

    回傳:
        單一影像: dict (校正結果與統計資訊)
        資料夾:   list[dict] (每張影像各自的校正結果)
    """
    os.makedirs(output_dir, exist_ok=True)

    if os.path.isdir(image_path):
        img_files = []
        for ext in IMAGE_EXTS:
            img_files.extend(glob.glob(os.path.join(image_path, f"*{ext}")))
        img_files = sorted(img_files)

        if not img_files:
            raise FileNotFoundError(f"資料夾內找不到任何影像: {image_path}")

        print(f"偵測到資料夾模式，共 {len(img_files)} 張影像待處理")

        results = []
        for img_file in img_files:
            try:
                single_result = _correct_single_image(
                    img_file, output_dir, canny_low, canny_high, debug, show
                )
                #print(f"校正完成: {single_result['result_path']}")
                results.append(single_result)
            except Exception as e:
                print(f"校正失敗: {e}")
                continue
        print(f"校正結果已儲存至: {output_dir}")
            

        return results

    else:
        print(f"單張影像模式: {image_path}")
        result = _correct_single_image(
            image_path, output_dir, canny_low, canny_high, debug, show
        )

        print(f"校正結果已儲存: {result['result_path']}")
        #print(f"輪廓預覽已儲存: {result['contour_preview_path']}")

        return result