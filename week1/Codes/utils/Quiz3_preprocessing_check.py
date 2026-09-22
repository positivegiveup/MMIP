"""
輸出九項前處理，確認四角捕捉
from Codes.Quiz3_preprocessing import run_debug

run_debug(
    "Data/test",
    "Data/test_result/"
)
"""

import os
import glob

import cv2
import numpy as np


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def visualize_quiz3(image_path, output_dir,
                     canny_low=75, canny_high=200):

    os.makedirs(output_dir, exist_ok=True)

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"無法讀取影像: {image_path}")

    basename = os.path.splitext(os.path.basename(image_path))[0]

    # ============================================================
    # 1. Resize
    # ============================================================

    orig_h, orig_w = image.shape[:2]

    ratio = 500.0 / orig_h if orig_h > 500 else 1.0

    resized = cv2.resize(
        image,
        (int(orig_w * ratio), int(orig_h * ratio))
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_01_resized.jpg"),
        resized
    )

    # ============================================================
    # 2. Gray
    # ============================================================

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_02_gray.jpg"),
        gray
    )

    # ============================================================
    # 3. Gaussian Blur
    # ============================================================

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_03_blur.jpg"),
        blurred
    )

    # ============================================================
    # 4. Canny
    # ============================================================

    edged = cv2.Canny(
        blurred,
        canny_low,
        canny_high
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_04_canny.jpg"),
        edged
    )

    # ============================================================
    # 5. Dilate
    # ============================================================

    dilated = cv2.dilate(
        edged,
        None,
        iterations=1
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_05_dilate.jpg"),
        dilated
    )

    # ============================================================
    # 6. Erode
    # ============================================================

    morphed = cv2.erode(
        dilated,
        None,
        iterations=1
    )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_06_erode.jpg"),
        morphed
    )

    # ============================================================
    # 7. Find Contours
    # ============================================================

    contours, _ = cv2.findContours(
        morphed.copy(),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    # ============================================================
    # 8. 顯示所有前 10 大 Contours
    # ============================================================

    contour_all = resized.copy()

    for i, contour in enumerate(contours[:10]):

        area = cv2.contourArea(contour)

        cv2.drawContours(
            contour_all,
            [contour],
            -1,
            (0, 255, 0),
            2
        )

        # 找 contour 中心位置
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                contour_all,
                f"#{i+1} area={int(area)}",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

    cv2.imwrite(
        os.path.join(output_dir, f"{basename}_07_top10_contours.jpg"),
        contour_all
    )

    # ============================================================
    # 9. 顯示前 5 個 contour 的 polygon approximation
    # ============================================================

    approx_preview = resized.copy()

    for i, contour in enumerate(contours[:5]):

        peri = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * peri,
            True
        )

        cv2.drawContours(
            approx_preview,
            [approx],
            -1,
            (0, 255, 0),
            3
        )

        # 標示編號
        M = cv2.moments(contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.putText(
                approx_preview,
                f"#{i+1}: {len(approx)} points",
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

    cv2.imwrite(
        os.path.join(
            output_dir,
            f"{basename}_08_approx_contours.jpg"
        ),
        approx_preview
    )

    # ============================================================
    # 10. 找出 Quiz3 最後選到的四邊形
    # ============================================================

    doc_cnt = None

    image_area = resized.shape[0] * resized.shape[1]

    for i, contour in enumerate(contours[:5]):

        peri = cv2.arcLength(contour, True)

        approx = cv2.approxPolyDP(
            contour,
            0.02 * peri,
            True
        )

        area = cv2.contourArea(approx)

        if (
            len(approx) == 4
            and area > 0.1 * image_area
        ):
            doc_cnt = approx.reshape(4, 2)

            print(
                f"[Debug] 選中的 contour #{i+1}"
            )

            print(
                f"[Debug] 面積 = {area:.2f}"
            )

            print(
                f"[Debug] 四個點:\n{doc_cnt}"
            )

            break

    # ============================================================
    # 11. 顯示最後選中的四邊形
    # ============================================================

    final_preview = resized.copy()

    if doc_cnt is not None:

        cv2.drawContours(
            final_preview,
            [doc_cnt.astype(int)],
            -1,
            (0, 255, 0),
            3
        )

        for i, (x, y) in enumerate(
            doc_cnt.astype(int)
        ):

            cv2.circle(
                final_preview,
                (x, y),
                8,
                (0, 0, 255),
                -1
            )

            cv2.putText(
                final_preview,
                str(i + 1),
                (x + 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

    else:

        print("[Debug] 沒有找到符合條件的四邊形")

    cv2.imwrite(
        os.path.join(
            output_dir,
            f"{basename}_09_final_contour.jpg"
        ),
        final_preview
    )

    print(
        f"[Debug] 完成: {basename}"
    )


def run_debug(image_path, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    # 如果是資料夾
    if os.path.isdir(image_path):

        img_files = []

        for ext in IMAGE_EXTS:
            img_files.extend(
                glob.glob(
                    os.path.join(
                        image_path,
                        f"*{ext}"
                    )
                )
            )

        img_files = sorted(img_files)

        print(
            f"[Debug] 共 {len(img_files)} 張圖片"
        )

        for img_path in img_files:

            visualize_quiz3(
                img_path,
                output_dir
            )

    # 如果是單張圖片
    else:

        visualize_quiz3(
            image_path,
            output_dir
        )