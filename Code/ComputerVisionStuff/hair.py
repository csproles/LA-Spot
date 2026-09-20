import cv2
import numpy as np


def measure_hair_width_px(
    img,
    kernel_size=17,
    threshold=10,
):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel   = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    if np.sum(hair_mask > 0) < 50:
        print("no hair detected")
        return None

    dist = cv2.distanceTransform(hair_mask, cv2.DIST_L2, 5)

    kernel_sk    = np.ones((3, 3), np.uint8)
    dist_dilated = cv2.dilate(dist, kernel_sk)
    skeleton     = (dist == dist_dilated) & (dist > 0)

    half_widths = dist[skeleton]
    half_widths = half_widths[(half_widths >= 0.5) & (half_widths <= 8.0)]

    if len(half_widths) < 10:
        print(f"too few points ({len(half_widths)})")
        return None

    hair_width_px = float(np.median(half_widths) * 2)
    return hair_width_px


def remove_hair(
    img,
    kernel_size=17,
    threshold=10,
):
    gray   = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)
    result = cv2.inpaint(img, hair_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    return result
#