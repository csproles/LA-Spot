"""Preprocessing chain: vignette removal, denoise, bilateral filter, hair
removal. Consolidated, UNCHANGED (only relocated), from:
  - Code/ComputerVisionStuff/vignette_remover.py  (remove_vignette)
  - Code/ComputerVisionStuff/median_filter.py     (remove_salt_pepper_noise)
  - Code/ComputerVisionStuff/bilateral_filter.py  (apply_bilateral_filter)
  - Code/ComputerVisionStuff/hair.py              (remove_hair)
"""
import cv2
import numpy as np


# ---- from Code/ComputerVisionStuff/vignette_remover.py --------------------

def remove_vignette(img, shrink=0.95):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bright = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("no circle found")
        return img, None

    largest = max(contours, key=cv2.contourArea)
    (cx, cy), radius = cv2.minEnclosingCircle(largest)

    circle_area = cv2.contourArea(largest)
    img_area = img.shape[0] * img.shape[1]

    if circle_area / img_area < 0.3:
        print("circle too small")
        return img, None

    circle_mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.circle(circle_mask, (int(cx), int(cy)), int(radius * shrink), 255, -1)

    skin_color = cv2.mean(img, mask=circle_mask)[:3]
    result     = img.copy()
    result[circle_mask == 0] = [int(c) for c in skin_color]

    print(f"Vignette removed center=({cx:.0f},{cy:.0f}) radius={radius:.0f}px")
    return result, (int(cx), int(cy), int(radius))


# ---- from Code/ComputerVisionStuff/median_filter.py ------------------------

def remove_salt_pepper_noise(img, kernel_size=3):
    denoised = cv2.medianBlur(img, kernel_size)
    return denoised
#


# ---- from Code/ComputerVisionStuff/bilateral_filter.py ---------------------

def apply_bilateral_filter(
    img,
    diameter=9,
    sigma_color=75,
    sigma_space=75,
):
    filtered = cv2.bilateralFilter(img, diameter, sigma_color, sigma_space)
    return filtered
#


# ---- from Code/ComputerVisionStuff/hair.py ----------------------------------

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
