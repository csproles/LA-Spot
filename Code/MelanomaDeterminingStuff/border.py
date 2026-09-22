import cv2
import numpy as np


def score_border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours:
        contour     = max(contours, key=cv2.contourArea)
        perimeter   = cv2.arcLength(contour, True)
        area        = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        border_irreg = 1 - circularity
    else:
        border_irreg = 0.0

    concern = border_irreg > 0.50
    print(f"Border irregularity: {border_irreg:.3f} {'⚠' if concern else '✔'}")

    return {
        "value":   round(border_irreg, 3),
        "concern": concern,
        "label":   f"Border irregularity: {border_irreg:.3f} {'⚠ Irregular' if concern else '✔ Regular'}",
    }