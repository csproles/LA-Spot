import cv2
import numpy as np


def score_diameter(
    mask,
    mm_per_px=None,
):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if contours and mm_per_px is not None:
        contour = max(contours, key=cv2.contourArea)
        (_, _), radius_px = cv2.minEnclosingCircle(contour)
        diameter_mm = radius_px * 2 * mm_per_px
        concern     = diameter_mm > 10.0

        print(f"Diameter: {diameter_mm:.1f}mm scale={mm_per_px*1000:.3f}µm/px {'⚠' if concern else '✔'}")

        return {
            "value":   round(diameter_mm, 2),
            "concern": concern,
            "label":   f"Est. diameter: {diameter_mm:.1f}mm (hair-calibrated) {'⚠ >10mm' if concern else '✔ ≤10mm'}",
        }
    else:
        reason = "no hair detected" if mm_per_px is None else "no lesion contour"
        print(f"Diameter: skipped ({reason})")
        return {
            "value":   None,
            "concern": False,
            "label":   f"Diameter: N/A ({reason})",
        }