"""Revised D: the lesion's diameter in pixels is ALWAYS computed and retained; millimetres and the
concern flag only exist when a calibration (mm per pixel) is supplied. This application never
supplies one, so D is reported in pixels only. The concern threshold (10.0mm) is UNCHANGED from
the original diameter scoring.
"""
import cv2

DIAMETER_CONCERN_MM = 10.0  # UNCHANGED from the original diameter scoring



def score_diameter_revised(mask, mm_per_px):
    """diameter_px is ALWAYS populated when a contour exists. D_mm/concern
    only exist when a verified (hair-shape-gated) mm_per_px is available.
    Concern threshold (10.0mm) is UNCHANGED."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"diameter_px": None, "value": None, "concern": False,
                "label": "Diameter: N/A (no lesion contour)"}

    contour = max(contours, key=cv2.contourArea)
    (_, _), radius_px = cv2.minEnclosingCircle(contour)
    diameter_px = round(float(radius_px * 2), 1)

    if mm_per_px is None:
        return {"diameter_px": diameter_px, "value": None, "concern": False,
                "label": f"Diameter: {diameter_px}px (no verified hair-based calibration; mm N/A)"}

    diameter_mm = diameter_px * mm_per_px
    concern = diameter_mm > DIAMETER_CONCERN_MM
    return {
        "diameter_px": diameter_px,
        "value": round(diameter_mm, 2),
        "concern": bool(concern),
        "label": f"Est. diameter: {diameter_mm:.1f}mm (hair-calibrated, shape-verified) "
                 f"{'⚠ >10mm' if concern else '✔ <=10mm'}",
    }
