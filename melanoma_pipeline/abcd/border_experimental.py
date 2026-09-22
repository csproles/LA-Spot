"""B_experimental: normalized total convexity-defect depth (recorded for
analysis only, does NOT feed any of the 17 selected features, but IS
computed as part of the shared score_instance() glue in pipeline.py -- kept
here for exact fidelity to the original scoring chain). Consolidated,
UNCHANGED, from revised_abcd/revised_border.py.
"""
import cv2
import numpy as np

NOISE_FLOOR_PX = 1.0  # numerical noise floor, NOT a diagnostic threshold


def score_border_experimental(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"value": None, "n_significant_defects": 0,
                "label": "Border (experimental): N/A (no contour)"}

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area <= 0 or len(contour) < 5:
        return {"value": None, "n_significant_defects": 0,
                "label": "Border (experimental): N/A (degenerate contour)"}

    equivalent_diameter = 2.0 * np.sqrt(area / np.pi)  # scale-invariant normalizer

    hull_indices = cv2.convexHull(contour, returnPoints=False)
    if hull_indices is None or len(hull_indices) < 4:
        return {"value": 0.0, "n_significant_defects": 0,
                "label": "Border (experimental): 0.000 (fully convex)"}

    hull_indices = np.sort(hull_indices, axis=0)
    try:
        defects = cv2.convexityDefects(contour, hull_indices)
    except cv2.error:
        return {"value": None, "n_significant_defects": 0,
                "label": "Border (experimental): N/A (defect computation failed)"}

    if defects is None:
        return {"value": 0.0, "n_significant_defects": 0,
                "label": "Border (experimental): 0.000 (fully convex)"}

    # cv2.convexityDefects returns shape (N,1,4) on some OpenCV builds and
    # (N,4) on others; reshape defensively rather than assuming either.
    defects = defects.reshape(-1, 4)
    depths_px = defects[:, 3] / 256.0  # cv2 stores depth in fixed-point (1/256 px)
    significant = depths_px[depths_px > NOISE_FLOOR_PX]
    normalized_total_depth = float(np.sum(significant) / equivalent_diameter) if len(significant) else 0.0

    return {
        "value": round(normalized_total_depth, 4),
        "n_significant_defects": int(len(significant)),
        "label": f"Border (experimental, convexity-defect): {normalized_total_depth:.3f} "
                 f"({len(significant)} localized indentations)",
    }
