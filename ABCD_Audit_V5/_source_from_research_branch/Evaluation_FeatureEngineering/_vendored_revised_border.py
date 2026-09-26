"""Revised B: the EXISTING circularity-based score_border from
Code/MelanomaDeterminingStuff/border.py is used UNCHANGED (imported
directly in the runner, not duplicated or modified) and continues to be
the ONLY thing that feeds the LOW/HIGH decision.

This module adds one additional, separately-recorded experimental metric —
normalized total convexity-defect depth — which targets the specific gap
identified in investigation/investigate_border.py: circularity is a single
global scalar that is (confirmed empirically) blind to moderate localized
protrusions/indentations once a mask is already near-convex and smooth,
which is true of essentially all masks in this dataset (smoothed-perimeter
ratio ~1.005-1.008 for both classes).

Convexity-defect analysis is an established general shape-analysis
technique (a defect is exactly a local indentation between the contour and
its convex hull) and is the natural computational analogue of the
"protrusions and indentations" sensitivity the border-irregularity
literature calls out as missing from plain compactness/circularity.

B_experimental carries NO concern threshold and does not participate in the
LOW/HIGH decision — it is recorded for comparison only, per instruction not
to invent or tune a new classification threshold. The only numeric cutoff
here (NOISE_FLOOR_PX) is a sub-pixel noise filter, not a diagnostic
threshold: it exists purely so single-pixel segmentation-boundary jitter
isn't counted as a "real" indentation.
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
