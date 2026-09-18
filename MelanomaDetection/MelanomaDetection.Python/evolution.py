"""Scores the "E" (Evolving) ABCDE criterion by comparing a spot's two most recent checks.

Why this exists:
    Evolving was the one ABCDE criterion the pipeline always reported as None,
    because it describes change over time and a single photo cannot show it.
    Now that checks are grouped under a tracked spot (see store.py), the prior
    check's mask, size calibration and lesion color are on hand, so change can
    actually be measured.

What it is honest about:
    These are consumer phone photos with no fixed distance, no fixed lighting
    and no image registration, so this deliberately measures only
    scale-invariant, rotation-tolerant quantities:

      * shape change  -- IoU of the two masks after normalizing them to equal
                         area, aligning their centroids, and taking the best
                         match across rotations. Isolates morphology change from
                         "I held the camera differently".
      * color change  -- LAB distance between mean lesion colors.
      * growth        -- only reported when BOTH checks carried a hair-based
                         mm-per-pixel calibration. Without it, apparent size is
                         dominated by how far away the phone was held, so a
                         number here would look authoritative and mean nothing.

    The result is a change indicator to review with a clinician, not a finding.
"""

import numpy as np

import cv2

# Each raw measurement is scaled so its clinical concern threshold lands at 5.0
# on the 0-10 scale, matching the convention the ABCD scorers in
# image_processor.py already use.
SHAPE_CONCERN_THRESHOLD = 0.25   # 1 - IoU, after equal-area normalization
COLOR_CONCERN_THRESHOLD = 12.0   # LAB distance; ~12 is clearly perceptible
GROWTH_CONCERN_THRESHOLD = 0.30  # +30% lesion area between checks

# Masks are compared at this many rotations to tolerate camera orientation.
_ROTATION_STEP_DEGREES = 15


def lesion_area_px(mask) -> int:
    """Number of lesion pixels in a segmentation mask."""
    if mask is None:
        return 0
    return int(np.count_nonzero(mask > 0))


def lesion_lab_mean(image_bgr, mask):
    """Mean LAB color of the lesion pixels, or None if the mask is empty.

    LAB rather than BGR because LAB distances approximate perceived color
    difference, which is what "the mole changed color" actually means.
    """
    if image_bgr is None or mask is None or lesion_area_px(mask) == 0:
        return None

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    selected = lab[mask > 0]
    if selected.size == 0:
        return None
    mean = selected.mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _centroid(mask):
    moments = cv2.moments((mask > 0).astype(np.uint8), binaryImage=True)
    if moments["m00"] == 0:
        return None
    return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def _best_shape_iou(mask_a, mask_b) -> float:
    """Best IoU between two masks after equal-area, centroid-aligned, rotated fitting.

    Warps mask_b into mask_a's frame: scaled so both cover the same area,
    translated so their centroids coincide, and rotated through a full turn to
    find the orientation that fits best. Returns the highest IoU found, so a
    photo taken at a different distance or angle does not read as shape change.
    """
    area_a = lesion_area_px(mask_a)
    area_b = lesion_area_px(mask_b)
    if area_a == 0 or area_b == 0:
        return 0.0

    centroid_a = _centroid(mask_a)
    centroid_b = _centroid(mask_b)
    if centroid_a is None or centroid_b is None:
        return 0.0

    scale = float(np.sqrt(area_a / area_b))
    binary_a = (mask_a > 0).astype(np.uint8)
    height, width = binary_a.shape[:2]
    binary_b = (mask_b > 0).astype(np.uint8)

    best_iou = 0.0
    for degrees in range(0, 360, _ROTATION_STEP_DEGREES):
        # Rotate+scale about mask_b's centroid, then shift that centroid onto mask_a's.
        matrix = cv2.getRotationMatrix2D(centroid_b, degrees, scale)
        matrix[0, 2] += centroid_a[0] - centroid_b[0]
        matrix[1, 2] += centroid_a[1] - centroid_b[1]

        warped = cv2.warpAffine(
            binary_b, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0
        )

        intersection = np.count_nonzero((binary_a > 0) & (warped > 0))
        union = np.count_nonzero((binary_a > 0) | (warped > 0))
        if union:
            best_iou = max(best_iou, intersection / union)

    return float(best_iou)


def _scaled(raw: float, threshold: float) -> float:
    """Map a raw measurement onto 0-10 with its concern threshold at 5.0."""
    return float(np.clip((raw / threshold) * 5.0, 0.0, 10.0))


def score_change(current: dict, prior: dict):
    """Score how much a lesion changed between its prior and current check.

    Args:
        current: dict with "mask" (ndarray), "area_px" (int), "lab" (3-tuple or
            None), "mm_per_px" (float or None) and "risk_score" (float).
        prior: the same shape, for the previous check of the same spot.

    Returns:
        (score, details) where score is 0-10 (5.0 = at the concern threshold)
        or None if nothing comparable could be measured, and details is a dict
        of the raw measurements plus a "signals" list naming what contributed.

    Why the maximum rather than an average:
        One unmistakable change is the thing worth surfacing; averaging it
        against two stable measurements would bury it. This mirrors how
        _calculate_color_variation already takes the max of its two sub-signals.
    """
    details = {
        "risk_score_delta": round(current["risk_score"] - prior["risk_score"], 2),
        "signals": [],
    }
    components = []

    shape_iou = _best_shape_iou(current.get("mask"), prior.get("mask"))
    if shape_iou > 0.0:
        shape_change = 1.0 - shape_iou
        details["shape_iou"] = round(shape_iou, 3)
        details["shape_change"] = round(shape_change, 3)
        details["shape_concern_threshold"] = SHAPE_CONCERN_THRESHOLD
        components.append(_scaled(shape_change, SHAPE_CONCERN_THRESHOLD))
        details["signals"].append("shape")

    current_lab = current.get("lab")
    prior_lab = prior.get("lab")
    if current_lab and prior_lab:
        color_distance = float(np.linalg.norm(np.array(current_lab) - np.array(prior_lab)))
        details["color_distance"] = round(color_distance, 2)
        details["color_concern_threshold"] = COLOR_CONCERN_THRESHOLD
        components.append(_scaled(color_distance, COLOR_CONCERN_THRESHOLD))
        details["signals"].append("color")

    current_mm = current.get("mm_per_px")
    prior_mm = prior.get("mm_per_px")
    if current_mm and prior_mm:
        current_area_mm2 = current["area_px"] * (current_mm ** 2)
        prior_area_mm2 = prior["area_px"] * (prior_mm ** 2)
        if prior_area_mm2 > 0:
            growth = (current_area_mm2 - prior_area_mm2) / prior_area_mm2
            details["area_mm2"] = round(current_area_mm2, 2)
            details["prior_area_mm2"] = round(prior_area_mm2, 2)
            details["area_growth_ratio"] = round(growth, 3)
            details["growth_concern_threshold"] = GROWTH_CONCERN_THRESHOLD
            # Only growth counts, not shrinkage -- a lesion getting smaller is
            # not the evolution melanoma screening is looking for.
            components.append(_scaled(max(growth, 0.0), GROWTH_CONCERN_THRESHOLD))
            details["signals"].append("growth")
            details["size_calibrated"] = True
    else:
        details["size_calibrated"] = False
        details["size_note"] = (
            "Size change not measured: both photos need visible fine hair for "
            "mm calibration."
        )

    if not components:
        return None, {**details, "reason": "no comparable measurements between the two checks"}

    return round(max(components), 2), details
