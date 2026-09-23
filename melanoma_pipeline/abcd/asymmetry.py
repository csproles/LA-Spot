"""Revised A (asymmetry): principal-axis alignment before the existing
flip/IoU calculation. Consolidated, UNCHANGED, from
revised_abcd/revised_asymmetry.py.

The crop+flip+IoU math below (_fold_asymmetry) is an exact line-for-line
reuse of the original classical asymmetry.py's algorithm — the ONLY change
from that original (unconsolidated here, since it's not part of the
feature set this package computes) is that the mask is first rotated so
its own principal axis (via image moments / PCA of foreground-pixel
coordinates) is horizontal, instead of folding along fixed image x/y axes.

The concern threshold (0.20) is UNCHANGED, per the original project's
instruction.
"""
import cv2
import numpy as np

CONCERN_THRESHOLD = 0.20  # unchanged from the original (classical) asymmetry scoring
MIN_PIXELS_FOR_ALIGNMENT = 5  # below this, PCA is degenerate; skip rotation (angle=0)


def _principal_axis_angle_deg(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < MIN_PIXELS_FOR_ALIGNMENT:
        h, w = mask.shape
        return 0.0, (w / 2.0, h / 2.0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return float(angle), (float(mean[0]), float(mean[1]))


def _fold_asymmetry(mask):
    """Identical crop+flip+IoU math to the original classical asymmetry
    scoring, applied to whichever mask (aligned or not) is passed in."""
    h, w = mask.shape
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    coords = np.argwhere(mask > 0)
    if len(coords) > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        half = max(r_max - r_min, c_max - c_min) // 2 + 10
        r0 = max(cy - half, 0); r1 = min(cy + half, h)
        c0 = max(cx - half, 0); c1 = min(cx + half, w)
        crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
    else:
        crop = np.zeros((1, 1), np.uint8)

    if crop.size == 0:
        return 0.0
    flip_h = np.fliplr(crop)
    overlap_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)
    flip_v = np.flipud(crop)
    overlap_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)
    asymmetry = 1 - (overlap_h + overlap_v) / 2
    return float(asymmetry)


def score_asymmetry_revised(mask):
    """Principal-axis-aligned asymmetry. Returns the rotation angle
    actually applied alongside the usual value/concern/label fields."""
    if np.sum(mask > 0) == 0:
        return {"value": None, "concern": False,
                "label": "Asymmetry (aligned): N/A (empty mask)",
                "principal_axis_deg": None}

    h, w = mask.shape
    angle, center = _principal_axis_angle_deg(mask)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_mask = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    asymmetry = _fold_asymmetry(aligned_mask)
    concern = asymmetry > CONCERN_THRESHOLD
    return {
        "value": round(asymmetry, 3),
        "concern": bool(concern),
        "label": f"Asymmetry (aligned) score: {asymmetry:.3f} {'⚠ Irregular' if concern else '✔ Regular'}",
        "principal_axis_deg": round(angle, 1),
    }
