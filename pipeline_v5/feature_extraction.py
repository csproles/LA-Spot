"""FROZEN V5 new-feature extraction.

STATUS: FROZEN 2026-09-21, based on Evaluation_FeatureEngineering/'s
ablation study and Evaluation_V5Candidate/'s development + locked-test
evaluation. Not modified after this freeze.

This module computes the 11 features V5 adds on top of V4's existing 6
(A_value, B_circularity, C_value, D_px, confidence, lesion_fraction, all
already produced unchanged by revised_abcd/pipeline_v2.py): color_entropy,
lab_a_std, lab_b_std, red_fraction, bluegray_fraction, dark_fraction,
skin_contrast, solidity, turning_angle_std, eccentricity, D_px_normalized.

The math here is copied UNCHANGED, function-for-function, from
Evaluation_FeatureEngineering/extract_features.py (the exact code the
ablation study, development CV, and one-time locked-test evaluation all
ran) -- relocated into a self-contained module (no dependency on the
Evaluation_* research folders or their cached CSVs) so it can run as a
runtime artifact on a single new image. Nothing about the computation was
changed in the process; only unused functions (texture/GLCM,
color_asymmetry, texture_asymmetry -- tested in the ablation study but NOT
part of the final V5 feature set) were dropped.

Inputs required, all already produced by the existing, unchanged pipeline:
  - `mask`: uint8 array, {0, 255}, the YOLO instance mask (same raster mask
    revised_abcd/pipeline_v2.py already produces -- retina_masks=True,
    thresholded at 0.5, single-image inference).
  - `no_hair`: BGR uint8 image, the same hair-removed image
    revised_abcd/pipeline_v2.py's preprocess_image() already produces
    (remove_vignette -> remove_salt_pepper_noise -> apply_bilateral_filter
    -> remove_hair, in that order, with the same parameters). This module
    does NOT run that preprocessing itself -- callers must reuse the
    existing chain's output rather than re-deriving it, to guarantee
    identical numbers to what was evaluated.
  - `circle_info`: the vignette-removal output already produced alongside
    `no_hair` (needed by the existing, unchanged
    Code/MelanomaDeterminingStuff/color.py::score_color for the
    dark/red/blue-gray fractions).
  - `D_px`: V4's existing diameter-in-pixels feature (already computed).
"""

import cv2
import numpy as np


# ---------------------------------------------------------------- geometry --

def largest_contour(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def turning_angle_std(contour, n_points=100):
    """Std of the discrete turning angle along an arc-length-resampled
    contour. Resampling to a fixed point count makes this far less sensitive
    to raw per-pixel contour noise than using cv2's native point list."""
    pts = contour.reshape(-1, 2).astype(np.float64)
    if len(pts) < 8:
        return np.nan
    pts_closed = np.vstack([pts, pts[0]])
    seglen = np.sqrt(np.sum(np.diff(pts_closed, axis=0) ** 2, axis=1))
    arclen = np.concatenate([[0], np.cumsum(seglen)])
    total = arclen[-1]
    if total <= 0:
        return np.nan
    sample_s = np.linspace(0, total, n_points, endpoint=False)
    resampled = np.empty((n_points, 2))
    for i, s in enumerate(sample_s):
        idx = np.searchsorted(arclen, s, side="right") - 1
        idx = min(max(idx, 0), len(pts_closed) - 2)
        seg_frac = (s - arclen[idx]) / (seglen[idx] + 1e-9)
        resampled[i] = pts_closed[idx] + seg_frac * (pts_closed[idx + 1] - pts_closed[idx])
    diffs = np.diff(np.vstack([resampled, resampled[0]]), axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    dtheta = np.diff(np.concatenate([angles, angles[:1]]))
    dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
    return float(np.std(dtheta))


def geometry_v5_features(mask):
    """Returns solidity, turning_angle_std, eccentricity (the three
    geometry-group members of V5's final feature set)."""
    feats = {"solidity": np.nan, "turning_angle_std": np.nan, "eccentricity": np.nan}
    area = float(np.sum(mask > 0))
    contour = largest_contour(mask)
    if contour is None or area <= 0:
        return feats

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area > 0:
        feats["solidity"] = contour_area / hull_area

    feats["turning_angle_std"] = turning_angle_std(contour, n_points=100)

    if len(contour) >= 5:
        (_, _), (MA, ma), _angle = cv2.fitEllipse(contour)
        major, minor = max(MA, ma), min(MA, ma)
    else:
        (_, (rw, rh), _a) = cv2.minAreaRect(contour)
        major, minor = max(rw, rh), min(rw, rh)
    if major > 0 and minor > 0:
        feats["eccentricity"] = float(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))
    return feats


def d_px_normalized(d_px, image_shape):
    """Scale-normalized diameter: D_px / sqrt(image area). `image_shape` is
    the (H, W) of the image the mask/D_px were computed on."""
    if d_px is None:
        return np.nan
    h, w = image_shape[:2]
    return float(d_px) / float(np.sqrt(h * w))


# -------------------------------------------------------------------- color --

def lab_of(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def color_stats_v5_features(img_bgr, mask):
    """Returns lab_a_std, lab_b_std, color_entropy (the color-statistics
    members of V5's final feature set; lab_L_std and color_cluster_count
    were computed in the ablation study but are NOT part of the final V5
    feature set and are intentionally not reproduced here)."""
    feats = {"lab_a_std": np.nan, "lab_b_std": np.nan, "color_entropy": np.nan}
    lab = lab_of(img_bgr)
    m = mask > 0
    if m.sum() < 20:
        return feats
    A, B = lab[..., 1][m], lab[..., 2][m]
    feats["lab_a_std"] = float(np.std(A))
    feats["lab_b_std"] = float(np.std(B))

    # OpenCV's 8-bit LAB encoding stores a/b as 0..255 (not signed
    # -128..127), matching how these arrays were produced (cv2.cvtColor on
    # 8-bit input, no rescaling).
    hist, _, _ = np.histogram2d(A, B, bins=16, range=[[0, 255], [0, 255]])
    p = hist.flatten()
    p = p[p > 0]
    if p.size > 0:
        p = p / p.sum()
        feats["color_entropy"] = float(-np.sum(p * np.log2(p)))
    return feats


def skin_contrast_feature(img_bgr, mask):
    m = (mask > 0).astype(np.uint8)
    area = int(m.sum())
    if area <= 0:
        return np.nan
    r = max(5, int(0.05 * np.sqrt(area)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    dilated = cv2.dilate(m, kernel)
    ring = (dilated > 0) & (m == 0)
    if ring.sum() < 20:
        return np.nan
    lab = lab_of(img_bgr)
    inside_mean = lab[m > 0].mean(axis=0)
    ring_mean = lab[ring].mean(axis=0)
    return float(np.linalg.norm(inside_mean - ring_mean))


# ------------------------------------------------------------- entry point --

def extract_v5_new_features(mask, no_hair, circle_info, d_px):
    """Compute all 11 new V5 features from the already-produced mask,
    hair-removed image, vignette circle_info, and D_px. Reuses the existing,
    unmodified Code/MelanomaDeterminingStuff/color.py::score_color for
    red_fraction/bluegray_fraction/dark_fraction (imported lazily so this
    module has no hard dependency on Code/'s package layout at import time).

    Returns a dict with exactly the 11 keys in
    decision_model.V5_CONFIG['new_features'].
    """
    from MelanomaDeterminingStuff.color import score_color  # existing, unchanged

    feats = {}
    feats.update(geometry_v5_features(mask))
    feats["D_px_normalized"] = d_px_normalized(d_px, mask.shape)
    feats.update(color_stats_v5_features(no_hair, mask))
    feats["skin_contrast"] = skin_contrast_feature(no_hair, mask)

    _color_result, pink_red, blue_gray, _white_px, black_px = score_color(mask, no_hair, circle_info)
    feats["red_fraction"] = pink_red
    feats["bluegray_fraction"] = blue_gray
    feats["dark_fraction"] = black_px

    return feats
