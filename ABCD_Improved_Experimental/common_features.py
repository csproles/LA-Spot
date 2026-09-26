"""Shared, read-only-with-respect-to-V5 feature helpers for the ABCD
Improved Experimental feature set.

READ-ONLY with respect to pipeline_v5/, revised_abcd/, Code/, and every
existing Evaluation_* directory -- this module only imports/adapts logic
from them, never edits them. All new files live under
ABCD_Improved_Experimental/.

Ports/adapts (documented per-function below):
  - `largest_contour`, `turning_angle_std`, `crop_bounds_like_fold`,
    `half_split_diff` -- copied near-verbatim from
    ABCD_Audit_V5/_source_from_research_branch/Evaluation_FeatureEngineering/
    extract_features.py (the exact reference implementation named in the
    task brief).
  - `_principal_axis_angle_deg` -- imported unmodified from the vendored
    copy in the same folder (itself an unmodified extraction of
    revised_abcd/revised_asymmetry.py's alignment function).
  - Border/geometry formulas (B_circularity = 1 - 4*pi*Area/Perimeter^2,
    solidity = contour_area/hull_area, D_px = 2*minEnclosingCircle radius)
    match pipeline_v5/feature_extraction.py and
    Code/MelanomaDeterminingStuff/border.py exactly, just restricted here
    to the PRIMARY-COMPONENT-ONLY mask (Phase 1 fix).
  - Color formulas reuse Code/MelanomaDeterminingStuff/color.py::score_color
    unmodified (called with the primary-component mask instead of the raw
    mask) plus pipeline_v5/feature_extraction.py's color_stats_v5_features/
    skin_contrast_feature formulas, again restricted to the primary
    component.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENDORED_DIR = ROOT / "ABCD_Audit_V5" / "_source_from_research_branch" / "Evaluation_FeatureEngineering"
if str(VENDORED_DIR) not in sys.path:
    sys.path.insert(0, str(VENDORED_DIR))
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from _vendored_revised_asymmetry import _principal_axis_angle_deg  # noqa: E402  (unmodified, read-only)

# --------------------------------------------------------------- constants --

# Reused unchanged from revised_abcd/segmentation_quality_v2.py, per the
# task's instruction to keep the rest of the codebase's existing
# conventions/thresholds where they overlap.
TINY_ARTIFACT_PX = 20
LARGE_AREA_FRACTION = 0.5

# New thresholds introduced here (documented fully in
# improved_feature_definitions.md): no existing constant in the codebase
# covers these two cases.
SECOND_COMPONENT_FRACTION = 0.30   # a 2nd component > 30% of the primary's area is a real 2nd-lesion candidate
EDGE_MARGIN_FRACTION = 0.01        # bbox within 1% of the shorter image dimension of any edge -> "edge adjacent"
EDGE_MARGIN_MIN_PX = 5             # floor so tiny images still get a sane minimum margin


# ---------------------------------------------------------- primary comp. --

def select_primary_component(mask):
    """Return a NEW mask (uint8, {0,255}) containing ONLY the single
    largest connected component of `mask` (by pixel area). This becomes
    "the" lesion region for every shape AND color feature computed from it
    here -- the Phase 1 fix for the V5 inconsistency where shape features
    used `largest_contour()` but color features indexed `mask > 0`
    (every fragment) directly."""
    m = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return np.zeros_like(mask, dtype=np.uint8), None
    areas = stats[1:, cv2.CC_STAT_AREA]
    primary_idx = 1 + int(np.argmax(areas))
    primary = (labels == primary_idx).astype(np.uint8) * 255
    return primary, (n_labels, labels, stats, primary_idx)


def component_summary(mask):
    """connectedComponentsWithStats summary used both for the primary-
    component selection and for the quality flags, so it is only computed
    once per mask."""
    m = (mask > 0).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    return n_labels - 1, areas, labels, stats


# -------------------------------------------------------------- geometry --

def largest_contour(mask):
    """Verbatim from extract_features.py. On a primary-component-only
    mask this is equivalent to "the" contour (only one component remains),
    kept as its own function for clarity/parity with the reference."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


def turning_angle_std(contour, n_points=100):
    """Verbatim from extract_features.py / pipeline_v5/feature_extraction.py."""
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


def shape_features_pc(primary_mask):
    """B_circularity_pc, solidity_pc, turning_angle_std_pc, D_px_pc -- same
    formulas as Code/MelanomaDeterminingStuff/border.py::score_border,
    pipeline_v5/feature_extraction.py::geometry_v5_features, and
    revised_abcd/revised_diameter.py::score_diameter_revised, computed on
    the primary-component-only mask."""
    feats = {"B_circularity_pc": np.nan, "solidity_pc": np.nan,
             "turning_angle_std_pc": np.nan, "D_px_pc": np.nan}
    area = float(np.sum(primary_mask > 0))
    if area <= 0:
        return feats
    contour = largest_contour(primary_mask)
    if contour is None:
        return feats

    perimeter = cv2.arcLength(contour, True)
    contour_area = cv2.contourArea(contour)
    circularity = (4 * np.pi * contour_area) / (perimeter ** 2 + 1e-6)
    feats["B_circularity_pc"] = 1 - circularity

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        feats["solidity_pc"] = contour_area / hull_area

    feats["turning_angle_std_pc"] = turning_angle_std(contour, n_points=100)

    (_, _), radius_px = cv2.minEnclosingCircle(contour)
    feats["D_px_pc"] = round(float(radius_px * 2), 1)
    return feats


# ------------------------------------------------------- asymmetry (h/v) --

def crop_bounds_like_fold(mask, cy, cx):
    """Verbatim from extract_features.py."""
    h, w = mask.shape
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return None
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    half = max(r_max - r_min, c_max - c_min) // 2 + 10
    r0 = max(int(cy) - half, 0)
    r1 = min(int(cy) + half, h)
    c0 = max(int(cx) - half, 0)
    c1 = min(int(cx) + half, w)
    return r0, r1, c0, c1


def fold_asymmetry_hv(primary_mask):
    """PCA-align the primary-component mask (same alignment
    `_principal_axis_angle_deg` V5's A_value already uses) and return BOTH
    fold-IoU-based asymmetry scores SEPARATELY, instead of V5's single
    averaged A_value. Per the verified convention in
    verify_axis_convention.py / improved_feature_definitions.md:
      - asym_h (np.fliplr, array axis=1) == asymmetry_major_axis
      - asym_v (np.flipud, array axis=0) == asymmetry_minor_axis
    Also returns A_value_pc = mean(asym_h, asym_v), i.e. V5's own A_value
    formula recomputed on the primary-component mask (Phase 1 fix)."""
    out = {"A_value_pc": np.nan, "asymmetry_h": np.nan, "asymmetry_v": np.nan}
    if np.sum(primary_mask > 0) == 0:
        return out
    h, w = primary_mask.shape
    angle, center = _principal_axis_angle_deg(primary_mask)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(primary_mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    M = cv2.moments(aligned)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
    bounds = crop_bounds_like_fold(aligned, cy, cx)
    if bounds is None:
        return out
    r0, r1, c0, c1 = bounds
    crop = (aligned[r0:r1, c0:c1] // 255).astype(np.uint8)
    if crop.size == 0:
        return out

    flip_h = np.fliplr(crop)
    iou_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)
    flip_v = np.flipud(crop)
    iou_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)

    asym_h = float(1 - iou_h)
    asym_v = float(1 - iou_v)
    out["asymmetry_h"] = asym_h
    out["asymmetry_v"] = asym_v
    out["A_value_pc"] = float((asym_h + asym_v) / 2)
    return out


# ----------------------------------------------------- fractal dimension --

def _box_count(binary_img, s):
    h, w = binary_img.shape
    pad_h = (-h) % s
    pad_w = (-w) % s
    if pad_h or pad_w:
        binary_img = np.pad(binary_img, ((0, pad_h), (0, pad_w)))
    h2, w2 = binary_img.shape
    reshaped = binary_img.reshape(h2 // s, s, w2 // s, s)
    occupied = reshaped.any(axis=(1, 3))
    return int(occupied.sum())


def border_fractal_dimension(primary_mask, min_box=2, max_box_frac=0.25, min_points=4):
    """Box-counting fractal-dimension estimate of the primary-component
    lesion boundary (Claridge et al. 1992; Piantanelli et al. 2005 --
    ABCD_Audit_V5/V5_ABCD_Literature_Review.md section B1).

    Box sizes: powers of two from `min_box` px up to
    floor(max_box_frac * min(bbox_w, bbox_h)) px (bbox of the boundary
    contour). log(box_count) vs log(1/box_size) is fit with an ordinary
    least-squares line; the slope is the fractal-dimension estimate.
    Returns (fractal_dimension, n_box_sizes_used); NaN/0 if fewer than
    `min_points` distinct box sizes are available (very small lesions)."""
    contour = largest_contour(primary_mask)
    if contour is None or len(contour) < 8:
        return np.nan, 0
    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return np.nan, 0

    boundary = np.zeros(primary_mask.shape, dtype=np.uint8)
    cv2.drawContours(boundary, [contour], -1, 1, thickness=1)
    pad = min_box
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, primary_mask.shape[1]), min(y + h + pad, primary_mask.shape[0])
    boundary_crop = boundary[y0:y1, x0:x1]

    max_box = int(max_box_frac * min(w, h))
    sizes = []
    s = min_box
    while s <= max_box:
        sizes.append(s)
        s *= 2
    if len(sizes) < min_points:
        return np.nan, len(sizes)

    counts = [_box_count(boundary_crop, s) for s in sizes]
    counts = np.array(counts, dtype=float)
    valid = counts > 0
    if valid.sum() < min_points:
        return np.nan, int(valid.sum())
    log_inv_s = np.log(1.0 / np.array(sizes, dtype=float)[valid])
    log_n = np.log(counts[valid])
    slope, _intercept = np.polyfit(log_inv_s, log_n, 1)
    return float(slope), int(valid.sum())


# ------------------------------------------------------------------ color --

def lab_of(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


def color_stats_pc(img_bgr, primary_mask):
    """color_entropy_pc, lab_a_std_pc, lab_b_std_pc -- same formulas as
    pipeline_v5/feature_extraction.py::color_stats_v5_features, restricted
    to the primary-component mask."""
    feats = {"color_entropy_pc": np.nan, "lab_a_std_pc": np.nan, "lab_b_std_pc": np.nan}
    lab = lab_of(img_bgr)
    m = primary_mask > 0
    if m.sum() < 20:
        return feats
    A, B = lab[..., 1][m], lab[..., 2][m]
    feats["lab_a_std_pc"] = float(np.std(A))
    feats["lab_b_std_pc"] = float(np.std(B))
    hist, _, _ = np.histogram2d(A, B, bins=16, range=[[0, 255], [0, 255]])
    p = hist.flatten()
    p = p[p > 0]
    if p.size > 0:
        p = p / p.sum()
        feats["color_entropy_pc"] = float(-np.sum(p * np.log2(p)))
    return feats


def skin_contrast_pc(img_bgr, primary_mask):
    """skin_contrast_pc -- same formula as
    pipeline_v5/feature_extraction.py::skin_contrast_feature, restricted to
    the primary-component mask (inside-mask mean vs. thin outside ring)."""
    m = (primary_mask > 0).astype(np.uint8)
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


def channel_entropy_lab(img_bgr, primary_mask, bins=16, value_range=(0, 255)):
    """Phase 3: 1D Shannon entropy of each LAB channel individually, over
    primary-component-masked in-lesion pixels. Bin count (16) matches the
    existing 16x16 2D a/b-plane histogram convention already used by
    color_entropy/color_entropy_pc, for consistency; range [0,255] matches
    OpenCV's unsigned 8-bit LAB encoding (same convention already used
    throughout this codebase's color code)."""
    feats = {"entropy_L": np.nan, "entropy_a": np.nan, "entropy_b": np.nan}
    lab = lab_of(img_bgr)
    m = primary_mask > 0
    if m.sum() < 20:
        return feats
    for key, ch in zip(["entropy_L", "entropy_a", "entropy_b"], [0, 1, 2]):
        vals = lab[..., ch][m]
        hist, _ = np.histogram(vals, bins=bins, range=value_range)
        p = hist[hist > 0]
        if p.size > 0:
            p = p / p.sum()
            feats[key] = float(-np.sum(p * np.log2(p)))
    return feats


def score_color_pc(mask_primary, no_hair, circle_info):
    """Wraps the existing, unmodified
    Code/MelanomaDeterminingStuff/color.py::score_color, called with the
    PRIMARY-COMPONENT mask instead of the raw (possibly fragmented) mask.
    Returns C_value_pc, red_fraction_pc, bluegray_fraction_pc,
    dark_fraction_pc."""
    from MelanomaDeterminingStuff.color import score_color  # existing, unchanged
    c_result, pink_red, blue_gray, _white_px, black_px = score_color(mask_primary, no_hair, circle_info)
    return {
        "C_value_pc": c_result["value"],
        "red_fraction_pc": pink_red,
        "bluegray_fraction_pc": blue_gray,
        "dark_fraction_pc": black_px,
    }


# --------------------------------------------------------- quality flags --

def compute_quality_flags(mask, primary_mask, primary_stats, very_small_floor_px):
    """Phase 1 point 4 / Phase 7. `primary_stats` is the
    (n_labels, labels, stats, primary_idx) tuple from
    `select_primary_component`. Returns a dict with the individual boolean
    flags plus the combined semicolon-joined `quality_flags` string.
    Nothing here excludes/drops an image -- flags only, per the task's
    explicit instruction (confirmed again in robustness_guard_report.md)."""
    h, w = mask.shape
    n_components, component_areas, labels, stats = component_summary(mask)
    meaningful_areas = sorted([a for a in component_areas if a > TINY_ARTIFACT_PX], reverse=True)

    flag_fragmented = len(meaningful_areas) > 1

    primary_area = float(np.sum(primary_mask > 0))
    flag_multi_substantial = False
    if len(meaningful_areas) > 1 and primary_area > 0:
        second_area = meaningful_areas[1]
        flag_multi_substantial = (second_area / primary_area) > SECOND_COMPONENT_FRACTION

    edge_margin = max(EDGE_MARGIN_MIN_PX, int(round(EDGE_MARGIN_FRACTION * min(h, w))))
    flag_edge_adjacent = False
    if primary_area > 0:
        ys, xs = np.nonzero(primary_mask > 0)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        flag_edge_adjacent = bool(
            y0 <= edge_margin or x0 <= edge_margin
            or (h - 1 - y1) <= edge_margin or (w - 1 - x1) <= edge_margin
        )

    flag_very_small = primary_area > 0 and primary_area < very_small_floor_px
    flag_oversized = primary_area > 0 and (primary_area / (h * w)) > LARGE_AREA_FRACTION

    names = []
    if flag_fragmented:
        names.append(f"fragmented({len(meaningful_areas)}_meaningful_blobs)")
    if flag_multi_substantial:
        names.append("multiple_substantial_components")
    if flag_edge_adjacent:
        names.append(f"edge_adjacent(margin_{edge_margin}px)")
    if flag_very_small:
        names.append(f"very_small_contour(lt_{int(very_small_floor_px)}px)")
    if flag_oversized:
        names.append("oversized_mask")

    return {
        "flag_fragmented": bool(flag_fragmented),
        "flag_multiple_substantial_components": bool(flag_multi_substantial),
        "flag_edge_adjacent": bool(flag_edge_adjacent),
        "flag_very_small_contour": bool(flag_very_small),
        "flag_oversized_mask": bool(flag_oversized),
        "quality_flags": ";".join(names) if names else "ok",
        "num_meaningful_components": len(meaningful_areas),
    }
