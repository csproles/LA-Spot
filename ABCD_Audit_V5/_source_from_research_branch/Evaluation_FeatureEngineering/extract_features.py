"""
Candidate ABCD-adjacent feature extraction for the V4-improvement
investigation.

READ-ONLY with respect to V4 (pipeline_v4/), YOLO, and the existing cached
V2 results. Reuses, rather than recomputes:
  - the existing frozen YOLO masks and A/B/C/D_px/confidence/lesion_fraction
    already in Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv --
    no re-segmentation, no YOLO rerun;
  - the existing PCA-alignment angle already computed for V4's A feature
    (A_principal_axis_deg), via a vendored read-only copy of
    revised_abcd/revised_asymmetry.py's alignment function;
  - the existing, unmodified Code/MelanomaDeterminingStuff/color.py::score_color
    for the dark/red/blue-gray region fractions, called exactly as the
    existing pipeline calls it;
  - the existing, unmodified Code/ classical preprocessing chain
    (remove_vignette -> remove_salt_pepper_noise -> apply_bilateral_filter
    -> remove_hair) to produce the same no_hair/circle_info inputs
    score_color already requires.

Restricted to the development cohort's SINGLE_LESION_EVALUABLE population
(2,090 images) -- the same population V4's feature table uses. The locked
test set is never read here.

Every per-feature computation is wrapped so one failing feature does not
drop the whole image's row; failures are recorded as NaN plus a per-row
error log, and aggregate failure rates are reported separately.
"""

import contextlib
import csv
import io
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import (  # noqa: E402
    remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair,
)
from MelanomaDeterminingStuff.color import score_color  # noqa: E402  (unchanged, existing)

from _vendored_revised_asymmetry import _principal_axis_angle_deg  # noqa: E402

import metrics_lib as ml  # noqa: E402  (existing, unmodified helper module)

COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
DEV_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "development_manifest.csv"


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


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


GEOMETRY_KEYS = ["lesion_area_px", "solidity", "convexity_deficit", "isoperimetric_ratio",
                 "radial_cv", "turning_angle_std", "major_axis_px", "minor_axis_px",
                 "aspect_ratio", "eccentricity"]


def geometry_features(mask):
    feats = {k: np.nan for k in GEOMETRY_KEYS}
    area = float(np.sum(mask > 0))
    feats["lesion_area_px"] = area
    contour = largest_contour(mask)
    if contour is None or area <= 0:
        return feats

    perimeter = cv2.arcLength(contour, True)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    if hull_area > 0:
        feats["solidity"] = contour_area / hull_area
        feats["convexity_deficit"] = 1 - feats["solidity"]
    if contour_area > 0:
        feats["isoperimetric_ratio"] = (perimeter ** 2) / (4 * np.pi * contour_area)

    M = cv2.moments(mask)
    cx, cy = (M["m10"] / M["m00"], M["m01"] / M["m00"]) if M["m00"] > 0 else (mask.shape[1] / 2, mask.shape[0] / 2)
    pts = contour.reshape(-1, 2).astype(np.float64)
    radial = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    feats["radial_cv"] = float(np.std(radial) / (np.mean(radial) + 1e-9))
    feats["turning_angle_std"] = turning_angle_std(contour, n_points=100)

    if len(contour) >= 5:
        (_, _), (MA, ma), _angle = cv2.fitEllipse(contour)
        major, minor = max(MA, ma), min(MA, ma)
    else:
        (_, (rw, rh), _a) = cv2.minAreaRect(contour)
        major, minor = max(rw, rh), min(rw, rh)
    feats["major_axis_px"] = float(major)
    feats["minor_axis_px"] = float(minor)
    if minor > 0:
        feats["aspect_ratio"] = float(major / minor)
        feats["eccentricity"] = float(np.sqrt(max(0.0, 1 - (minor / major) ** 2)))
    return feats


# -------------------------------------------------------------------- color --

def lab_of(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)


COLOR_STAT_KEYS = ["lab_L_std", "lab_a_std", "lab_b_std", "color_entropy", "color_cluster_count"]


def color_stats_features(img_bgr, mask):
    feats = {k: np.nan for k in COLOR_STAT_KEYS}
    lab = lab_of(img_bgr)
    m = mask > 0
    if m.sum() < 20:
        return feats
    L, A, B = lab[..., 0][m], lab[..., 1][m], lab[..., 2][m]
    feats["lab_L_std"] = float(np.std(L))
    feats["lab_a_std"] = float(np.std(A))
    feats["lab_b_std"] = float(np.std(B))

    # OpenCV's 8-bit LAB encoding stores a/b as 0..255 (not signed -128..127),
    # since these arrays came from cv2.cvtColor(..., COLOR_BGR2LAB) on 8-bit
    # input without rescaling -- match that range here.
    hist, _, _ = np.histogram2d(A, B, bins=16, range=[[0, 255], [0, 255]])
    p = hist.flatten()
    p = p[p > 0]
    if p.size > 0:
        p = p / p.sum()
        feats["color_entropy"] = float(-np.sum(p * np.log2(p)))

    pix = np.column_stack([L, A, B]).astype(np.float32)
    feats["color_cluster_count"] = color_cluster_count(pix)
    return feats


def color_cluster_count(pix, k=6, min_frac=0.05, max_samples=3000):
    if len(pix) > max_samples:
        idx = np.random.RandomState(0).choice(len(pix), max_samples, replace=False)
        pix = pix[idx]
    if len(pix) < k:
        return np.nan
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.3)
    _compactness, labels, _centers = cv2.kmeans(pix, k, None, criteria, 4, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(labels.flatten(), minlength=k)
    frac = counts / counts.sum()
    return int(np.sum(frac >= min_frac))


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


# ---------------------------------------------------------- PCA asymmetry ---

def crop_bounds_like_fold(mask, cy, cx):
    """Mirrors Code/MelanomaDeterminingStuff/asymmetry.py's crop window
    exactly, so the color/texture asymmetry candidates use the same
    lesion-centered crop as the existing shape-asymmetry feature."""
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


def half_split_diff(vcrop, mcrop, axis):
    """Split the lesion crop into two halves along `axis`, mirror the
    second half onto the first, and compare mean values over the pixels
    where both halves' lesion masks agree."""
    n = vcrop.shape[axis]
    mid = n // 2
    if axis == 1:
        v1, m1 = vcrop[:, :mid], mcrop[:, :mid]
        v2, m2 = vcrop[:, mid:], mcrop[:, mid:]
        v2, m2 = np.fliplr(v2), np.fliplr(m2)
    else:
        v1, m1 = vcrop[:mid, :], mcrop[:mid, :]
        v2, m2 = vcrop[mid:, :], mcrop[mid:, :]
        v2, m2 = np.flipud(v2), np.flipud(m2)
    hmin = min(v1.shape[0], v2.shape[0])
    wmin = min(v1.shape[1], v2.shape[1])
    v1, m1 = v1[:hmin, :wmin], m1[:hmin, :wmin]
    v2, m2 = v2[:hmin, :wmin], m2[:hmin, :wmin]
    both = m1 & m2
    if both.sum() < 20:
        return np.nan
    a, b = v1[both], v2[both]
    if a.ndim == 2:
        return float(np.linalg.norm(a.mean(axis=0) - b.mean(axis=0)))
    return float(abs(a.mean() - b.mean()))


def pca_aligned_asymmetry_generic(mask, value_map):
    """Normalized bilateral difference of `value_map` (color or texture)
    within the lesion, in the SAME principal-axis-aligned frame V4's own
    shape-asymmetry feature already uses (same alignment function, vendored
    unmodified from revised_abcd/revised_asymmetry.py)."""
    if np.sum(mask > 0) == 0:
        return np.nan
    h, w = mask.shape
    angle, center = _principal_axis_angle_deg(mask)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_mask = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    aligned_val = cv2.warpAffine(value_map, M_rot, (w, h), flags=cv2.INTER_LINEAR)

    M = cv2.moments(aligned_mask)
    if M["m00"] == 0:
        return np.nan
    cx, cy = M["m10"] / M["m00"], M["m01"] / M["m00"]
    bounds = crop_bounds_like_fold(aligned_mask, cy, cx)
    if bounds is None:
        return np.nan
    r0, r1, c0, c1 = bounds
    mcrop = aligned_mask[r0:r1, c0:c1] > 0
    vcrop = aligned_val[r0:r1, c0:c1]
    if mcrop.sum() < 20:
        return np.nan

    pooled = vcrop[mcrop]
    pooled_std = float(np.linalg.norm(np.std(pooled, axis=0))) if pooled.ndim == 2 else float(np.std(pooled))
    if pooled_std < 1e-6:
        return 0.0

    dh = half_split_diff(vcrop, mcrop, axis=1)
    dv = half_split_diff(vcrop, mcrop, axis=0)
    vals = [x for x in (dh, dv) if x == x]
    if not vals:
        return np.nan
    return float(np.mean(vals) / (pooled_std + 1e-6))


def local_variance_map(gray):
    gray = gray.astype(np.float32)
    mean = cv2.blur(gray, (9, 9))
    sq_mean = cv2.blur(gray * gray, (9, 9))
    return np.clip(sq_mean - mean * mean, 0, None)


# ------------------------------------------------------------------ texture --

TEXTURE_KEYS = ["entropy_intensity", "local_contrast", "glcm_contrast", "glcm_homogeneity", "glcm_energy"]


def texture_features(gray_full, mask):
    feats = {k: np.nan for k in TEXTURE_KEYS}
    m = mask > 0
    if m.sum() < 20:
        return feats
    vals = gray_full[m]
    hist, _ = np.histogram(vals, bins=32, range=(0, 255))
    p = hist[hist > 0]
    if p.size > 0:
        p = p / p.sum()
        feats["entropy_intensity"] = float(-np.sum(p * np.log2(p)))

    lv = local_variance_map(gray_full)
    feats["local_contrast"] = float(np.sqrt(np.mean(lv[m])))

    ys, xs = np.nonzero(m)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop_gray = gray_full[y0:y1, x0:x1]
    crop_mask = m[y0:y1, x0:x1]
    if crop_mask.sum() < 20:
        return feats
    fill_val = crop_gray[crop_mask].mean()
    crop_gray2 = crop_gray.astype(np.float32).copy()
    crop_gray2[~crop_mask] = fill_val
    q = np.clip((crop_gray2 / 256.0 * 32).astype(np.uint8), 0, 31)
    glcm = graycomatrix(q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                         levels=32, symmetric=True, normed=True)
    feats["glcm_contrast"] = float(np.mean(graycoprops(glcm, "contrast")))
    feats["glcm_homogeneity"] = float(np.mean(graycoprops(glcm, "homogeneity")))
    feats["glcm_energy"] = float(np.mean(graycoprops(glcm, "energy")))
    return feats


# --------------------------------------------------------------- per-image --

ALL_NEW_KEYS = (GEOMETRY_KEYS + ["D_px_normalized"]
                + ["red_fraction", "bluegray_fraction", "white_fraction", "dark_fraction"]
                + COLOR_STAT_KEYS + ["skin_contrast", "color_asymmetry"]
                + TEXTURE_KEYS + ["texture_asymmetry"])


def process_one(row, image_path):
    iid = ml.image_id_from_name(row["image_name"])
    result = {"image_id": iid}
    errors = []

    mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = (mask_raw > 127).astype(np.uint8) * 255

    try:
        geo = geometry_features(mask)
    except Exception as e:  # noqa: BLE001
        geo = {k: np.nan for k in GEOMETRY_KEYS}
        errors.append(f"geometry:{e!r}")
    result.update(geo)

    d_px = to_float(row["D_px"])
    h, w = mask.shape
    result["D_px_normalized"] = (d_px / np.sqrt(h * w)) if d_px is not None else np.nan

    stdout_buf = io.StringIO()
    no_hair = None
    mask_use = mask
    try:
        with contextlib.redirect_stdout(stdout_buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            original = load_image(str(image_path))
            no_vignette, circle_info = remove_vignette(original)
            denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
            bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
            no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)
            if no_hair.shape[:2] != mask.shape:
                mask_use = cv2.resize(mask, (no_hair.shape[1], no_hair.shape[0]), interpolation=cv2.INTER_NEAREST)
            c_result, pink_red, blue_gray, white_px, black_px = score_color(mask_use, no_hair, circle_info)
        result["red_fraction"] = pink_red
        result["bluegray_fraction"] = blue_gray
        result["white_fraction"] = white_px
        result["dark_fraction"] = black_px
    except Exception as e:  # noqa: BLE001
        for k in ["red_fraction", "bluegray_fraction", "white_fraction", "dark_fraction"]:
            result[k] = np.nan
        errors.append(f"preproc_or_score_color:{e!r}")
        no_hair = None

    if no_hair is not None:
        try:
            result.update(color_stats_features(no_hair, mask_use))
        except Exception as e:  # noqa: BLE001
            result.update({k: np.nan for k in COLOR_STAT_KEYS})
            errors.append(f"color_stats:{e!r}")

        try:
            result["skin_contrast"] = skin_contrast_feature(no_hair, mask_use)
        except Exception as e:  # noqa: BLE001
            result["skin_contrast"] = np.nan
            errors.append(f"skin_contrast:{e!r}")

        gray = cv2.cvtColor(no_hair, cv2.COLOR_BGR2GRAY)
        try:
            result.update(texture_features(gray, mask_use))
        except Exception as e:  # noqa: BLE001
            result.update({k: np.nan for k in TEXTURE_KEYS})
            errors.append(f"texture:{e!r}")

        try:
            result["color_asymmetry"] = pca_aligned_asymmetry_generic(mask_use, lab_of(no_hair))
        except Exception as e:  # noqa: BLE001
            result["color_asymmetry"] = np.nan
            errors.append(f"color_asymmetry:{e!r}")

        try:
            result["texture_asymmetry"] = pca_aligned_asymmetry_generic(mask_use, local_variance_map(gray))
        except Exception as e:  # noqa: BLE001
            result["texture_asymmetry"] = np.nan
            errors.append(f"texture_asymmetry:{e!r}")
    else:
        for k in COLOR_STAT_KEYS + ["skin_contrast"] + TEXTURE_KEYS + ["texture_asymmetry", "color_asymmetry"]:
            result[k] = np.nan

    result["_errors"] = "; ".join(errors)
    return result


def main(limit=None):
    with open(DEV_MANIFEST, newline="", encoding="utf-8") as f:
        dev_rows = list(csv.DictReader(f))
    dev_ids = {r["image_id"] for r in dev_rows}
    path_by_id = {r["image_id"]: r["image_path"] for r in dev_rows}

    all_results = ml.load_merged_results()
    evaluable_rows, status_counts, n_images = ml.build_split_rows(all_results, dev_ids)
    print(f"Development coverage (shared with V4): {status_counts}")
    print(f"SINGLE_LESION_EVALUABLE rows to process: {len(evaluable_rows)}")

    if limit:
        evaluable_rows = evaluable_rows[:limit]

    out_rows = []
    t0 = time.time()
    for i, row in enumerate(evaluable_rows):
        iid = ml.image_id_from_name(row["image_name"])
        img_path = path_by_id[iid]
        rec = process_one(row, img_path)
        out_rows.append(rec)
        if i % 100 == 0 or i == len(evaluable_rows) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"[{i+1}/{len(evaluable_rows)}] {iid}  elapsed={elapsed:.0f}s  rate={rate:.2f} img/s")

    fieldnames = ["image_id"] + ALL_NEW_KEYS + ["_errors"]
    out_path = BENCH_DIR / "candidate_features_raw.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    n_with_errors = sum(1 for r in out_rows if r["_errors"])
    print(f"\nWrote {len(out_rows)} rows to {out_path}")
    print(f"Rows with at least one feature-computation error: {n_with_errors}")
    return out_rows


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    main(limit=args.limit)
