"""Mechanism investigation: why does the existing (production) A calculation
fail to trigger on many melanoma false negatives?

READ-ONLY: does not modify or rerun YOLO, ABCD, thresholds, scoring, masks,
or Evaluation_3500. Uses only already-saved original images, raster masks,
and Evaluation_3500/full_run/results.csv.

The "existing A calculation" for this experiment is
revised_abcd/revised_asymmetry.py::score_asymmetry_revised — that is the
function that actually produced every A_value/A_concern in
Evaluation_3500/full_run/results.csv (NOT the older, unaligned
Code/MelanomaDeterminingStuff/asymmetry.py, which was superseded for this
experiment). This script does not import that module's private helpers;
instead it reproduces its logic line-for-line in an instrumented function so
every intermediate array (aligned mask, crop, flips, per-axis IoU) can be
extracted for visualization, then VERIFIES the reproduction against the
saved A_value for every selected case before trusting any of it.
"""

import csv
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
MANIFEST_CSV = ROOT / "Evaluation_3500" / "evaluation_manifest_3500.csv"
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")
OUT_DIR = ROOT / "Evaluation_3500" / "MelanomaAsymmetryInvestigation"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST_OUT = OUT_DIR / "asymmetry_investigation_manifest.csv"

SEED = 20260918
MELANOMA_DIAG3 = {"Melanoma Invasive", "Melanoma in situ", "Melanoma, NOS", "Melanoma metastasis"}
CONCERN_THRESHOLD = 0.20  # unchanged, matches revised_abcd/revised_asymmetry.py

COLOR_OVERLAP = (40, 200, 40)     # green: present in both crop and flip
COLOR_ORIG_ONLY = (220, 60, 60)   # red: present in crop only
COLOR_FLIP_ONLY = (60, 90, 230)   # blue: present in flip only


def image_id_from_name(name):
    return Path(name).stem


# ---- exact, faithful reproduction of revised_abcd/revised_asymmetry.py, ---
# ---- instrumented to expose every intermediate for visualization ----------
def principal_axis_angle_deg(mask):
    ys, xs = np.nonzero(mask > 0)
    if len(xs) < 5:
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


def instrumented_asymmetry(mask):
    h, w = mask.shape
    angle, center = principal_axis_angle_deg(mask)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_mask = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)

    M = cv2.moments(aligned_mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    coords = np.argwhere(aligned_mask > 0)
    if len(coords) > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        half = max(r_max - r_min, c_max - c_min) // 2 + 10
        r0 = max(cy - half, 0); r1 = min(cy + half, h)
        c0 = max(cx - half, 0); c1 = min(cx + half, w)
        crop = (aligned_mask[r0:r1, c0:c1] // 255).astype(np.uint8)
    else:
        crop = np.zeros((1, 1), np.uint8)
        r0 = r1 = c0 = c1 = 0

    if crop.size == 0:
        flip_h = flip_v = crop
        overlap_h = overlap_v = 0.0
        asymmetry = 0.0
    else:
        flip_h = np.fliplr(crop)
        overlap_h = float(np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6))
        flip_v = np.flipud(crop)
        overlap_v = float(np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6))
        asymmetry = 1 - (overlap_h + overlap_v) / 2

    concern = asymmetry > CONCERN_THRESHOLD
    return {
        "angle": angle, "center": center, "aligned_mask": aligned_mask,
        "cx": cx, "cy": cy, "crop_bounds": (r0, r1, c0, c1), "crop": crop,
        "flip_h": flip_h, "flip_v": flip_v,
        "overlap_h": overlap_h, "overlap_v": overlap_v,
        "asymmetry": round(float(asymmetry), 3), "concern": bool(concern),
    }


def overlap_diff_image(crop, flip):
    """green=overlap, red=crop-only, blue=flip-only, on black."""
    h, w = crop.shape
    img = np.zeros((h, w, 3), dtype=np.uint8)
    both = (crop > 0) & (flip > 0)
    orig_only = (crop > 0) & (flip == 0)
    flip_only = (crop == 0) & (flip > 0)
    img[both] = COLOR_OVERLAP
    img[orig_only] = COLOR_ORIG_ONLY
    img[flip_only] = COLOR_FLIP_ONLY
    return img


def draw_panel(image_id, row, image_path, mask_path, group_label, diag, save_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_raw > 127).astype(np.uint8) * 255

    aligned = diag["aligned_mask"]
    r0, r1, c0, c1 = diag["crop_bounds"]
    crop = diag["crop"]
    flip_h = diag["flip_h"]
    flip_v = diag["flip_v"]

    # aligned mask with axis line + crop rectangle overlay
    aligned_vis = cv2.cvtColor(aligned, cv2.COLOR_GRAY2RGB)
    cx, cy = diag["cx"], diag["cy"]
    L = max(aligned.shape) // 3
    cv2.line(aligned_vis, (cx - L, cy), (cx + L, cy), (255, 220, 0), 3)  # aligned axis is horizontal by construction
    cv2.rectangle(aligned_vis, (c0, r0), (c1, r1), (0, 200, 255), 3)
    cv2.circle(aligned_vis, (cx, cy), max(4, aligned.shape[1] // 150), (255, 0, 255), -1)

    h_diff = overlap_diff_image(crop, flip_h)
    v_diff = overlap_diff_image(crop, flip_v)

    fig = plt.figure(figsize=(22, 9.5), facecolor="#fcfcfb")
    gs = gridspec.GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1.15], figure=fig)

    def show(ax, img, title, cmap=None):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.axis("off")

    show(fig.add_subplot(gs[0, 0]), image_rgb, "1. Original image")
    show(fig.add_subplot(gs[0, 1]), mask, "2. Original YOLO raster mask", cmap="gray")
    show(fig.add_subplot(gs[0, 2]), aligned_vis, "3. PCA-aligned mask\n(yellow=axis, cyan box=fold crop)")
    show(fig.add_subplot(gs[0, 3]), crop * 255, "Aligned + cropped region\n(this is what gets flipped)", cmap="gray")

    show(fig.add_subplot(gs[1, 0]), flip_h * 255, "4. Horizontal reflection", cmap="gray")
    show(fig.add_subplot(gs[1, 1]), h_diff, f"5. Horizontal overlap/disagreement\ngreen=overlap red=orig-only blue=flip-only")
    show(fig.add_subplot(gs[1, 2]), flip_v * 255, "6. Vertical reflection", cmap="gray")
    show(fig.add_subplot(gs[1, 3]), v_diff, f"7. Vertical overlap/disagreement\ngreen=overlap red=orig-only blue=flip-only")

    ax_text = fig.add_subplot(gs[:, 4])
    ax_text.axis("off")

    def yn(v):
        return "concern" if v == "True" else "no concern" if v == "False" else "N/A"

    lines = [
        f"ISIC ID: {image_id}",
        f"group: {group_label}",
        "",
        f"TP/FN status: {row['_tpfn']}",
        f"risk_level (LOW/HIGH): {row['risk_level']}",
        "",
        f"principal axis angle: {diag['angle']:.1f} deg",
        f"8. horizontal IoU: {diag['overlap_h']:.3f}",
        f"9. vertical IoU:   {diag['overlap_v']:.3f}",
        f"10. final A value: {diag['asymmetry']:.3f}",
        f"11. A concern (>{CONCERN_THRESHOLD}): {diag['concern']}",
        "",
        f"saved A_value (results.csv): {row['A_value']}",
        f"saved A_concern (results.csv): {yn(row['A_concern'])}",
        f"reproduction matches saved value: {row['_match']}",
        "",
        f"B raw: {row['B_circularity']}   B concern: {yn(row['B_circularity_concern'])}",
        f"C raw: {row['C_value']}   C concern: {yn(row['C_concern'])}",
        "",
        f"YOLO confidence: {row['confidence']}",
        f"mask area fraction: {row['lesion_fraction']}",
        f"quality_flags: {row['quality_flags']}",
    ]
    ax_text.text(0.02, 0.98, "\n".join(lines), transform=ax_text.transAxes, va="top", ha="left",
                fontsize=11, fontfamily="monospace")

    fig.suptitle(f"{image_id} — melanoma — {group_label}  (A={diag['asymmetry']:.3f}, "
                f"{'CONCERN' if diag['concern'] else 'no concern'})",
                fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(save_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    manifest = list(csv.DictReader(open(MANIFEST_CSV, newline="", encoding="utf-8")))
    results = list(csv.DictReader(open(RESULTS_CSV, newline="", encoding="utf-8")))
    meta = {r["isic_id"]: r for r in csv.DictReader(open(SOURCE_METADATA, newline="", encoding="utf-8"))}
    manifest_paths = {r["image_id"]: r["image_path"] for r in manifest}

    rows_by_image = defaultdict(list)
    for r in results:
        rows_by_image[image_id_from_name(r["image_name"])].append(r)

    melanoma_ids = [r["image_id"] for r in manifest if r["ground_truth"] == "malignant"
                    and (meta[r["image_id"]].get("diagnosis_3") or "").strip() in MELANOMA_DIAG3]

    fn_no_large, tp_no_large = [], []
    for image_id in melanoma_ids:
        image_rows = rows_by_image.get(image_id, [])
        if not image_rows or image_rows[0]["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
            continue
        row = image_rows[0]
        has_large = "large_area_fraction" in (row.get("quality_flags") or "")
        if has_large:
            continue
        pred = row["provisional_prediction"]
        if pred == "negative":
            row["_tpfn"] = "FN"
            fn_no_large.append((image_id, row))
        elif pred == "positive":
            row["_tpfn"] = "TP"
            tp_no_large.append((image_id, row))

    print(f"Pools available: FN_no_large={len(fn_no_large)}  TP_no_large={len(tp_no_large)}")

    rng = random.Random(SEED)
    selected_fn = rng.sample(fn_no_large, min(10, len(fn_no_large)))
    selected_tp = rng.sample(tp_no_large, min(10, len(tp_no_large)))

    print("Selected FN IDs:", [i for i, _ in selected_fn])
    print("Selected TP IDs:", [i for i, _ in selected_tp])

    manifest_rows = []
    fieldnames = ["group", "image_id", "tpfn_status", "risk_level",
                  "principal_axis_deg", "horizontal_IoU", "vertical_IoU", "A_value_reproduced",
                  "A_concern_reproduced", "A_value_saved", "A_concern_saved", "reproduction_matches",
                  "B_circularity", "B_circularity_concern", "C_value", "C_concern",
                  "confidence", "lesion_fraction", "quality_flags", "panel_path"]

    n_mismatches = 0
    for group_label, selection in [("FN_no_large_area", selected_fn), ("TP_no_large_area", selected_tp)]:
        for image_id, row in selection:
            image_path = Path(manifest_paths[image_id])
            mask_path = ROOT / row["mask_path"]
            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            mask = (mask_raw > 127).astype(np.uint8) * 255

            diag = instrumented_asymmetry(mask)
            saved_a = float(row["A_value"])
            matches = abs(diag["asymmetry"] - saved_a) < 0.002
            row["_match"] = matches
            if not matches:
                n_mismatches += 1
                print(f"  [MISMATCH] {image_id}: reproduced={diag['asymmetry']} saved={saved_a}")

            panel_path = PANELS_DIR / f"{group_label}_{image_id}.png"
            draw_panel(image_id, row, image_path, mask_path, group_label, diag, panel_path)

            manifest_rows.append({
                "group": group_label, "image_id": image_id, "tpfn_status": row["_tpfn"],
                "risk_level": row["risk_level"],
                "principal_axis_deg": round(diag["angle"], 1),
                "horizontal_IoU": round(diag["overlap_h"], 4), "vertical_IoU": round(diag["overlap_v"], 4),
                "A_value_reproduced": diag["asymmetry"], "A_concern_reproduced": diag["concern"],
                "A_value_saved": saved_a, "A_concern_saved": row["A_concern"],
                "reproduction_matches": matches,
                "B_circularity": row["B_circularity"], "B_circularity_concern": row["B_circularity_concern"],
                "C_value": row["C_value"], "C_concern": row["C_concern"],
                "confidence": row["confidence"], "lesion_fraction": row["lesion_fraction"],
                "quality_flags": row["quality_flags"],
                "panel_path": str(panel_path.relative_to(ROOT)),
            })

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(manifest_rows)

    print(f"\nReproduction mismatches: {n_mismatches} / {len(manifest_rows)}")
    print(f"Wrote {len(manifest_rows)} panels + manifest to {OUT_DIR}")
    return manifest_rows, n_mismatches


if __name__ == "__main__":
    main()
