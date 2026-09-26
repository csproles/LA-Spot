"""Diagnostic-only investigation of the A (asymmetry) criterion.

Does NOT modify Code/MelanomaDeterminingStuff/asymmetry.py. It reproduces
that file's exact algorithm here (verified against the master CSV) purely so
intermediate quantities (crop bounds, truncation, connected components,
solidity, principal axis) can be inspected, and adds one literature-grounded
alternative (principal-axis-aligned folding) for comparison ONLY — nothing
here is wired into the production pipeline.
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "raster_masks"
IMAGES_ROOT = ROOT / "Images"
OUT_DIR = Path(__file__).resolve().parent / "asymmetry"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_manifest():
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"image_path": p, "ground_truth": label}
    return manifest


# ---- exact reproduction of Code/MelanomaDeterminingStuff/asymmetry.py -----
def current_score_asymmetry(mask):
    h, w = mask.shape
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2
    coords = np.argwhere(mask > 0)
    truncated = False
    if len(coords) > 0:
        r_min, c_min = coords.min(axis=0)
        r_max, c_max = coords.max(axis=0)
        half = max(r_max - r_min, c_max - c_min) // 2 + 10
        r0_raw, r1_raw = cy - half, cy + half
        c0_raw, c1_raw = cx - half, cx + half
        r0 = max(r0_raw, 0); r1 = min(r1_raw, h)
        c0 = max(c0_raw, 0); c1 = min(c1_raw, w)
        truncated = (r0 != r0_raw) or (r1 != r1_raw) or (c0 != c0_raw) or (c1 != c1_raw)
        crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)
    else:
        crop = np.zeros((1, 1), np.uint8)

    if crop.size == 0:
        asymmetry = 0.0
    else:
        flip_h = np.fliplr(crop)
        overlap_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)
        flip_v = np.flipud(crop)
        overlap_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)
        asymmetry = 1 - (overlap_h + overlap_v) / 2

    return asymmetry, (cx, cy), truncated, crop


# ---- literature-grounded alternative: principal-axis alignment first -----
# (Principal-axes / moment-of-inertia based asymmetry is an established
# approach in the skin-lesion image-analysis literature — e.g. inertia-moment
# axis-of-symmetry methods that partition the lesion by its own axes of
# inertia before folding, rather than by fixed image x/y axes.)
def principal_axis_angle_deg(mask):
    ys, xs = np.nonzero(mask > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    mean = pts.mean(axis=0)
    pts_c = pts - mean
    cov = np.cov(pts_c.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    eccentricity = np.sqrt(max(eigvals) / (min(eigvals) + 1e-9))
    return angle, mean, eccentricity


def aligned_score_asymmetry(mask):
    """Rotate the mask so its principal axis is horizontal, then re-run the
    SAME crop+flip procedure used by the existing implementation. Isolates
    the effect of axis alignment alone."""
    h, w = mask.shape
    angle, mean, eccentricity = principal_axis_angle_deg(mask)
    M_rot = cv2.getRotationMatrix2D((float(mean[0]), float(mean[1])), angle, 1.0)
    rotated = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    asymmetry, centroid, truncated, crop = current_score_asymmetry(rotated)
    return asymmetry, angle, eccentricity, rotated, crop


def shape_descriptors(mask):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    num_blobs = n_labels - 1
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(biggest)
        hull = cv2.convexHull(biggest)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else None
    else:
        solidity = None
    return num_blobs, solidity


def main():
    manifest = build_manifest()
    rows = []
    for stem, info in sorted(manifest.items()):
        mask_path = MASK_DIR / f"{stem}_mask.png"
        if not mask_path.exists():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        if np.sum(mask > 0) == 0:
            continue

        a_current, centroid, truncated, _ = current_score_asymmetry(mask)
        a_aligned, angle, eccentricity, _, _ = aligned_score_asymmetry(mask)
        num_blobs, solidity = shape_descriptors(mask)

        rows.append({
            "stem": stem, "ground_truth": info["ground_truth"],
            "A_current": round(a_current, 4), "A_aligned": round(a_aligned, 4),
            "delta": round(a_current - a_aligned, 4),
            "principal_axis_deg": round(angle, 1),
            "eccentricity": round(eccentricity, 3),
            "num_blobs": num_blobs,
            "solidity": round(solidity, 4) if solidity is not None else "",
            "crop_truncated": truncated,
        })

    csv_path = OUT_DIR / "asymmetry_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    for gt in ("benign", "malignant"):
        sub = [r for r in rows if r["ground_truth"] == gt]
        print(f"\n{gt} (n={len(sub)}):")
        print(f"  mean A_current  = {np.mean([r['A_current'] for r in sub]):.3f}")
        print(f"  mean A_aligned  = {np.mean([r['A_aligned'] for r in sub]):.3f}")
        print(f"  mean num_blobs  = {np.mean([r['num_blobs'] for r in sub]):.2f}")
        print(f"  mean solidity   = {np.mean([r['solidity'] for r in sub if r['solidity'] != '']):.3f}")
        print(f"  mean eccentricity = {np.mean([r['eccentricity'] for r in sub]):.3f}")
        print(f"  num truncated   = {sum(1 for r in sub if r['crop_truncated'])}")
        print(f"  num multi-blob (>1) = {sum(1 for r in sub if r['num_blobs'] > 1)}")

    # --- visualize extreme cases: highest-A benign, lowest-A malignant
    rows_sorted_benign = sorted([r for r in rows if r["ground_truth"] == "benign"], key=lambda r: -r["A_current"])
    rows_sorted_malignant = sorted([r for r in rows if r["ground_truth"] == "malignant"], key=lambda r: r["A_current"])
    highlight_stems = [r["stem"] for r in rows_sorted_benign[:2]] + [r["stem"] for r in rows_sorted_malignant[:2]]

    for stem in highlight_stems:
        mask_path = MASK_DIR / f"{stem}_mask.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        info = manifest[stem]
        image = cv2.imread(str(info["image_path"]), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        a_current, centroid, truncated, crop_current = current_score_asymmetry(mask)
        a_aligned, angle, eccentricity, rotated, crop_aligned = aligned_score_asymmetry(mask)
        num_blobs, solidity = shape_descriptors(mask)

        fig, axes = plt.subplots(1, 4, figsize=(18, 5), facecolor="#fcfcfb")
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"{stem} ({info['ground_truth']})\nblobs={num_blobs} solidity={solidity}")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        cx, cy = centroid
        L = 150
        rad = np.radians(angle)
        axes[1].plot([cx - L * np.cos(rad), cx + L * np.cos(rad)],
                     [cy - L * np.sin(rad), cy + L * np.sin(rad)], color="lime", linewidth=2)
        axes[1].scatter([cx], [cy], color="yellow", s=40)
        axes[1].set_title(f"mask + PCA principal axis\nangle={angle:.0f} deg  ecc={eccentricity:.2f}")
        axes[1].axis("off")

        axes[2].imshow(crop_current, cmap="gray")
        axes[2].set_title(f"current crop (image-axis flip)\nA_current={a_current:.3f}  truncated={truncated}")
        axes[2].axis("off")

        axes[3].imshow(crop_aligned, cmap="gray")
        axes[3].set_title(f"principal-axis-aligned crop\nA_aligned={a_aligned:.3f}")
        axes[3].axis("off")

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}_asymmetry_diagnostic.png", dpi=130)
        plt.close(fig)

    print(f"\nSaved diagnostic visualizations for: {highlight_stems}")


if __name__ == "__main__":
    main()
