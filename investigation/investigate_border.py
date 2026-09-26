"""Diagnostic-only investigation of the B (border) criterion.

Does NOT modify Code/MelanomaDeterminingStuff/border.py. Reproduces its
circularity formula exactly (verified against the master CSV) and adds
established alternative shape descriptors (solidity/convexity, a smoothing-
sensitivity probe) purely for comparison.
"""

import csv
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MASK_DIR = ROOT / "raster_masks"
IMAGES_ROOT = ROOT / "Images"
OUT_DIR = Path(__file__).resolve().parent / "border"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_manifest():
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"image_path": p, "ground_truth": label}
    return manifest


# ---- exact reproduction of Code/MelanomaDeterminingStuff/border.py -------
def current_score_border(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
        border_irreg = 1 - circularity
    else:
        contour = None
        border_irreg = 0.0
        perimeter = area = 0
    return border_irreg, contour, perimeter, area


def solidity_irregularity(contour):
    if contour is None:
        return None
    area = cv2.contourArea(contour)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else None
    return (1 - solidity) if solidity is not None else None


def smoothed_perimeter_ratio(mask, contour, blur_kernel=15):
    """How much of the raw perimeter is high-frequency 'texture' vs the
    macroscopic outline: smooth the mask heavily, re-measure perimeter, and
    compare. A ratio near 1.0 means the mask was already smooth (little for
    circularity to detect); a ratio >>1 means substantial fine boundary
    detail exists that circularity is (or isn't) picking up."""
    if contour is None:
        return None
    raw_perimeter = cv2.arcLength(contour, True)
    blurred = cv2.GaussianBlur(mask, (blur_kernel, blur_kernel), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(smoothed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours2:
        return None
    smooth_contour = max(contours2, key=cv2.contourArea)
    smooth_perimeter = cv2.arcLength(smooth_contour, True)
    return raw_perimeter / (smooth_perimeter + 1e-6)


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

        border_irreg, contour, perimeter, area = current_score_border(mask)
        convexity_irreg = solidity_irregularity(contour)
        smooth_ratio = smoothed_perimeter_ratio(mask, contour)
        n_contour_pts = len(contour) if contour is not None else 0

        rows.append({
            "stem": stem, "ground_truth": info["ground_truth"],
            "B_current_circularity_irreg": round(border_irreg, 4),
            "convexity_irregularity": round(convexity_irreg, 4) if convexity_irreg is not None else "",
            "smoothed_perimeter_ratio": round(smooth_ratio, 4) if smooth_ratio is not None else "",
            "contour_points": n_contour_pts,
            "perimeter_px": round(perimeter, 1),
            "area_px": round(area, 1),
        })

    csv_path = OUT_DIR / "border_diagnostics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")

    for gt in ("benign", "malignant"):
        sub = [r for r in rows if r["ground_truth"] == gt]
        print(f"\n{gt} (n={len(sub)}):")
        print(f"  mean B_current (circularity irreg) = {np.mean([r['B_current_circularity_irreg'] for r in sub]):.4f}  "
              f"std={np.std([r['B_current_circularity_irreg'] for r in sub]):.4f}  "
              f"range=[{min(r['B_current_circularity_irreg'] for r in sub):.3f}, {max(r['B_current_circularity_irreg'] for r in sub):.3f}]")
        print(f"  mean convexity_irregularity        = {np.mean([r['convexity_irregularity'] for r in sub]):.4f}  "
              f"std={np.std([r['convexity_irregularity'] for r in sub]):.4f}")
        print(f"  mean smoothed_perimeter_ratio       = {np.mean([r['smoothed_perimeter_ratio'] for r in sub]):.4f}  "
              f"(1.0 = already smooth; higher = more fine detail present)")
        print(f"  mean contour_points                 = {np.mean([r['contour_points'] for r in sub]):.0f}")

    # correlation check: does circularity even track convexity/smoothness on this data?
    b_vals = np.array([r["B_current_circularity_irreg"] for r in rows])
    conv_vals = np.array([r["convexity_irregularity"] for r in rows])
    smooth_vals = np.array([r["smoothed_perimeter_ratio"] for r in rows])
    from scipy.stats import pearsonr
    print(f"\ncorr(circularity_irreg, convexity_irreg) = {pearsonr(b_vals, conv_vals)[0]:.3f}")
    print(f"corr(circularity_irreg, smoothed_perimeter_ratio) = {pearsonr(b_vals, smooth_vals)[0]:.3f}")

    # --- visualize: most "regular" vs most "irregular" by current metric, both classes
    rows_sorted = sorted(rows, key=lambda r: r["B_current_circularity_irreg"])
    lowest = rows_sorted[:2]   # "most regular" by circularity
    highest = rows_sorted[-2:]  # "most irregular" by circularity
    highlight = lowest + highest

    for r in highlight:
        stem = r["stem"]
        info = manifest[stem]
        mask = cv2.imread(str(MASK_DIR / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.uint8) * 255
        image = cv2.cvtColor(cv2.imread(str(info["image_path"])), cv2.COLOR_BGR2RGB)
        _, contour, _, _ = current_score_border(mask)
        hull = cv2.convexHull(contour) if contour is not None else None

        vis = image.copy()
        if contour is not None:
            cv2.drawContours(vis, [contour], -1, (255, 80, 80), 4)
        if hull is not None:
            cv2.drawContours(vis, [hull], -1, (80, 200, 255), 2)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5), facecolor="#fcfcfb")
        axes[0].imshow(image)
        axes[0].set_title(f"{stem} ({r['ground_truth']})")
        axes[0].axis("off")
        axes[1].imshow(vis)
        axes[1].set_title(f"red=contour blue=convex hull\nB={r['B_current_circularity_irreg']:.3f} "
                          f"convexity_irreg={r['convexity_irregularity']}")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}_border_diagnostic.png", dpi=130)
        plt.close(fig)

    print(f"\nSaved diagnostic visualizations for: {[r['stem'] for r in highlight]}")


if __name__ == "__main__":
    main()
