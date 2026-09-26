"""Diagnostic-only investigation of the 5 images flagged for segmentation
quality problems by run_revised_abcd_pipeline.py (3 fragmented, 2 oversized).

Read-only: does not modify Code/, raster_masks/, YoloMaskABCDTest/, or
RevisedABCDTest/. Pulls the already-computed A/B numbers straight from
RevisedABCDTest/baseline_vs_revised.csv rather than recomputing the
pipeline, and only adds connected-component geometry on top.
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
REVISED_CSV = ROOT / "RevisedABCDTest" / "baseline_vs_revised.csv"
OUT_DIR = Path(__file__).resolve().parent / "segmentation_flags"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLAGGED_STEMS = [
    "ISIC_0000006", "ISIC_0000069", "ISIC_0000072",  # fragmented
    "ISIC_0000075", "ISIC_0000076",                  # oversized
]


def find_image_path(stem):
    for folder in ("Benign", "Malignant"):
        p = IMAGES_ROOT / folder / f"{stem}.jpg"
        if p.exists():
            return p
    return None


def component_report(mask):
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    total_area = int(np.sum(mask > 0))
    components = []
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        components.append({
            "id": i, "area_px": area,
            "pct_of_lesion": round(100 * area / total_area, 1) if total_area else 0,
            "bbox_w": w, "bbox_h": h,
            "centroid": (round(float(centroids[i][0]), 1), round(float(centroids[i][1]), 1)),
        })
    components.sort(key=lambda c: -c["area_px"])
    return components, labels


def load_revised_row(stem):
    if not REVISED_CSV.exists():
        return None
    with open(REVISED_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if Path(row["image_name"]).stem == stem:
                return row
    return None


def main():
    summary_lines = []
    for stem in FLAGGED_STEMS:
        image_path = find_image_path(stem)
        mask_path = MASK_DIR / f"{stem}_mask.png"
        if image_path is None or not mask_path.exists():
            print(f"SKIP {stem}: missing image or mask")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = (mask_raw > 127).astype(np.uint8) * 255

        components, labels = component_report(mask)
        lesion_fraction = float(np.mean(mask > 0))
        frame_area = h * w

        revised_row = load_revised_row(stem)

        # --- overlay: color each component differently ---
        overlay = image_rgb.copy()
        rng = np.random.default_rng(42)
        colors = (rng.integers(60, 255, size=(len(components) + 1, 3)))
        for idx, comp in enumerate(components):
            color = colors[idx]
            region = (labels == comp["id"])
            overlay[region] = (overlay[region] * 0.4 + np.array(color) * 0.6).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), max(2, w // 300))

        fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), facecolor="#fcfcfb")
        axes[0].imshow(image_rgb)
        axes[0].set_title(f"{stem}\noriginal ({w}x{h})")
        axes[0].axis("off")

        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title(f"YOLO raster mask\narea={lesion_fraction:.1%} of frame, "
                          f"{len(components)} component(s)")
        axes[1].axis("off")

        axes[2].imshow(overlay)
        axes[2].set_title("overlay: each component a different color\nyellow = outer contour(s)")
        axes[2].axis("off")

        fig.tight_layout()
        fig.savefig(OUT_DIR / f"{stem}_segmentation_flag.png", dpi=130)
        plt.close(fig)

        comp_str = "; ".join(
            f"#{c['id']}: {c['area_px']}px ({c['pct_of_lesion']}% of mask, bbox {c['bbox_w']}x{c['bbox_h']})"
            for c in components
        )
        line = (
            f"\n=== {stem} ===\n"
            f"  image size: {w}x{h} ({frame_area} px total)\n"
            f"  mask area: {int(np.sum(mask > 0))} px = {lesion_fraction:.1%} of frame\n"
            f"  connected components: {len(components)}\n"
            f"    {comp_str}\n"
        )
        if revised_row:
            line += (
                f"  A_original={revised_row['A_original']} -> A_revised(aligned)={revised_row['A_revised']}\n"
                f"  B_circularity={revised_row['B_circularity']} B_experimental={revised_row['B_experimental']} "
                f"(n_defects={revised_row['B_experimental_n_defects']})\n"
                f"  segmentation_quality_flags: {revised_row['segmentation_quality_flags']}\n"
            )
        summary_lines.append(line)
        print(line)

    report_path = OUT_DIR / "segmentation_flags_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(summary_lines)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
