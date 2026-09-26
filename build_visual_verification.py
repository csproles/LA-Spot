"""Visual verification study for the completed Evaluation_3500 run.

READ-ONLY: does not modify or rerun YOLO, ABCD, masks, thresholds, or the
3500-image experiment. Uses only already-saved data: original ISIC images
(from the path recorded in the manifest), already-saved raster instance
masks, Evaluation_3500/full_run/results.csv, the manifest, and the ISIC
source metadata.csv (for diagnosis_1..5 / melanocytic, which are not all
carried in results.csv).

Selection is a fixed-seed random sample per group (no cherry-picking), from:
  - FN with large_area_fraction flag        (~15)
  - FN without large_area_fraction flag      (~10)
  - TP with large_area_fraction flag         (~10)
  - TP without large_area_fraction flag      (~10)
where TP/FN is restricted to malignant, SINGLE_LESION_EVALUABLE rows, exactly
matching the definitions already verified in the earlier error analysis.
"""

import csv
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
MANIFEST_CSV = ROOT / "Evaluation_3500" / "evaluation_manifest_3500.csv"
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")

OUT_DIR = ROOT / "Evaluation_3500" / "VisualVerification"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST_OUT = OUT_DIR / "visual_verification_manifest.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PANELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260918  # same seed used for the original 3500-image sampling, reused for full reproducibility

SELECTION_PLAN = [
    ("FN_large_area", "FN", True, 15),
    ("FN_no_large_area", "FN", False, 10),
    ("TP_large_area", "TP", True, 10),
    ("TP_no_large_area", "TP", False, 10),
]


def load_results():
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_manifest_paths():
    with open(MANIFEST_CSV, newline="", encoding="utf-8") as f:
        return {r["image_id"]: r["image_path"] for r in csv.DictReader(f)}


def load_source_metadata():
    with open(SOURCE_METADATA, newline="", encoding="utf-8") as f:
        return {r["isic_id"]: r for r in csv.DictReader(f)}


def classify(row):
    if row["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
        return None
    if row["ground_truth"] != "malignant":
        return None
    gt = row["ground_truth_binary"]
    pred = row["provisional_prediction"]
    if gt == "1" and pred == "positive":
        return "TP"
    if gt == "1" and pred == "negative":
        return "FN"
    return None


def image_id_from_name(image_name):
    return Path(image_name).stem


def draw_panel(image_id, row, image_path, mask_path, meta_row, group_label, save_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"could not read original image: {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        raise FileNotFoundError(f"could not read mask: {mask_path}")
    mask = (mask_raw > 127).astype(np.uint8) * 255

    # overlay: translucent fill + contour, matching the style used in the
    # earlier segmentation-flags investigation (read-only visualization only)
    overlay = image_rgb.copy()
    fill = overlay.copy()
    fill[mask > 0] = (255, 80, 80)
    blended = cv2.addWeighted(overlay, 0.72, fill, 0.28, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, image.shape[1] // 300)
    cv2.drawContours(blended, contours, -1, (255, 230, 0), thickness)

    fig = plt.figure(figsize=(20, 5.5), facecolor="#fcfcfb")
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(image_rgb)
    ax1.set_title("Original ISIC image", fontsize=11, fontweight="bold")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(mask, cmap="gray")
    ax2.set_title("Binary YOLO raster mask", fontsize=11, fontweight="bold")
    ax2.axis("off")

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(blended)
    ax3.set_title("Original + mask boundary/overlay", fontsize=11, fontweight="bold")
    ax3.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis("off")

    def yn(v):
        return "concern" if v == "True" else "no concern" if v == "False" else "N/A"

    lines = [
        f"ISIC ID: {image_id}",
        f"group: {group_label}",
        "",
        f"diagnosis_1: {meta_row.get('diagnosis_1','')}",
        f"diagnosis_2: {meta_row.get('diagnosis_2','')}",
        f"diagnosis_3: {meta_row.get('diagnosis_3','')}",
        f"melanocytic: {meta_row.get('melanocytic','')}",
        "",
        f"TP/FN status: {row['_tpfn']}",
        f"risk_level: {row['risk_level']}",
        f"provisional_prediction: {row['provisional_prediction']}",
        "",
        f"A raw: {row['A_value']}   A concern: {yn(row['A_concern'])}",
        f"B raw: {row['B_circularity']}   B concern: {yn(row['B_circularity_concern'])}",
        f"C raw: {row['C_value']}   C concern: {yn(row['C_concern'])}",
        "",
        f"YOLO confidence: {row['confidence']}",
        f"mask area fraction: {row['lesion_fraction']}",
        f"D_px: {row['D_px']}   (D_mm intentionally unused)",
        "",
        f"quality_flags: {row['quality_flags']}",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), transform=ax4.transAxes, va="top", ha="left",
              fontsize=10.5, fontfamily="monospace")

    fig.suptitle(f"{image_id}  —  {group_label}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    results = load_results()
    manifest_paths = load_manifest_paths()
    source_meta = load_source_metadata()

    groups = {"TP_large": [], "TP_no_large": [], "FN_large": [], "FN_no_large": []}
    for r in results:
        tpfn = classify(r)
        if tpfn is None:
            continue
        has_large = "large_area_fraction" in (r.get("quality_flags") or "")
        key = f"{tpfn}_{'large' if has_large else 'no_large'}"
        r["_tpfn"] = tpfn
        groups[key].append(r)

    print("Available pool sizes:")
    for k, v in groups.items():
        print(f"  {k}: {len(v)}")

    rng = random.Random(SEED)
    manifest_rows = []
    manifest_fields = [
        "selection_group", "image_id", "tpfn_status", "risk_level", "provisional_prediction",
        "diagnosis_1", "diagnosis_2", "diagnosis_3", "melanocytic",
        "A_value", "A_concern", "B_circularity", "B_circularity_concern",
        "C_value", "C_concern", "confidence", "lesion_fraction", "D_px",
        "quality_flags", "image_path", "mask_path", "panel_path",
    ]

    pool_key_map = {
        "FN_large_area": "FN_large", "FN_no_large_area": "FN_no_large",
        "TP_large_area": "TP_large", "TP_no_large_area": "TP_no_large",
    }

    for group_label, tpfn, has_large, n in SELECTION_PLAN:
        pool = groups[pool_key_map[group_label]]
        n_select = min(n, len(pool))
        selected = rng.sample(pool, n_select)
        print(f"Selected {len(selected)}/{n} requested for {group_label} (pool had {len(pool)})")

        for r in selected:
            image_id = image_id_from_name(r["image_name"])
            image_path = manifest_paths.get(image_id)
            mask_path = ROOT / r["mask_path"]
            meta_row = source_meta.get(image_id, {})

            if not image_path or not Path(image_path).exists():
                print(f"  [WARN] missing original image for {image_id}, skipping")
                continue
            if not mask_path.exists():
                print(f"  [WARN] missing mask for {image_id}, skipping")
                continue

            panel_path = PANELS_DIR / f"{group_label}_{image_id}.png"
            draw_panel(image_id, r, Path(image_path), mask_path, meta_row, group_label, panel_path)

            manifest_rows.append({
                "selection_group": group_label,
                "image_id": image_id,
                "tpfn_status": r["_tpfn"],
                "risk_level": r["risk_level"],
                "provisional_prediction": r["provisional_prediction"],
                "diagnosis_1": meta_row.get("diagnosis_1", ""),
                "diagnosis_2": meta_row.get("diagnosis_2", ""),
                "diagnosis_3": meta_row.get("diagnosis_3", ""),
                "melanocytic": meta_row.get("melanocytic", ""),
                "A_value": r["A_value"], "A_concern": r["A_concern"],
                "B_circularity": r["B_circularity"], "B_circularity_concern": r["B_circularity_concern"],
                "C_value": r["C_value"], "C_concern": r["C_concern"],
                "confidence": r["confidence"], "lesion_fraction": r["lesion_fraction"], "D_px": r["D_px"],
                "quality_flags": r["quality_flags"],
                "image_path": str(image_path), "mask_path": str(mask_path),
                "panel_path": str(panel_path.relative_to(ROOT)),
            })

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} panels and manifest to {OUT_DIR}")
    print(f"Seed used: {SEED}")


if __name__ == "__main__":
    main()
