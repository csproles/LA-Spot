"""Reproducible visual review panels for melanoma TP/FN cases (Evaluation_3500).

READ-ONLY: uses only already-saved original images, raster masks,
results.csv, manifest, and ISIC source metadata. Does not create a new
segmentation or modify any existing mask.
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

ROOT = Path(__file__).resolve().parent
RESULTS_CSV = ROOT / "Evaluation_3500" / "full_run" / "results.csv"
MANIFEST_CSV = ROOT / "Evaluation_3500" / "evaluation_manifest_3500.csv"
SOURCE_METADATA = Path(r"C:\Users\sirjanaa\Downloads\ISIC-images\metadata.csv")
OUT_DIR = ROOT / "Evaluation_3500" / "MelanomaFailureAnalysis"
PANELS_DIR = OUT_DIR / "panels"
MANIFEST_OUT = OUT_DIR / "melanoma_visual_review_manifest.csv"

SEED = 20260918
MELANOMA_DIAG3 = {"Melanoma Invasive", "Melanoma in situ", "Melanoma, NOS", "Melanoma metastasis"}

SELECTION_PLAN = [
    ("FN_large_area", "FN", True, 15),
    ("FN_no_large_area", "FN", False, 15),
    ("TP_large_area", "TP", True, 10),
    ("TP_no_large_area", "TP", False, 10),
]


def image_id_from_name(name):
    return Path(name).stem


def draw_panel(image_id, row, image_path, mask_path, meta_row, group_label, save_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    mask = (mask_raw > 127).astype(np.uint8) * 255

    overlay = image_rgb.copy()
    fill = overlay.copy()
    fill[mask > 0] = (255, 80, 80)
    blended = cv2.addWeighted(overlay, 0.72, fill, 0.28, 0)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, image.shape[1] // 300)
    cv2.drawContours(blended, contours, -1, (255, 230, 0), thickness)

    fig = plt.figure(figsize=(20, 5.5), facecolor="#fcfcfb")
    for i, (img, title) in enumerate([(image_rgb, "Original ISIC image"),
                                       (mask, "Binary YOLO raster mask"),
                                       (blended, "Original + mask boundary/overlay")], start=1):
        ax = fig.add_subplot(1, 4, i)
        ax.imshow(img, cmap="gray" if img is mask else None)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis("off")

    def yn(v):
        return "concern" if v == "True" else "no concern" if v == "False" else "N/A"

    lines = [
        f"ISIC ID: {image_id}",
        f"group: {group_label}",
        "",
        f"diagnosis_3 (subtype): {meta_row.get('diagnosis_3','')}",
        f"source/institution: {meta_row.get('attribution','')}",
        "",
        f"TP/FN status: {row['_tpfn']}",
        f"risk_level: {row['risk_level']}",
        "",
        f"A raw: {row['A_value']}   A concern: {yn(row['A_concern'])}",
        f"B raw: {row['B_circularity']}   B concern: {yn(row['B_circularity_concern'])}",
        f"C raw: {row['C_value']}   C concern: {yn(row['C_concern'])}",
        "",
        f"YOLO confidence: {row['confidence']}",
        f"mask area fraction: {row['lesion_fraction']}",
        f"D_px: {row['D_px']}",
        "",
        f"quality_flags: {row['quality_flags']}",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), transform=ax4.transAxes, va="top", ha="left",
              fontsize=10.5, fontfamily="monospace")

    fig.suptitle(f"{image_id}  —  melanoma  —  {group_label}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
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

    groups = {"TP_large": [], "TP_no_large": [], "FN_large": [], "FN_no_large": []}
    for image_id in melanoma_ids:
        image_rows = rows_by_image.get(image_id, [])
        if not image_rows or image_rows[0]["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
            continue
        row = image_rows[0]
        pred = row["provisional_prediction"]
        if pred == "positive":
            tpfn = "TP"
        elif pred == "negative":
            tpfn = "FN"
        else:
            continue
        row["_tpfn"] = tpfn
        has_large = "large_area_fraction" in (row.get("quality_flags") or "")
        groups[f"{tpfn}_{'large' if has_large else 'no_large'}"].append((image_id, row))

    print("Pool sizes:", {k: len(v) for k, v in groups.items()})

    rng = random.Random(SEED)
    manifest_rows = []
    fieldnames = ["selection_group", "image_id", "tpfn_status", "risk_level", "diagnosis_3", "attribution",
                  "A_value", "A_concern", "B_circularity", "B_circularity_concern", "C_value", "C_concern",
                  "confidence", "lesion_fraction", "D_px", "quality_flags", "image_path", "mask_path", "panel_path"]
    pool_key_map = {"FN_large_area": "FN_large", "FN_no_large_area": "FN_no_large",
                    "TP_large_area": "TP_large", "TP_no_large_area": "TP_no_large"}

    for group_label, tpfn, has_large, n in SELECTION_PLAN:
        pool = groups[pool_key_map[group_label]]
        n_select = min(n, len(pool))
        selected = rng.sample(pool, n_select)
        print(f"Selected {len(selected)}/{n} for {group_label} (pool={len(pool)})")

        for image_id, row in selected:
            image_path = manifest_paths.get(image_id)
            mask_path = ROOT / row["mask_path"]
            meta_row = meta.get(image_id, {})
            if not image_path or not Path(image_path).exists() or not mask_path.exists():
                print(f"  [WARN] missing file for {image_id}, skipping")
                continue

            panel_path = PANELS_DIR / f"{group_label}_{image_id}.png"
            draw_panel(image_id, row, Path(image_path), mask_path, meta_row, group_label, panel_path)

            manifest_rows.append({
                "selection_group": group_label, "image_id": image_id, "tpfn_status": row["_tpfn"],
                "risk_level": row["risk_level"], "diagnosis_3": meta_row.get("diagnosis_3", ""),
                "attribution": meta_row.get("attribution", ""),
                "A_value": row["A_value"], "A_concern": row["A_concern"],
                "B_circularity": row["B_circularity"], "B_circularity_concern": row["B_circularity_concern"],
                "C_value": row["C_value"], "C_concern": row["C_concern"],
                "confidence": row["confidence"], "lesion_fraction": row["lesion_fraction"], "D_px": row["D_px"],
                "quality_flags": row["quality_flags"], "image_path": str(image_path), "mask_path": str(mask_path),
                "panel_path": str(panel_path.relative_to(ROOT)),
            })

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} panels. Seed used: {SEED}")


if __name__ == "__main__":
    main()
