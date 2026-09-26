"""Visual review of the FROZEN V4 locked-test predictions.

READ-ONLY / interpretation only. Does not retrain, retune, change the 0.25
threshold, change features, or modify V4/V3/V2 in any way. Reuses the
existing frozen model (pipeline_v4/frozen_model.pkl) and the existing
locked-test manifest exactly as already evaluated in
Evaluation_V4/freeze_and_locked_test.py — no new fitting happens here.
"""

import csv
import random
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
TEST_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
OUT_DIR = ROOT / "Evaluation_V4" / "VisualReview"
MANIFEST_OUT = OUT_DIR / "visual_review_manifest.csv"

SEED = 20260918
MAX_PER_CATEGORY = 20

sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))
from metrics_lib import v2_predict, infer_critical_override  # noqa: E402
sys.path.insert(0, str(ROOT))
from pipeline_v3.decision import v3_predict_from_row  # noqa: E402
from pipeline_v4.decision_model import v4_predict_from_row, load_frozen_pipeline  # noqa: E402


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def image_id_from_name(name):
    return Path(name).stem


def build_evaluable_rows():
    with open(COHORT_RESULTS, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    with open(TEST_MANIFEST, newline="", encoding="utf-8") as f:
        manifest_rows = list(csv.DictReader(f))
    manifest_by_id = {r["image_id"]: r for r in manifest_rows}
    test_ids = set(manifest_by_id.keys())

    by_image = defaultdict(list)
    for r in results:
        image_id = image_id_from_name(r["image_name"])
        if image_id in test_ids:
            by_image[image_id].append(r)

    pipeline = load_frozen_pipeline()
    missing_masks = []
    rows_out = []
    for image_id, rows in by_image.items():
        if rows[0]["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
            continue
        r = rows[0]
        mask_path = ROOT / r["mask_path"]
        if not mask_path.exists():
            missing_masks.append(image_id)
            continue

        override = infer_critical_override(r)
        v2_pred = v2_predict(r)
        v3_pred = v3_predict_from_row(r, override)
        v4_elevated, v4_score = v4_predict_from_row(
            {"A_value": to_float(r["A_value"]), "B_circularity": to_float(r["B_circularity"]),
             "C_value": to_float(r["C_value"]), "D_px": to_float(r["D_px"]),
             "confidence": to_float(r["confidence"]), "lesion_fraction": to_float(r["lesion_fraction"])},
            pipeline,
        )
        v4_pred = 1 if v4_elevated else 0
        gt = int(r["ground_truth_binary"])

        if gt == 1 and v4_pred == 1:
            v4_class = "TP"
        elif gt == 0 and v4_pred == 0:
            v4_class = "TN"
        elif gt == 0 and v4_pred == 1:
            v4_class = "FP"
        else:
            v4_class = "FN"

        rows_out.append({
            "image_id": image_id,
            "ground_truth": "Melanoma" if gt == 1 else "Benign",
            "ground_truth_binary": gt,
            "v2_prediction": "ELEVATED" if v2_pred == 1 else "LOWER",
            "v3_prediction": "ELEVATED" if v3_pred == 1 else "LOWER",
            "v4_prediction": "ELEVATED" if v4_pred == 1 else "LOWER",
            "v3_pred_binary": v3_pred, "v4_pred_binary": v4_pred,
            "v4_decision_score": round(v4_score, 4),
            "classification": v4_class,
            "A": r["A_value"], "B": r["B_circularity"], "C": r["C_value"], "D_px": r["D_px"],
            "mask_area_fraction": r["lesion_fraction"], "yolo_confidence": r["confidence"],
            "quality_flags": r["quality_flags"],
            "original_image_path": manifest_by_id[image_id]["image_path"],
            "mask_path": str(mask_path),
        })

    return rows_out, missing_masks


def draw_panel(row, save_path):
    image = cv2.imread(row["original_image_path"], cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = (mask_raw > 127).astype(np.uint8) * 255

    overlay = image_rgb.copy()
    fill = overlay.copy()
    fill[mask > 0] = (255, 60, 60)
    blended = cv2.addWeighted(overlay, 0.82, fill, 0.18, 0)  # light fill so boundary is easy to judge
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    thickness = max(2, image.shape[1] // 300)
    cv2.drawContours(blended, contours, -1, (255, 230, 0), thickness)

    fig = plt.figure(figsize=(16, 5.5), facecolor="#fcfcfb")
    for i, (img, title) in enumerate([(image_rgb, "Original image"), (mask, "Binary raster mask"),
                                       (blended, "Mask overlay (light fill + boundary)")], start=1):
        ax = fig.add_subplot(1, 4, i)
        ax.imshow(img, cmap="gray" if img is mask else None)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axis("off")

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.axis("off")
    lines = [
        f"image_id: {row['image_id']}",
        f"ground truth: {row['ground_truth']}",
        "",
        f"V2 result: {row['v2_prediction']}",
        f"V3 result: {row['v3_prediction']}",
        f"V4 result: {row['v4_prediction']}  (score={row['v4_decision_score']})",
        f"V4 classification: {row['classification']}",
        "",
        f"A: {row['A']}",
        f"B: {row['B']}",
        f"C: {row['C']}",
        f"D_px: {row['D_px']}",
        f"mask area fraction: {row['mask_area_fraction']}",
        f"YOLO confidence: {row['yolo_confidence']}",
        "",
        f"quality_flags: {row['quality_flags']}",
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), transform=ax4.transAxes, va="top", ha="left",
             fontsize=10.5, fontfamily="monospace")

    fig.suptitle(f"{row['image_id']} — {row['ground_truth']} — V4:{row['classification']}",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=125, bbox_inches="tight")
    plt.close(fig)


def main():
    all_rows, missing_masks = build_evaluable_rows()
    print(f"Evaluable locked-test rows with usable masks: {len(all_rows)}")
    if missing_masks:
        print(f"Missing masks (excluded from review): {len(missing_masks)} -> {missing_masks}")

    by_id = {r["image_id"]: r for r in all_rows}

    pools = {
        "True_Positive": [r for r in all_rows if r["classification"] == "TP"],
        "True_Negative": [r for r in all_rows if r["classification"] == "TN"],
        "False_Positive": [r for r in all_rows if r["classification"] == "FP"],
        "False_Negative": [r for r in all_rows if r["classification"] == "FN"],
        "V3_to_V4_Changes/FN_rescued": [
            r for r in all_rows if r["ground_truth_binary"] == 1 and r["v3_pred_binary"] == 0 and r["v4_pred_binary"] == 1
        ],
        "V3_to_V4_Changes/FP_corrected": [
            r for r in all_rows if r["ground_truth_binary"] == 0 and r["v3_pred_binary"] == 1 and r["v4_pred_binary"] == 0
        ],
    }

    for name, pool in pools.items():
        print(f"Pool {name}: {len(pool)} available")

    rng = random.Random(SEED)
    manifest_rows = []
    fieldnames = ["category", "image_id", "ground_truth", "v2_prediction", "v3_prediction",
                  "v4_prediction", "classification", "A", "B", "C", "D_px",
                  "mask_area_fraction", "yolo_confidence", "original_image_path", "mask_path", "panel_path"]

    for category, pool in pools.items():
        n_select = min(MAX_PER_CATEGORY, len(pool))
        selected = rng.sample(pool, n_select)
        print(f"Selected {n_select}/{min(MAX_PER_CATEGORY, len(pool))} for {category}")
        category_dir = OUT_DIR / category
        category_dir.mkdir(parents=True, exist_ok=True)
        for r in selected:
            panel_path = category_dir / f"{r['image_id']}.png"
            draw_panel(r, panel_path)
            manifest_rows.append({
                "category": category, "image_id": r["image_id"], "ground_truth": r["ground_truth"],
                "v2_prediction": r["v2_prediction"], "v3_prediction": r["v3_prediction"],
                "v4_prediction": r["v4_prediction"], "classification": r["classification"],
                "A": r["A"], "B": r["B"], "C": r["C"], "D_px": r["D_px"],
                "mask_area_fraction": r["mask_area_fraction"], "yolo_confidence": r["yolo_confidence"],
                "original_image_path": r["original_image_path"], "mask_path": r["mask_path"],
                "panel_path": str(panel_path.relative_to(ROOT)),
            })

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader(); w.writerows(manifest_rows)

    print(f"\nWrote {len(manifest_rows)} panels + manifest to {OUT_DIR}")
    return manifest_rows, missing_masks, pools


if __name__ == "__main__":
    main()
