"""
Independent segmentation benchmark: original classical LAB/Otsu segmentation
vs. the frozen YOLO segmentation model, evaluated against OFFICIAL ISIC 2018
Challenge Task 1 test-set ground-truth lesion-boundary masks.

THIS IS A SEPARATE EXPERIMENT. The 100 images here are drawn from the public
ISIC Archive's "Challenge 2018: Task 1-2: Test" collection (collection id 64)
-- they are NOT part of, and share zero image IDs with, the 3,482-image
melanoma-vs-benign classification cohort used elsewhere in this repo
(Evaluation_FinalTargeted/, Evaluation_DevComparison/). No result here feeds
back into, or is merged with, that cohort.

Neither segmentation method is tuned, retrained, or modified. The classical
segmentation is the exact, unmodified Code/ pipeline chain used by
Code/main.py. YOLO inference uses the exact, unmodified
revised_abcd/yolo_single_image.py function (vendored as a read-only copy in
this folder, extracted from git history without touching user-shree) and the
same frozen checkpoint already used for V2/V3/V4.

No threshold is adjusted after seeing ground truth. No YOLO retraining
occurs.
"""

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
BENCH_DIR = Path(__file__).resolve().parent
if str(BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(BENCH_DIR))

# --- existing, UNMODIFIED classical pipeline code, same chain as Code/main.py ---
from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import (  # noqa: E402
    remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair,
    segment_lesion,
)

# --- vendored, unmodified, frozen YOLO inference (see file header) ---
from _vendored_yolo_single_image import run_single_image_inference  # noqa: E402

from ultralytics import YOLO  # noqa: E402

YOLO_WEIGHTS = Path(r"C:\Users\sirjanaa\Downloads\runs\runs\segment\melanoma_yolo26n_seg\weights\best.pt")
YOLO_CONF = 0.25  # unchanged, matches revised_abcd/pipeline_v2.py's default

MANIFEST = BENCH_DIR / "segmentation_benchmark_manifest.csv"
IMAGES_DIR = BENCH_DIR / "images"
GT_DIR = BENCH_DIR / "ground_truth"
OTSU_MASK_DIR = BENCH_DIR / "otsu_masks"
YOLO_MASK_DIR = BENCH_DIR / "yolo_masks"


def run_classical_otsu(image_path):
    """Exact, unmodified Code/ chain up through segment_lesion(), as used by
    Code/main.py. Returns a 0/255 uint8 mask, or None on failure."""
    original = load_image(str(image_path))
    no_vignette, circle_info = remove_vignette(original)
    denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)
    mask, _masked = segment_lesion(no_hair)
    return mask


def iou_dice(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    pred_area = pred_bin.sum()
    gt_area = gt_bin.sum()
    if union == 0:
        # only possible if both GT and prediction are empty; GT is never
        # empty for an official annotation, so this branch is defensive only
        return 1.0, 1.0
    iou = inter / union
    dice = (2 * inter) / (pred_area + gt_area) if (pred_area + gt_area) > 0 else 0.0
    return float(iou), float(dice)


def main():
    rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
    print(f"Running segmentation benchmark on {len(rows)} images (independent ISIC-2018 "
          f"Task-1 test-set sample, seed={rows[0]['sample_seed']}).")

    model = YOLO(str(YOLO_WEIGHTS))
    print(f"Loaded frozen YOLO checkpoint: {YOLO_WEIGHTS}")

    results = []
    for i, row in enumerate(rows):
        iid = row["image_id"]
        img_path = IMAGES_DIR / f"{iid}.jpg"
        gt_path = GT_DIR / f"{iid}_gt.png"

        gt_raw = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        gt_bin = (gt_raw > 127)
        gt_area = int(gt_bin.sum())
        h, w = gt_bin.shape

        rec = dict(image_id=iid, gt_area_px=gt_area, image_h=h, image_w=w)

        # --- classical LAB/Otsu ---
        try:
            otsu_mask = run_classical_otsu(img_path)
            if otsu_mask.shape[:2] != (h, w):
                otsu_mask = cv2.resize(otsu_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            otsu_bin = otsu_mask > 0
            otsu_area = int(otsu_bin.sum())
            otsu_empty = otsu_area == 0
            otsu_iou, otsu_dice = iou_dice(otsu_bin, gt_bin)
            otsu_error = ""
        except Exception as e:
            otsu_bin = np.zeros((h, w), dtype=bool)
            otsu_area = 0
            otsu_empty = True
            otsu_iou, otsu_dice = 0.0, 0.0
            otsu_error = repr(e)
        cv2.imwrite(str(OTSU_MASK_DIR / f"{iid}_otsu.png"), (otsu_bin.astype(np.uint8) * 255))
        rec.update(otsu_area_px=otsu_area, otsu_iou=round(otsu_iou, 4), otsu_dice=round(otsu_dice, 4),
                   otsu_empty=otsu_empty, otsu_error=otsu_error)

        # --- frozen YOLO (single-image, unbatched, retina_masks) ---
        try:
            instances, orig_shape = run_single_image_inference(model, img_path, conf=YOLO_CONF)
            num_instances = len(instances)
            if num_instances == 0:
                yolo_bin = np.zeros((h, w), dtype=bool)
                yolo_area = 0
                yolo_empty = True
                yolo_error = ""
            else:
                primary = instances[0]  # existing primary-instance policy: highest confidence
                yolo_mask = primary["mask"]
                if yolo_mask.shape[:2] != (h, w):
                    yolo_mask = cv2.resize(yolo_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                yolo_bin = yolo_mask > 0
                yolo_area = int(yolo_bin.sum())
                yolo_empty = yolo_area == 0
                yolo_error = ""
            yolo_iou, yolo_dice = iou_dice(yolo_bin, gt_bin)
        except Exception as e:
            yolo_bin = np.zeros((h, w), dtype=bool)
            yolo_area = 0
            yolo_empty = True
            num_instances = 0
            yolo_iou, yolo_dice = 0.0, 0.0
            yolo_error = repr(e)
        cv2.imwrite(str(YOLO_MASK_DIR / f"{iid}_yolo.png"), (yolo_bin.astype(np.uint8) * 255))
        rec.update(yolo_area_px=yolo_area, yolo_iou=round(yolo_iou, 4), yolo_dice=round(yolo_dice, 4),
                   yolo_empty=yolo_empty, yolo_num_instances=num_instances,
                   yolo_multi_instance=num_instances > 1, yolo_error=yolo_error)

        results.append(rec)
        if i % 10 == 0:
            print(f"  [{i+1}/{len(rows)}] {iid}  otsu_iou={rec['otsu_iou']}  yolo_iou={rec['yolo_iou']}"
                  f"  yolo_instances={num_instances}")

    fieldnames = ["image_id", "gt_area_px", "image_h", "image_w",
                  "otsu_area_px", "otsu_iou", "otsu_dice", "otsu_empty", "otsu_error",
                  "yolo_area_px", "yolo_iou", "yolo_dice", "yolo_empty", "yolo_num_instances",
                  "yolo_multi_instance", "yolo_error"]
    with open(BENCH_DIR / "segmentation_metrics_per_image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(r)

    # -------- aggregate --------
    def stats(vals):
        arr = np.array(vals, dtype=float)
        return float(np.mean(arr)), float(np.median(arr))

    otsu_ious = [r["otsu_iou"] for r in results]
    otsu_dices = [r["otsu_dice"] for r in results]
    yolo_ious = [r["yolo_iou"] for r in results]
    yolo_dices = [r["yolo_dice"] for r in results]
    otsu_failures = sum(1 for r in results if r["otsu_empty"] or r["otsu_error"])
    yolo_failures = sum(1 for r in results if r["yolo_empty"] or r["yolo_error"])

    otsu_mean_iou, otsu_med_iou = stats(otsu_ious)
    otsu_mean_dice, otsu_med_dice = stats(otsu_dices)
    yolo_mean_iou, yolo_med_iou = stats(yolo_ious)
    yolo_mean_dice, yolo_med_dice = stats(yolo_dices)

    yolo_wins_iou = sum(1 for r in results if r["yolo_iou"] > r["otsu_iou"])
    otsu_wins_iou = sum(1 for r in results if r["otsu_iou"] > r["yolo_iou"])
    ties_iou = len(results) - yolo_wins_iou - otsu_wins_iou

    yolo_wins_dice = sum(1 for r in results if r["yolo_dice"] > r["otsu_dice"])
    otsu_wins_dice = sum(1 for r in results if r["otsu_dice"] > r["yolo_dice"])
    ties_dice = len(results) - yolo_wins_dice - otsu_wins_dice

    paired_diff_iou = [r["yolo_iou"] - r["otsu_iou"] for r in results]
    mean_paired_diff, median_paired_diff = stats(paired_diff_iou)

    multi_instance_count = sum(1 for r in results if r["yolo_multi_instance"])
    no_detection_count = sum(1 for r in results if r["yolo_num_instances"] == 0)

    summary = {
        "n_images": len(results),
        "Original_LAB_Otsu": {
            "mean_iou": round(otsu_mean_iou, 4), "median_iou": round(otsu_med_iou, 4),
            "mean_dice": round(otsu_mean_dice, 4), "median_dice": round(otsu_med_dice, 4),
            "failures": otsu_failures,
        },
        "YOLO": {
            "mean_iou": round(yolo_mean_iou, 4), "median_iou": round(yolo_med_iou, 4),
            "mean_dice": round(yolo_mean_dice, 4), "median_dice": round(yolo_med_dice, 4),
            "failures": yolo_failures,
            "no_detection_count": no_detection_count,
            "multi_instance_count": multi_instance_count,
        },
        "paired_comparison": {
            "yolo_wins_iou": yolo_wins_iou, "otsu_wins_iou": otsu_wins_iou, "ties_iou": ties_iou,
            "yolo_wins_dice": yolo_wins_dice, "otsu_wins_dice": otsu_wins_dice, "ties_dice": ties_dice,
            "mean_paired_diff_iou_yolo_minus_otsu": round(mean_paired_diff, 4),
            "median_paired_diff_iou_yolo_minus_otsu": round(median_paired_diff, 4),
        },
    }

    with open(BENCH_DIR / "segmentation_metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    return results, summary


if __name__ == "__main__":
    main()
