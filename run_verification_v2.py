"""V2 pipeline verification: rerun every image in Images/ with the
segmentation-investigation fixes applied, before touching any larger
dataset.

Changes from RevisedABCDTest (see revised_abcd/pipeline_v2.py docstring for
the full rationale, grounded in investigation/ and the segmentation-flags
follow-up):
  - single-image YOLO inference only (no batching, no resize distortion)
  - every detected instance kept and scored SEPARATELY (no np.max union)
  - per-instance quality flags, with a tiny-artifact floor and a
    reporting-only low-confidence flag
  - D_mm is now unconditionally None (no calibration attempted at all);
    D_px is always recorded
  - A/B/C scoring and all existing thresholds/LOW-HIGH rules unchanged

Does NOT modify Code/, raster_masks/, YoloMaskABCDTest/, or RevisedABCDTest/.
New masks go to raster_masks_v2/; new results go to VerificationV2/.

UNATTENDED: no input()/GUI calls anywhere in this script or what it imports.
Every image's row(s) are written and flushed immediately, so an
interruption loses at most the image in flight.
"""

import csv
import sys
import time
import traceback
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
import raster_mask_inference as rmi  # noqa: E402  (reused only for DEFAULT_MODEL constant)
from revised_abcd.pipeline_v2 import process_image  # noqa: E402
from revised_abcd.evaluation_policy import determine_evaluation_status  # noqa: E402

IMAGES_ROOT = ROOT / "Images"
MASK_DIR_V2 = ROOT / "raster_masks_v2"
OUTPUT_DIR = ROOT / "VerificationV2"
OUTPUT_CSV = OUTPUT_DIR / "verification_v2_results.csv"
CONF = 0.25  # UNCHANGED — same as every prior run, never tuned against these images

FIELDNAMES = [
    "image_name", "ground_truth", "ground_truth_binary", "status", "evaluation_status",
    "num_instances_detected", "lesion_instance_id", "confidence",
    "num_total_components", "num_meaningful_components", "num_tiny_artifacts_ignored",
    "lesion_fraction", "quality_flags",
    "A_value", "A_concern", "A_principal_axis_deg",
    "B_circularity", "B_circularity_concern", "B_experimental", "B_experimental_n_defects",
    "C_value", "C_concern", "C_label",
    "D_px", "D_mm", "D_concern",
    "concerns", "risk_level", "provisional_prediction",
    "is_primary_instance", "mask_path", "error_or_warning",
]


def build_manifest():
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"image_path": p, "ground_truth": label}
    for p in sorted(IMAGES_ROOT.glob("*")):
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.stem not in manifest:
            manifest[p.stem] = {"image_path": p, "ground_truth": "unknown"}
    return manifest


def write_header(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_rows(path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for row in rows:
            writer.writerow(row)
        f.flush()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR_V2.mkdir(parents=True, exist_ok=True)
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    write_header(OUTPUT_CSV)

    manifest = build_manifest()
    print(f"Loaded manifest: {len(manifest)} images "
          f"({sum(1 for v in manifest.values() if v['ground_truth'] != 'unknown')} labeled, "
          f"{sum(1 for v in manifest.values() if v['ground_truth'] == 'unknown')} unlabeled)")

    print(f"Loading model: {rmi.DEFAULT_MODEL}")
    model = YOLO(rmi.DEFAULT_MODEL)

    n_images = len(manifest)
    n_processed_images = 0
    n_no_detection = 0
    n_failed = 0
    n_instances_total = 0
    n_multi_lesion_images = 0

    all_rows = []  # accumulated for the end-of-run evaluation summary (43 images — trivial memory cost)

    t_start = time.time()
    for idx, (stem, info) in enumerate(sorted(manifest.items()), 1):
        image_path = info["image_path"]
        ground_truth = info["ground_truth"]
        ground_truth_binary = {"benign": 0, "malignant": 1}.get(ground_truth, "")

        base_row = {k: "" for k in FIELDNAMES}
        base_row.update({
            "image_name": image_path.name,
            "ground_truth": ground_truth,
            "ground_truth_binary": ground_truth_binary,
        })

        try:
            rows, orig_shape = process_image(model, image_path, conf=CONF)
        except Exception:
            err = traceback.format_exc(limit=3).replace("\n", " | ")
            row = dict(base_row)
            row["status"] = "FAILED"
            row["evaluation_status"] = determine_evaluation_status(ground_truth, "FAILED", 0)
            row["error_or_warning"] = err
            append_rows(OUTPUT_CSV, [row])
            all_rows.append(row)
            n_failed += 1
            print(f"[{idx}/{n_images}] {stem}: FAILED: {err[:150]}")
            continue

        if not rows:
            row = dict(base_row)
            row["status"] = "NO_DETECTION"
            row["evaluation_status"] = determine_evaluation_status(ground_truth, "NO_DETECTION", 0)
            row["num_instances_detected"] = 0
            append_rows(OUTPUT_CSV, [row])
            all_rows.append(row)
            n_no_detection += 1
            print(f"[{idx}/{n_images}] {stem}: NO_DETECTION (evaluation_status={row['evaluation_status']})")
            continue

        n_processed_images += 1
        n_instances_total += len(rows)
        eval_status = determine_evaluation_status(ground_truth, "PROCESSED", len(rows))
        if len(rows) > 1:
            n_multi_lesion_images += 1

        out_rows = []
        for r in rows:
            stem_mask_path = MASK_DIR_V2 / f"{stem}_instance{r['lesion_instance_id']}_mask.png"
            cv2.imwrite(str(stem_mask_path), r["mask"])

            row = dict(base_row)
            row["status"] = "PROCESSED"
            row["evaluation_status"] = eval_status
            row["num_instances_detected"] = len(rows)
            # Informational ranking only (highest confidence) — NOT a claim
            # that this instance is the one the image's label refers to.
            # See revised_abcd/evaluation_policy.py.
            row["is_primary_instance"] = (r["lesion_instance_id"] == 0)
            row["mask_path"] = str(stem_mask_path.relative_to(ROOT))
            for key in FIELDNAMES:
                if key in r:
                    row[key] = r[key]
            out_rows.append(row)

        append_rows(OUTPUT_CSV, out_rows)
        all_rows.extend(out_rows)
        confs = [f"{r['confidence']:.2f}" for r in rows]
        print(f"[{idx}/{n_images}] {stem}: {len(rows)} instance(s), confs={confs}, "
              f"evaluation_status={eval_status}, flags={[r['quality_flags'] for r in rows]}")

    t_end = time.time()
    print(f"\nDone. images_processed={n_processed_images} no_detection={n_no_detection} "
          f"failed={n_failed} total_instances={n_instances_total} "
          f"multi_lesion_images={n_multi_lesion_images} "
          f"runtime={t_end - t_start:.1f}s")
    print(f"Results: {OUTPUT_CSV}")
    print(f"Masks: {MASK_DIR_V2}")

    print_evaluation_summary(all_rows)


def print_evaluation_summary(all_rows):
    """Reports every evaluation_status bucket explicitly — nothing is
    silently excluded — and computes the confusion matrix ONLY over
    SINGLE_LESION_EVALUABLE rows, per the multi-lesion evaluation policy."""
    by_image = {}
    for r in all_rows:
        by_image.setdefault(r["image_name"], []).append(r)

    buckets = {}
    for name, rows in by_image.items():
        status = rows[0]["evaluation_status"]
        buckets.setdefault(status, []).append(name)

    print("\n" + "=" * 60)
    print("EVALUATION-STATUS BREAKDOWN (every image accounted for)")
    print("=" * 60)
    for status in ("SINGLE_LESION_EVALUABLE", "MULTI_LESION_AMBIGUOUS", "NO_DETECTION", "FAILED", "UNLABELED"):
        names = buckets.get(status, [])
        print(f"  {status}: {len(names)}")
        if status in ("MULTI_LESION_AMBIGUOUS", "NO_DETECTION", "FAILED") and names:
            print(f"    {sorted(names)}")

    evaluable_rows = [r for r in all_rows if r["evaluation_status"] == "SINGLE_LESION_EVALUABLE"]
    tp = tn = fp = fn = 0
    for r in evaluable_rows:
        gt = int(r["ground_truth_binary"])
        pred = 1 if r["provisional_prediction"] == "positive" else 0
        if gt == 1 and pred == 1:
            tp += 1
        elif gt == 0 and pred == 0:
            tn += 1
        elif gt == 0 and pred == 1:
            fp += 1
        elif gt == 1 and pred == 0:
            fn += 1
    n = tp + tn + fp + fn
    print(f"\nClassification metrics — SINGLE_LESION_EVALUABLE images only (n={n}):")
    if n:
        acc = (tp + tn) / n
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else float("nan")
        print(f"  TP={tp} TN={tn} FP={fp} FN={fn}")
        print(f"  Accuracy={acc:.3f} Sensitivity={sens:.3f} Specificity={spec:.3f} Precision={prec:.3f} F1={f1:.3f}")
    else:
        print("  (no evaluable images)")
    print(f"\nMULTI_LESION_AMBIGUOUS images are excluded from the above metrics but NOT from the "
          f"dataset — {len(buckets.get('MULTI_LESION_AMBIGUOUS', []))} such image(s) are reported "
          f"separately above, with every instance still fully scored in the CSV.")


if __name__ == "__main__":
    main()
