"""Production script for a LARGE dataset run, using the same v2 pipeline
verified in run_verification_v2.py (single-image inference, per-instance
masks/scoring, D_mm always None, all existing ABCD thresholds unchanged).

Not run as part of this change — created and ready, per instruction, for
when a larger dataset is actually supplied.

Design points (per the production requirements):
  - processes ONE IMAGE AT A TIME: images are discovered via a directory
    glob (paths only) and read from disk one at a time inside
    revised_abcd.pipeline_v2.process_image(); the full dataset is never
    loaded into memory at once, only whichever single image is currently
    being scored
  - runs unattended: no input()/GUI calls anywhere in this script or in
    revised_abcd/*, matching run_verification_v2.py
  - saves incrementally: every image's row(s) are appended and flushed to
    the results CSV immediately, exactly as in run_verification_v2.py
  - checkpoint/resume: on startup, if the output CSV already exists, every
    image_name already present (with ANY status — PROCESSED, NO_DETECTION,
    or FAILED) is treated as done and skipped, so a killed/interrupted run
    can simply be re-invoked with the same command. Pass --fresh to ignore
    prior progress and start over. Pass --retry-failed to additionally
    re-attempt images that previously ended in FAILED (but not
    NO_DETECTION, which is a real, final result, not an error).
  - logs errors and continues: identical per-image try/except to the
    verification script
  - logs no detections: identical NO_DETECTION status handling
  - records every lesion instance and confidence: identical per-instance
    row schema
  - no dashboards/overlays are generated for any image (matches
    verification; this script only ever writes CSV rows and per-instance
    mask PNGs, both small)
  - reports progress, remaining images, runtime, and ETA: printed after
    each image, based on a running average of per-image processing time
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import cv2
import psutil

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402
import torch  # noqa: E402
import raster_mask_inference as rmi  # noqa: E402  (DEFAULT_MODEL constant only)
from revised_abcd.pipeline_v2 import process_image  # noqa: E402
from revised_abcd.evaluation_policy import determine_evaluation_status  # noqa: E402

CONF = 0.25  # UNCHANGED — never tune this against dataset results
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

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


def discover_images(source_dir: Path):
    """Yield (stem, path, ground_truth) one at a time — never materializes
    the full list of loaded images, only paths. Used when --source (a
    Benign/Malignant-subfoldered directory) is given instead of --manifest."""
    for label, folder_name in (("benign", "Benign"), ("malignant", "Malignant")):
        folder = source_dir / folder_name
        if folder.is_dir():
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMAGE_EXTS:
                    yield p.stem, p, label
    for p in sorted(source_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p.stem, p, "unknown"


def discover_images_from_manifest(manifest_path: Path):
    """Yield (stem, path, ground_truth) from a fixed evaluation manifest
    (image_id, image_path, ground_truth, ...) — e.g.
    Evaluation_3500/evaluation_manifest_3500.csv. Reads the manifest once
    (a few thousand rows of text, negligible), but never loads any image
    data itself; images are opened one at a time inside process_image().
    Original image paths are used directly — nothing is copied."""
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            path = Path(row["image_path"])
            yield row["image_id"], path, row["ground_truth"]


def load_done_stems(csv_path: Path, retry_failed: bool):
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if retry_failed and row.get("status") == "FAILED":
                continue
            done.add(row["image_name"])
    return done


def write_header_if_new(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_rows(path: Path, rows):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        for row in rows:
            writer.writerow(row)
        f.flush()


def format_eta(seconds):
    if seconds != seconds or seconds < 0:  # NaN guard
        return "unknown"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s"


def main():
    parser = argparse.ArgumentParser(description="Production v2 YOLO->ABCD run for a large dataset.")
    parser.add_argument("--source", default=None,
                        help="Dataset root, expected to contain Benign/ and Malignant/ subfolders "
                             "(and optionally loose unlabeled images directly inside). "
                             "Ignored if --manifest is given.")
    parser.add_argument("--manifest", default=None,
                        help="Path to a fixed evaluation manifest CSV (image_id, image_path, "
                             "ground_truth, ...), e.g. Evaluation_3500/evaluation_manifest_3500.csv. "
                             "Takes precedence over --source. Images are read from their paths in "
                             "the manifest directly — never copied.")
    parser.add_argument("--output-dir", default=str(ROOT / "LargeDatasetV2"),
                        help="Where to write results.csv and instance masks. Default: LargeDatasetV2/")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore any existing results.csv and start over from scratch.")
    parser.add_argument("--retry-failed", action="store_true",
                        help="On resume, also re-attempt images previously logged as FAILED "
                             "(NO_DETECTION is left alone — it is a real result, not an error).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most this many of the remaining (not-yet-done) images in "
                             "this invocation, then stop. For benchmarking — does not change any "
                             "inference or scoring behavior, only how many iterations run.")
    args = parser.parse_args()

    if not args.manifest and not args.source:
        args.source = str(ROOT / "Images")  # preserve old default when neither is given

    output_dir = Path(args.output_dir)
    output_csv = output_dir / "results.csv"
    mask_dir = output_dir / "masks"

    if args.fresh and output_csv.exists():
        output_csv.unlink()

    write_header_if_new(output_csv)
    mask_dir.mkdir(parents=True, exist_ok=True)

    done_stems_by_name = load_done_stems(output_csv, args.retry_failed)
    print(f"Resuming: {len(done_stems_by_name)} image(s) already logged and will be skipped "
          f"(pass --fresh to ignore prior progress).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch reports CUDA available: {torch.cuda.is_available()} -> using device: {device}")
    print(f"Loading model: {rmi.DEFAULT_MODEL}")
    model = YOLO(rmi.DEFAULT_MODEL)
    model.to(device)

    if args.manifest:
        print(f"Discovering images from manifest: {args.manifest}")
        all_images = list(discover_images_from_manifest(Path(args.manifest)))
    else:
        print(f"Discovering images from source directory: {args.source}")
        all_images = list(discover_images(Path(args.source)))  # paths only — cheap
    total = len(all_images)
    remaining = [(stem, path, gt) for stem, path, gt in all_images if path.name not in done_stems_by_name]
    if args.limit is not None:
        remaining = remaining[:args.limit]
        print(f"--limit {args.limit} applied: only the first {len(remaining)} remaining image(s) "
              f"will be processed in this invocation.")
    print(f"Dataset: {total} image(s) total, {len(remaining)} to process in this invocation.")

    process = psutil.Process()
    peak_rss_mb = process.memory_info().rss / 1e6

    n_ok = n_no_detection = n_failed = n_instances_total = n_multi = 0
    per_image_times = []
    t_start = time.time()

    for i, (stem, image_path, ground_truth) in enumerate(remaining, 1):
        t0 = time.time()
        ground_truth_binary = {"benign": 0, "malignant": 1}.get(ground_truth, "")
        base_row = {k: "" for k in FIELDNAMES}
        base_row.update({
            "image_name": image_path.name,
            "ground_truth": ground_truth,
            "ground_truth_binary": ground_truth_binary,
        })

        try:
            rows, _ = process_image(model, image_path, conf=CONF)
            failed_error = None
        except Exception:
            rows = None
            failed_error = traceback.format_exc(limit=3).replace("\n", " | ")

        if failed_error is not None:
            row = dict(base_row)
            row["status"] = "FAILED"
            row["evaluation_status"] = determine_evaluation_status(ground_truth, "FAILED", 0)
            row["error_or_warning"] = failed_error
            append_rows(output_csv, [row])
            n_failed += 1
            status_str = f"FAILED: {failed_error[:120]}"
        elif not rows:
            row = dict(base_row)
            row["status"] = "NO_DETECTION"
            row["evaluation_status"] = determine_evaluation_status(ground_truth, "NO_DETECTION", 0)
            row["num_instances_detected"] = 0
            append_rows(output_csv, [row])
            n_no_detection += 1
            status_str = "NO_DETECTION"
        else:
            n_ok += 1
            n_instances_total += len(rows)
            eval_status = determine_evaluation_status(ground_truth, "PROCESSED", len(rows))
            if len(rows) > 1:
                n_multi += 1
            out_rows = []
            for r in rows:
                mask_path = mask_dir / f"{stem}_instance{r['lesion_instance_id']}_mask.png"
                cv2.imwrite(str(mask_path), r["mask"])
                row = dict(base_row)
                row["status"] = "PROCESSED"
                row["evaluation_status"] = eval_status
                row["num_instances_detected"] = len(rows)
                # Informational ranking only (highest confidence) — NOT a
                # claim that this instance is the one the image's label
                # refers to. See revised_abcd/evaluation_policy.py.
                row["is_primary_instance"] = (r["lesion_instance_id"] == 0)
                row["mask_path"] = str(mask_path.relative_to(ROOT)) if ROOT in mask_path.parents else str(mask_path)
                for key in FIELDNAMES:
                    if key in r:
                        row[key] = r[key]
                out_rows.append(row)
            append_rows(output_csv, out_rows)
            status_str = f"{len(rows)} instance(s), evaluation_status={eval_status}"

        elapsed = time.time() - t0
        per_image_times.append(elapsed)
        if len(per_image_times) > 200:  # bound memory for a very large run
            per_image_times.pop(0)
        avg_time = sum(per_image_times) / len(per_image_times)
        images_left = len(remaining) - i
        eta = avg_time * images_left
        total_elapsed = time.time() - t_start
        current_rss_mb = process.memory_info().rss / 1e6
        peak_rss_mb = max(peak_rss_mb, current_rss_mb)

        print(f"[{i}/{len(remaining)}] {stem}: {status_str}  "
              f"| elapsed={format_eta(total_elapsed)} avg={avg_time:.2f}s/img "
              f"remaining={images_left} eta={format_eta(eta)} rss={current_rss_mb:.0f}MB")

    total_elapsed = time.time() - t_start
    print(f"\nDone. this_run: processed_ok={n_ok} no_detection={n_no_detection} failed={n_failed} "
          f"instances={n_instances_total} multi_lesion_images={n_multi} "
          f"runtime={format_eta(total_elapsed)} device={device} peak_rss_mb={peak_rss_mb:.0f}")
    print(f"Results: {output_csv}")
    print(f"Masks: {mask_dir}")
    print("To resume a later run (e.g. after an interruption), re-run this exact command — "
          "already-logged images are skipped automatically.")


if __name__ == "__main__":
    main()
