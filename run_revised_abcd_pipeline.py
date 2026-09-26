"""Revised/experimental ABCD pipeline runner.

Applies the corrections identified in the A/B/D root-cause investigation
(see investigation/) to the SAME 40 labeled ISIC images and the SAME YOLO
raster masks already on disk (raster_masks/) — no retraining, no new masks,
no field-of-view calibration.

Does NOT modify anything under Code/, does NOT touch YoloMaskABCDTest/ or
raster_masks/. Every existing threshold (A>0.20, B>0.50, C's 0.35/8%/50%
color rules, D>10mm, critical-color-override, "2+ flags = HIGH") is
reproduced UNCHANGED from Code/MelanomaDeterminingStuff/score.py — nothing
here was tuned against these 40 labels.

Per image, both the ORIGINAL and REVISED numbers are computed side by side
in the same pass (both call directly into the unmodified Code/ functions for
the "original" figures), so the comparison CSV is a true apples-to-apples
diff, not a join across two separate runs.

UNATTENDED: no input()/confirmation/GUI calls anywhere in this script or the
modules it imports (matplotlib is never invoked here). Every image's row is
written and flushed to CSV immediately after it's computed, so an
interruption loses at most the one image in flight. Any per-image exception
is caught, logged into that row, and processing continues automatically.
"""

import csv
import io
import contextlib
import sys
import time
import traceback
import warnings
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
CODE_DIR = ROOT / "Code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# --- existing, UNMODIFIED pipeline code ------------------------------------
from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import (  # noqa: E402
    remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair,
)
from ComputerVisionStuff.hair import measure_hair_width_px as measure_hair_width_px_original  # noqa: E402
from MelanomaDeterminingStuff.asymmetry import score_asymmetry as score_asymmetry_original  # noqa: E402
from MelanomaDeterminingStuff.border import score_border  # noqa: E402  (used UNCHANGED for both baseline and revised)
from MelanomaDeterminingStuff.color import score_color  # noqa: E402   (used UNCHANGED, C is not revised)
from MelanomaDeterminingStuff.diameter import score_diameter as score_diameter_original  # noqa: E402

# --- new, non-invasive revised modules --------------------------------------
from revised_abcd.revised_asymmetry import score_asymmetry_revised  # noqa: E402
from revised_abcd.revised_border import score_border_experimental  # noqa: E402
from revised_abcd.revised_diameter import measure_calibration_revised, score_diameter_revised  # noqa: E402

VELLUS_HAIR_UM = 70.0  # UNCHANGED, matches Code/main.py
CRITICAL_COLOR_OVERRIDE = 0.50  # UNCHANGED, matches Code/MelanomaDeterminingStuff/score.py
CONCERNS_HIGH_THRESHOLD = 2     # UNCHANGED

IMAGES_ROOT = ROOT / "Images"
MASK_DIR = ROOT / "raster_masks"
OUTPUT_DIR = ROOT / "RevisedABCDTest"
OUTPUT_CSV = OUTPUT_DIR / "baseline_vs_revised.csv"
REPORT_PATH = OUTPUT_DIR / "revised_evaluation_report.txt"

LARGE_AREA_FRACTION = 0.5  # segmentation quality flag threshold — an
# engineering/data-quality flag, NOT a diagnostic threshold; identical
# heuristic already used (and validated against real over-segmentation
# cases, e.g. ISIC_0000076) in the earlier dataset evaluation.

FIELDNAMES = [
    "image_name", "mask_name", "ground_truth", "ground_truth_binary", "status",
    # segmentation quality
    "num_blobs", "mask_lesion_fraction", "flag_fragmented", "flag_large_area",
    "segmentation_quality_flags",
    # A
    "A_original", "A_original_concern", "A_revised", "A_revised_concern",
    "A_principal_axis_deg",
    # B
    "B_circularity", "B_circularity_concern",
    "B_experimental", "B_experimental_n_defects",
    # C (unchanged)
    "C_value", "C_concern", "C_label",
    # D
    "hair_width_px_original", "mm_per_px_original", "D_mm_original", "D_original_concern",
    "hair_calibration_reason_revised", "n_hair_like_components", "n_blob_like_components",
    "mm_per_px_revised", "D_px", "D_mm_revised", "D_revised_concern",
    # combined scoring, baseline vs revised
    "concerns_original", "risk_level_original",
    "concerns_revised", "risk_level_revised",
    "risk_level_changed",
    "provisional_prediction_original", "provisional_prediction_revised",
    "error_or_warning",
]


def build_manifest():
    manifest = {}
    for label, folder in (("benign", "Benign"), ("malignant", "Malignant")):
        for p in sorted((IMAGES_ROOT / folder).glob("*.jpg")):
            manifest[p.stem] = {"image_path": p, "ground_truth": label}
    return manifest


def segmentation_quality_flags(mask):
    n_labels, _, _, _ = cv2.connectedComponentsWithStats(mask)
    num_blobs = n_labels - 1
    lesion_fraction = float(np.mean(mask > 0))
    flag_fragmented = num_blobs > 1
    flag_large_area = lesion_fraction > LARGE_AREA_FRACTION
    flags = []
    if flag_fragmented:
        flags.append(f"fragmented({num_blobs}_blobs)")
    if flag_large_area:
        flags.append(f"large_area_fraction({lesion_fraction:.0%})_possible_over_segmentation")
    return num_blobs, lesion_fraction, flag_fragmented, flag_large_area, (";".join(flags) if flags else "ok")


def combine_scores(a_result, b_result, c_result, d_result):
    """Reproduces Code/MelanomaDeterminingStuff/score.py::analyze_abcde's
    combination logic exactly (concerns count, critical-color override,
    2-flags-means-HIGH rule) — NOT modified, just re-expressed here so it
    can be applied to either the original or the revised A/D inputs while
    always using the SAME unchanged B (circularity) and C."""
    concerns = sum(1 for r in (a_result, b_result, c_result, d_result) if r["concern"])
    return concerns


def risk_level_from(concerns, critical_override):
    if critical_override:
        return "HIGH"
    return "LOW" if concerns <= CONCERNS_HIGH_THRESHOLD - 1 else "HIGH"


def process_one(stem, image_path, mask_path, ground_truth):
    row = {
        "image_name": image_path.name, "mask_name": mask_path.name if mask_path else "",
        "ground_truth": ground_truth,
        "ground_truth_binary": 1 if ground_truth == "malignant" else 0,
    }
    warnings_list = []
    errors_list = []

    stdout_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buf), warnings.catch_warnings(record=True) as wlist:
            warnings.simplefilter("always")

            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                raise ValueError(f"could not read mask: {mask_path}")
            mask = (mask_raw > 127).astype(np.uint8) * 255

            lesion_fraction_check = float(np.mean(mask > 0))
            if lesion_fraction_check == 0.0:
                row["status"] = "NO_DETECTION"
                for key in FIELDNAMES:
                    row.setdefault(key, "")
                row["error_or_warning"] = "EMPTY MASK: YOLO detected no lesion; ABCD not run"
                return row

            num_blobs, lesion_fraction, flag_fragmented, flag_large_area, quality_str = segmentation_quality_flags(mask)
            row["num_blobs"] = num_blobs
            row["mask_lesion_fraction"] = round(lesion_fraction, 4)
            row["flag_fragmented"] = flag_fragmented
            row["flag_large_area"] = flag_large_area
            row["segmentation_quality_flags"] = quality_str

            # --- existing preprocessing chain, UNCHANGED ---
            original = load_image(str(image_path))
            no_vignette, circle_info = remove_vignette(original)
            denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
            bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
            no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)

            # --- A: original (unchanged) vs revised (principal-axis aligned) ---
            a_original = score_asymmetry_original(mask)
            a_revised = score_asymmetry_revised(mask)
            row["A_original"] = a_original["value"]
            row["A_original_concern"] = a_original["concern"]
            row["A_revised"] = a_revised["value"]
            row["A_revised_concern"] = a_revised["concern"]
            row["A_principal_axis_deg"] = a_revised["principal_axis_deg"]

            # --- B: circularity UNCHANGED (drives the decision) + experimental ---
            b_result = score_border(mask)
            b_experimental = score_border_experimental(mask)
            row["B_circularity"] = b_result["value"]
            row["B_circularity_concern"] = b_result["concern"]
            row["B_experimental"] = b_experimental["value"]
            row["B_experimental_n_defects"] = b_experimental["n_significant_defects"]

            # --- C: UNCHANGED, uses no_hair exactly as the existing pipeline does ---
            c_result, pink_red_px, blue_gray_px, white_px, black_px = score_color(mask, no_hair, circle_info)
            row["C_value"] = c_result["value"]
            row["C_concern"] = c_result["concern"]
            row["C_label"] = c_result["label"]

            # --- D: original (unchanged) vs revised (hair-shape-verified) ---
            hair_width_px_original = measure_hair_width_px_original(bilateral)
            mm_per_px_original = (VELLUS_HAIR_UM / hair_width_px_original) / 1000.0 if hair_width_px_original else None
            d_original = score_diameter_original(mask, mm_per_px_original)
            row["hair_width_px_original"] = hair_width_px_original
            row["mm_per_px_original"] = mm_per_px_original
            row["D_mm_original"] = d_original["value"]
            row["D_original_concern"] = d_original["concern"]

            mm_per_px_revised, reason, n_hair_like, n_blob_like = measure_calibration_revised(bilateral)
            d_revised = score_diameter_revised(mask, mm_per_px_revised)
            row["hair_calibration_reason_revised"] = reason
            row["n_hair_like_components"] = n_hair_like
            row["n_blob_like_components"] = n_blob_like
            row["mm_per_px_revised"] = mm_per_px_revised
            row["D_px"] = d_revised["diameter_px"]
            row["D_mm_revised"] = d_revised["value"]
            row["D_revised_concern"] = d_revised["concern"]

            # --- combine, using UNCHANGED thresholds/rules both times ---
            critical_color = max(pink_red_px, blue_gray_px, white_px, black_px)
            critical_override = critical_color > CRITICAL_COLOR_OVERRIDE

            concerns_original = combine_scores(a_original, b_result, c_result, d_original)
            concerns_revised = combine_scores(a_revised, b_result, c_result, d_revised)
            risk_original = risk_level_from(concerns_original, critical_override)
            risk_revised = risk_level_from(concerns_revised, critical_override)

            row["concerns_original"] = concerns_original
            row["risk_level_original"] = risk_original
            row["concerns_revised"] = concerns_revised
            row["risk_level_revised"] = risk_revised
            row["risk_level_changed"] = risk_original != risk_revised
            row["provisional_prediction_original"] = "positive" if risk_original == "HIGH" else "negative"
            row["provisional_prediction_revised"] = "positive" if risk_revised == "HIGH" else "negative"

            # Only reached if every step above succeeded.
            row["status"] = "PROCESSED"

        for w in wlist:
            warnings_list.append(f"{w.category.__name__}: {w.message}")
        for line in stdout_buf.getvalue().splitlines():
            lower = line.lower()
            if any(kw in lower for kw in ("no circle found", "circle too small", "no hair detected",
                                           "too few points", "few clean pixels", "baseline too dark")):
                warnings_list.append(line.strip())

    except Exception:
        errors_list.append(traceback.format_exc(limit=3).replace("\n", " | "))
        # An exception means this row is incomplete regardless of whatever
        # partial fields were set before the failure — always mark FAILED,
        # never leave a stale "PROCESSED" from an earlier line in the try block.
        row["status"] = "FAILED"

    row["error_or_warning"] = " || ".join(x for x in (" | ".join(warnings_list), " | ".join(errors_list)) if x)
    for key in FIELDNAMES:
        row.setdefault(key, "")
    return row


def write_header_if_needed(path):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()


def append_row(path, row):
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow(row)
        f.flush()


def already_done_stems(path):
    done = set()
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(Path(r["image_name"]).stem)
    return done


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Fresh run each invocation: start a clean CSV (no resume-merge ambiguity).
    if OUTPUT_CSV.exists():
        OUTPUT_CSV.unlink()
    write_header_if_needed(OUTPUT_CSV)

    manifest = build_manifest()
    print(f"Processing {len(manifest)} labeled images (fully unattended, incremental save)...")

    t_start = time.time()
    n_ok = n_no_detection = n_failed = 0
    for stem, info in sorted(manifest.items()):
        mask_path = MASK_DIR / f"{stem}_mask.png"
        if not mask_path.exists():
            row = {k: "" for k in FIELDNAMES}
            row["image_name"] = info["image_path"].name
            row["ground_truth"] = info["ground_truth"]
            row["ground_truth_binary"] = 1 if info["ground_truth"] == "malignant" else 0
            row["status"] = "FAILED"
            row["error_or_warning"] = f"no mask file found at {mask_path}"
            append_row(OUTPUT_CSV, row)
            n_failed += 1
            print(f"--- {stem}: FAILED (no mask file) ---")
            continue

        row = process_one(stem, info["image_path"], mask_path, info["ground_truth"])
        append_row(OUTPUT_CSV, row)

        if row["status"] == "PROCESSED":
            n_ok += 1
            print(f"--- {stem}: A {row['A_original']}->{row['A_revised']}  "
                  f"D_mm {row['D_mm_original']}->{row['D_mm_revised']}  "
                  f"risk {row['risk_level_original']}->{row['risk_level_revised']}  "
                  f"flags={row['segmentation_quality_flags']} ---")
        elif row["status"] == "NO_DETECTION":
            n_no_detection += 1
            print(f"--- {stem}: NO_DETECTION ---")
        else:
            n_failed += 1
            print(f"--- {stem}: FAILED: {row['error_or_warning'][:200]} ---")

    t_end = time.time()
    print(f"\nDone. processed_ok={n_ok} no_detection={n_no_detection} failed={n_failed} "
          f"runtime={t_end - t_start:.1f}s avg={(t_end - t_start) / max(len(manifest), 1):.2f}s/image")
    print(f"Results: {OUTPUT_CSV}")

    return {"n_ok": n_ok, "n_no_detection": n_no_detection, "n_failed": n_failed,
            "runtime_sec": t_end - t_start, "csv_path": str(OUTPUT_CSV)}


if __name__ == "__main__":
    main()
