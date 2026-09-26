"""
Reconstruct the discarded 17-feature values for the V5 locked-test audit.

CONTEXT: run_locked_test_evaluation.py (the original, one-time locked-test
scoring script, vendored read-only at
ABCD_Audit_V5/_source_from_research_branch/Evaluation_V5Candidate/LockedTest/
run_locked_test_evaluation.py) scored 615 locked-test images with the frozen
V5 model but only persisted image_id/ground_truth/probabilities/predictions
to locked_test_per_image.csv -- it discarded the 17 feature values it
computed in memory before feeding them to the model.

This script is READ-ONLY reconstruction, not re-modeling:
  - ground_truth_binary, v4_proba, v4_pred, v5_proba, v5_pred,
    v5_features_valid are copied verbatim from locked_test_per_image.csv
    -- never recomputed.
  - The 6 base features (A_value, B_circularity, C_value, D_px, confidence,
    lesion_fraction) are copied verbatim from the matched
    Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv row -- never
    recomputed.
  - Only the 11 new V5 features (+ extra candidate features also computed
    by the same function) are recomputed, using the exact same unmodified
    extract_features.process_one() that run_locked_test_evaluation.py
    itself called -- same code, same inputs, same population, first and
    only re-application to reproduce what was thrown away.
  - The population (615 SINGLE_LESION_EVALUABLE locked-test images) is
    reproduced by copying build_locked_evaluable_rows()'s exact filtering
    logic from run_locked_test_evaluation.py, unchanged.

No model is retrained, no threshold is changed, no prediction is
recomputed. Output: ABCD_Audit_V5/V5_locked_test_master.csv
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
AUDIT_DIR = ROOT / "ABCD_Audit_V5"
SOURCE_DIR = AUDIT_DIR / "_source_from_research_branch"
FEATENG_DIR = SOURCE_DIR / "Evaluation_FeatureEngineering"
LOCKED_DIR = SOURCE_DIR / "Evaluation_V5Candidate" / "LockedTest"

sys.path.insert(0, str(FEATENG_DIR))
sys.path.insert(0, str(ROOT / "Code"))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

from extract_features import process_one  # noqa: E402 -- unmodified, reused as-is
import metrics_lib as ml  # noqa: E402

COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
LOCKED_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
PER_IMAGE_CSV = LOCKED_DIR / "locked_test_per_image.csv"
OUT_CSV = AUDIT_DIR / "V5_locked_test_master.csv"

BASE_FEATURES = ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"]
NEW_FEATURES_17 = ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                    "dark_fraction", "skin_contrast", "solidity", "turning_angle_std", "eccentricity",
                    "D_px_normalized"]
FINAL_17 = BASE_FEATURES + NEW_FEATURES_17  # exact order required by the frozen V5 model

EXTRA_CANDIDATE_FEATURES = [
    "lesion_area_px", "convexity_deficit", "isoperimetric_ratio", "radial_cv",
    "major_axis_px", "minor_axis_px", "aspect_ratio", "lab_L_std", "color_cluster_count",
    "white_fraction", "color_asymmetry", "texture_asymmetry", "entropy_intensity",
    "local_contrast", "glcm_contrast", "glcm_homogeneity", "glcm_energy",
]

EXPECTED_V5_LOCKED = {"TP": 98, "TN": 340, "FP": 137, "FN": 40, "n": 615}


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_locked_evaluable_rows():
    """Exact copy of run_locked_test_evaluation.py's build_locked_evaluable_rows()."""
    with open(COHORT_RESULTS, newline="", encoding="utf-8") as f:
        results = list(csv.DictReader(f))
    with open(LOCKED_MANIFEST, newline="", encoding="utf-8") as f:
        locked_rows = list(csv.DictReader(f))
    locked_ids = {r["image_id"] for r in locked_rows}
    path_by_id = {r["image_id"]: r["image_path"] for r in locked_rows}

    by_image = {}
    for r in results:
        image_id = ml.image_id_from_name(r["image_name"])
        if image_id not in locked_ids:
            continue
        by_image.setdefault(image_id, []).append(r)

    status_counts = {"SINGLE_LESION_EVALUABLE": 0, "MULTI_LESION_AMBIGUOUS": 0,
                     "NO_DETECTION": 0, "FAILED": 0}
    evaluable = []
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status in status_counts:
            status_counts[status] += 1
        if status == "SINGLE_LESION_EVALUABLE":
            evaluable.append(rows[0])
    return evaluable, status_counts, len(by_image), path_by_id


def main():
    evaluable, status_counts, n_images, path_by_id = build_locked_evaluable_rows()
    print(f"Locked test: {n_images} images total, coverage: {status_counts}")
    print(f"Evaluable (SINGLE_LESION_EVALUABLE): {len(evaluable)}")
    if len(evaluable) != 615:
        raise RuntimeError(
            f"Evaluable population is {len(evaluable)}, expected 615 -- STOPPING. "
            "Do not force a match; investigate build_locked_evaluable_rows() / input files."
        )

    image_ids = [ml.image_id_from_name(r["image_name"]) for r in evaluable]

    # Load the already-computed, frozen per-image results -- copied verbatim, never recomputed.
    with open(PER_IMAGE_CSV, newline="", encoding="utf-8") as f:
        per_image_rows = list(csv.DictReader(f))
    per_image_by_id = {r["image_id"]: r for r in per_image_rows}

    missing_in_per_image = [iid for iid in image_ids if iid not in per_image_by_id]
    if missing_in_per_image:
        raise RuntimeError(
            f"{len(missing_in_per_image)} evaluable image_ids not found in locked_test_per_image.csv "
            f"(e.g. {missing_in_per_image[:5]}). Stopping -- population mismatch."
        )

    print(f"\nExtracting V5 features (11 final + {len(EXTRA_CANDIDATE_FEATURES)} extra candidate) "
          f"for {len(evaluable)} locked-test images via extract_features.process_one "
          "(unmodified)...")

    t0 = time.time()
    out_rows = []
    for i, r in enumerate(evaluable):
        iid = image_ids[i]
        img_path = path_by_id[iid]
        rec = process_one(r, img_path)

        pi = per_image_by_id[iid]
        gt_bin = int(pi["ground_truth_binary"])

        out = {
            "image_id": iid,
            "ground_truth_binary": gt_bin,
            "ground_truth_label": "malignant" if gt_bin == 1 else "benign",
            "v5_proba": pi["v5_proba"],
            "v5_pred": pi["v5_pred"],
            "v4_proba": pi["v4_proba"],
            "v4_pred": pi["v4_pred"],
            "v5_features_valid": pi["v5_features_valid"],
        }
        for f in BASE_FEATURES:
            out[f] = to_float(r[f])
        for f in NEW_FEATURES_17:
            v = rec.get(f, np.nan)
            out[f] = v if v == v else ""  # NaN -> empty in CSV
        for f in EXTRA_CANDIDATE_FEATURES:
            v = rec.get(f, np.nan)
            out[f] = v if v == v else ""
        out["feature_extraction_errors"] = rec.get("_errors", "")

        out_rows.append(out)
        if (i + 1) % 50 == 0 or i == len(evaluable) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{len(evaluable)}] {iid}  elapsed={elapsed:.0f}s  rate={rate:.2f} img/s")

    fieldnames = (
        ["image_id", "ground_truth_binary", "ground_truth_label",
         "v5_proba", "v5_pred", "v4_proba", "v4_pred", "v5_features_valid"]
        + FINAL_17
        + EXTRA_CANDIDATE_FEATURES
        + ["feature_extraction_errors"]
    )
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"\nWrote {len(out_rows)} rows to {OUT_CSV}")

    # ---------------------------------------------------------------- integrity check --
    print("\n--- Integrity check: reconstructed rows vs. published locked_test_v5_vs_v4.json ---")
    valid_rows = [r for r in out_rows if str(r["v5_features_valid"]) == "True"]
    y_true = np.array([r["ground_truth_binary"] for r in valid_rows])
    v5_pred = np.array([int(r["v5_pred"]) for r in valid_rows])
    tp = int(((v5_pred == 1) & (y_true == 1)).sum())
    tn = int(((v5_pred == 0) & (y_true == 0)).sum())
    fp = int(((v5_pred == 1) & (y_true == 0)).sum())
    fn = int(((v5_pred == 0) & (y_true == 1)).sum())
    n = tp + tn + fp + fn
    actual = {"TP": tp, "TN": tn, "FP": fp, "FN": fn, "n": n}
    print(f"  Reconstructed (v5_features_valid==True rows): {actual}")
    print(f"  Published (locked_test_v5_vs_v4.json V5 block): {EXPECTED_V5_LOCKED}")
    match = actual == EXPECTED_V5_LOCKED
    print(f"  MATCH: {match}")
    if not match:
        print("  *** MISMATCH -- see report, do not silently proceed as if verified. ***")

    n_errors = sum(1 for r in out_rows if r["feature_extraction_errors"])
    print(f"\nRows with non-empty feature_extraction_errors: {n_errors}/{len(out_rows)}")

    def is_nan_val(v):
        return v == "" or v is None

    rows_with_nan17 = []
    for r in out_rows:
        if any(is_nan_val(r[f]) for f in FINAL_17):
            rows_with_nan17.append(r)
    print(f"Rows with a NaN in at least one of the 17 final model features: {len(rows_with_nan17)}/{len(out_rows)}")

    # cross-check: NaN-in-17 <=> v5_features_valid == False
    # (v5_features_valid was originally computed from np.isnan over ALL 17 features,
    # base + new -- see run_locked_test_evaluation.py lines 192-198 -- so the check
    # below must cover all 17, not just the 11 newly-recomputed ones.)
    inconsistent = []
    for r in out_rows:
        has_nan17 = any(is_nan_val(r[f]) for f in FINAL_17)
        valid = str(r["v5_features_valid"]) == "True"
        if has_nan17 == valid:  # nan present but marked valid, OR no nan but marked invalid
            inconsistent.append((r["image_id"], has_nan17, valid))
    print(f"Consistency check (NaN-in-17 <=> v5_features_valid==False): "
          f"{'OK, no inconsistencies' if not inconsistent else f'{len(inconsistent)} INCONSISTENCIES FOUND'}")
    if inconsistent:
        for iid, has_nan17, valid in inconsistent[:20]:
            print(f"    {iid}: has_nan_in_17={has_nan17} v5_features_valid={valid}")


if __name__ == "__main__":
    main()
