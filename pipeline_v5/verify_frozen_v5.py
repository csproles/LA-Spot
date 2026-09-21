"""Reproducibility check for the FROZEN pipeline_v5/ runtime artifact.

Confirms that pipeline_v5/feature_extraction.py + pipeline_v5/decision_model.py
+ pipeline_v5/frozen_model.pkl -- the clean, relocated runtime artifact, NOT
the research scripts under Evaluation_FeatureEngineering/ or
Evaluation_V5Candidate/ -- reproduce EXACTLY the same per-image decision
scores already recorded in
Evaluation_V5Candidate/LockedTest/locked_test_per_image.csv, the file that
backed the official one-time locked-test evaluation and the KEEP-V4-vs-
FREEZE-V5-AS-FINAL decision.

This is a verification run only. It does not touch YOLO, does not refit
anything, and does not change any feature, threshold, or coefficient. If
it finds a mismatch, that indicates a transcription error in relocating the
code into pipeline_v5/, to be fixed in pipeline_v5/ itself (not in the
already-frozen research results, which stay as the historical record of
what was actually evaluated).
"""

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

PIPELINE_V5_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_V5_DIR.parent
CODE_DIR = ROOT / "Code"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(PIPELINE_V5_DIR))
sys.path.insert(0, str(ROOT / "Evaluation_FinalTargeted"))

from HandlingStuff import load_image  # noqa: E402
from ComputerVisionStuff import remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair  # noqa: E402
import metrics_lib as ml  # noqa: E402

from feature_extraction import extract_v5_new_features  # noqa: E402
from decision_model import V5_CONFIG, featurize, load_frozen_pipeline  # noqa: E402

COHORT_RESULTS = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "cohort_v2_results.csv"
LOCKED_MANIFEST = ROOT / "Evaluation_FinalTargeted" / "Cohort" / "locked_test_manifest.csv"
REFERENCE_PER_IMAGE = ROOT / "Evaluation_V5Candidate" / "LockedTest" / "locked_test_per_image.csv"


def to_float(x):
    if x in (None, ""):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def build_locked_evaluable_rows():
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

    evaluable = []
    for image_id, rows in by_image.items():
        status = rows[0]["evaluation_status"] or rows[0]["status"]
        if status == "SINGLE_LESION_EVALUABLE":
            evaluable.append(rows[0])
    return evaluable, path_by_id


def score_one_via_frozen_runtime(row, image_path, pipeline):
    """Runs the SAME preprocessing chain revised_abcd/pipeline_v2.py already
    uses (unchanged), loads the existing cached YOLO mask (no re-YOLO), then
    scores via the pipeline_v5/ runtime artifact only."""
    mask_raw = cv2.imread(row["mask_path"], cv2.IMREAD_GRAYSCALE)
    mask = (mask_raw > 127).astype(np.uint8) * 255

    original = load_image(str(image_path))
    no_vignette, circle_info = remove_vignette(original)
    denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)
    if no_hair.shape[:2] != mask.shape:
        mask = cv2.resize(mask, (no_hair.shape[1], no_hair.shape[0]), interpolation=cv2.INTER_NEAREST)

    d_px = to_float(row["D_px"])
    new_feats = extract_v5_new_features(mask, no_hair, circle_info, d_px)

    full_row = {f: to_float(row[f]) for f in V5_CONFIG["base_features"]}
    full_row.update(new_feats)
    if any(v is None or v != v for v in full_row.values()):
        return None  # missing feature -- consistent with how the original run excluded such rows

    X = featurize(full_row)
    proba = float(pipeline.predict_proba(X)[0, 1])
    return proba


def main():
    with open(REFERENCE_PER_IMAGE, newline="", encoding="utf-8") as f:
        reference = {r["image_id"]: r for r in csv.DictReader(f)}

    evaluable, path_by_id = build_locked_evaluable_rows()
    print(f"Verifying pipeline_v5/ runtime artifact against {len(evaluable)} locked-test images "
          f"from the official evaluation record...")

    pipeline = load_frozen_pipeline()

    n_checked = 0
    n_match = 0
    n_mismatch = 0
    n_skipped_no_reference_score = 0
    mismatches = []
    max_abs_diff = 0.0

    for i, row in enumerate(evaluable):
        iid = ml.image_id_from_name(row["image_name"])
        ref = reference.get(iid)
        if ref is None:
            continue
        ref_proba_str = ref.get("v5_proba", "")
        if ref_proba_str == "":
            n_skipped_no_reference_score += 1
            continue

        proba = score_one_via_frozen_runtime(row, path_by_id[iid], pipeline)
        n_checked += 1
        if proba is None:
            mismatches.append((iid, "runtime produced no score (missing feature) but reference had one"))
            n_mismatch += 1
            continue

        ref_proba = float(ref_proba_str)
        diff = abs(proba - ref_proba)
        max_abs_diff = max(max_abs_diff, diff)
        if diff < 1e-6:
            n_match += 1
        else:
            n_mismatch += 1
            mismatches.append((iid, f"runtime={proba:.6f} reference={ref_proba:.6f} diff={diff:.2e}"))

        if i % 100 == 0 or i == len(evaluable) - 1:
            print(f"  [{i+1}/{len(evaluable)}] checked={n_checked} match={n_match} mismatch={n_mismatch}")

    print(f"\nChecked: {n_checked}  Exact matches (<1e-6): {n_match}  Mismatches: {n_mismatch}  "
          f"Skipped (no reference score, e.g. NO_DETECTION-adjacent rows): {n_skipped_no_reference_score}")
    print(f"Max absolute probability difference across all checked images: {max_abs_diff:.2e}")
    if mismatches:
        print("\nFirst 10 mismatches:")
        for iid, detail in mismatches[:10]:
            print(f"  {iid}: {detail}")

    if n_mismatch == 0 and n_checked > 0:
        print("\nREPRODUCIBILITY CHECK PASSED: pipeline_v5/'s runtime artifact reproduces every "
              "checked locked-test decision score exactly.")
    else:
        print("\nREPRODUCIBILITY CHECK FAILED -- investigate pipeline_v5/feature_extraction.py "
              "before treating this artifact as equivalent to what was evaluated.")
        sys.exit(1)


if __name__ == "__main__":
    main()
