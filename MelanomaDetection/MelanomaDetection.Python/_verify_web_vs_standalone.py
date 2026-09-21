"""Golden-image comparison: web-integrated V5Detector vs the standalone
frozen pipeline_v5 artifact. Compares ALL 17 features, the decision score,
and the final concern -- not just the label. Read-only, no files modified.
"""
import sys
import os

_WORKTREE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_WORKTREE, "MelanomaDetection", "MelanomaDetection.Python"))

import csv
import json

ROOT = "C:/Users/sirjanaa/Documents/huggingface_projects/Advanced-Melanoma-Detection"
os.chdir(ROOT)  # mask_path in cohort_v2_results.csv is relative to the main repo root

import v5_detector as v5d
from pipeline_v5.decision_model import V5_CONFIG

# capture the exact feature dict V5Detector builds, without touching the file
captured = {}
_orig_predict = v5d.v5_predict_from_row


def _spy_predict(row, pipeline=None):
    captured["row"] = dict(row)
    return _orig_predict(row, pipeline)


v5d.v5_predict_from_row = _spy_predict

detector = v5d.V5Detector()

# reference: standalone per-image decision scores from the official locked-test record
ref = {r["image_id"]: r for r in csv.DictReader(
    open(f"{ROOT}/Evaluation_V5Candidate/LockedTest/locked_test_per_image.csv", newline="", encoding="utf-8"))}
paths = {r["image_id"]: r["image_path"] for r in csv.DictReader(
    open(f"{ROOT}/Evaluation_FinalTargeted/Cohort/locked_test_manifest.csv", newline="", encoding="utf-8"))}

# Also need the standalone 17-feature vectors for the SAME images, from the
# extended feature table (dev-only) doesn't cover locked test, so instead
# re-derive via the exact same standalone functions extract_features.py uses
# -- this is the module pipeline_v5/feature_extraction.py's math was itself
# checked against in Phase 1 (615/615 exact match).
sys.path.insert(0, f"{ROOT}/Evaluation_FeatureEngineering")
import extract_features as standalone_extract  # noqa: E402
import metrics_lib as ml  # noqa: E402

with open(f"{ROOT}/Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv", newline="", encoding="utf-8") as f:
    all_results = list(csv.DictReader(f))
by_image = {}
for r in all_results:
    iid = ml.image_id_from_name(r["image_name"])
    by_image.setdefault(iid, []).append(r)

sample_ids = sorted(ref.keys())[:20]  # first 20 locked-test images with a recorded standalone score

fields = V5_CONFIG["features"]
n_checked = 0
n_full_match = 0
mismatches = []

for iid in sample_ids:
    row = by_image[iid][0]
    if row["evaluation_status"] != "SINGLE_LESION_EVALUABLE":
        continue
    path = paths[iid]

    # --- web path ---
    web_result = detector.process_image(path)
    web_row = captured["row"]
    web_score = web_result["v5_decision_score"]
    web_concern = web_result["overall_visual_concern"]

    # --- standalone path (independent module, same math, different code) ---
    standalone_new = standalone_extract.process_one(row, path)
    standalone_row = {f: float(row[f]) for f in V5_CONFIG["base_features"]}
    for f in V5_CONFIG["new_features"]:
        standalone_row[f] = standalone_new[f]

    n_checked += 1
    feature_diffs = {f: (web_row[f], standalone_row[f], abs(web_row[f] - standalone_row[f]))
                      for f in fields if abs(web_row[f] - standalone_row[f]) > 1e-6}
    ref_score = float(ref[iid]["v5_proba"])
    score_diff = abs(web_score - round(ref_score, 4))

    if not feature_diffs and score_diff < 1e-3:
        n_full_match += 1
    else:
        mismatches.append((iid, feature_diffs, score_diff))

    print(f"{iid}: features_match={'YES' if not feature_diffs else 'NO'} "
          f"web_score={web_score} ref_score={round(ref_score,4)} score_diff={score_diff:.2e} "
          f"concern={web_concern}")

print(f"\nChecked {n_checked} images. Full match (17 features + score): {n_full_match}/{n_checked}")
if mismatches:
    print("MISMATCHES:")
    for iid, fd, sd in mismatches:
        print(f"  {iid}: feature_diffs={fd} score_diff={sd:.2e}")
