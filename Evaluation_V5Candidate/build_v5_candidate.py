"""
Build the experimental V5 candidate: frozen V4's 6 features plus the exact
11-feature "best combined" (color + border + geometry) set identified in
Evaluation_FeatureEngineering/'s ablation study. NO new features, NO further
search -- the feature list is imported unchanged from that prior experiment.

Frozen V4 (pipeline_v4/) is not touched. YOLO is not touched. This script
only reads V4's existing cached OOF (Evaluation_V4/CrossValidation/
oof_predictions_full.json) for later comparison; it never refits or
modifies it.

Reuses the EXACT same CV recipe as both V4's own build_feature_table_and_cv.py
and the prior feature-engineering ablation: patient/lesion GroupKFold(5),
StandardScaler -> LogisticRegression(max_iter=1000, random_state=20260918)
fit fresh inside each fold. Locked test set is never read.

Outputs:
  - v5_oof_predictions.json      (OOF probabilities + y_true, for threshold
                                   selection and comparison)
  - v5_reference_model.pkl        (Pipeline fit on ALL 2,090 dev rows, purely
                                   for coefficient inspection -- NOT used for
                                   any of the CV/threshold numbers below, and
                                   NOT wired into any production path)
  - v5_coefficients.csv           (feature, standardized coefficient, |coef|
                                    rank)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

V5_DIR = Path(__file__).resolve().parent
ROOT = V5_DIR.parent
FEATENG_DIR = ROOT / "Evaluation_FeatureEngineering"
sys.path.insert(0, str(FEATENG_DIR))

from run_ablation_cv import (  # noqa: E402  -- exact, unmodified reuse of the prior experiment's code
    load_table, run_group_cv, BASE_FEATURES, BEST_COMBO_FEATURES, SEED,
)

# The exact, frozen V5 feature set -- imported, not redefined, so it cannot
# silently drift from what the ablation study actually tested.
V5_NEW_FEATURES = list(dict.fromkeys(BEST_COMBO_FEATURES))
V5_ALL_FEATURES = BASE_FEATURES + V5_NEW_FEATURES

EXPECTED_FROM_ABLATION = {
    "roc_auc": 0.8008, "sensitivity": 0.7712, "specificity": 0.6733, "balanced_accuracy": 0.7222,
}


def main():
    print(f"V5 feature set ({len(V5_ALL_FEATURES)} features): {V5_ALL_FEATURES}")
    rows = load_table()
    result = run_group_cv(rows, V5_NEW_FEATURES)

    m = result["overall_metrics"]
    print(f"\nV5 OOF (re-run, same code/data/seed as the ablation study):")
    print(f"  n_used={result['n_used']}  n_dropped_missing={result['n_dropped_missing']}")
    print(f"  ROC-AUC={result['overall_auc']}  sens={m['sensitivity']:.4f}  spec={m['specificity']:.4f}  "
          f"bal_acc={m['balanced_accuracy']:.4f}")

    # Sanity check: this MUST reproduce the ablation study's reported numbers
    # exactly, since it is the same code, data, and seed -- if it doesn't,
    # something about the feature list or data has drifted and that must be
    # investigated before proceeding, not silently accepted.
    checks = {
        "roc_auc": result["overall_auc"],
        "sensitivity": round(m["sensitivity"], 4),
        "specificity": round(m["specificity"], 4),
        "balanced_accuracy": round(m["balanced_accuracy"], 4),
    }
    for k, expected in EXPECTED_FROM_ABLATION.items():
        actual = checks[k]
        status = "OK" if abs(actual - expected) < 0.0005 else "MISMATCH"
        print(f"  reproducibility check [{k}]: expected={expected} actual={actual} -> {status}")
        if status == "MISMATCH":
            raise RuntimeError(f"V5 CV result does not reproduce the prior ablation study for {k}: "
                                f"expected {expected}, got {actual}")

    # ---- save OOF predictions (for threshold selection + comparison) ----
    oof_ids = sorted(result["oof_by_id"].keys())
    payload = {
        "feature_set": V5_ALL_FEATURES,
        "n_rows_used": result["n_used"],
        "n_dropped_missing": result["n_dropped_missing"],
        "oof_roc_auc": result["overall_auc"],
        "image_ids": oof_ids,
        "oof_probabilities": [result["oof_by_id"][i] for i in oof_ids],
        "y_true": [result["y_by_id"][i] for i in oof_ids],
        "fold_records": result["fold_records"],
    }
    with open(V5_DIR / "v5_oof_predictions.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote v5_oof_predictions.json ({len(oof_ids)} images)")

    # ---- reference model fit on ALL dev rows, coefficient inspection only ----
    field_rows = [r for r in rows if all(r[f] == r[f] for f in V5_ALL_FEATURES)]  # drop NaNs, same as CV
    X = np.array([[r[f] for f in V5_ALL_FEATURES] for r in field_rows], dtype=float)
    y = np.array([r["ground_truth_binary"] for r in field_rows])
    pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=SEED))])
    pipe.fit(X, y)

    with open(V5_DIR / "v5_reference_model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print(f"Wrote v5_reference_model.pkl (fit on all {len(field_rows)} dev rows; "
          f"for coefficient inspection ONLY -- not used for any CV/threshold number in this report, "
          f"not wired into any production path)")

    coefs = pipe.named_steps["clf"].coef_[0]
    order = np.argsort(-np.abs(coefs))
    with open(V5_DIR / "v5_coefficients.csv", "w", encoding="utf-8") as f:
        f.write("feature,standardized_coefficient,abs_rank\n")
        for rank, idx in enumerate(order, start=1):
            f.write(f"{V5_ALL_FEATURES[idx]},{coefs[idx]:.5f},{rank}\n")
        f.write(f"intercept,{pipe.named_steps['clf'].intercept_[0]:.5f},\n")
    print("Wrote v5_coefficients.csv")
    for rank, idx in enumerate(order, start=1):
        print(f"  #{rank}: {V5_ALL_FEATURES[idx]:20s} coef={coefs[idx]:+.4f}")

    return result, payload


if __name__ == "__main__":
    main()
