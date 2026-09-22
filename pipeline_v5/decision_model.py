"""FROZEN V5 decision layer.

STATUS: FROZEN 2026-09-21, based on Evaluation_FeatureEngineering/'s
ablation study (development-only) and Evaluation_V5Candidate/'s one-time
locked-test evaluation. Not modified after this freeze.

V4 (pipeline_v4/) is untouched by this file's existence. V5 is an
additional, separate candidate; nothing here overwrites or is imported by
pipeline_v4/.

SELECTED MODEL: Logistic Regression (scikit-learn), on 17 features — V4's
existing 6 (A_value, B_circularity, C_value, D_px, confidence,
lesion_fraction) plus 11 new ones identified in the color+border+geometry
ablation (Evaluation_FeatureEngineering/run_ablation_cv.py's
BEST_COMBO_FEATURES, unchanged, no further feature search performed):
color_entropy, lab_a_std, lab_b_std, red_fraction, bluegray_fraction,
dark_fraction, skin_contrast, solidity, turning_angle_std, eccentricity,
D_px_normalized.

DEVELOPMENT EVIDENCE (5-fold, patient/lesion-grouped CV, n=2089 evaluable,
same recipe as pipeline_v4/decision_model.py): OOF ROC-AUC 0.8008 vs. V4's
0.7787; balanced accuracy 0.7222 vs. V4's 0.7181 at threshold 0.25.
Bootstrap 95% CI for the paired AUC difference: [0.0097, 0.0353] (real, not
noise). See Evaluation_FeatureEngineering/experiment_log.md.

LOCKED-TEST EVIDENCE (run once, 2026-09-21, n=615 evaluable): V5 improved
EVERY confusion-matrix count over V4 (TP 98 vs 95, TN 340 vs 336, FP 137 vs
141, FN 40 vs 43). Paired ROC-AUC difference +0.0498, bootstrap 95% CI
[0.0218, 0.0791] — larger than the development-observed gain, not smaller.
See Evaluation_V5Candidate/LockedTest/experiment_log.md.

OPERATING THRESHOLD: 0.25 — selected on development-only grouped-OOF
probabilities using the SAME rule as V4 (maximize balanced accuracy among
thresholds with sensitivity>=0.55 AND specificity>=0.55; see
Evaluation_V5Candidate/experiment_log.md section 2). Not re-tuned after
seeing the locked-test result; it happens to equal V4's own threshold,
which is a coincidence of the development sweep, not something chosen to
match V4.

The score this model produces is a MODEL DECISION SCORE (a calibrated-ish
logistic-regression probability of the "melanoma" training label), not a
clinical melanoma/cancer probability. This remains a research/prototype
visual-concern system, not a diagnostic tool.
"""

import pickle
from pathlib import Path

import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "frozen_model.pkl"

V5_CONFIG = {
    "version": "v5-logreg-color-border-geometry",
    "method": "logistic_regression",
    "base_features": ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"],
    "new_features": ["color_entropy", "lab_a_std", "lab_b_std", "red_fraction", "bluegray_fraction",
                      "dark_fraction", "skin_contrast", "solidity", "turning_angle_std", "eccentricity",
                      "D_px_normalized"],
    "operating_threshold": 0.25,
}
# Exact, ordered feature list the frozen pipeline was fit on. Order matters:
# StandardScaler's fitted mean_/scale_ are positional, not name-keyed.
V5_CONFIG["features"] = V5_CONFIG["base_features"] + V5_CONFIG["new_features"]


def featurize(row):
    """row: dict containing V4's 6 existing fields (as already produced by
    revised_abcd/pipeline_v2.py) PLUS the 11 new fields produced by
    feature_extraction.extract_v5_new_features(). Raises KeyError if any
    required feature is missing -- callers should not call this for
    NO_DETECTION/FAILED rows, exactly as with V4."""
    return np.array([[row[f] for f in V5_CONFIG["features"]]], dtype=float)


def load_frozen_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def v5_predict_from_row(row, pipeline=None):
    """Returns (elevated: bool, probability: float). `probability` is a
    model decision score, not a melanoma/cancer probability."""
    pipeline = pipeline or load_frozen_pipeline()
    X = featurize(row)
    proba = float(pipeline.predict_proba(X)[0, 1])
    elevated = proba >= V5_CONFIG["operating_threshold"]
    return elevated, proba
