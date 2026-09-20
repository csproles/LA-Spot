"""FROZEN V4 decision layer.

STATUS: FROZEN 2026-09-20, based ENTIRELY on DEVELOPMENT/cross-validation
evidence (Evaluation_V4/CrossValidation/, ThresholdSelection/). Not
modified after this freeze, including after locked-test results are seen.

SELECTED MODEL: Logistic Regression (scikit-learn), on 6 features —
A_value, B_circularity, C_value, D_px, confidence, lesion_fraction — all
already produced by the unchanged V2 ABCD extraction
(revised_abcd/pipeline_v2.py). No new measurements, no deep learning, no
YOLO/segmentation change.

WHY THIS FEATURE SET AND MODEL (5-fold, patient/lesion-grouped CV on
development, n=2090 evaluable):

  Feature set                    Model                   OOF AUC   BalAcc@0.5
  core ABC                       logistic regression     0.712     0.603
  core ABC                       shallow tree (depth 3)  0.687     0.611
  core ABC + D_px                logistic regression     0.729     0.610
  core ABC + confidence          logistic regression     0.711     0.599
  full (ABC+D+conf+fraction)     logistic regression     0.779     0.652   <- selected
  full (ABC+D+conf+fraction)     shallow tree (depth 3)  0.747     0.654

Confidence alone added essentially nothing (0.712 -> 0.711 AUC), matching
the earlier finding that YOLO confidence isn't meaningfully associated with
TP/FN. D_px alone added a small amount (0.712 -> 0.729). Only the FULL
combination (adding mask_area_fraction alongside D_px and confidence)
produced a genuinely higher AUC (0.779) — a threshold-independent measure,
so this is evidence of real added discrimination, not a threshold artifact.
Logistic regression was preferred over the shallow tree at equal-ish
balanced accuracy for its higher AUC (better ranking quality) and because
it was the required primary candidate.

OPERATING THRESHOLD: 0.25 (not the default 0.50), selected from a 7-point
sweep (0.20-0.50) over the SAME cross-validation out-of-fold probabilities,
using the rule "maximize balanced accuracy among thresholds where both
sensitivity and specificity are >=0.55" (this floor rules out a threshold
that reaches high sensitivity only by flagging nearly everything, or the
mirror-image failure). Result at t=0.25: sensitivity=0.745,
specificity=0.691, balanced_accuracy=0.718, F1=0.609 — the same underlying
classifier improved BOTH sensitivity and specificity over the frozen V3
rule simultaneously (V3 dev: sens=0.612, spec=0.617), which a simple
threshold shift on V3's own decision layer could not do (V3's own A-
threshold sweep showed sensitivity and specificity trading off ~1-for-1
with flat balanced accuracy).

The locked test (run once, after this freeze, never revisited) is the real
test of whether this holds up out of sample.
"""

import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(__file__).resolve().parent / "frozen_model.pkl"

V4_CONFIG = {
    "version": "v4-logreg-full-features",
    "method": "logistic_regression",
    "features": ["A_value", "B_circularity", "C_value", "D_px", "confidence", "lesion_fraction"],
    "operating_threshold": 0.25,
}


def featurize(row):
    """row: dict with the raw pipeline_v2 field names (as in results.csv /
    revised_abcd.pipeline_v2.process_image's per-instance output)."""
    return np.array([[
        row["A_value"], row["B_circularity"], row["C_value"],
        row["D_px"], row["confidence"], row["lesion_fraction"],
    ]], dtype=float)


def load_frozen_pipeline():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def v4_predict_from_row(row, pipeline=None):
    """Returns (elevated: bool, probability: float). `row` must have all of
    V4_CONFIG['features'] populated (i.e. status==PROCESSED,
    evaluation_status usable) — callers should not call this for
    NO_DETECTION/FAILED rows."""
    pipeline = pipeline or load_frozen_pipeline()
    X = featurize(row)
    proba = float(pipeline.predict_proba(X)[0, 1])
    elevated = proba >= V4_CONFIG["operating_threshold"]
    return elevated, proba
