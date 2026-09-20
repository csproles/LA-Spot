"""FROZEN V3 decision layer.

STATUS: FROZEN 2026-09-20, based on DEVELOPMENT-set-only analysis
(Evaluation_FinalTargeted/ThresholdAnalysis/, DecisionModelAnalysis/). Not
touched again after this freeze, including after locked-test results were
seen (Phase 7).

SELECTED: Candidate B — the existing binary concern-count rule, unchanged
in every respect except the A (asymmetry) concern threshold, lowered from
0.20 to 0.14. B's threshold (0.50), C's scoring, the "2-of-3 concerns"
rule, and the critical-color override are ALL unchanged from V2.

WHY 0.14, AND WHY THIS CANDIDATE OVER THE ALTERNATIVES (development set,
n=2090 evaluable, targeted melanoma-vs-benign-melanocytic cohort):

  Candidate                          Sens    Spec    BalAcc   F1
  A: current (A>0.20)                0.410   0.815   0.612    0.448
  B: revised A>0.14  <- SELECTED     0.612   0.617   0.614    0.493
  C: continuous score, cutoff=2.0    0.901   0.292   0.597    0.514
  C: continuous score, cutoff=1.8    0.962   0.170   0.566    0.500
  C: continuous score, cutoff=1.6    0.986   0.076   0.531    0.483

- The full A-threshold sweep (0.20 down to 0.08) showed balanced accuracy
  essentially FLAT (0.610-0.615) across the entire range — moving the
  threshold trades sensitivity for specificity roughly 1-for-1 along the
  same underlying ROC-like curve; it is not a "free" improvement, and 0.14
  sits at the sweep's balanced-accuracy/F1 peak, not an arbitrary pick.
- The continuous ABC score (a simple, interpretable normalized sum of
  A/B/C relative to their own thresholds — see the version-controlled
  history of this file / phase5_decision_layer.py for its exact
  definition) was tested at three cutoffs. All three post nominally higher
  raw sensitivity and, at cutoff 2.0, even a higher F1 than candidate B —
  but every one drives specificity down to 7.6-29.2%, i.e. flags most
  clearly-benign nevi as elevated concern. This is exactly the
  "maximizing sensitivity by letting specificity collapse" outcome the
  project rules explicitly rule out, regardless of F1. It was not selected
  DESPITE the higher F1, precisely because F1 alone does not capture that
  failure mode.
- Candidate B keeps specificity at a still-usable 61.7% (vs. 81.5% before),
  a real and disclosed cost, in exchange for a genuine, large sensitivity
  gain (+20.2 points) directly targeting the melanoma-FN problem this
  whole investigation was about, at no meaningful net change in balanced
  accuracy.
- Candidate B is also the simplest possible change (one number in one
  existing threshold) — fully interpretable and easy to explain in a
  presentation, versus a bespoke weighted-sum formula for a
  non-superior-on-balance result.

Does not depend on or modify revised_abcd/revised_asymmetry.py,
Code/MelanomaDeterminingStuff/*, or any YOLO/segmentation code — this file
ONLY consumes already-computed A/B/C values and decides elevated/lower.
"""

V3_CONFIG = {
    "version": "v3-A-threshold-0.14",
    "method": "binary_concern_count",
    "a_threshold": 0.14,  # FROZEN — was 0.20 in V2; see selection rationale above
    "b_threshold": 0.50,  # UNCHANGED from V2 — informational only, B's concern flag is precomputed
    "c_note": "C concern is taken as already computed by the unchanged existing score_color logic",
    "concerns_required_for_elevated": 2,  # UNCHANGED from V2
}


def v3_predict_from_features(a_value, a_concern, b_value, b_concern, c_value, c_concern):
    """True (elevated) / False (lower). Takes raw values + already-computed
    concern booleans for A/B/C (the concern booleans are unchanged from the
    existing scoring code for B and C in every version considered)."""
    if a_value is None:
        a_flag = bool(a_concern)
    else:
        a_flag = a_value > V3_CONFIG["a_threshold"]
    concerns = sum([a_flag, bool(b_concern), bool(c_concern)])
    return concerns >= V3_CONFIG["concerns_required_for_elevated"]


def v3_predict_from_row(row, override_fired=False):
    """Convenience wrapper for a results.csv-style row (as used by the
    Phase 7 locked-test evaluation), preserving the inferred critical-color
    override exactly as V2 would have applied it."""
    if override_fired:
        return 1
    a_value = row["A_value"]
    a_value = float(a_value) if a_value not in (None, "") else None
    elevated = v3_predict_from_features(
        a_value=a_value, a_concern=row["A_concern"] == "True",
        b_value=row["B_circularity"], b_concern=row["B_circularity_concern"] == "True",
        c_value=row["C_value"], c_concern=row["C_concern"] == "True",
    )
    return 1 if elevated else 0
