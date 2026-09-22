# pipeline_v5 — FROZEN, 2026-09-21

Experimental V5 candidate, approved as the final frozen model after
development-only feature engineering, development-only threshold
selection, and a one-time locked-test evaluation. See
`v5_model_metadata.json` for the full, structured spec (feature list and
order, threshold, preprocessing, YOLO dependency, dev/locked-test metrics).

## Files

- `frozen_model.pkl` — the fitted `StandardScaler -> LogisticRegression`
  pipeline (17 features), fit once on all 2,089 valid development rows.
  Byte-identical to `Evaluation_V5Candidate/v5_reference_model.pkl`. Never
  refit, including after locked-test results were seen.
- `feature_extraction.py` — computes the 11 new features V5 adds over V4's
  6, from the already-produced YOLO mask + hair-removed image + vignette
  circle_info + D_px. Unchanged math, relocated (not rewritten) from
  `Evaluation_FeatureEngineering/extract_features.py`.
- `decision_model.py` — `V5_CONFIG` (exact 17-feature order, threshold
  0.25), `featurize()`, `v5_predict_from_row()`. Mirrors
  `pipeline_v4/decision_model.py`'s structure exactly.
- `v5_model_metadata.json` — structured documentation of everything above,
  plus dev/locked-test metrics and explicit provenance.
- `verify_frozen_v5.py` — reproducibility check: confirms this runtime
  artifact reproduces the exact per-image decision scores recorded in
  `Evaluation_V5Candidate/LockedTest/locked_test_per_image.csv`, the file
  that backed the actual KEEP-V4-vs-FREEZE-V5 decision.

## What this is NOT

- Not wired into `MelanomaDetection/MelanomaDetection.Python/` or any
  Flask/Blazor path yet. Integration is a separate, explicitly-approved
  step (see `Evaluation_V5Candidate/` for the phased evaluation history).
- Not a change to `pipeline_v4/`, which remains untouched and available.
- Not a change to YOLO, segmentation, or any preprocessing step — V5 reuses
  every one of those exactly as V4 already does.
- Not a melanoma/cancer probability — the model's output is a decision
  score, described that way everywhere in this codebase's documentation.

## Provenance trail

1. `Evaluation_FeatureEngineering/` — candidate feature investigation,
   development-only ablation, KEEP/WARRANTED recommendation.
2. `Evaluation_V5Candidate/` — V5 build, development-only threshold
   selection (`v5_selected_threshold.json`), development comparison vs.
   frozen V4.
3. `Evaluation_V5Candidate/LockedTest/` — the one-time locked-test
   evaluation and FREEZE-V5-AS-FINAL conclusion.
4. `pipeline_v5/` (this folder) — the resulting frozen runtime artifact.
