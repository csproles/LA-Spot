# V5 vs. frozen V4: one-time locked-test evaluation

**Run exactly once. No feature, coefficient, threshold, preprocessing, YOLO,
or ABCD-calculation change was made before, during, or after this run.** No
case was excluded based on any result. V4 (`pipeline_v4/`), YOLO, and
`user-shree` were not touched. `Evaluation_V4/LockedTest/` (V4's own
existing locked-test results) was read but not modified. This evaluation
is preserved separately in this folder.

## What was used, exactly

- **V4**: the exact, already-frozen `pipeline_v4/frozen_model.pkl` (vendored
  read-only copy, extracted via `git show user-shree:pipeline_v4/
  frozen_model.pkl`, never refit), its existing 6 features, threshold 0.25.
- **V5**: the exact, already-fitted `Evaluation_V5Candidate/
  v5_reference_model.pkl` (fit on all 2,089 development rows in the prior
  step, never refit here), its existing 17 features (unchanged from the
  ablation study -- no new features, no further search), development-
  selected threshold 0.25 (unchanged).
- **Locked-test population**: identical to `Evaluation_V4/
  freeze_and_locked_test.py`'s own -- the 615 `SINGLE_LESION_EVALUABLE`
  images among the 786-image locked test set. Coverage counts
  (615/118/53/0 single/multi/no-detection/failed) match exactly.
- The only new computation performed: extracting V5's 11 new features for
  these 615 locked-test images, for the first time, using the exact,
  unmodified `Evaluation_FeatureEngineering/extract_features.py::process_one`
  function that was already finalized before this run and before any
  locked-test result existed. Zero computation errors, zero missing
  feature values -- no locked-test image needed to be excluded on data
  grounds.

**Correctness gate, checked before any V5 result was computed**: re-scoring
V4's frozen model on these 615 images reproduced the existing published
locked-test result (`Evaluation_V4/LockedTest/locked_test_v2_v3_v4.json`)
exactly -- TP=95, TN=336, FP=141, FN=43, sensitivity and specificity exact
to 13 decimal places, ROC-AUC 0.7484. The script was written to hard-stop
if this check failed, precisely so no V5 result could be seen or reasoned
about before confirming the harness itself was correct.

**Identical evaluable population confirmed**: all 615 locked-test images
had valid V5 features (0 dropped), so the paired comparison uses the exact
same 615 images for both models, with no divergence.

## Results

| Model | Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Frozen V4 | 0.25 | 0.6884 | 0.7044 | 0.4025 | 0.5080 | 0.7008 | 0.6964 | 0.7484 |
| **V5 Candidate** | 0.25 | **0.7101** | **0.7128** | **0.4170** | **0.5255** | **0.7122** | **0.7115** | **0.7981** |

Confusion matrices (n=615 both):

| | V4 | V5 |
|---|---|---|
| TP | 95 | 98 |
| TN | 336 | 340 |
| FP | 141 | 137 |
| FN | 43 | 40 |

**V5 improves on every single count** -- more true positives, more true
negatives, fewer false positives, fewer false negatives. This is not a
sensitivity-for-specificity trade (which is what development showed); on
the locked test, both sensitivity and specificity improve together.

Paired shift (same 615 images):

| | Count |
|---|---|
| Melanoma false negatives rescued by V5 | 16 |
| New melanoma false negatives introduced | 13 |
| V4 false positives corrected by V5 | 37 |
| New false positives introduced | 33 |
| **Net change** | TP +3, TN +4, FP -4, FN -3 |

Paired ROC-AUC difference (V5-V4): **+0.0498**, bootstrap 95% CI **[0.0218,
0.0791]** -- entirely above zero, and a *larger* margin than development
showed (+0.023 there), not a smaller one. No sign of the development
advantage being a dev-only artifact.

Paired balanced-accuracy difference: **+0.0151**, bootstrap 95% CI
**[-0.027, 0.0557]**. As in development, this specific interval still
touches zero -- expected given the locked test has less than a third as
many evaluable images (615 vs. 2,089), so any paired-difference CI is
mechanically wider here. The point estimate is positive and larger than
development's own (+0.0151 vs. +0.0044), and it is now the second
independent dataset (development CV, and now locked test) both pointing the
same direction with a dominant confusion-matrix improvement, which is
stronger evidence than either alone.

## Complexity check, revisited with locked-test evidence

V5 nearly triples V4's feature count (6 -> 17). On development alone, that
increase bought a real but modest AUC gain with a sensitivity/specificity
trade. On the untouched locked test, the same fixed model and threshold
delivered a *larger* AUC gain and improved *every* confusion-matrix count
simultaneously -- the complexity is not just development-fitted noise that
evaporated on new data, which is exactly the failure mode a locked-test
check exists to catch. That it didn't happen here is the strongest
available justification for the added complexity.

## Conclusion: FREEZE V5 AS FINAL

The development-observed improvement not only generalized to the untouched
locked test, it strengthened: a larger, still bootstrap-significant ROC-AUC
gain, and a clean dominance across TP/TN/FP/FN rather than the
sensitivity-for-specificity trade seen in development. This is not a
marginal or fragile result being talked up -- every count moved in V5's
favor. Combined with the earlier development evidence (fold-consistency,
coefficient analysis tracing the gain to the border and geometry feature
groups with real independent signal), this clears the bar set for freezing
V5: a meaningful, reasonably consistent improvement across two independent
evaluations, not a trivial fluctuation.

**This report stops here, per instructions.** V5 has not been wired into
any production path, the web application has not been modified, and no
further changes have been made to V5 based on these results (the only
post-hoc code change in this entire step was fixing a rounding-tolerance
bug in my own verification check, made *before* any V5 number existed).
Freezing V5 as the production model -- updating `pipeline_v4/` or its
successor, retraining nothing, changing the web app -- is a separate action
requiring your explicit go-ahead.

## Files

- `run_locked_test_evaluation.py` -- the one-time evaluation script.
- `locked_test_v5_vs_v4.json` -- full results (this log's numbers).
- `locked_test_per_image.csv` -- per-image V4/V5 probabilities and
  predictions, for audit.
- `_vendored_v4_frozen_model.pkl` -- read-only copy of V4's frozen model
  (see `README.md`).
- `Evaluation_V4/LockedTest/locked_test_v2_v3_v4.json` -- V4's own existing
  locked-test result, read but **not modified**, preserved exactly as it
  was.
