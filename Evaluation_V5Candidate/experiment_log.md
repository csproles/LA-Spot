# V5 experimental candidate: build, threshold selection, and development comparison vs. frozen V4

**Frozen V4 (`pipeline_v4/`) was not modified, retrained, or touched.** YOLO
was not touched. `user-shree` and the web app were not touched. The locked
test set was not read anywhere in this work. No feature search was
performed -- the feature set is imported unchanged from
`Evaluation_FeatureEngineering/run_ablation_cv.py`'s `BEST_COMBO_FEATURES`.
V5 is not wired into any production path; `v5_reference_model.pkl` exists
only for coefficient inspection.

## 1. V5 feature set (unchanged from the prior ablation study)

17 features = V4's 6 (`A_value`, `B_circularity`, `C_value`, `D_px`,
`confidence`, `lesion_fraction`) + 11 new: `color_entropy`, `lab_a_std`,
`lab_b_std`, `red_fraction`, `bluegray_fraction`, `dark_fraction`,
`skin_contrast`, `solidity`, `turning_angle_std`, `eccentricity`,
`D_px_normalized`.

CV recipe: identical to `Evaluation_V4/build_feature_table_and_cv.py` and
the prior ablation study -- patient/lesion-grouped `GroupKFold(n_splits=5)`,
`StandardScaler -> LogisticRegression(max_iter=1000, random_state=20260918)`
fit fresh inside each fold, on the same 2,090 `SINGLE_LESION_EVALUABLE`
development images (2,089 after dropping 1 row missing `skin_contrast`).

**Reproducibility check** (`build_v5_candidate.py`): re-running this exact
recipe reproduced the prior ablation study's numbers to 4 decimal places --
ROC-AUC 0.8008, sensitivity 0.7712, specificity 0.6733, balanced accuracy
0.7222, all exact matches. This is expected (same code/data/seed) and
confirms nothing drifted between the two experiments.

## 2. Development-only threshold selection

Same grid and rule as `Evaluation_V4/select_threshold.py`: sweep {0.20,
0.25, 0.30, 0.35, 0.40, 0.45, 0.50} on V5's own grouped-OOF probabilities
(never a same-sample-trained model), pick the highest-balanced-accuracy
point among thresholds with sensitivity >= 0.55 AND specificity >= 0.55.

| Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced Accuracy |
|---|---|---|---|---|---|---|
| 0.20 | 0.8323 | 0.5892 | 0.4712 | 0.6017 | 0.6635 | 0.7108 |
| **0.25** | **0.7712** | **0.6733** | 0.5093 | 0.6135 | 0.7032 | **0.7222** |
| 0.30 | 0.7022 | 0.7422 | 0.5450 | 0.6137 | 0.7300 | **0.7222** |
| 0.35 | 0.6442 | 0.8001 | 0.5863 | 0.6139 | 0.7525 | **0.7222** |
| 0.40 | 0.5862 | 0.8429 | 0.6213 | 0.6032 | 0.7645 | 0.7145 |
| 0.45 | 0.5376 | 0.8773 | 0.6583 | 0.5919 | 0.7736 | 0.7075 |
| 0.50 | 0.4812 | 0.9035 | 0.6868 | 0.5659 | 0.7745 | 0.6924 |

**Important, disclosed transparently rather than papered over**: 0.25, 0.30,
and 0.35 are tied at balanced accuracy 0.7222 to 4 decimal places -- this is
a plateau, not a single unambiguous peak. The selection rule (identical to
V4's own script) takes the first qualifying maximum by threshold order,
which lands on **0.25**. That this exactly matches V4's own threshold is a
genuine coincidence of the sweep, not something engineered -- but the choice
among 0.25/0.30/0.35 is a real sensitivity-vs-specificity value judgment
(0.25 favors sensitivity 0.771/specificity 0.673; 0.35 favors specificity
0.800/sensitivity 0.644) that the balanced-accuracy criterion alone does not
resolve. No threshold was hand-picked to favor a particular result; the same
mechanical rule V4 used was applied as-is.

**Selected V5 threshold: 0.25.**

## 3. V5 vs. frozen V4, development comparison (common 2,089-image OOF population)

| Model | Threshold | Sensitivity | Specificity | Precision | F1 | Accuracy | Balanced Accuracy | ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Frozen V4 | 0.25 | 0.7445 | 0.6912 | 0.5146 | 0.6086 | 0.7075 | 0.7179 | 0.7784 |
| **V5 Candidate** | 0.25 | **0.7712** | 0.6733 | 0.5093 | 0.6135 | 0.7032 | **0.7222** | **0.8008** |

Paired shift (same 2,089 images, V4 at 0.25 vs. V5 at 0.25):

| | Count |
|---|---|
| Melanoma false negatives rescued by V5 | 53 |
| New melanoma false negatives introduced by V5 | 36 |
| V4 false positives corrected by V5 | 89 |
| New false positives introduced by V5 | 115 |
| **Net change** | TP +17, TN -26, FP +26, FN -17 |

Fold-by-fold (V5, from the CV run itself; V4's own fold numbers from the
prior ablation study for direct comparison):

| Fold | V5 AUC | V5 Sens@0.25 | V5 Spec@0.25 | V5 BalAcc@0.25 | V4 AUC | V4 Sens@0.25 | V4 Spec@0.25 | V4 BalAcc@0.25 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.8087 | 0.7368 | 0.7018 | 0.7193 | 0.752 | 0.634 | 0.704 | 0.669 |
| 1 | 0.7993 | 0.7656 | 0.6586 | 0.7121 | 0.831 | 0.844 | 0.703 | 0.774 |
| 2 | 0.8084 | 0.8065 | 0.6565 | 0.7315 | 0.775 | 0.776 | 0.672 | 0.724 |
| 3 | 0.7840 | 0.7717 | 0.6701 | 0.7209 | 0.790 | 0.764 | 0.671 | 0.718 |
| 4 | 0.8061 | 0.7778 | 0.6804 | 0.7291 | 0.756 | 0.719 | 0.707 | 0.713 |
| **Range** | **0.784-0.809** | | | **0.712-0.732** | **0.752-0.831** | | | **0.669-0.774** |

V5's AUC and balanced-accuracy are both **less variable fold-to-fold** than
V4's -- V4 swings from 0.669 to 0.774 balanced accuracy depending on the
fold (a 0.105 spread), while V5 stays within 0.712-0.732 (a 0.020 spread).
This mirrors what the earlier ablation study found and is not new, but it
is now confirmed at V5's own properly-selected threshold, not V4's.

## 4. Complexity check: exact V5 coefficients

`v5_reference_model.pkl` fit on all 2,089 valid development rows (for
inspection only -- not the source of any number above, which all come from
out-of-fold CV predictions):

| Rank | Feature | Standardized coefficient |
|---|---|---|
| 1 | solidity | -0.728 |
| 2 | D_px_normalized | +0.720 |
| 3 | B_circularity | -0.621 |
| 4 | A_value | +0.258 |
| 5 | skin_contrast | -0.246 |
| 6 | lab_b_std | +0.243 |
| 7 | lab_a_std | +0.230 |
| 8 | C_value | +0.167 |
| 9 | red_fraction | +0.157 |
| 10 | color_entropy | +0.149 |
| 11 | D_px | +0.143 |
| 12 | eccentricity | -0.100 |
| 13 | turning_angle_std | +0.094 |
| 14 | lesion_fraction | +0.062 |
| 15 | confidence | -0.059 |
| 16 | dark_fraction | +0.028 |
| 17 | bluegray_fraction | +0.004 |
| | intercept | -1.048 |

**Read this with one important caveat.** `B_circularity`'s coefficient is
negative here, even though higher border irregularity is univariately
associated with melanoma (its own standalone AUC was 0.683, in the expected
direction). This is a standard multicollinearity/suppression effect: `solidity`
and `B_circularity` both measure boundary irregularity and are correlated
with each other, so in the *joint* model one absorbs the shared signal while
the other's coefficient sign flips. The two should be read together as "the
model relies heavily on boundary-shape irregularity," not interpreted as
"circularity works backwards." The three largest-magnitude features --
`solidity`, `D_px_normalized`, `B_circularity` -- are a border-shape measure,
a scale-normalized diameter, and the existing border measure; color features
contribute real but individually smaller weight (`skin_contrast`,
`lab_b_std`, `lab_a_std`, `red_fraction`, `color_entropy`, ranks 5-10).

**Is 6 -> 17 features justified?** The three dominant coefficients by
magnitude come from exactly the two feature groups (border, geometry) that
the prior ablation identified as carrying real independent signal; the
weaker-but-still-present color-feature coefficients corroborate that group's
own real (CI-supported) contribution. No single new feature dominates by
accident, and no near-zero-variance or redundant feature survived into this
final list (all were pruned before this stage). This is not "17 features for
a coin-flip effect" -- but it is also not a dramatic jump: ROC-AUC moves from
0.778 to 0.801, balanced accuracy from 0.718 to 0.722.

## 5. Decision gate

Per the criteria set for this investigation (meaningful, reasonably
consistent development improvement, evaluated at V5's *own* properly
selected threshold, not V4's):

- **Consistency**: yes, and now confirmed at V5's own threshold, not
  borrowed from V4 -- fold-to-fold balanced accuracy is markedly more stable
  for V5 (0.712-0.732) than V4 (0.669-0.774).
- **Sensitivity vs. specificity**: sensitivity improves by a real, material
  margin (+2.7 points, 53 melanomas rescued net 17 more than introduced)
  without a specificity collapse (-1.8 points, a proportionate trade, not a
  cliff).
- **ROC-AUC**: improves from 0.778 to 0.801, consistent with the
  bootstrap-confirmed real gain (95% CI [0.010, 0.035], from the prior
  ablation study) -- not a fluctuation.
- **Balanced accuracy**: improves from 0.718 to 0.722 at the shared 0.25
  threshold, and holds at exactly 0.722 across the entire 0.25-0.35 plateau
  -- a small but *consistent* (not threshold-cherry-picked) gain, though the
  prior ablation's own paired bootstrap CI on this specific metric alone
  straddled zero. This residual uncertainty is disclosed, not hidden.
- **Complexity**: justified by the coefficient analysis above -- the
  largest-magnitude features track directly back to the two groups shown to
  carry real independent signal.

## Conclusion: V5 READY FOR ONE-TIME LOCKED TEST

Development evidence is consistent across every angle this investigation
checked -- fold stability, a real sensitivity gain without a specificity
collapse, a statistically supported ROC-AUC improvement, and a small but
non-cherry-picked balanced-accuracy gain that holds across a plateau of
threshold choices, not one lucky point. This clears the bar of "meaningful
and reasonably consistent," not a trivial metric fluctuation. The one
honestly-disclosed soft spot is that balanced accuracy's own paired
confidence interval isn't independently conclusive -- which is exactly what
a locked-test check is for.

**This report stops here. The locked test set has not been read or used
anywhere in this work, and V5 will not be run against it without your
explicit approval.** If approved, that would be a single, one-time run
(mirroring V4's own Phase 7 protocol) -- not a repeated or iterated
evaluation.

## Files

- `build_v5_candidate.py` -- builds V5's OOF predictions (reused, unmodified
  ablation code) and fits `v5_reference_model.pkl` for coefficient
  inspection only.
- `select_v5_threshold.py` -- development-only threshold sweep/selection.
- `compare_v5_vs_v4.py` -- the full comparison in this log.
- `v5_oof_predictions.json`, `v5_threshold_sweep.json`,
  `v5_selected_threshold.json`, `v5_vs_v4_comparison.json`,
  `v5_coefficients.csv`, `v5_reference_model.pkl` -- outputs.
