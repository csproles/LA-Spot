# Part 13 — Development-Only Experiment Plan

**Hard constraint, repeated from the audit instructions and honored throughout this plan: every
experiment below runs ONLY on the development cohort
(`Evaluation_FinalTargeted/Cohort/development_manifest.csv`, n=2089/2090 evaluable, the same population
`pipeline_v5`/`pipeline_v4` were fit on), using the same GroupKFold(5) patient/lesion-grouped
cross-validation recipe the original V4/V5 ablation studies used
(`Evaluation_FeatureEngineering/run_ablation_cv.py`, `Evaluation_V4/build_feature_table_and_cv.py`). The
locked test set (`Evaluation_FinalTargeted/Cohort/locked_test_manifest.csv`, n=786/615 evaluable) is
NEVER read, touched, or referenced by any of these experiments. If, after a full development-only
sweep, a candidate change looks genuinely justified, a *separate*, explicitly-approved, one-time
locked-test confirmation (mirroring exactly how V5 itself was confirmed) would be the only appropriate
next step — not performed here, not scoped here.**

Each experiment below corresponds to a recommendation in `V5_ABCD_Improvement_Recommendations.md`.

---

## Experiment 1 — Drop `B_circularity` / replace with a non-collinear border descriptor

- **Hypothesis**: `B_circularity`'s multivariate sign flip is a pure multicollinearity artifact (driven
  by its rho=1.00 relationship to the excluded `isoperimetric_ratio` and correlation with
  `solidity`/`turning_angle_std`), so removing it should not meaningfully hurt performance, and may even
  reduce noise in the fitted coefficients.
- **Feature being changed**: `B_circularity` (removed in arm B; replaced in arm C).
- **Baseline calculation**: current V5 17-feature vector, current frozen preprocessing.
- **New calculation**: Arm B = 16 features (drop `B_circularity`). Arm C = 17 features, but
  `B_circularity` replaced by a DFT-magnitude descriptor of the already-computed 100-point resampled
  contour turning-angle sequence (Part 11, B3) — low implementation cost since the resampled sequence
  already exists inside `turning_angle_std`'s computation.
- **Development dataset**: `development_manifest.csv`, GroupKFold(5), identical recipe to the original
  ablation study.
- **Validation dataset**: the same development cohort's out-of-fold (OOF) predictions — no separate
  held-out split beyond the existing grouped CV folds, matching the original V4/V5 methodology exactly.
- **Metric to compare**: OOF ROC-AUC (primary, matches how V5 itself was justified over V4), balanced
  accuracy at a freshly-swept threshold (not V5's existing 0.25 — re-sweep per arm, same selection rule:
  maximize balanced accuracy subject to sensitivity≥0.55 and specificity≥0.55), plus paired bootstrap CI
  on the AUC difference vs. current V5 (same procedure as
  `Evaluation_V5Candidate/LockedTest/run_locked_test_evaluation.py`'s `paired_bootstrap_ci`, applied to
  development OOF probabilities instead).
- **Success criterion**: Arm B (drop) succeeds if its OOF AUC is not meaningfully worse than current V5
  (95% CI of the paired AUC difference includes 0 or favors the simpler model) — a genuine "no loss from
  removing a confusing feature" result. Arm C (replace) succeeds only if its AUC improvement's 95% CI is
  entirely above zero, matching the same bar the original ablation study itself used to justify V5 over
  V4.

## Experiment 2 — `lesion_fraction` / `D_px_normalized` redundancy

- **Hypothesis**: `D_px_normalized` contributes negligible independent information beyond
  `lesion_fraction` (already suspected at rho=0.989); dropping it should not measurably hurt performance.
- **Feature being changed**: `D_px_normalized` (removed).
- **Baseline calculation**: current 17 features.
- **New calculation**: 16 features (`lesion_fraction` retained as the pre-existing V4 feature with the
  longer track record; `D_px_normalized` dropped).
- **Development dataset / validation**: as in Experiment 1.
- **Metric**: OOF ROC-AUC, balanced accuracy, paired bootstrap CI vs. current V5.
- **Success criterion**: same as Experiment 1 Arm B — success means "no meaningful loss," which would
  support treating `D_px_normalized` as redundant complexity rather than added signal.
- **Companion (non-modeling) experiment**: a data-only check (no CV, no model fit) of whether
  `lesion_fraction`/`D_px_normalized` values differ systematically by ISIC source collection/contributing
  institution, independent of ground-truth label — if available metadata supports it. This doesn't
  produce a performance metric; it produces evidence for or against the "photo-framing confound"
  hypothesis and should be reported alongside, not instead of, the modeling result.

## Experiment 3 — `skin_contrast` formula variants (gated on Part 5 visual review)

- **Hypothesis**: *Not yet formed.* This experiment is explicitly gated on the Part 5 human visual
  review identifying a plausible mechanism for the reproducible-but-counter-intuitive direction. Do not
  run this experiment until that review is complete.
- **Feature being changed**: `skin_contrast` (formula variant, not removal).
- **Baseline calculation**: current area-scaled-ring LAB-distance formula.
- **New calculation** (candidates, to be narrowed after visual review): fixed-radius ring instead of
  area-scaled radius; or the Kaya et al. multi-scale texture-homogeneity border-cutoff measure (Part 11).
- **Development dataset / validation**: as above.
- **Metric**: OOF ROC-AUC contribution of `skin_contrast` alone (single-feature AUC, as already computed
  in `feature_analysis_table.csv`'s style) before and after the formula change, plus full-model OOF AUC
  with the variant substituted in.
- **Success criterion**: the new variant's direction should be *at least as reproducible* (consistent
  sign across CV folds) as the current formula, and ideally more consistent with the mechanism identified
  in visual review; a pure "does the number change" result without a mechanism is not sufficient grounds
  to change a feature that already replicates cleanly across two independent cohorts.

## Experiment 4 — Remove `eccentricity`; replace color-fraction features with data-driven clustering

- **Hypothesis (a)**: `eccentricity` (chance-level univariate AUC on both cohorts) contributes nothing
  once its correlated geometry siblings (`aspect_ratio`, `radial_cv` — both already excluded) are
  accounted for; removing it should not hurt performance.
- **Hypothesis (b)**: replacing `red_fraction`/`bluegray_fraction`/`dark_fraction`'s hand-tuned fixed
  thresholds with a data-driven color-cluster-based description (Part 11, C1) captures the same
  "presence of a specific dangerous hue" concept more robustly, without the 0.0-median step-function
  problem currently observed.
- **Feature being changed**: `eccentricity` (arm A, removed); `red/bluegray/dark_fraction` (arm B,
  replaced by k-means color-cluster features in LAB space, k and cluster-significance threshold to be
  swept on development data only).
- **Baseline calculation**: current 17 features.
- **Development dataset / validation**: as above.
- **Metric**: OOF ROC-AUC, balanced accuracy, paired bootstrap CI.
- **Success criterion**: Arm A succeeds if AUC is not meaningfully worse (removal justified). Arm B
  succeeds only if AUC improvement's 95% CI is entirely above zero (same bar as Experiment 1 Arm C) —
  this is a bigger, riskier change (three features replaced by a differently-shaped feature set) so the
  bar for keeping it should be at least as strict as the one V5 itself had to clear over V4.

## Experiment 5 — Channel-specific color entropy addition

- **Hypothesis**: per-channel (L, a, b) 1D Shannon entropy adds independent signal beyond the existing
  joint 2D (a,b) `color_entropy` and the existing `lab_a_std`/`lab_b_std` spread measures (per Part 11,
  C3) — though this is genuinely uncertain, since per-channel entropy may turn out to be redundant with
  the std features that already exist.
- **Feature being changed**: addition of 1-3 new candidate features (`lab_L_entropy`, `lab_a_entropy`,
  `lab_b_entropy`), not a replacement.
- **Baseline calculation**: current 17 features.
- **New calculation**: 17 + up to 3 new per-channel entropy features.
- **Development dataset / validation**: as above; additionally, compute Spearman correlation between the
  new entropy features and the existing `color_entropy`/`lab_a_std`/`lab_b_std` (redundancy_report.md-
  style check) BEFORE running any CV, exactly as the original ablation study pruned redundant candidates
  before modeling — abandon any new feature exceeding |rho|>0.90 with an existing one, per the project's
  own established redundancy threshold.
- **Metric**: univariate AUC of each new feature (screening step, cheap); only features that individually
  clear a meaningful univariate bar AND survive the redundancy check proceed to a full-model OOF AUC
  comparison.
- **Success criterion**: same "95% CI of paired AUC difference entirely above zero" bar. Given this
  pipeline's own prior finding that texture/improved-asymmetry additions showed no defensible signal
  (Part 1/2, `experiment_log.md`), this experiment should be run with the expectation that it may
  legitimately fail the bar — that is a valid, useful outcome, not a wasted experiment.

## Experiment 6 — Robustness guards for asymmetry/border/diameter (edge-clamp, small-contour floor)

- **Hypothesis**: adding a minimum-contour-size floor (consistent across A/B/D features) and rejecting/
  flagging alignment for border-adjacent lesions will not regress performance, since it only changes
  behavior on cases the current code already handles unreliably (near-zero-size masks, off-center crops).
- **Feature being changed**: no new feature; a robustness modification to `A_value`, `B_circularity`,
  `solidity`, `turning_angle_std`, `D_px`'s underlying contour/crop handling.
- **Gating step**: first confirm via Part 5 visual review that edge-adjacent-lesion and near-noise-scale-
  mask cases actually occur in this dataset in numbers large enough to matter (if they're vanishingly
  rare, this drops in priority).
- **Baseline calculation**: current implementation (no floor/guard).
- **New calculation**: add a shared minimum-contour-area floor (e.g., matching the color features'
  existing 20px convention) returning NaN below it, and an edge-margin check that flags (does not
  silently clamp) alignment for masks whose bounding box is within some margin of the image border.
- **Development dataset / validation**: as above.
- **Metric**: OOF ROC-AUC/balanced accuracy before/after; additionally, count of rows newly flagged/NaN'd
  by the guard, to confirm the change is targeted (a handful of genuinely risky cases) rather than
  sweeping (a large fraction of the cohort suddenly excluded, which would indicate the guard is
  miscalibrated).
- **Success criterion**: no meaningful AUC regression, and the guard fires on a small, explicable subset
  of cases (ideally cross-checked against the visual review's specific flagged examples).

---

## Sequencing recommendation

Run Experiments 1, 2, and 4a (all cheap ablations, all currently well-evidenced) first, in a single
combined CV sweep alongside the current-V5 baseline, since they're independent enough to test together
or separately. Experiment 5 next (moderate cost, genuinely uncertain outcome). Experiments 3 and 6 are
explicitly gated on the Part 5 visual review and should not be started before that review is complete.
Experiment 4b (color clustering) is the largest, riskiest change and should come last, informed by
whatever Experiments 1/2/4a/5 reveal about how much headroom remains.
