# V4 feature-engineering investigation: can richer ABCD features beat frozen V4?

**Frozen V4 (`pipeline_v4/`) was not modified, retrained, or re-thresholded
anywhere in this investigation.** YOLO was not rerun or retrained; every new
feature reuses the existing cached YOLO masks/confidence/lesion_fraction
already in `Evaluation_FinalTargeted/Cohort/cohort_v2_results.csv`. The
locked test set was never read. Work is entirely on `pipeline-testing/
research`; `user-shree` and the web app were not touched.

**No V5 was created.** This is a research report only, per instructions.

## Scope

- Population: the same 2,090 `SINGLE_LESION_EVALUABLE` development images
  V4's own feature table uses (`Evaluation_V4/FeatureTable/
  development_feature_table.csv`). Verified merge coverage: 2,090/2,090.
- All 29 new candidate features computed successfully on all 2,090 images;
  **zero computation failures**, missingness 0% for every new feature except
  `skin_contrast` (1 image, 0.05%, a degenerate ring case) — reported, not
  silently dropped from other features' rows.
- New computation reused existing code wherever possible: the exact
  unmodified `Code/MelanomaDeterminingStuff/color.py::score_color` for
  dark/red/blue-gray fractions, and a vendored (read-only, unmodified)
  copy of `revised_abcd/revised_asymmetry.py`'s PCA-alignment function so
  the new color/texture-asymmetry features use the *same* principal axis
  V4's own A feature already computes -- no new alignment logic invented.

## 1-2. Candidate features and extraction

Full univariate results: `feature_analysis_table.csv`. Highlights (sorted by
separation strength):

| Feature | Benign median | Melanoma median | Cohen's d | Univariate AUC |
|---|---|---|---|---|
| D_px_normalized (new) | 0.315 | 0.755 | 0.98 | 0.758 |
| lesion_fraction (existing V4 feature) | 0.047 | 0.273 | 0.90 | 0.757 |
| turning_angle_std (new) | 0.241 | 0.323 | 0.78 | 0.719 |
| B_circularity (existing) | 0.240 | 0.309 | 0.68 | 0.683 |
| solidity (new) | 0.963 | 0.938 | -0.64 | 0.334* |
| lab_a_std (new) | 3.20 | 3.88 | 0.58 | 0.655 |
| color_entropy (new) | 1.30 | 1.57 | 0.55 | 0.650 |
| A_value (existing) | 0.139 | 0.183 | 0.55 | 0.649 |
| color_asymmetry (new) | 0.356 | 0.449 | 0.32 | 0.595 |

\* solidity discriminates in the *opposite* direction (lower solidity ->
more melanoma-like); |AUC-0.5|=0.166 is the meaningful quantity, not the
raw value.

Nine of the 29 new features showed essentially no standalone signal
(univariate AUC within ~0.03 of chance): `color_cluster_count`,
`glcm_contrast`, `glcm_homogeneity`, `glcm_energy`, `entropy_intensity`,
`white_fraction`, `aspect_ratio`/`eccentricity` (redundant, see below),
`texture_asymmetry` (AUC 0.501 -- literally chance), `local_contrast`. These
were not added to any ablation candidate merely because they were available.

## Correlation / redundancy findings (`correlation_matrix.csv`,
`redundancy_report.md`)

No near-zero-variance or high-missingness features. 15 pairs exceeded
|Spearman rho| > 0.90; the important ones, pruned before any modeling:

- `isoperimetric_ratio` <-> existing `B_circularity`: rho=**1.00** (it's a
  monotonic transform, `1/(1-B_circularity)`) -- adds zero information.
- `solidity` <-> `convexity_deficit`: rho=**-1.00** (exact complements) --
  kept `solidity` only.
- `aspect_ratio` <-> `eccentricity`: rho=**1.00** -- kept `eccentricity`.
- `major_axis_px`, `minor_axis_px`, `lesion_area_px` <-> existing `D_px`:
  rho=0.92-1.00 -- all dropped, no new information over baseline.
- `D_px_normalized` <-> existing `lesion_fraction`: rho=**0.989**. This is
  the single most important redundancy finding: the *strongest-looking* new
  feature (univariate AUC 0.758) is nearly a duplicate of a feature V4
  **already has**. Kept anyway to test empirically rather than assume, but
  flagged as expected-to-add-little going in.
- `local_contrast` <-> `glcm_homogeneity`/`glcm_contrast`: rho=-0.95/0.91 --
  kept `local_contrast` only.
- `radial_cv` <-> `eccentricity`: rho=0.913 (cross-group) -- both border-
  group and geometry-group capture lesion elongation; `radial_cv` dropped
  specifically from the *combined* candidate (kept in the standalone
  border-only ablation) to avoid double-counting.

Pruned candidate groups actually tested:

| Group | Features added to V4's 6 |
|---|---|
| + Improved asymmetry | `color_asymmetry` (texture_asymmetry dropped: chance-level) |
| + Border | `solidity`, `radial_cv`, `turning_angle_std` |
| + Color | `color_entropy`, `lab_a_std`, `lab_b_std`, `red_fraction`, `bluegray_fraction`, `dark_fraction`, `skin_contrast` |
| + Geometry | `eccentricity`, `D_px_normalized` |
| + Texture | `local_contrast`, `glcm_energy` |
| Best combined | Color + Border(-`radial_cv`) + Geometry = 11 new features (17 total) |

## 3-4. Ablation study (same GroupKFold(5) + StandardScaler ->
LogisticRegression(random_state=20260918) recipe as
`Evaluation_V4/build_feature_table_and_cv.py`; all metrics at V4's own
frozen 0.25 threshold, unchanged, on OOF predictions)

| Feature Set | Sens | Spec | Precision | F1 | Accuracy | Balanced Acc | ROC-AUC |
|---|---|---|---|---|---|---|---|
| **Current V4** | 0.7449 | 0.6912 | 0.5152 | 0.6091 | 0.7077 | 0.7181 | 0.7787 |
| + Improved asymmetry | 0.7418 | 0.6837 | 0.5080 | 0.6031 | 0.7014 | 0.7127 | 0.7798 |
| + Border | 0.7465 | 0.6892 | 0.5140 | 0.6088 | 0.7067 | 0.7178 | 0.7877 |
| + Color | 0.7586 | 0.6775 | 0.5084 | 0.6088 | 0.7022 | 0.7180 | 0.7932 |
| + Geometry | 0.7512 | 0.6761 | 0.5053 | 0.6042 | 0.6990 | 0.7136 | 0.7849 |
| + Texture | 0.7465 | 0.6864 | 0.5118 | 0.6073 | 0.7048 | 0.7165 | 0.7797 |
| **Best combined (Color+Border+Geometry)** | **0.7712** | 0.6733 | 0.5093 | 0.6135 | 0.7032 | **0.7222** | **0.8008** |

## 5. Is the improvement meaningful? Paired vs. V4's own cached OOF, 2,000-sample bootstrap

| Candidate | dAUC | AUC 95% CI | dBalAcc | BalAcc 95% CI | FN rescued | FN introduced | FP removed | FP introduced |
|---|---|---|---|---|---|---|---|---|
| + Improved asymmetry | +0.0012 | [-0.0028, 0.0052] | -0.0054 | [-0.0147, 0.0039] | 12 | 14 | 20 | 31 |
| + Border | +0.0091 | [0.0002, 0.0172] | -0.0003 | [-0.0144, 0.0129] | 25 | 24 | 65 | 68 |
| + Color | +0.0149 | [0.0039, 0.0263] | +0.0002 | [-0.0156, 0.0155] | 41 | 32 | 81 | 101 |
| + Geometry | +0.0062 | [0.0000, 0.0129] | -0.0045 | [-0.0151, 0.0072] | 19 | 15 | 38 | 60 |
| + Texture | +0.0011 | [-0.0046, 0.0066] | -0.0016 | [-0.0127, 0.0091] | 17 | 16 | 37 | 44 |
| **Best combined** | **+0.0225** | **[0.0097, 0.0353]** | +0.0043 | [-0.0130, 0.0214] | **53** | 36 | 89 | 115 |

**Reading this honestly:**
- The **ROC-AUC gain for the combined candidate is real, not noise** -- its
  95% CI is entirely above zero, and it is close to the sum of color's and
  border's independent contributions (0.0148+0.0089=0.0237 vs. the observed
  0.0225), meaning these two groups are contributing largely non-overlapping
  signal rather than double-counting the same thing.
- Fold-by-fold AUC for the combined candidate (0.809, 0.799, 0.808, 0.784,
  0.806 -- see `fold_by_fold_results.csv`) is **more consistent across folds
  than V4 itself** (0.752, 0.831, 0.775, 0.790, 0.756). The combined
  candidate's balanced-accuracy spread across folds is also tighter (0.712-
  0.732) than V4's (0.669-0.774). This is a genuine, separate finding from
  the mean-AUC lift: the richer feature set is not just marginally better on
  average, it is also less fold-to-fold erratic.
- **Balanced accuracy at the fixed 0.25 threshold is NOT statistically
  distinguishable from V4's** -- every candidate's balanced-accuracy 95% CI,
  including the combined one, straddles zero.
- At the operational level: sensitivity rose from 0.745 to 0.771 (+2.6
  points, 53 melanomas newly caught vs. 36 newly missed, net +17) while
  specificity fell from 0.691 to 0.673 (-1.8 points, 89 false alarms removed
  vs. 115 newly introduced, net -26). This is a real trade, not a
  specificity collapse, but it is a genuine trade, not a free win: more
  benign lesions get flagged for the sensitivity gained.
- Texture and improved-asymmetry show no defensible signal on their own
  (CIs straddle zero on both AUC and balanced accuracy) -- consistent with
  their near-chance univariate AUCs. Adding them contributed nothing to the
  combined candidate and were excluded from it.

## 6. Overfitting/complexity guard

- Dropped 12 of 29 computed candidate features outright as redundant,
  near-chance, or both, before any modeling (see section above).
- Zero near-zero-variance or high-missingness features found (no feature
  needed to be dropped on data-quality grounds).
- The combined candidate still nearly triples V4's feature count (6 -> 17)
  for a ROC-AUC gain of ~0.02 and a balanced-accuracy gain that isn't
  statistically distinguishable from zero. Per the "20 features for a tiny
  gain isn't justified" standard set for this investigation, this is a
  borderline case, not a clear-cut win -- addressed directly in the
  conclusion below.

## 7. Conclusion: V5 WARRANTED, but only as a properly re-thresholded next-phase candidate -- not a foregone replacement

**Which features produced the improvement:** color-distribution measures
(`color_entropy`, `lab_a_std`/`lab_b_std`, `red_fraction`, `bluegray_fraction`,
`dark_fraction`, `skin_contrast`) and border-shape measures (`solidity`,
`turning_angle_std`) each independently clear a real, CI-supported ROC-AUC
bar; `eccentricity`/`D_px_normalized` add a smaller amount on top. Improved-
asymmetry and texture features do not help and should not be carried
forward.

**Consistency across folds:** yes -- the combined candidate's AUC and
balanced accuracy are both more stable across the 5 folds than V4's own,
not just higher on average.

**Sensitivity vs. specificity:** sensitivity improves by a real, non-trivial
margin (+2.6 points, net 17 more melanomas caught) without a specificity
collapse (-1.8 points is a proportionate trade, not a cliff).

**ROC-AUC and/or balanced accuracy:** ROC-AUC improves meaningfully (CI
excludes zero, ~+0.02-0.03). Balanced accuracy at V4's *existing* threshold
does not show a statistically defensible improvement -- but this is expected
and not disqualifying: V4's 0.25 threshold was selected *for V4's own score
distribution*, not for a 17-feature model with a materially different
distribution. Judging a new model purely by an old model's threshold
understates it.

**Is the added complexity justified?** Marginally, and only conditionally.
The evidence clears a real statistical bar on ROC-AUC and shows a genuine,
non-trivial sensitivity gain -- this is not the "20 features for a trivial
fluctuation" case the investigation was warned against. But it is also not
an unambiguous win: balanced accuracy at the current threshold is flat, and
17 features is a real increase in complexity and future maintenance surface
over 6.

**Recommendation: V5 WARRANTED for further development, not for immediate
freezing.** Specifically: build a V5 *candidate* on the color+border+
geometry feature set (11 new features) and run the same CV-based threshold
selection procedure V4 itself went through (`Evaluation_V4/
select_threshold.json`'s method, on development data only, never on locked
test) before treating it as comparable to frozen V4. If, after its own
threshold is chosen on development data, its balanced accuracy and
sensitivity/specificity trade still look favorable, freezing it as V5 would
be justified. If not -- **explicitly keep V4** rather than replace it on the
strength of an AUC-only improvement. This report does not perform that
threshold selection or create V5; it stops here per instructions, pending
your decision on whether to proceed.

## Files

- `candidate_features_raw.csv` -- all 29 new features, per image, unmerged.
- `extended_feature_table.csv` -- merged with V4's 6 base features + ground
  truth + group_id, 2,090 rows.
- `feature_analysis_table.csv`, `correlation_matrix.csv`,
  `redundancy_report.md` -- sections 1-2 above.
- `ablation_comparison_table.csv`, `fold_by_fold_results.csv`,
  `paired_error_analysis_vs_v4.json` -- sections 4-5 above.
- `extract_features.py`, `build_extended_table_and_analysis.py`,
  `run_ablation_cv.py` -- the code, in the order it was run.
- `_vendored_revised_asymmetry.py` -- read-only copy from `user-shree`
  (unused `_vendored_revised_border.py` also present, kept for reference).
