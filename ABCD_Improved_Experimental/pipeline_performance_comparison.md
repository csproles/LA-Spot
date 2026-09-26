# Pipeline Performance Comparison

Full narrative companion to `pipeline_performance_comparison.csv` (identical data, machine-readable).
All 32 rows use the SAME development population and CV methodology:
5-fold `GroupKFold` (grouped by `group_id`) out-of-fold (OOF) predictions from a fresh
`Pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=20260918))` fit inside
each fold, on `dev_feature_table_v2.csv` (2090 development-cohort rows). No row in this file, or in
this document, touches the locked test set. Each experiment selects its own operating threshold by
sweeping `{0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50}` and picking the balanced-accuracy-maximizing
threshold among those with both sensitivity>=0.55 and specificity>=0.55 (falling back to max
balanced accuracy overall if none qualify) — the same rule `select_v5_threshold.py` used for frozen
V5.

## Sanity gate: EXP0 reproducibility check

Before running anything else, the frozen V5 17-feature set was re-run through this harness and
checked against the known-correct numbers from `Evaluation_V5Candidate/build_v5_candidate.py`:

| Metric | Expected (frozen V5) | Actual (this harness) | Status |
|---|---|---|---|
| ROC-AUC | 0.8008 | 0.8008 | OK |
| Sensitivity | 0.7712 | 0.7712 | OK |
| Specificity | 0.6733 | 0.6733 | OK |
| Balanced accuracy | 0.7222 | 0.7222 | OK |

All four matched to well within the 0.0005 tolerance (exact match after rounding). **The harness is
verified correct; every experiment below uses this same, confirmed-correct code path.**

## Phase-by-phase results

### Phase 0 — Baseline

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std |
|---|---|---|---|---|---|---|---|---|
| EXP0 (frozen V5) | 17 | 0.25 | 0.7712 | 0.6733 | 0.7222 | 0.8008 | 0.7226 | 0.0078 |

### Phase 1 — Primary-component-consistent features (EXP1)

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP1a | 17 | 0.35 | 0.6473 | 0.8015 | 0.7244 | 0.8013 | 0.7247 | 0.0248 | Isolates A_value->A_value_pc only |
| EXP1b | 17 | 0.35 | 0.6473 | 0.8001 | 0.7237 | 0.8010 | 0.7240 | 0.0230 | Isolates color group -> _pc only |
| **EXP1** | 17 | 0.25 | 0.7727 | 0.6761 | 0.7244 | 0.8014 | 0.7247 | 0.0066 | Both combined |

Both sub-changes individually produce small, within-CV-std gains (delta balanced accuracy +0.0022
and +0.0015 respectively vs. EXP0); combined, EXP1 nets +0.0022 balanced accuracy and +0.0006
ROC-AUC over EXP0. Notably, EXP1's own threshold sweep re-selects 0.25 (same as EXP0), while the two
isolated sub-variants (EXP1a/EXP1b) each individually prefer 0.35 — a reminder that per-experiment
threshold selection genuinely differs and combining changes can shift the optimum back.

### Phase 2 — Drop 4 weak/near-chance features (EXP2)

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP2_minus_eccentricity_only | 16 | 0.30 | 0.7085 | 0.7491 | 0.7288 | 0.8016 | 0.7291 | 0.0118 | EXP1 minus eccentricity alone |
| EXP2_minus_dark_fraction_pc_only | 16 | 0.30 | 0.7053 | 0.7478 | 0.7265 | 0.8016 | 0.7269 | 0.0135 | EXP1 minus dark_fraction_pc alone |
| EXP2_minus_bluegray_fraction_pc_only | 16 | 0.35 | 0.6473 | 0.8050 | 0.7261 | 0.8026 | 0.7264 | 0.0227 | EXP1 minus bluegray_fraction_pc alone |
| EXP2_minus_D_px_only | 16 | 0.30 | 0.7069 | 0.7443 | 0.7256 | 0.8002 | 0.7260 | 0.0151 | EXP1 minus D_px alone |
| **EXP2** | 13 | 0.30 | 0.7116 | 0.7478 | 0.7297 | 0.8019 | 0.7300 | 0.0108 | All 4 dropped together |

Every individual removal is a small improvement over EXP1's 0.7244 baseline; removing all four
together (EXP2) reaches 0.7297, the single largest-magnitude stage gain in the whole cascade
(+0.0053 vs. EXP1), while cutting feature count from 17 to 13.

### Phase 3 — Size-feature redundancy: lesion_fraction vs. D_px_normalized

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP3A (both kept, =EXP2) | 13 | 0.30 | 0.7116 | 0.7478 | 0.7297 | 0.8019 | 0.7300 | 0.0108 | |
| EXP3B (lesion_fraction only) | 12 | 0.25 | 0.7649 | 0.6871 | 0.7260 | 0.7996 | 0.7263 | 0.0173 | Drop D_px_normalized |
| **EXP3C (D_px_normalized only)** | 12 | 0.35 | 0.6661 | 0.7988 | **0.7325** | **0.8031** | 0.7327 | 0.0135 | Drop lesion_fraction — WINNER |
| EXP3D (neither) | 11 | 0.30 | 0.6803 | 0.7181 | 0.6992 | 0.7654 | 0.6995 | 0.0227 | Drop both — clearly harmful |

This is the clearest result in the study: the two size features are NOT interchangeable in practice.
Keeping `D_px_normalized` alone (EXP3C) beats keeping both (EXP3A, +0.0028) and clearly beats keeping
`lesion_fraction` alone (EXP3B, +0.0065). Dropping both (EXP3D) is a severe regression (-0.0305
balanced accuracy, -0.0365 ROC-AUC vs. EXP3A) — by far the largest negative effect measured anywhere
in this study, confirming this feature group carries real, non-redundant signal despite its high
(rho=0.989) pairwise correlation.

### Phase 4 — Channel-specific entropy + color-fraction re-test

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| EXP4a (color_entropy_pc only, =EXP3C) | 12 | 0.35 | 0.6661 | 0.7988 | 0.7325 | 0.8031 | 0.7327 | 0.0135 | |
| **EXP4b (entropy_L/a/b replacing color_entropy_pc)** | 14 | 0.30 | 0.7210 | 0.7478 | **0.7344** | 0.8016 | 0.7348 | **0.0096** | WINNER — also lowest CV std of the three |
| EXP4c (both together) | 15 | 0.35 | 0.6693 | 0.7967 | 0.7330 | 0.8020 | 0.7333 | 0.0126 | |

Color-fraction group re-test (on top of EXP4b's entropy config, `frac_base` = EXP4b minus any
fraction feature):

| Experiment | Features | Threshold | Bal.Acc | ROC-AUC | CV std | Notes |
|---|---|---|---|---|---|---|
| EXP4_frac_none | 13 | 0.35 | 0.7323 | 0.8016 | 0.0100 | No fraction features |
| **EXP4_frac_red** | 14 | 0.30 | **0.7344** | 0.8016 | 0.0096 | WINNER — red_fraction_pc alone |
| EXP4_frac_bluegray | 14 | 0.35 | 0.7300 | 0.8005 | 0.0083 | bluegray_fraction_pc alone |
| EXP4_frac_dark | 14 | 0.35 | 0.7326 | 0.8012 | 0.0101 | dark_fraction_pc alone |
| EXP4_frac_red+bluegray | 15 | 0.30 | 0.7331 | 0.8006 | 0.0069 | |
| EXP4_frac_red+dark | 15 | 0.30 | 0.7344 | 0.8015 | 0.0075 | Essentially tied with red-only |
| EXP4_frac_bluegray+dark | 15 | 0.35 | 0.7300 | 0.8003 | 0.0091 | |
| EXP4_frac_red+bluegray+dark (all 3) | 16 | 0.30 | 0.7331 | 0.8005 | 0.0069 | |
| **EXP4 (final combined)** | 14 | 0.30 | **0.7344** | 0.8016 | 0.0096 | entropy_L/a/b + red_fraction_pc only |

Channel-specific entropy (`entropy_L/a/b` replacing `color_entropy_pc`) is a genuine, if modest,
improvement (+0.0019 balanced accuracy vs. keeping joint entropy alone) and also the config with the
lowest fold-to-fold CV std among the three entropy sub-variants — a stability win, not just a point
estimate win. The color-fraction re-test **confirms** EXP2's earlier decision: adding back
`bluegray_fraction_pc` and/or `dark_fraction_pc` alongside channel entropy does not meaningfully
help in any combination tested; `red_fraction_pc` alone remains the best (or tied-best) choice.

### Phase 5 — Split major/minor-axis asymmetry vs. single A_value_pc

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **EXP5_single_A_value_pc (=EXP4, WINNER)** | 14 | 0.30 | 0.7210 | 0.7478 | **0.7344** | 0.8016 | 0.7348 | 0.0096 | |
| EXP5_split_major_minor | 15 | 0.35 | 0.6599 | 0.7981 | 0.7290 | 0.8017 | 0.7292 | 0.0114 | asymmetry_major_axis + asymmetry_minor_axis |

Splitting asymmetry into major/minor-axis components does NOT help here — balanced accuracy falls by
0.0054 despite adding a feature. The single combined `A_value_pc` is both simpler and better within
this feature set.

### Phase 6 — Border fractal dimension

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std | Notes |
|---|---|---|---|---|---|---|---|---|---|
| **EXP6_without_fractal (=EXP5, WINNER)** | 14 | 0.30 | 0.7210 | 0.7478 | **0.7344** | 0.8016 | 0.7348 | **0.0096** | |
| EXP6_with_fractal | 15 | 0.35 | 0.6536 | 0.8021 | 0.7278 | 0.8012 | 0.7291 | 0.0511 | +border_fractal_dimension; 1 NaN row excluded |

Adding `border_fractal_dimension` hurts both the point estimate (-0.0066 balanced accuracy) and,
sharply, fold-to-fold stability (CV std jumps more than 5x, from 0.0096 to 0.0511) — one fold is
evidently very sensitive to this feature at this sample size. Not recommended for inclusion.

### FINAL

| Experiment | Features | Threshold | Sens | Spec | Bal.Acc | ROC-AUC | CV mean | CV std |
|---|---|---|---|---|---|---|---|---|
| **FINAL** | 14 | 0.30 | 0.7210 | 0.7478 | **0.7344** | **0.8016** | 0.7348 | 0.0096 |

Best balanced accuracy achieved: **0.7344** (FINAL / EXP4 / EXP5 / EXP6, all identical — the cascade
converged after Phase 4 and neither Phase 5 nor Phase 6's tested changes improved on it).
Best ROC-AUC achieved: **0.8031** (EXP3C / EXP4a, both = the 12-feature D_px_normalized-only,
color_entropy_pc-only config) — marginally higher than FINAL's 0.8016, but FINAL's balanced accuracy
is higher (0.7344 vs. 0.7325) and its CV std is lower (0.0096 vs. 0.0135), so FINAL wins on the
stated priority order (balanced accuracy first, ROC-AUC second, only as a tie-break).

See `experiment_summary.csv` for the condensed one-row-per-phase version and
`feature_ablation_results.csv` for every individually-tested single change with its own delta and
verdict.
