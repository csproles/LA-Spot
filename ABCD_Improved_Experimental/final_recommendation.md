# Final Recommendation — Development-Only Candidate

> **This candidate has NOT been run on the locked test set and must NOT be run on the locked test
> set until explicitly approved by a human.** Everything in this document is derived exclusively
> from 5-fold `GroupKFold` out-of-fold (OOF) cross-validation on the 2090-row development cohort in
> `dev_feature_table_v2.csv`. It is a development-only candidate, not a frozen or validated pipeline.

## FINAL feature set (14 features)

```
A_value_pc          C_value_pc           entropy_L
B_circularity        confidence           entropy_a
solidity             lab_a_std_pc         entropy_b
turning_angle_std    lab_b_std_pc         red_fraction_pc
D_px_normalized      skin_contrast_pc
```

Full list in canonical order: `A_value_pc, B_circularity, C_value_pc, confidence, lab_a_std_pc,
lab_b_std_pc, skin_contrast_pc, solidity, turning_angle_std, D_px_normalized, entropy_L, entropy_a,
entropy_b, red_fraction_pc`.

This is the endpoint of the staged cascade EXP1 -> EXP2 -> EXP3 -> EXP4 -> EXP5 -> EXP6 described
below (each stage keeping only the best-performing sub-variant and carrying it forward, per the
brief's own experiment ordering).

## FINAL configuration's full metrics

Selected threshold: **0.30** (via the standard sweep rule: max balanced accuracy among
`{0.20,...,0.50}` where both sensitivity>=0.55 AND specificity>=0.55; this FINAL config's own OOF
qualified at multiple thresholds and 0.30 was the balanced-accuracy-maximizing one among them).

| Metric | Value |
|---|---|
| N (used) | 2089 / 2090 (1 row dropped: `skin_contrast_pc` NaN for `ISIC_9895318`) |
| N benign / malignant | 1451 / 638 |
| TP / TN / FP / FN | 460 / 1085 / 366 / 178 |
| Accuracy | 0.7396 |
| **Balanced accuracy** | **0.7344** |
| Sensitivity | 0.7210 |
| Specificity | 0.7478 |
| Precision | 0.5569 |
| F1 | 0.6284 |
| **ROC-AUC** | **0.8016** |
| CV mean balanced accuracy (5-fold) | 0.7348 |
| CV std balanced accuracy (5-fold) | 0.0096 |

Compared to frozen V5 (EXP0, reproduced exactly): balanced accuracy **+0.0122** (0.7222 -> 0.7344),
ROC-AUC **+0.0008** (0.8008 -> 0.8016), at a smaller feature count (14 vs. 17) and a different
operating threshold (0.30 vs. 0.25 — this FINAL candidate's own threshold, independently swept, not
V5's reused).

## How the FINAL configuration was reached (cascade summary)

1. **EXP0** — frozen V5's 17 original features. Reproduced exactly (sanity gate, see below).
2. **EXP1** — swapped `A_value` and the 8 color features for their primary-component-consistent
   (`_pc`) versions. Balanced accuracy 0.7244 (+0.0022 vs EXP0), within CV std — a real but small
   improvement, consistent with most dev images already being single-component masks.
3. **EXP2** — dropped `eccentricity`, `dark_fraction_pc`, `bluegray_fraction_pc`, `D_px` (the 4
   features the prior audit flagged as weak). Balanced accuracy 0.7297 (+0.0053 vs EXP1) — also
   within CV std individually, but this is the largest single-stage gain in the cascade and pushed
   the operating threshold from 0.25 to 0.30.
4. **EXP3** — tested `lesion_fraction`/`D_px_normalized` redundancy (4 variants). **EXP3C**
   (D_px_normalized only, drop lesion_fraction) won: balanced accuracy 0.7325, beating "both kept"
   (0.7297) and "lesion_fraction only" (0.7260). Dropping **both** (EXP3D) was clearly harmful
   (balanced accuracy fell to 0.6992, ROC-AUC to 0.7654) — the only ablation in this entire study
   that "hurt" outside CV-std noise.
5. **EXP4** — tested channel-specific entropy (3 sub-variants) and re-tested the color-fraction
   group. **EXP4b** (replace `color_entropy_pc` with `entropy_L/entropy_a/entropy_b`) won over
   keeping the joint entropy alone (EXP4a) or keeping both (EXP4c). Within the fraction-group test,
   keeping `red_fraction_pc` alone won over dropping it, adding back `bluegray_fraction_pc`/
   `dark_fraction_pc`, or adding all three — confirming EXP2's original decision to drop
   `bluegray_fraction_pc`/`dark_fraction_pc` still holds even after channel entropy is added.
   Balanced accuracy 0.7344 (+0.0019 vs EXP3C).
6. **EXP5** — tested splitting `A_value_pc` into `asymmetry_major_axis` + `asymmetry_minor_axis`.
   The single combined `A_value_pc` **won** (0.7344 vs. 0.7290 split) — splitting did not help and
   is not recommended.
7. **EXP6** — tested adding `border_fractal_dimension`. Adding it **did not help** (0.7278 vs. 0.7344
   without) and also sharply destabilized the fold-to-fold CV std (0.0511 vs. 0.0096) — the 1 NaN
   row plus this feature's narrow observed range (~0.95-1.05) appear to add mostly noise at this
   feature count. Not recommended.
8. **FINAL** — the cascade endpoint after all 6 stages: EXP4's entropy+fraction config, unchanged
   by EXP5/EXP6 since both of their tested additions lost.

## Recommended vs. tested-but-not-recommended

### Recommended (included in FINAL)

| Change | Effect |
|---|---|
| `A_value` -> `A_value_pc` | +0.0022 balanced accuracy (small, within CV std, but directionally consistent and theoretically well-motivated — fixes a real mask-fragmentation inconsistency) |
| Color features -> `_pc` versions | +0.0015 balanced accuracy (same rationale) |
| Drop `eccentricity` | part of EXP2's combined +0.0053; individually -0.0044 impact if removed alone (i.e. removing it alone *improved* balanced accuracy by 0.0044) |
| Drop `dark_fraction_pc` | individually +0.0021 if removed alone |
| Drop `bluegray_fraction_pc` (from EXP1's superset) | individually +0.0017 if removed alone |
| Drop `D_px` (raw pixels) | individually +0.0012 if removed alone |
| Drop `lesion_fraction`, keep `D_px_normalized` | +0.0028 balanced accuracy vs. keeping only lesion_fraction; beat "keep both" too |
| Replace `color_entropy_pc` with `entropy_L`/`entropy_a`/`entropy_b` | +0.0019 balanced accuracy vs. keeping joint entropy alone |
| Keep `red_fraction_pc` | best of the color-fraction subsets tested |

### Tested but NOT recommended (excluded from FINAL)

| Change | Why not |
|---|---|
| Dropping BOTH `lesion_fraction` and `D_px_normalized` | Clearly harmful: -0.0305 balanced accuracy, -0.0365 ROC-AUC — the one unambiguous "hurt" result in the whole study |
| Adding back `bluegray_fraction_pc` and/or `dark_fraction_pc` alongside channel entropy | No meaningful improvement in any combination tested (individually, pairwise, or all three); best fraction subset remained `red_fraction_pc` alone |
| Splitting `A_value_pc` into `asymmetry_major_axis` + `asymmetry_minor_axis` | Balanced accuracy fell from 0.7344 to 0.7290; the combined single feature is simpler AND better here |
| Adding `border_fractal_dimension` | Balanced accuracy fell from 0.7344 to 0.7278 and fold-to-fold CV std ballooned (0.0096 -> 0.0511); not recommended at this feature-set size, though it may still be worth testing in combination with other new features in a future round |
| Adding `eccentricity`, `dark_fraction_pc`, `bluegray_fraction_pc`, `D_px` back individually | Each one's removal alone was mildly positive or neutral; none justified keeping |

### A note on interpretability (priority 7 of the stated ranking)

**EXP2** (13 features: EXP1's set minus the 4 weak features, still using `color_entropy_pc` rather
than channel entropy, D_px_normalized+lesion_fraction both retained) achieves balanced accuracy
0.7297 — a delta of only **0.0047** from FINAL's 0.7344, which is **within FINAL's own fold-to-fold
CV standard deviation (0.0096)**, i.e. "essentially equivalent" by this study's explicit definition.
FINAL is still recommended as the primary candidate because it wins on **both** of the two
highest-priority criteria (balanced accuracy AND ROC-AUC, priorities 1 and 2) — interpretability
(priority 7) only serves as a tie-break among genuinely equivalent options, and FINAL is not
strictly tied with EXP2 on the top two criteria, only close. If a future reviewer prefers a smaller,
simpler feature set and is willing to accept ~0.005 lower balanced accuracy, **EXP2's 13-feature
configuration is a reasonable, well-tested fallback** — see `pipeline_performance_comparison.csv`
for its full metrics (row `EXP2`).

## Multicollinearity caveat

See `multicollinearity_report.md` for full detail. FINAL still contains `B_circularity`, `solidity`,
and `turning_angle_std` together (VIFs of 6.94 / 6.11 / 3.09 respectively within the FINAL set —
moderate, not severe, collinearity). **B_circularity's coefficient sign-flip is NOT resolved by
FINAL**: its univariate correlation with malignancy is positive (+0.30, the clinically expected
direction), but its coefficient in a plain LogisticRegression fit on the FINAL feature set is
**-0.719** — still inverted, essentially unchanged from frozen V5's own -0.621. None of the changes
tested in this modeling phase targeted this specific multicollinearity, and it remains an open,
documented limitation.

## Explicit statement of scope

- This FINAL configuration is a **development-only candidate**. It has been evaluated exclusively
  via 5-fold grouped cross-validation on the 2090-image development cohort.
- It has **NOT** been evaluated on `Evaluation_FinalTargeted/Cohort/locked_test_manifest.csv` or any
  data under that path — that file/directory was never opened, read, or referenced by any script in
  this modeling phase (verified by grep; see the parent report for the exact commands run).
- It is **NOT frozen** and **NOT validated** for production or clinical use.
- Running this candidate (or any candidate) against the locked test set requires **explicit human
  approval** before proceeding — that approval has not been sought or given as part of this task.
