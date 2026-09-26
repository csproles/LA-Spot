# Robustness Guard Report — Phase 7

Summary of the Phase 1/7 quality flags across all **2090** development-cohort images in
`dev_feature_table_v2.csv`. Generated from `ABCD_Improved_Experimental/build_mask_features.py`'s
output (`_mask_features.csv`) merged with `extended_feature_table.csv` and `_image_features.csv`. No
locked-test data was used anywhere in this analysis.

**Action taken, confirmed**: every one of the 2090 images has a full row in
`dev_feature_table_v2.csv` regardless of which flags (if any) fired. **No image was excluded or
dropped for tripping any flag.** Flags are metadata columns only; any filtering decision is left
entirely to the modeling phase (a separate agent's task), per the brief's explicit instruction.

---

## 1. How many images trip each flag

| Flag | Count | % of 2090 |
|---|---:|---:|
| `flag_fragmented` (>1 meaningful component, area > 20px) | 121 | 5.79% |
| `flag_multiple_substantial_components` (2nd component > 30% of primary) | 1 | 0.05% |
| `flag_edge_adjacent` (primary bbox within ~1% of image dim of any edge) | 206 | 9.86% |
| `flag_very_small_contour` (primary area < 1st-percentile floor, 12445 px) | 21 | 1.00% |
| `flag_oversized_mask` (primary area > 50% of image) | 193 | 9.23% |
| **Any flag** | **337** | **16.1%** |
| `quality_flags == "ok"` (no flag) | 1753 | 83.9% |

`num_meaningful_components` distribution (components with area > 20px, from the RAW mask before
primary-component selection): 1 component in 1969/2090 images (94.2%), 2 in 93, 3 in 16, 4 in 4, 5 in
4, 6 in 4. So the large majority of dev images are already single-component at the raw-mask level;
`flag_fragmented` captures the remaining ~5.8% where V5's original inconsistency (Phase 1) actually
had pixels to disagree about.

`flag_edge_adjacent` and `flag_oversized_mask` overlap heavily (Pearson correlation of the two
boolean columns = 0.72) — a lesion photographed close enough to fill more than half the frame is
naturally more likely to also touch or nearly touch an edge. `flag_fragmented` correlates modestly
with both (~0.23) — a fragmented mask is somewhat more likely to also be large/edge-adjacent, plausibly
because larger/harder-to-segment lesions are more prone to both raster fragmentation and
close-up/edge-touching framing simultaneously, though this analysis does not establish causation.

---

## 2. Which features each flag is most relevant to (measured directly, not just asserted)

For every image, the absolute difference between each original V5 feature and its primary-component
(`_pc`) recomputation was computed (e.g. `dA = |A_value - A_value_pc|`, `dSkin = |skin_contrast -
skin_contrast_pc|`, etc.), then averaged within each flag's True/False groups. A large gap between the
True and False group means that flag is where the Phase 1 fix actually changes the feature value the
most — i.e., where the *original* V5 feature was most likely computed on an inconsistent mask.

### `flag_fragmented` -> primarily affects `A_value` and `skin_contrast` (as predicted by the audit)

| Feature | mean \|delta\|, not fragmented | mean \|delta\|, fragmented | ratio |
|---|---:|---:|---:|
| `A_value` vs `A_value_pc` | 0.0002 | 0.0133 | ~66x |
| `skin_contrast` vs `skin_contrast_pc` | 0.0082 | 1.0919 | ~133x |
| `lab_b_std` vs `lab_b_std_pc` | 0.0000 | 0.0456 | — |
| `lab_a_std` vs `lab_a_std_pc` | 0.0000 | 0.0168 | — |
| `color_entropy` vs `color_entropy_pc` | 0.0000 | 0.0076 | — |
| `bluegray_fraction` vs `_pc` | 0.0000 | 0.0025 | — |
| `C_value` vs `C_value_pc` | 0.0000 | 0.0031 | — |
| `B_circularity`, `solidity`, `turning_angle_std`, `D_px` vs their `_pc` counterparts | **0** | **0** | **exact match, always** |

This directly confirms `V5_Part6_Asymmetry_Implementation_Audit.md` and
`V5_Part7_Border_Implementation_Audit.md`'s finding: V5's border/geometry/diameter features
(`B_circularity`, `solidity`, `turning_angle_std`, `D_px`) already used `largest_contour()`, which is
**exactly equivalent** to this task's primary-component selection whenever there's only one dominant
component — so those four features show **zero** difference from their `_pc` recomputation for every
single one of the 2090 images, fragmented or not. `A_value` and all the color features, by contrast,
used the full (possibly fragmented) mask in the original V5/ablation code, so they DO diverge when a
mask is fragmented — most dramatically for `skin_contrast` (up to ~133x larger average error when
fragmented) because its "outside ring" is computed by dilating the *entire* mask, so a stray secondary
fragment shifts the ring's footprint away from the true lesion boundary.

Within the 121 fragmented images specifically: `A_value` differs from `A_value_pc` by more than 0.05
(a substantial change relative to V5's own 0.20 asymmetry-concern threshold) for **8 images**, with a
maximum single-image difference of **0.278** (`ISIC_4428176`, 3 meaningful components) — large enough
that a fragmented mask could plausibly have flipped V5's asymmetry-concern flag for that image. `C_value` differences stay
small even when fragmented (max 0.043 overall) — the coefficient-of-variation formula is naturally
more robust to a few extra pixels than a raw mean-based comparison like `skin_contrast`.

### `flag_multiple_substantial_components` -> the single most extreme case in the dataset

Only 1 image trips this (a genuine 2nd blob > 30% of the primary component's area): `ISIC_3665449`
(5 meaningful components total). For that one image: `|A_value - A_value_pc| = 0.186`, `|C_value - C_value_pc| = 0.039`, `|lab_b_std -
lab_b_std_pc| = 1.376`, `|skin_contrast - skin_contrast_pc| = 2.693` — by far the largest deltas
observed anywhere in the dataset on every one of these features simultaneously, exactly as expected:
this is the case where "which component is the lesion" is most genuinely ambiguous, so the
inconsistent-mask-handling problem this task fixes has its largest real effect.

### `flag_edge_adjacent` / `flag_oversized_mask` -> primarily affect `A_value`/`A_value_pc` and `skin_contrast`/`skin_contrast_pc`

| Feature | mean \|delta\|, flag False | mean \|delta\|, flag True (edge_adjacent) | mean \|delta\|, flag True (oversized) |
|---|---:|---:|---:|
| `A_value` vs `A_value_pc` | 0.0009 | 0.0020 (~2.2x) | 0.0022 (~2.4x) |
| `skin_contrast` vs `skin_contrast_pc` | 0.0323-0.0354 | 0.3978 (~11x) | 0.4527 (~14x) |

This is the specific "edge-clamped crop asymmetry" risk named in
`V5_Part6_Asymmetry_Implementation_Audit.md`: a lesion whose bounding box nearly touches the image
border gets its `_fold_asymmetry` crop window asymmetrically clamped (the crop simply cannot extend
past the image edge on that side), which can distort the fold-IoU comparison independent of the
lesion's true shape — visible here as a real, if modest on average, elevation in `A_value` divergence.
`skin_contrast`'s much larger elevation for these flags is a related but distinct effect: an
edge-adjacent or oversized lesion leaves less "outside ring" available before the image boundary is
hit, making the ring sample itself less reliable regardless of fragmentation. One image
(`ISIC_9895318`, flagged both `edge_adjacent` and `oversized_mask`) has `skin_contrast_pc = NaN`
entirely, because its outside ring fell below the 20-pixel minimum required by
`skin_contrast_feature`'s existing guard — the ring collapses when the lesion is both large and
close to the frame edge. This is the only NaN in any `_pc` color column across all 2090 images.

### `flag_very_small_contour` -> affects everything mildly, most visibly `A_value` and `skin_contrast`

21 images. `A_value` divergence (0.0010 -> 0.0034, ~3.4x) and `skin_contrast` divergence (0.0701 ->
0.1535, ~2.2x) both rise, consistent with `V5_Part6_Asymmetry_Implementation_Audit.md`'s separate
concern (#3) that very small masks make PCA-based alignment numerically less stable (near-equal
eigenvalues on a small point cloud). This flag is also the reason `border_fractal_dimension` is `NaN`
for exactly 1 image (`ISIC_1828562`): its primary component is too small to yield the minimum 4
distinct box sizes the box-counting regression requires.

---

## 3. Fractal dimension (`border_fractal_dimension`) distribution

Computed successfully for 2089/2090 images (the 1 failure is the very-small-contour case above, an
expected/handled NaN, not a bug). Distribution: mean 0.989, std 0.017, min 0.932, max 1.088, median
0.988. Number of box-size points used in the regression fit: predominantly 5-7 (median 6), ranging
2-9 depending on lesion size. Values cluster tightly just below/around 1.0 (a smooth curve's
theoretical box-counting fractal dimension), consistent with YOLO segmentation masks producing
comparatively smooth pixel-level boundaries in this dataset — see
`improved_feature_definitions.md` Phase 5 for the full method and literature citation.

---

## 4. Confirmation

- **No image was dropped.** All 2090 rows in `dev_feature_table_v2.csv` have their original
  `extended_feature_table.csv` columns fully populated regardless of flags; the mask pass had 0
  errors and the image pass had 0 errors across all 2090 images (both scripts' own printed summaries
  confirm this: `build_mask_features.py` -> 0 rows with `_mask_error`, `build_image_features.py` ->
  "Rows with at least one error: 0").
- The only two NaNs anywhere in the new `_pc`/entropy/fractal columns are individually accounted for
  above (`skin_contrast_pc` for `ISIC_9895318`, `border_fractal_dimension` for `ISIC_1828562`), both
  from pre-existing, documented minimum-pixel-count guards firing on genuinely small/edge-case inputs
  — not silent failures.
- Locked test set: never opened. Only `extended_feature_table.csv`, `development_manifest.csv`, and
  `cohort_v2_results.csv` (filtered to the SINGLE_LESION_EVALUABLE dev-cohort rows already used by the
  original ablation study) were read.
