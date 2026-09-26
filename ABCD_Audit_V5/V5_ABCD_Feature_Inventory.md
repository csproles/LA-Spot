# V5 ABCD Feature Inventory (Parts 1–2)

**Scope note:** the request asked for "the 12 features." Reading the actual frozen model code
(`pipeline_v5/decision_model.py::V5_CONFIG`), the current V5 pipeline uses **17 features**, not 12:
6 base features carried over unchanged from V4 (`revised_abcd/pipeline_v2.py`) plus 11 new features
added by V5 (`pipeline_v5/feature_extraction.py`). This document audits all 17, in the model's exact
fitting order. Nothing here modifies the frozen model, thresholds, or any locked result.

Per-feature evidence (medians, Cohen's d, univariate AUC, Spearman correlations) is drawn from the
project's own prior ablation-study artifacts (`Evaluation_FeatureEngineering/feature_analysis_table.csv`,
`redundancy_report.md`, `experiment_log.md`) and the frozen model's own standardized coefficients
(`Evaluation_V5Candidate/v5_coefficients.csv`) — all copied read-only into
`ABCD_Audit_V5/_source_from_research_branch/` from the `pipeline-testing/research` branch (they don't
exist on `user`). These are *development-cohort* statistics (n=2090), not locked-test statistics —
locked-test per-image statistics are computed separately in Part 4.

The full machine-readable version of everything below is `V5_ABCD_Feature_Inventory.csv`.

---

## The 17 features, in the model's exact fitting order

```
Base 6 (unchanged from V4): A_value, B_circularity, C_value, D_px, confidence, lesion_fraction
New 11 (V5 only):           color_entropy, lab_a_std, lab_b_std, red_fraction, bluegray_fraction,
                             dark_fraction, skin_contrast, solidity, turning_angle_std, eccentricity,
                             D_px_normalized
```

Order matters — `StandardScaler`'s fitted `mean_`/`scale_` are positional (`pipeline_v5/decision_model.py:62-64`).

---

## Asymmetry (A_value) — exact mechanics

Source: `revised_abcd/revised_asymmetry.py`.

1. **Alignment**: PCA on the foreground-pixel `(x, y)` coordinates of the mask
   (`_principal_axis_angle_deg`) — covariance matrix of centered coordinates, eigendecomposed via
   `np.linalg.eigh`, dominant eigenvector's angle taken as the lesion's principal axis. **Yes, PCA is
   used.** This replaces the *legacy* asymmetry code's fixed image x/y-axis folding — the revision's
   stated motivation (module docstring) is that folding along fixed image axes conflates a lesion's
   *orientation in the photograph* with genuine shape asymmetry (a documented example: one elongated
   but otherwise unremarkable lesion scored 0.630 unaligned vs. 0.265 once aligned to its own axis).
2. **Rotation**: `cv2.getRotationMatrix2D(center, angle, 1.0)` + `cv2.warpAffine` (nearest-neighbor,
   to keep the mask binary) rotates the mask so the principal axis becomes horizontal. If fewer than 5
   foreground pixels exist, alignment is skipped (angle=0) rather than computed on a degenerate PCA.
3. **Cropping**: a centroid-centered square window is cropped, half-width = `max(bbox_h, bbox_w)//2 + 10`
   (a fixed 10px padding beyond the tightest bounding box).
4. **Flip/overlap**: the crop is compared against both its horizontal (`np.fliplr`) and vertical
   (`np.flipud`) mirror. For each: `IoU = sum(crop AND flip) / (sum(crop OR flip) + 1e-6)`.
5. **Final score**: `asymmetry = 1 - (IoU_horizontal + IoU_vertical) / 2` — i.e., the *average* of the
   two fold directions is used, both directions always contribute (not "whichever axis is worse").
6. **Normalization**: no separate normalization step; the IoU-based formula is bounded to `[0,1]` by
   construction. Concern flag at `>0.20` (legacy display-only, not used by the ML model).

## Border (B_circularity, solidity, turning_angle_std) — exact mechanics

Three distinct measurements are in the final feature set, plus two more computed but excluded:

- **B_circularity** (`Code/MelanomaDeterminingStuff/border.py`, unchanged legacy code): despite its
  name, this stores `1 - circularity` where `circularity = 4π·Area / Perimeter²` — i.e. it is
  **border irregularity**, not circularity. This is a **global shape** measure (how far the whole
  outline departs from a circle), not a local-notch detector.
- **solidity** (`pipeline_v5/feature_extraction.py`): `contour_area / convex_hull_area`. This *does*
  capture local concavities (notches, indentations) because the convex hull only "sees" outward
  projections — a boundary with deep inward notches has low solidity even if its overall aspect
  ratio is circular. This is the most locally-sensitive border measurement in the final set.
- **turning_angle_std** (`pipeline_v5/feature_extraction.py`): std of the discrete turning angle along
  a contour *resampled to 100 equal arc-length points* (not cv2's native, noise-sensitive point list).
  This is explicitly a **local** roughness measure — how much the tangent direction wiggles as you walk
  the boundary — deliberately de-noised by the fixed-point resampling so raw per-pixel rasterization
  jitter from the YOLO mask doesn't dominate the signal.
- **Computed but NOT in the final 17**: `isoperimetric_ratio` (perfectly redundant with B_circularity,
  Spearman rho=1.00 — literally a monotonic transform of it) and `convexity_deficit` (`1 - solidity`,
  rho=-1.00 with solidity — an exact complement, dropped to avoid encoding the same number twice).
  `B_experimental` (`revised_abcd/revised_border.py`, a convexity-defect count) is computed and logged
  for analysis but is **not** one of the 17 model features.

**Contour extraction**: `cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)`, then the
single largest-area contour is used everywhere — holes inside the mask and secondary
fragments/components are ignored by every border measurement (see Part 7 for the implications).

## Color — exact mechanics

**Color space**: LAB throughout (`cv2.cvtColor(..., COLOR_BGR2LAB)`), OpenCV's 8-bit unsigned encoding
(L, a, b each in `[0,255]`, not the signed `-128..127` convention). Every color feature operates on
`no_hair` — the vignette-removed, denoised, bilateral-filtered, hair-removed image — never the raw
original.

**Which statistics are used:**
- **C_value** (legacy `score_color`): coefficient of variation (`std/mean`) of each lesion pixel's LAB
  distance from a sampled skin-baseline LAB color. Not a raw channel stat — it's a *distance-from-skin*
  dispersion measure.
- **color_entropy**: Shannon entropy of a 16×16 2D histogram over the (a,b) plane, i.e. how many
  distinct color "buckets" are populated and how evenly.
- **lab_a_std, lab_b_std**: plain per-channel standard deviation over in-mask pixels (a and b channels
  only — `lab_L_std`, the lightness-channel std, was computed in the ablation study but **excluded**
  from V5's final 17).
- **red_fraction, bluegray_fraction, dark_fraction** (legacy `score_color`, reused unchanged): each is
  the *fraction of lesion pixels* meeting a hand-tuned rule relative to the sampled skin baseline (see
  the CSV for each rule's exact thresholds). These are fractions/proportions, not means or variances.
- **skin_contrast**: mean LAB color inside the mask vs. mean LAB color in a thin ring immediately
  outside it (dilated mask minus mask) — a local, not global, comparison.
- **white_fraction** and `color_cluster_count` (k-means color-cluster count) were computed in the
  ablation study but are **not** in the final 17.

**Masking discipline**: every color feature explicitly indexes `lab[mask > 0]` or equivalent — pixels
outside the lesion mask are excluded by construction (Part 8 audits whether this holds up in practice,
e.g. near ragged mask boundaries).

**Illumination / skin-baseline sensitivity**: `C_value`, `red_fraction`, `bluegray_fraction`, and
`dark_fraction` all depend on the *sampled skin baseline* (ring-sampling with a corner-pixel and
brightness-based fallback chain in `sample_skin_color`) — if that baseline sampling is wrong (e.g. the
ring accidentally includes shadow, another lesion, or a border artifact), every one of these four
features is affected simultaneously, since they share the same baseline call. `color_entropy`,
`lab_a_std`, `lab_b_std`, and `skin_contrast` do **not** depend on the sampled skin baseline at all —
they're computed purely from in-mask (and, for skin_contrast, immediate-ring) pixels.

## Diameter — exact mechanics

- **Definition**: `D_px = 2 × radius`, where `radius` comes from `cv2.minEnclosingCircle()` around the
  largest contour — i.e., the diameter of the *smallest circle that fully encloses the lesion outline*,
  not a bounding-box diagonal and not the maximum pairwise contour-point distance (Feret diameter).
  For an elongated lesion, minEnclosingCircle diameter is close to the longest axis but can slightly
  overstate it depending on shape.
- **Physical units**: **never applied.** `D_mm` is always `None` in the live pipeline
  (`revised_abcd/pipeline_v2.py::score_instance` calls `score_diameter_revised(mask, mm_per_px=None)`
  unconditionally). `revised_diameter.py` *contains* a hair-width-based calibration routine
  (`measure_calibration_revised`, gated by an elongation/aspect-ratio check on detected dark blobs) but
  it is dead code on this path — an earlier investigation found 0 of 40 sampled images had a hair-shaped
  majority among detected dark structures (mostly pigment texture/noise, not actual hair shafts), so the
  calibration was judged untrustworthy and is never invoked to produce `D_mm`.
- **D_px_normalized**: `D_px / sqrt(image_height × image_width)` — a *scale* normalization (making pixel
  diameter comparable across differently-sized source images), explicitly **not** a physical-unit
  normalization. See the CSV entry and Part 9 for why this is flagged as a possible framing/zoom confound.

---

## Summary of direction/quality flags surfaced purely from code + the project's own prior research
(full detail and evidence in the CSV; Part 4 will add locked-test-specific numbers)

| Concern | Features | Type |
|---|---|---|
| Sign flips vs. univariate/clinical direction | `B_circularity` (multivariate coef negative, univariate positive) | Direction contradiction — likely multicollinearity |
| Near-duplicate features (rho≈0.99) | `lesion_fraction` ↔ `D_px_normalized` | Redundancy — both may be a photo-framing/zoom confound rather than true diameter |
| Chance-level univariate signal, nonzero model weight | `eccentricity` (AUC 0.503), `bluegray_fraction` (AUC 0.558), `dark_fraction` (AUC 0.541) | Weak discrimination |
| Counter-intuitive direction, unresolved | `skin_contrast` (lower in melanoma) | Needs human/clinical review |
| Sparse/skewed distribution driving AUC | `red_fraction` (both class medians ≈0) | Needs case-level review |
| Not an ABCD signal at all | `confidence` (YOLO detection confidence, AUC 0.470) | By design — quality covariate, not clinical feature |

These are preliminary, code+prior-research-derived flags. Part 4 recomputes these statistics directly
on the locked-test set (rather than relying on the dev-set ablation numbers above) once the
reconstructed per-image feature file is available.
