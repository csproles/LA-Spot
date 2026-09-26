# ABCD Improved Experimental — Feature Definitions

Authoritative reference for every new column in `dev_feature_table_v2.csv` that is not already
present in `extended_feature_table.csv`. This is a **feature-engineering deliverable only** — no
model was fit or trained here (that is the next agent's job). Nothing in `pipeline_v5/`,
`revised_abcd/`, `Code/`, or any existing `Evaluation_*` directory was modified; the locked test
set was never opened.

All code lives in `ABCD_Improved_Experimental/`:
- `common_features.py` — shared feature-computation functions.
- `verify_axis_convention.py` — the empirical major/minor-axis verification for Phase 4 (run,
  output captured below).
- `build_mask_features.py` — mask-only pass (Phase 1 shape half, Phase 4, Phase 5, Phase 7).
- `build_image_features.py` — image pass (Phase 1 color half, Phase 3).
- `build_dev_feature_table_v2.py` — merges everything into `dev_feature_table_v2.csv`.

---

## Phase 1 — Consistent primary-component mask handling

**Problem being fixed** (per `ABCD_Audit_V5/V5_Part6_Asymmetry_Implementation_Audit.md` and
`V5_Part7_Border_Implementation_Audit.md`): V5's shape features (asymmetry, border, diameter) call
`largest_contour(mask)` — only the single largest connected contour — while its color features
index `mask > 0` directly, i.e. every foreground pixel including secondary mask fragments. A
fragmented mask therefore produces internally inconsistent inputs across the 17-feature vector.

**Fix**: `common_features.select_primary_component(mask)` runs
`cv2.connectedComponentsWithStats(mask, connectivity=8)` and returns a new mask containing ONLY the
single largest connected component (by pixel area). Every `_pc`-suffixed feature below is computed
from this one consistent mask, for BOTH shape and color.

### Recomputed shape features (mask-only; `common_features.shape_features_pc` / `fold_asymmetry_hv`)

| Column | Formula | Source formula reused from |
|---|---|---|
| `B_circularity_pc` | `1 - 4*pi*contour_area / (perimeter^2 + 1e-6)` on the primary-component contour | `Code/MelanomaDeterminingStuff/border.py::score_border` |
| `solidity_pc` | `contour_area / convex_hull_area` | `pipeline_v5/feature_extraction.py::geometry_v5_features` |
| `turning_angle_std_pc` | std of discrete turning angle over a 100-point arc-length-resampled contour | same, `turning_angle_std` |
| `D_px_pc` | `2 * radius` from `cv2.minEnclosingCircle()` on the primary-component contour | `revised_abcd/revised_diameter.py::score_diameter_revised` |
| `A_value_pc` | mean of `asymmetry_h`/`asymmetry_v` below — i.e. V5's own `A_value` formula, recomputed on the primary-component mask | `revised_abcd/revised_asymmetry.py::score_asymmetry_revised` |

### Recomputed color features (image pass; `common_features.score_color_pc` / `color_stats_pc` / `skin_contrast_pc`)

| Column | Formula | Source formula reused from |
|---|---|---|
| `C_value_pc` | coefficient of variation of primary-component-pixel LAB distance from sampled skin baseline | `Code/MelanomaDeterminingStuff/color.py::score_color` (called with the primary-component mask) |
| `red_fraction_pc`, `bluegray_fraction_pc`, `dark_fraction_pc` | same hand-tuned skin-baseline-relative rules as V5, evaluated only over primary-component pixels | same `score_color` call |
| `color_entropy_pc` | Shannon entropy of a 16x16 2D histogram over the (a,b) LAB plane, primary-component pixels only | `pipeline_v5/feature_extraction.py::color_stats_v5_features` |
| `lab_a_std_pc`, `lab_b_std_pc` | per-channel std over primary-component pixels | same |
| `skin_contrast_pc` | `\|\|mean(LAB inside primary component) - mean(LAB in thin ring outside it)\|\|` | `pipeline_v5/feature_extraction.py::skin_contrast_feature` |

`white_fraction` was never one of V5's 17 features and is not recomputed here (out of scope, per
the task brief's focus on the 17 named features + the audit's fragmentation issue).

**Validation**: for every image whose original raw mask is already a single connected component
(the majority of the 2090 dev images, since these are per-instance YOLO masks), every `_pc` column
reproduces its corresponding original V5 column to full numerical precision — confirmed directly
(e.g. `ISIC_0073863`: `B_circularity`=0.299 vs `B_circularity_pc`=0.298545 rounding difference only
from the original CSV's 3-decimal rounding; `C_value`=0.314 vs `C_value_pc`=0.314 exact;
`color_entropy`=1.116461 vs `color_entropy_pc`=1.116461 exact; `lab_a_std`, `lab_b_std`,
`skin_contrast` all exact). Divergence between a `_pc` column and its original V5 counterpart is
therefore a direct, interpretable signal of how much a fragmented/multi-component mask was
affecting that original feature.

### Quality flags (`common_features.compute_quality_flags`, all mask-only)

Reuses `revised_abcd/segmentation_quality_v2.py`'s existing constants where they overlap
(`TINY_ARTIFACT_PX = 20`, `LARGE_AREA_FRACTION = 0.5`), unchanged.

| Flag | Definition | Threshold source |
|---|---|---|
| `flag_fragmented` | more than 1 connected component with area > `TINY_ARTIFACT_PX` (20px) | reused unchanged from `segmentation_quality_v2.py` |
| `flag_multiple_substantial_components` | a stricter version: a 2nd meaningful component whose area is > 30% of the primary component's area | **new, our own choice** — `SECOND_COMPONENT_FRACTION = 0.30` in `common_features.py`. No existing constant in the codebase covers "is the 2nd blob a real 2nd-lesion candidate rather than raster noise"; 30% was chosen as a round, conservative threshold clearly above raster-noise scale (which `TINY_ARTIFACT_PX` already filters at 20px) but well below "roughly equal-sized blobs." Not fit to this dataset's numbers. |
| `flag_edge_adjacent` | the primary component's bounding box comes within `max(5px, 1% of min(image_h, image_w))` of any image edge | **new, our own choice** — `EDGE_MARGIN_FRACTION = 0.01`, `EDGE_MARGIN_MIN_PX = 5` in `common_features.py`. This is exactly the risk documented in `V5_Part6_Asymmetry_Implementation_Audit.md` ("edge-clamped crop asymmetry"): a lesion whose bbox nearly touches the frame edge gets its `_fold_asymmetry` crop window asymmetrically clamped, which can inflate asymmetry independent of true lesion shape. |
| `flag_very_small_contour` | primary component area < the 1st percentile of `lesion_area_px` across all 2090 dev images (computed from `extended_feature_table.csv`, **12445.1 px** at run time — see `build_mask_features.py`'s printed value for the exact number used) | **new, our own choice**, but grounded in the dataset's own distribution rather than an arbitrary pixel count, per the task's suggestion. The 1st percentile (not the 5th) was chosen to keep this a genuine-outlier flag (~1% of images) rather than flagging a routine 5% of the cohort as "very small." |
| `flag_oversized_mask` | primary component covers > `LARGE_AREA_FRACTION` (0.5) of the full image | reused unchanged from `segmentation_quality_v2.py` |

`quality_flags` is the semicolon-joined string of whichever flag names fired (`"ok"` if none) — same
convention as `segmentation_quality_v2.py`. **No image is dropped for tripping any flag** — every one
of the 2090 images gets a full row regardless; filtering, if any, is left entirely to the modeling
phase. See `robustness_guard_report.md` for the full per-flag counts and a discussion of which
features each flag is most relevant to.

---

## Phase 3 — Channel-specific color entropy

`entropy_L`, `entropy_a`, `entropy_b` (`common_features.channel_entropy_lab`): 1D Shannon entropy of
each LAB channel individually, computed as `-sum(p * log2(p))` over a 16-bin histogram (range
`[0,255]`, matching OpenCV's unsigned 8-bit LAB encoding already used throughout this codebase) of
that channel's values, restricted to the primary-component mask, on the same preprocessed
(vignette-removed, denoised, bilateral-filtered, hair-removed) image already loaded for the Phase 1
color features — no second image load.

**Bin-count choice**: 16 bins was chosen to match the existing `color_entropy`/`color_entropy_pc`
2D-histogram convention (`16x16` bins over the a/b plane in `color_stats_v5_features`), for direct
comparability rather than introducing a second, differently-binned entropy convention into the same
feature table. Range `[0,255]` matches the unsigned OpenCV LAB encoding, again for consistency with
every other color feature already in this codebase.

**Literature basis** (`ABCD_Audit_V5/V5_ABCD_Literature_Review.md` section C3): Martínez-Ortega &
Martinez-Jaramillo, "Beyond Global Shannon Entropy: A Channel-Specific Approach to Quantify
Polychromia in Melanoma" (Cureus, 2026, DOI 10.7759/cureus.102257) directly argue that a single
global Shannon entropy value does not reliably capture polychromia (multi-color variegation) under
real-world imaging conditions, and that channel-specific entropy discriminates chromatically
heterogeneous lesions better. V5's existing `color_entropy`/`color_entropy_pc` is a single value
computed over the joint (a,b) 2D histogram — `entropy_L/a/b` add the channel-specific decomposition
this paper recommends, without removing or altering the existing joint-entropy feature.

---

## Phase 4 — Separate major/minor-axis asymmetry

`asymmetry_h` and `asymmetry_v` (`common_features.fold_asymmetry_hv`): the same PCA-aligned
fold-IoU math `_fold_asymmetry`/`score_asymmetry_revised` already use (vendored unmodified from
`revised_abcd/revised_asymmetry.py` via `_vendored_revised_asymmetry.py`), applied to the
**primary-component mask** (Phase 1 fix), but keeping the two fold directions SEPARATE instead of
V5's single averaged `A_value = 1 - mean(IoU_h, IoU_v)`:

- `asymmetry_h = 1 - IoU(crop, np.fliplr(crop))` — mirror across a VERTICAL line (array axis=1,
  compares the left half of the aligned lesion crop against the mirrored right half).
- `asymmetry_v = 1 - IoU(crop, np.flipud(crop))` — mirror across a HORIZONTAL line (array axis=0,
  compares the top half against the mirrored bottom half).
- `A_value_pc = mean(asymmetry_h, asymmetry_v)` is also kept, as the direct primary-component-mask
  analogue of V5's original `A_value` (Phase 1).

### Empirical verification of the major/minor-axis convention

Per the task's explicit instruction not to assign major/minor labels without verifying, this was
tested empirically in `verify_axis_convention.py` rather than assumed, using synthetic masks with a
precisely known geometry (not real lesion images). Two things were checked:

**1. Does PCA alignment (`_principal_axis_angle_deg` + `cv2.getRotationMatrix2D`) actually make the
mask's major (elongation) axis horizontal, as the source module's docstring claims?**
Confirmed directly: for a plain ellipse (semi-major=480px, semi-minor=180px) constructed at five
different photographed rotations (0, 25, 60, 90, -40 degrees), the ALIGNED mask's own
independently-recomputed principal-axis angle was within 0.01 degrees of 0 or 180 (mod 180, i.e.
horizontal) in every case. **The major axis is always horizontal after alignment**, for every image
this pipeline processes — this is what makes a fixed fold-direction-to-axis mapping possible at all
(rather than needing to be re-derived per image).

**2. Given a horizontal major axis, which fold — `asymmetry_h` (fliplr, axis=1) or `asymmetry_v`
(flipud, axis=0) — responds to asymmetry placed between the two ENDS of the major axis, and which
responds to asymmetry placed between the two SIDES of the minor axis?**
Two families of synthetic shapes were built, each asymmetric in exactly one of these two ways by
construction (an implicit-boundary "egg" shape with a different semi-axis length on its `+`/`-` half
along one axis only, leaving the perpendicular axis exactly symmetric), each tested at the same five
rotation angles, with results compared against a plain symmetric ellipse at the same rotation to
isolate genuine asymmetry response from baseline rasterization noise (a small, real, direction-
dependent effect present even for a perfectly symmetric ellipse, roughly 0.003-0.007 on `asymmetry_h`
vs 0.007 on `asymmetry_v` in this test geometry — see the "Step 1" baseline rows in the script's
output — attributable to `cv2.warpAffine(..., INTER_NEAREST)` boundary aliasing being a larger
relative effect on the shorter axis).

Results (mean delta vs. the same-rotation symmetric-ellipse baseline, across all 5 rotations):

```
MAJOR-axis-end asymmetry (egg_x)  -> mean d(asym_h)=+0.1042  mean d(asym_v)=+0.0337
MINOR-axis-side asymmetry (egg_y) -> mean d(asym_h)=+0.0006  mean d(asym_v)=+0.1856
```

`egg_y` (minor-axis-side asymmetry) gives a completely clean signal: `asymmetry_v` rises by ~0.186 at
every single one of the 5 rotations tested with zero exceptions, while `asymmetry_h` stays at
~0.0006 (indistinguishable from baseline noise) at every rotation. `egg_x` (major-axis-end asymmetry)
gives a consistent but noisier signal: `asymmetry_h` rises by ~0.10 at every rotation, while
`asymmetry_v` rises by a smaller but non-negligible and less consistent amount (dominated by one
rotation, 60 degrees, where nearest-neighbor rotation-resampling of this specific elongated synthetic
shape introduced extra boundary aliasing — noted directly in the script's output). The 3x gap between
the mean effect sizes (0.104 vs 0.034), combined with the completely clean `egg_y` result, is
considered sufficient to assign the labels with confidence, though the `egg_x` result is somewhat
noisier than `egg_y`'s and this is stated plainly rather than glossed over.

**Verified convention** (used to construct the alias columns in `dev_feature_table_v2.csv`):

```
asymmetry_h (np.fliplr, array axis=1) == asymmetry_major_axis
asymmetry_v (np.flipud, array axis=0) == asymmetry_minor_axis
```

i.e. `asymmetry_major_axis` measures how differently the lesion's shape behaves at the two ENDS of
its own long axis (e.g. a teardrop/pear shape scores high here), and `asymmetry_minor_axis` measures
how differently it behaves on the two SIDES of its own short axis (e.g. a lesion with a bulge on one
long edge only scores high here). Both are carried into `dev_feature_table_v2.csv` as both the
`asymmetry_h`/`asymmetry_v` raw names and the `asymmetry_major_axis`/`asymmetry_minor_axis` aliases
(identical values, both names kept so downstream consumers can use whichever is clearer).

### Color/pigment asymmetry (Phase 4's other request)

`color_asymmetry` — already exists, precomputed, in `extended_feature_table.csv` for all 2090 dev
images (via `pca_aligned_asymmetry_generic(mask, lab_of(no_hair))` in the original ablation study's
`extract_features.py`, itself using the SAME PCA alignment as the shape asymmetry above but comparing
LAB-value distributions between PCA-aligned halves rather than mask overlap). This satisfies Phase
4's color/pigment-asymmetry request with already-existing data — it is carried through unchanged into
`dev_feature_table_v2.csv` and was **not** recomputed.

---

## Phase 5 — Fractal dimension border feature

`border_fractal_dimension` (`common_features.border_fractal_dimension`): box-counting estimate of the
fractal dimension of the primary-component lesion **boundary** (the contour, drawn as a 1px-thick
line via `cv2.drawContours(..., thickness=1)` — not the filled lesion area).

**Algorithm**: for box sizes `s` in powers of two from `min_box=2` px up to
`floor(0.25 * min(bbox_w, bbox_h))` px (bbox of the boundary contour), count the number of `s x s`
grid boxes that contain at least one boundary pixel (`_box_count`, a vectorized reshape-and-`any()`
over the boundary image, padded to a multiple of `s`). Fit an ordinary-least-squares line to
`log(box_count)` vs. `log(1/box_size)` (`np.polyfit`, degree 1); the slope is the fractal-dimension
estimate. `border_fractal_dimension_n_box_sizes` records how many distinct box sizes were actually
used in the regression fit for that image (typically 5-7 for this dataset's lesion sizes; images
whose primary component is too small to yield at least 4 box sizes get `NaN`).

**Literature basis** (`V5_ABCD_Literature_Review.md` section B1): Claridge, Hall, Keefe, Allen,
"Shape analysis for classification of malignant melanoma" (J Biomed Eng, 1992, PMID 1588780) —
foundational box-counting-style boundary fractal dimension for melanoma; Piantanelli et al.,
"Fractal characterisation of boundary irregularity in skin pigmented lesions" (Med Biol Eng Comput,
2005) extends this with a box-counting fractal-dimension estimator reporting ~85% nevus/melanoma
discrimination using fractal dimension alone. V5 currently has no multi-scale roughness measure — its
three border features (`B_circularity`, `solidity`, `turning_angle_std`) all operate at a single
scale (whole-contour or fixed 100-point resampling); `border_fractal_dimension` is a genuinely new
axis of information, not a redundant reformulation of any existing V5 feature.

**Observed range on this dataset**: predominantly close to 1.0 (roughly 0.95-1.05 in a first look at
mask-pass output), consistent with the fact that box-counting fractal dimension of a smooth curve
approaches 1.0, and YOLO segmentation masks tend to produce fairly smooth boundaries at pixel
resolution; see `robustness_guard_report.md` for the full distribution once the mask pass completed.

---

## Phase 6 — Size/diameter terminology (no new computation)

`D_px`/`D_px_pc` are **pixel-space** enclosing-circle diameters (`2 * cv2.minEnclosingCircle()
radius`) — never physical millimeters, in either the original V5 pipeline or this recomputation
(per `V5_Part9_Diameter_Implementation_Audit.md`, `D_mm` is never populated anywhere in the live
pipeline). `lesion_fraction` and `D_px_normalized` (both carried through unchanged from
`extended_feature_table.csv`) are explicitly flagged, per the same audit's Part 9 finding, as
**potentially framing/zoom-sensitive rather than confirmed physical-size measurements** —
`D_px_normalized` is nearly a duplicate of `lesion_fraction` (Spearman rho=0.989 per the prior
ablation study), and both may reflect how close-up a photo was taken rather than the lesion's true
physical size. No code change was needed or made for this phase; this is a documentation-only note
carried into this table's consumers.

---

## Phase 7 — Robustness guards

Covered by the Phase 1 quality flags above. Full per-flag counts, which features each flag is most
relevant to, and confirmation that flagging (never dropping) was the only action taken are in
`ABCD_Improved_Experimental/robustness_guard_report.md`.
