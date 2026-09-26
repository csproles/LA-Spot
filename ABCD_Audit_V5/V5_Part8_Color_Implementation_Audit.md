# Part 8 — Color Implementation Audit

Sources audited: `Code/MelanomaDeterminingStuff/color.py` (`score_color`, `sample_skin_color`,
`pixelate` — legacy, unchanged, drives `C_value`/`red_fraction`/`bluegray_fraction`/`dark_fraction`),
`pipeline_v5/feature_extraction.py` (`color_stats_v5_features`, `skin_contrast_feature` — new),
`Code/ComputerVisionStuff/vignette_remover.py` (`remove_vignette`, the source of `circle_info`).

Code-level audit only; no clinical judgment about whether a color pattern looks malignant.

## Checklist

**Only pixels inside the lesion mask used?** — Yes, confirmed by direct indexing in every function:
`lab[mask > 0]` (`color_stats_v5_features`, `skin_contrast_feature`'s "inside" term), `lab[m > 0]`
(the ring term), and `pix_lab[mask > 0]` / `lesion_lab_dc = pix_small_lab[mask > 0]` (`score_color`).
No function in this group operates on the full unmasked image. Background/off-lesion pixels are not
accidentally included through this pathway.

**Color space** — LAB exclusively (`cv2.cvtColor(..., COLOR_BGR2LAB)`), OpenCV's 8-bit unsigned
convention (a/b channels in `[0,255]`, offset-encoded, not the signed CIE `-128..127` range). This is
applied consistently — every color feature's histogram range (`range=[[0,255],[0,255]]` in
`color_stats_v5_features`) correctly matches this encoding; a common bug in this kind of code would be
mixing the signed and unsigned conventions, which does not appear to happen here.

**Normalization** — None applied to the LAB values themselves (no per-image contrast/white-balance
normalization). `C_value` and the red/bluegray/dark fraction rules are all *relative to a sampled skin
baseline* (see below) rather than absolute thresholds, which is a soft form of per-image normalization,
but there is no explicit histogram equalization, color constancy correction, or similar step anywhere in
this chain — every measurement is only as reliable as (a) the raw image's own color fidelity/white
balance as captured, and (b) the quality of the sampled skin baseline.

**Illumination sensitivity — this is the single biggest implementation risk in the color group.**
`C_value`, `red_fraction`, `bluegray_fraction`, and `dark_fraction` all depend on `sample_skin_color`,
which has a **three-tier fallback chain**:
1. Primary: median LAB in a ring at 60–80% of the vignette-detected circle radius, excluding lesion
   pixels (`circle_info` required, from `remove_vignette`).
2. If that ring yields <100 clean (non-lesion) pixels: re-sample the *same* ring but *without* excluding
   lesion pixels (i.e., the fallback ring can include lesion pixels in the "skin" baseline) — a
   `print("few clean pixels, using full ring")` side-effect note, easy to miss in a batch run.
3. If the resulting baseline's L channel is still <60 (interpreted as "too dark," e.g. a poorly-lit
   photo or a baseline still contaminated by lesion pixels): fall back again to sampling the four image
   **corners**, with an additional bright-pixel-only fallback if even the corners are too dark.

Separately, `remove_vignette` itself can return `circle_info=None` whenever no sufficiently large bright
circular region is found (its own threshold: detected bright-region area must be ≥30% of total image
area) — in that case `score_color` skips `sample_skin_color` entirely and goes straight to the
corner-sampling branch. **For any ISIC image that is not a classic circular-vignette dermoscope capture
(e.g., a clinical/macro photo, or a dermoscope image with a non-standard border), the skin baseline is
being estimated from image corners, not from tissue actually adjacent to the lesion.** Whether this
happens often in this dataset is an empirical question this audit cannot answer from code alone — Part 4
should check what fraction of the reconstructed locked-test rows likely went through non-primary
fallback paths (this isn't logged into any of the 17 feature columns, so it would need to be inferred,
e.g. by re-running `remove_vignette` standalone and checking how often it returns `None`).

**Number-of-colors logic** — `color_cluster_count` (k-means, k=6, LAB pixels, counts clusters with
≥5% population share) exists in `pipeline_v5/feature_extraction.py`'s ablation-era sibling module
(`Evaluation_FeatureEngineering/extract_features.py`) but is **explicitly excluded** from V5's final
`pipeline_v5/feature_extraction.py` (the module docstring says so directly: "lab_L_std and
color_cluster_count were computed in the ablation study but are NOT part of the final V5 feature set and
are intentionally not reproduced here"). So there is currently **no** "number of distinct colors" feature
in the live 17-feature model at all — color variegation is captured only via entropy/std/fraction
measures, not a literal color-count.

**Mean/std calculations** — `lab_a_std`/`lab_b_std` are plain per-channel `np.std` over in-mask pixels.
`C_value` is `std(distances)/mean(distances)` — a coefficient of variation of *distance from baseline*,
not a raw channel statistic. `skin_contrast` compares `.mean(axis=0)` of inside-mask vs. ring pixels.
All are straightforward, correctly-masked NumPy reductions; no numerical issues found in the formulas
themselves beyond the small-mask risk below.

**Thresholds** — `red/bluegray/dark_fraction` each apply hand-tuned, hard-coded thresholds relative to
the sampled skin baseline (e.g. `A_channel > skin_A + 10`, `L_channel < skin_L - 60`, etc. — see the
Feature Inventory CSV for the exact values). These constants are not derived from this dataset or
validated against it in the code; they read as generic dermatology heuristics carried over from the
original implementation. `bluegray_fraction`/`dark_fraction` additionally have a **hard 0.15 / 0.12
floor** ("if condition True, value is AT LEAST 0.15/0.12, never something smoothly in between") — this
step-function behavior partly explains why both features show identical benign/malignant medians of 0.0
in the dev-set analysis (Part 1/2): the *relative* boolean condition either doesn't fire (value stays at
the `_abs` fraction, often 0) or fires and jumps straight to 0.15/0.12, with no smooth values observed
in between at the population level.

**Extremely small masks** — Inconsistent handling across the group:
- `color_stats_v5_features` and `skin_contrast_feature` both have an explicit `if m.sum() < 20: return
  NaN` guard.
- `score_color` (the legacy function driving `C_value`/`red_fraction`/`bluegray_fraction`/`dark_fraction`)
  has **no minimum-mask-size guard at all** — `lesion_lab = pix_lab[mask > 0]` and the downstream
  `np.std`/`np.mean`/`np.median` calls will run on however many pixels are in the mask, including a
  mask with only 1–2 foreground pixels. `np.std` of a single value is `0`, which would make `color_cv`
  evaluate to `0/mean` — not a crash, but a silently misleading near-zero CV for a lesion whose color
  variation genuinely could not be assessed from so little data. This asymmetry (new V5 functions guard
  against tiny masks; the legacy function that four of the eleven-plus-six features depend on does not)
  is a real, code-confirmed gap.

**Hair/glare/artifact sensitivity** — Hair is explicitly addressed upstream: every color feature
operates on `no_hair` (post `remove_hair`, a blackhat-filter-based hair inpainting/removal step), not the
raw image. **Glare/specular highlights are not specifically addressed anywhere in this chain** — a bright
specular reflection would read as a very high-L pixel, which is exactly the definition used for
`white_fraction` in the (ablation-only, excluded) legacy code — but since `white_fraction` is **not** one
of the final 17 features, a glare-heavy image doesn't get flagged via that channel. Glare could still
distort `lab_a_std`/`lab_b_std`/`color_entropy`/`C_value` (a cluster of near-white glare pixels pulls the
LAB distribution and its entropy) without any dedicated safeguard.

## Summary of implementation issues found

1. **Illumination/skin-baseline sensitivity via a 3-tier fallback chain**, ultimately falling back to
   image corners for any image without a detectable circular vignette — the single most consequential
   risk in this group, affecting 4 of the 17 model features simultaneously (`C_value`, `red_fraction`,
   `bluegray_fraction`, `dark_fraction`) since they all share one `sample_skin_color` call.
2. **No small-mask guard in the legacy `score_color` path** (in contrast to the new V5 color functions,
   which do guard at 20px) — a real inconsistency, not present in the newer code.
3. **No dedicated glare/specular-highlight handling**; the one feature that would have flagged it
   (`white_fraction`) is excluded from the final 17.
4. `bluegray_fraction`/`dark_fraction`'s hard 0.12/0.15 step-function floors plausibly explain their
   near-zero discriminative power observed in Part 1/2/4 — a code-level explanation for a
   statistically-weak feature, not just an empirical curiosity.

Items 1 and 4 are best illustrated with example visualizations (lesion + computed skin baseline + color
statistic overlay) on representative cases once Part 5's case selection is available — particularly
cases lacking a detected circular vignette, to directly show the corner-fallback baseline in action.
