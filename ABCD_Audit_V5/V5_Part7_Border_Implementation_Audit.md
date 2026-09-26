# Part 7 — Border Implementation Audit

Sources audited: `Code/MelanomaDeterminingStuff/border.py::score_border` (B_circularity, legacy,
unchanged), `pipeline_v5/feature_extraction.py::geometry_v5_features`/`turning_angle_std` (solidity,
turning_angle_std), `revised_abcd/revised_border.py::score_border_experimental` (B_experimental —
computed and logged, **not** one of the 17 model features).

This is a code-level audit; it does not render a clinical judgment about whether a border looks
malignant. Debug visualizations for representative cases are added once Part 5's case list is final.

## Checklist

**Contour extraction** — All three border measurements independently call
`cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)` and then take
`max(contours, key=cv2.contourArea)` — i.e. **only the single largest external contour** is used
everywhere in this group. `RETR_EXTERNAL` means **internal holes are structurally invisible** to every
border feature (see "holes inside masks" below).

**Circularity (B_circularity)** — `circularity = 4π·Area / (Perimeter² + 1e-6)`;
`B_circularity` (the stored feature name) = `1 - circularity`. As documented in Part 1/2, this is a
**global** shape-irregularity measure — sensitive to the overall departure from a circle, not to any
one specific local notch. A lesion with one small notch and an otherwise near-perfect circular outline
will show a *smaller* B_circularity value than a lesion whose entire outline is gently but uniformly
wavy, even if a human reviewer would call the first case "more irregular" — this is an inherent property
of using Area/Perimeter² as a *summary statistic*, not a code bug, but worth flagging in Part 5 review
(this is the "evaluates overall shape vs. actual local border irregularity" distinction the audit asked
for).

**Solidity** — `contour_area / convex_hull_area` (`cv2.convexHull`, `cv2.contourArea`). Unlike
B_circularity, solidity *is* locally sensitive: a single deep inward notch reduces the convex hull's
"coverage" of the contour area meaningfully even if the rest of the outline is smooth, because the hull
is computed from the *actual* contour points, not a smoothed approximation. Solidity and B_circularity
therefore measure genuinely different (if correlated — Spearman rho was not directly reported for this
pair, though both correlate with the excluded `isoperimetric_ratio`/`convexity_deficit`) aspects of
irregularity: one "how far the whole outline strays from round," the other "how much of the convex
envelope is filled in."

**Turning-angle std** — The one feature in this group specifically designed to measure **local** border
roughness: it resamples the contour to a fixed 100 equal-arc-length points, computes the discrete
turning angle between each consecutive pair, and takes the std. The fixed-point resampling is a genuine,
deliberate de-noising step (documented in the code comment) — using cv2's native, unresampled contour
point list directly would make this measure highly sensitive to per-pixel rasterization noise in the
YOLO mask rather than actual boundary shape. This is the most defensible "local irregularity" measure of
the three.

**Convexity ("B_experimental") — computed but not used in the decision** — `revised_border.py`'s
convexity-defect count is recorded per instance (`B_experimental`, `B_experimental_n_defects` columns in
`cohort_v2_results.csv`) purely for analysis/comparison; it is explicitly **not** part of the frozen
V5 17-feature vector (confirmed: absent from `V5_CONFIG["features"]` in `pipeline_v5/decision_model.py`).
This is worth surfacing because it means a border-irregularity signal *was already investigated and
computed at scale* but never adopted — Part 12/13 should treat re-examining it as a genuinely
low-cost next step (the code and cached values already exist) rather than a from-scratch idea.

**Perimeter calculations** — `cv2.arcLength(contour, True)` (closed-contour arc length over the raw,
un-resampled contour point list) is used for B_circularity; note this is a *different* perimeter
representation than the one `turning_angle_std` implicitly uses (its 100-point resampled polyline).
Neither is wrong, but it means "perimeter," in effect, means two different things depending on which
border feature is being computed — not a bug, but a documentation gap worth naming precisely so a future
reader doesn't assume the two features share an underlying perimeter value.

**Contour smoothing** — Only `turning_angle_std` performs any smoothing (via fixed-point resampling).
`B_circularity` and `solidity` operate on the raw `CHAIN_APPROX_NONE` contour (every boundary pixel, no
polygon simplification) — meaning both are, in principle, sensitive to single-pixel-level rasterization
jaggedness in the YOLO mask, not just genuine lesion-boundary shape. For a mask with a naturally jagged
raster edge (common with imperfect segmentation), this could inflate `1 - circularity` and depress
`solidity` for reasons that are a segmentation artifact rather than a real irregular border.

**Small contour handling** — None of the three functions have an explicit minimum-contour-size guard
(unlike, e.g., the color functions' `>= 20 pixels` floor, or asymmetry's `MIN_PIXELS_FOR_ALIGNMENT = 5`).
`geometry_v5_features` does check `area <= 0` (returns all-NaN) but does not guard against a *very small
but nonzero* contour — a 3-pixel sliver, for instance, would still produce a (likely wildly unstable)
`solidity`/`turning_angle_std`/`B_circularity` value rather than NaN. This is the same category of risk
flagged for asymmetry in Part 6 (§ "handling of very small lesions") and should be checked against the
same `lesion_area_px` data once available.

**Fragmented masks** — Because every border feature explicitly reduces to "the single largest external
contour," a fragmented mask (multiple disconnected components) is handled *by ignoring every component
except the largest* — silently. This is actually the *most defensible* of the three ways this repo's
code handles fragmentation (compare: asymmetry and color, Parts 6/8, both use *all* mask pixels
including fragments) — but it means border features and color/asymmetry features are looking at
different effective "shapes" for the same fragmented mask, which is the cross-cutting inconsistency
flagged in Part 6. There is also no flag or NaN propagated when this happens — a fragmented mask's
border features look exactly like a clean single-component mask's, numerically, with no signal in the
feature vector itself that a fragment was discarded (the `quality_flags`/`fragmented(...)` note exists
only in the separate metadata columns, not in any of the 17 model features).

**Holes inside masks** — `RETR_EXTERNAL` retrieval mode means internal holes (e.g. a ring-shaped
segmentation artifact, or a genuinely unsegmented lighter region inside the lesion) are **completely
invisible** to `cv2.contourArea`/`cv2.arcLength`/`cv2.convexHull` — the contour traced is always the
outer boundary only, regardless of what's inside it. A mask with a hole would report the *same*
B_circularity/solidity/turning_angle_std as an identical mask with the hole filled in. This is standard
`RETR_EXTERNAL` behavior, not a bug, but it does mean these features are blind to a specific class of
segmentation artifact (holes) that would visually look like a data-quality problem to a human reviewer.

**False-positive mask regions** — Not specifically guarded against; if YOLO's mask includes a
false-positive region attached to (touching/overlapping) the true lesion, it becomes part of "the
largest contour" and directly distorts every border measurement. If the false-positive region is
disconnected from the true lesion, it's discarded by the largest-contour selection (assuming it's
smaller) — but if it happens to be *larger* than the true lesion, the largest-contour logic would score
the false-positive region's shape instead of the lesion's, with no safeguard against this.

## Summary of implementation issues found

1. **No small-contour floor** for B_circularity/solidity/turning_angle_std (code-confirmed) — unlike
   asymmetry's 5px gate or color's 20px gate.
2. **Fragmentation handled by silent single-largest-contour selection**, with no flag propagated into
   the feature vector itself, and inconsistent with how asymmetry/color handle the same fragmented mask
   (Part 6 cross-reference).
3. **RETR_EXTERNAL blindness to internal holes** — standard OpenCV behavior, but a real blind spot for
   this specific application.
4. **No guard against a larger false-positive region dominating the largest-contour selection.**
5. B_circularity/solidity operate on the raw, un-smoothed pixel contour (rasterization-noise-sensitive);
   only turning_angle_std explicitly de-noises via resampling.

Items 1, 2 and 5 are best illustrated with contour overlays on selected cases (fragmented masks,
very-small lesions, ragged raster edges) once Part 5's case selection is available.
