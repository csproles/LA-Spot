# Part 9 — Diameter Implementation Audit

Sources audited: `revised_abcd/revised_diameter.py` (`score_diameter_revised`,
`measure_calibration_revised`, `_measure_hair_width_px_verified`), `pipeline_v5/feature_extraction.py`
(`d_px_normalized`), `revised_abcd/pipeline_v2.py::score_instance` (call site).

## Checklist

**Exact definition of diameter** — `D_px = 2 × radius`, where `radius` is from
`cv2.minEnclosingCircle()` on the largest contour of the mask. This is the diameter of the *smallest
circle fully enclosing the lesion's outline* — **not** a bounding-box diagonal, and **not** the maximum
pairwise distance between any two contour points (the "Feret diameter," which is the more common
clinical-image-analysis definition of "longest diameter"). For a convex, roughly round lesion the two
definitions are nearly identical; for an elongated or irregular lesion, `minEnclosingCircle` diameter can
modestly *overstate* the longest axis (the enclosing circle must cover the full width span too, not just
the length), and will differ more the less circular/more elongated the shape is. This is a specific,
code-confirmed definitional choice worth naming precisely, since "diameter" reads as an obvious concept
but this implementation is not measuring the single most standard definition of it.

**Mask dimensions** — Operates directly in the mask's own pixel space (no cropping/resizing beforehand);
consistent with how border/asymmetry features are computed.

**Bounding box vs. maximum contour distance vs. enclosing circle** — As above: neither bounding-box
diagonal nor true max-pairwise-distance is used; `minEnclosingCircle` is a third, distinct definition.

**Pixel-based vs. physical measurement** — **Pixel-based only, permanently.** `D_mm` is always `None` in
the live pipeline: `revised_abcd/pipeline_v2.py::score_instance` calls
`score_diameter_revised(mask, mm_per_px=None)` unconditionally — this is not a fallback path that
sometimes fires, it is the only path ever exercised in production. The module's own docstring documents
why: an investigation (`investigation/investigate_diameter.py`, referenced but its output not
independently re-verified in this audit) found that across a 40-image sample, the hair-detecting
blackhat filter's output was dominated by small round blobs (pigment texture/noise), not actual
hair-shaft-shaped structures — so 0/40 images had a hair-like majority among detected dark structures,
meaning the "no reliable hair → None" fallback the calibration code *does* implement never actually gets
past its own elongation-majority gate in practice. **`D_px` is therefore never a physical size
measurement anywhere in this codebase's live path** — a fact worth stating unambiguously, since a
feature literally named "Diameter" strongly implies a physical-unit quantity to anyone reading model
output without the code in front of them.

**Any normalization applied** — `D_px_normalized = D_px / sqrt(image_height × image_width)`. This is a
**scale** normalization (making the pixel diameter comparable across images taken at different
resolutions or effective zoom levels) — it is explicitly *not* an attempt at physical-unit conversion,
and the code makes no such claim. See the cross-reference below for why this specific normalization
choice is itself a significant analytical concern.

**Hair-based scale estimation — present in code, never actually used** — `measure_calibration_revised`
implements a real, non-trivial improvement over the *original* (still-present, unused) hair-width
calibration in `Code/ComputerVisionStuff/hair.py`: it adds an elongation gate
(`MIN_ASPECT_RATIO_HAIR_LIKE = 4.0`, ellipse-fit aspect ratio) requiring detected dark-blob components to
actually be shaped like hair (elongated) rather than round pigment/texture blobs, and further requires
hair-like components to be a **majority** of what was detected, not merely present in any amount (the
module docstring explains this majority requirement directly targets a weakness found in the original
threshold-only gate). This is good, careful work — but because it is gated behind `mm_per_px=None` at
the call site, **none of this logic is reachable in the live pipeline.** This is worth flagging plainly:
a real improvement exists in the codebase and is simply not wired in, which is a very different, and much
cheaper, category of "future work" than needing to design a new calibration approach from scratch.

**Handling of missing hair** — When no hair-like evidence survives the gate,
`_measure_hair_width_px_verified` returns `(None, reason_string, n_hair_like, n_blob_like)` with
descriptive reason codes (`no_dark_structures_detected`, `hair_like_not_majority(...)`,
`too_few_hair_like_points(...)`) — good diagnostic granularity, but again, unreachable in production since
the calling code never passes anything but `mm_per_px=None`.

**Multiple hair detections** — Handled via `cv2.connectedComponentsWithStats` + per-component ellipse-fit
aspect-ratio classification (hair-like vs. blob-like), with the majority-vote requirement described
above; not reachable in production.

**Outliers** — No outlier detection/clipping for `D_px` itself; whatever the enclosing circle computes
is used as-is. Given `D_px` is derived from the same largest-contour logic as the border features, it
inherits the same "largest contour only, no small-contour floor" characteristics documented in Part 7 —
a spurious secondary/false-positive mask region larger than the true lesion would distort `D_px` exactly
as it would distort border measurements.

**Different source image resolutions** — This is precisely what `D_px_normalized` is meant to address,
and the fix is directionally sound (dividing by `sqrt(H×W)` does make the normalized value more
resolution-invariant than raw `D_px`). However — and this is the most important cross-cutting concern in
this section, already raised in Parts 1/2/4 — `D_px_normalized` turns out to be nearly a **duplicate of
`lesion_fraction`** (Spearman rho=0.989, per the project's own prior redundancy analysis). Both
quantities are fundamentally "how much of the photo frame does the lesion occupy," and *both* are
sensitive to a confound that has nothing to do with diameter or lesion size: **how the photo was framed/
cropped/zoomed at capture time.** A genuinely small lesion photographed in extreme close-up would score
high on both features; a genuinely large lesion photographed from further away would score low on both —
in neither case does the feature reflect the lesion's actual physical size. Because this dataset spans
multiple ISIC source collections/studies with presumably differing photography conventions, this is a
plausible dataset-level confound, not merely a redundancy-for-its-own-sake issue. This cannot be resolved
by reading code alone; it requires either (a) checking whether framing convention correlates with
ground-truth label independent of true lesion size (a dataset-level analysis, out of scope for this
code-focused audit), or (b) human visual review of specific high-D_px_normalized cases (Part 5) to check
whether they look like large lesions or like close-up photos of ordinary-sized ones.

## When should the calculated diameter be considered unreliable?

Based purely on the above:

1. **Always**, in the sense that `D_px`/`D_px_normalized` never represent a physical measurement — any
   consumer of this pipeline's output treating "Diameter" as millimeters would be flatly wrong; this is
   a labeling/communication risk as much as a measurement one.
2. When the largest contour is dominated by a false-positive mask region rather than the true lesion
   (same risk as Part 7's border features, since `D_px` shares the identical largest-contour logic).
3. When `D_px_normalized`/`lesion_fraction` are both elevated — the case where it is least possible to
   tell, from the number alone, whether that reflects a genuinely large lesion or a close-up/zoomed
   photograph.

## Summary of implementation issues found

1. **`D_mm` calibration code exists, is a genuine improvement over the legacy version, and is entirely
   unreachable in production** (`mm_per_px=None` is hard-coded at the only call site) — a low-cost,
   already-built piece of future work if physical calibration is ever revisited, distinct from needing
   new engineering.
2. **`D_px` uses `minEnclosingCircle`, not bounding-box diagonal or true Feret/max-pairwise-distance** —
   a specific, non-obvious definitional choice.
3. **No small-contour or outlier guard** on `D_px`, inheriting the same largest-contour risks as Part 7.
4. **`D_px_normalized` ≈ `lesion_fraction` (rho=0.989)** — the most important cross-cutting concern,
   carried over from Parts 1/2/4: both may be measuring photo framing/zoom rather than lesion diameter.
