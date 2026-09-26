# Part 6 — Asymmetry Implementation Audit

Source audited: `revised_abcd/revised_asymmetry.py` (`score_asymmetry_revised`, `_principal_axis_angle_deg`,
`_fold_asymmetry`), called from `revised_abcd/pipeline_v2.py::score_instance`. Upstream mask source:
`revised_abcd/yolo_single_image.py::run_single_image_inference`.

This is a **code-level audit**. It does not judge whether any lesion is clinically asymmetric — that is
explicitly reserved for manual review (Part 5). Debug visualizations for specific TP/TN/FP/FN cases are
generated once case selection from Part 5 is finalized (this file will be updated with links to them).

## Checklist

**Correct use of the segmentation mask** — Yes. The function receives exactly the same per-instance
binary mask (`{0,255}` uint8) that drives every other A/B/C/D feature for that instance; no
re-thresholding or re-binarization happens inside `revised_asymmetry.py` itself.

**Lesion orientation/alignment** — PCA-based, via `_principal_axis_angle_deg`: eigendecomposition
(`np.linalg.eigh`) of the covariance matrix of foreground-pixel `(x,y)` coordinates; the dominant
eigenvector's angle becomes the rotation applied before folding. **Confirmed: PCA is used**, and this
is a genuine, intentional improvement over the (still-present, unused) legacy fixed-axis fold in
`Code/MelanomaDeterminingStuff/asymmetry.py`.

**Centroid handling** — Two different centroid computations are used at two different stages, and they
are **not the same value**:
1. `_principal_axis_angle_deg` uses the *mean of foreground pixel coordinates* (`pts.mean(axis=0)`) as
   the rotation center — an unweighted point-cloud centroid.
2. `_fold_asymmetry` (applied to the *already-rotated* mask) recomputes the centroid via
   `cv2.moments(mask)` — `m10/m00, m01/m00` — the intensity-weighted centroid of the binary mask.
   For a binary mask these two should be numerically very close (both are just "average position of
   foreground pixels") but they are computed via genuinely different code paths, on different frames of
   reference (pre- vs. post-rotation), rather than the same centroid being carried through. This is not
   a correctness bug — cv2.moments on a binary mask reduces to the same pixel-average — but it means the
   two centroid values are never cross-validated against each other; if one implementation ever diverges
   (e.g. mask were multi-valued instead of binary) this would silently produce a subtly-off crop window
   rather than an error.

**Image/mask dimensions** — The mask is always full-image-resolution (YOLO `retina_masks=True`,
resized back to `orig_shape` if needed — see `yolo_single_image.py:50-56`), so no separate
image/mask-size mismatch handling is needed inside the asymmetry code itself; it operates entirely in
mask-pixel space.

**Flip/rotation logic** — `cv2.warpAffine` with `flags=cv2.INTER_NEAREST` is used for the rotation
specifically to keep the mask strictly binary post-rotation (bilinear interpolation would produce
fractional/gray values at the boundary). This is correct practice for a binary mask. The fold step then
uses `np.fliplr`/`np.flipud` (exact, lossless mirrors) on the *already-rotated, already-cropped* mask —
both directions (horizontal and vertical) are always computed and averaged; there is no data-dependent
choice of "worse axis."

**Overlap calculation** — `IoU = sum(crop & flip) / (sum(crop | flip) + 1e-6)`, standard Jaccard
overlap on the binary crop. `asymmetry = 1 - mean(IoU_h, IoU_v)`. The `1e-6` epsilon guards only against
an all-zero crop (empty union) — a legitimate, minimal safety term.

**Normalization** — None beyond the IoU ratio itself; the score is bounded to `[0,1]` by construction
(IoU is itself bounded to `[0,1]`, and the final value is `1 - average-of-two-IoUs`).

**Handling of very small lesions** — `MIN_PIXELS_FOR_ALIGNMENT = 5`: below 5 foreground pixels, PCA
alignment is skipped entirely (`angle=0.0`, identity rotation) rather than computed on a degenerate
covariance matrix. This is a real safety gate, but note the threshold is very low (5 px) — a mask with,
say, 15–30 pixels (still tiny relative to typical lesion sizes of tens of thousands of pixels, see the
`lesion_area_px` medians in the feature-engineering data) *will* get a full PCA alignment despite being
essentially noise-scale, and PCA on a handful of points can be numerically unstable (a near-circular tiny
blob has near-equal eigenvalues, making the "dominant" axis direction close to arbitrary / sensitive to
single-pixel changes). This is a plausible source of unstable A_value scores for very small/low-confidence
detections — worth checking directly against the reconstructed feature data (Part 4) for correlation
between small `lesion_area_px` and A_value volatility or extreme values.

**Handling of fragmented masks / multiple components** — **Not explicitly handled, and this is a
genuine implementation gap.** `_principal_axis_angle_deg` computes its PCA over *every* foreground pixel
in the mask, regardless of how many disconnected components it spans — so if a mask has, say, one large
blob plus a small disconnected fleck (the kind of raster artifact
`revised_abcd/segmentation_quality_v2.py` explicitly flags as `fragmented(...)`), the small fleck still
pulls on the principal-axis direction and on the crop-window's centroid. `_fold_asymmetry` similarly
operates on `mask // 255` directly — all components, not just the largest — unlike the border features
(Part 7), which explicitly call `largest_contour()` and discard everything but the single largest
contour. **This is an inconsistency between the asymmetry feature and the border/geometry features**:
asymmetry (and the raw color features, see Part 8) use *all* mask pixels including secondary fragments,
while border/geometry features use *only the largest connected contour*. A fragmented mask therefore
produces internally inconsistent inputs across the 17-feature vector — some features "see" the fragments,
others don't. This is worth flagging as a feature-calculation consistency concern (Part 10, category B).

**Multiple components** — same as above; there is no connected-component filtering step in this file at
all (contrast with `segmentation_quality_v2.py`, which computes `cv2.connectedComponentsWithStats` purely
to *report* a flag, not to gate any A/B/C/D calculation).

**Padding/cropping effects** — The crop window (`half = max(bbox_h,bbox_w)//2 + 10`, clamped to image
bounds via `max(...,0)`/`min(...,h or w)`) is generous relative to the mask's own bounding box, so for a
lesion well inside the frame, padding shouldn't meaningfully change the measurement. However, **if the
lesion is close to an image edge, the clamp silently makes the crop asymmetric relative to the lesion's
own centroid** (e.g. a lesion near the top of the frame gets `r0` clamped to 0 while `r1` is not
correspondingly reduced) — this means the "mirror" comparison in `_fold_asymmetry` is being computed on
an off-center crop for edge-adjacent lesions, which could inflate the asymmetry score for reasons that
have nothing to do with the lesion's actual shape (an edge-clipped crop will generically look
"asymmetric" against its own mirror even for a genuinely round lesion). This is a real, code-confirmed
failure mode, not a hypothetical — worth checking directly for images where the lesion's bounding box
touches or nearly touches the image border.

## Summary of implementation issues found

1. **Fragmented-mask inconsistency** (code-confirmed): asymmetry uses all mask pixels; border/geometry
   use only the largest contour. Same underlying mask, different effective "shape" seen by different
   features.
2. **Edge-clamped crop asymmetry** (code-confirmed): lesions near the image border get an off-center
   crop window, which can inflate the asymmetry score independent of true lesion shape.
3. **PCA instability on very small masks** (plausible, not yet empirically confirmed): the 5-pixel
   alignment floor is low enough that near-noise-scale masks still get a full PCA rotation; recommend
   cross-checking against `lesion_area_px` in the reconstructed feature data.
4. Two independently-computed centroids (point-mean vs. `cv2.moments`) are mathematically equivalent for
   a clean binary mask but are never cross-validated — low risk, noted for completeness only.

None of these require a code change to observe; #3 needs the locked-test feature data (Part 4) to
confirm empirically, and #1/#2 are best illustrated with debug visualizations on real fragmented/edge
cases once Part 5's case list is finalized.
