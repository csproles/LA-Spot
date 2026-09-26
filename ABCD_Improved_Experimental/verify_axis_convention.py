"""Empirical verification of the major/minor-axis fold convention used by
`_fold_asymmetry` (vendored from revised_abcd/revised_asymmetry.py) once a
mask has been PCA-aligned via `_principal_axis_angle_deg`.

Read-only with respect to revised_abcd/ -- this script imports the exact,
unmodified alignment function and re-derives the SAME fold math (copied
verbatim, not re-invented) to test it on synthetic shapes with known
geometry. Nothing here writes to revised_abcd/.

Two questions answered:
  1. After `_principal_axis_angle_deg` + `cv2.getRotationMatrix2D` rotation,
     does the mask's major (elongation) axis actually end up horizontal
     (array axis=1 / columns), as the docstring in
     revised_abcd/revised_asymmetry.py claims?
  2. Of the two folds `_fold_asymmetry` computes -- `flip_h = np.fliplr`
     (mirror across a VERTICAL line, i.e. compares left half vs right half,
     axis=1) and `flip_v = np.flipud` (mirror across a HORIZONTAL line,
     compares top half vs bottom half, axis=0) -- which one responds to an
     asymmetry deliberately placed at one END of the long (major) axis
     (e.g. a teardrop shape, pointed at one tip), and which responds to an
     asymmetry placed on one SIDE of the short (minor) axis (e.g. a bump on
     one long edge, symmetric between the two tips)?

Output: printed verdict, used verbatim in
ABCD_Improved_Experimental/improved_feature_definitions.md.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VENDORED_DIR = ROOT / "ABCD_Audit_V5" / "_source_from_research_branch" / "Evaluation_FeatureEngineering"
sys.path.insert(0, str(VENDORED_DIR))

from _vendored_revised_asymmetry import _principal_axis_angle_deg  # noqa: E402


def fold_ious(mask):
    """Same crop+flip+IoU math as `_fold_asymmetry`, but returns the two
    IoU values separately instead of only the averaged asymmetry score."""
    h, w = mask.shape
    M = cv2.moments(mask)
    if M["m00"] > 0:
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = w // 2, h // 2

    coords = np.argwhere(mask > 0)
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    half = max(r_max - r_min, c_max - c_min) // 2 + 10
    r0 = max(cy - half, 0); r1 = min(cy + half, h)
    c0 = max(cx - half, 0); c1 = min(cx + half, w)
    crop = (mask[r0:r1, c0:c1] // 255).astype(np.uint8)

    flip_h = np.fliplr(crop)  # axis=1, mirror across a VERTICAL line (left vs right)
    iou_h = np.sum(crop & flip_h) / (np.sum(crop | flip_h) + 1e-6)
    flip_v = np.flipud(crop)  # axis=0, mirror across a HORIZONTAL line (top vs bottom)
    iou_v = np.sum(crop & flip_v) / (np.sum(crop | flip_v) + 1e-6)
    return float(iou_h), float(iou_v)


def align(mask):
    h, w = mask.shape
    angle, center = _principal_axis_angle_deg(mask)
    M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(mask, M_rot, (w, h), flags=cv2.INTER_NEAREST)
    return aligned, angle


def _grid(canvas_hw, center, rot_deg):
    """Pixel-center (x, y) coordinates, rotated by -rot_deg about `center`
    so that evaluating a shape formula in the ROTATED frame produces a
    shape that appears rotated by +rot_deg in the actual image -- i.e. this
    lets every synthetic shape be defined once, in its own canonical
    (unrotated) frame, and then photographed at an arbitrary angle, exactly
    mimicking a real lesion photographed at an arbitrary orientation."""
    h, w = canvas_hw
    cx, cy = center
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    xs -= cx
    ys -= cy
    theta = np.radians(rot_deg)
    xr = xs * np.cos(theta) + ys * np.sin(theta)
    yr = -xs * np.sin(theta) + ys * np.cos(theta)
    return xr, yr


def make_ellipse(canvas_hw, center, axes, rot_deg):
    """Plain, perfectly symmetric ellipse (control shape). axes = (a, b)
    semi-major/semi-minor in its own canonical frame."""
    a, b = axes
    xr, yr = _grid(canvas_hw, center, rot_deg)
    inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return (inside.astype(np.uint8)) * 255


def make_egg_x(canvas_hw, center, a_pos, a_neg, b, rot_deg):
    """Asymmetric-cap ellipse ("egg"), asymmetric ONLY between its two ENDS
    along its own major (x, before photographing rotation) axis: the +x cap
    uses semi-axis `a_pos`, the -x cap uses `a_neg` (a_pos != a_neg), while
    BOTH caps share the same semi-minor `b` for every x -- so the shape is,
    by construction, an EXACT mirror of itself under a flip across its own
    minor axis (top vs bottom, in its own frame) and only breaks symmetry
    under a flip across its own major axis (left tip vs right tip)."""
    xr, yr = _grid(canvas_hw, center, rot_deg)
    a = np.where(xr >= 0, a_pos, a_neg)
    inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return (inside.astype(np.uint8)) * 255


def make_egg_y(canvas_hw, center, a, b_pos, b_neg, rot_deg):
    """Mirror-image construction of `make_egg_x`: asymmetric ONLY between
    the two SIDES along its own minor (y, before photographing rotation)
    axis (b_pos != b_neg on the +y/-y halves), symmetric between its own
    two major-axis tips (same `a` for all y)."""
    xr, yr = _grid(canvas_hw, center, rot_deg)
    b = np.where(yr >= 0, b_pos, b_neg)
    inside = (xr / a) ** 2 + (yr / b) ** 2 <= 1.0
    return (inside.astype(np.uint8)) * 255


def report(name, mask):
    aligned, angle = align(mask)
    # Recompute PCA angle of the ALIGNED mask to confirm the major axis is
    # now horizontal (angle should be close to 0 or 180 mod 180).
    post_angle, _ = _principal_axis_angle_deg(aligned)
    post_angle_mod = post_angle % 180.0
    iou_h, iou_v = fold_ious(aligned)
    print(f"{name:28s} pre_angle={angle:7.2f}  post_align_angle_mod180={post_angle_mod:6.2f}  "
          f"IoU_fliplr(h,axis=1)={iou_h:.4f}  IoU_flipud(v,axis=0)={iou_v:.4f}  "
          f"asym_h={1-iou_h:.4f}  asym_v={1-iou_v:.4f}")
    return post_angle_mod, iou_h, iou_v


def main():
    # Large canvas/axes relative to the earlier draft: nearest-neighbor
    # rotation resampling (required to keep masks strictly binary, exactly
    # as the real pipeline does) introduces boundary aliasing that is a
    # much larger RELATIVE error on a small mask -- scaling everything up
    # keeps the discretization noise a small fraction of the signal being
    # measured, per a first pass at these numbers.
    canvas = (1200, 2000)  # h, w
    center = (1000, 600)  # cx, cy  (x, y) as cv2 expects
    semi_major, semi_minor = 480, 180

    print("=== Step 1: does PCA alignment make the major axis horizontal? ===")
    baselines = {}
    for rot in [0, 25, 60, 90, -40]:
        ell = make_ellipse(canvas, center, (semi_major, semi_minor), rot)
        baselines[rot] = report(f"plain_ellipse rot={rot}", ell)

    print("\n=== Step 2: egg_x -- asymmetric ONLY between the two ENDS of the shape's own")
    print("    MAJOR (long) axis; exactly symmetric between its own minor-axis sides ===")
    results_tip = []
    for rot in [0, 25, 60, 90, -40]:
        eg = make_egg_x(canvas, center, semi_major * 1.35, semi_major * 0.75, semi_minor, rot)
        base = baselines[rot]
        post_angle, iou_h, iou_v = report(f"egg_x(major-end-asym) rot={rot}", eg)
        d_h = (1 - iou_h) - (1 - base[1])
        d_v = (1 - iou_v) - (1 - base[2])
        print(f"    delta vs symmetric baseline: d(asym_h)={d_h:+.4f}  d(asym_v)={d_v:+.4f}")
        results_tip.append((d_h, d_v))

    print("\n=== Step 3: egg_y -- asymmetric ONLY between the two SIDES of the shape's own")
    print("    MINOR (short) axis; exactly symmetric between its own major-axis ends ===")
    results_side = []
    for rot in [0, 25, 60, 90, -40]:
        eg = make_egg_y(canvas, center, semi_major, semi_minor * 1.6, semi_minor * 0.5, rot)
        base = baselines[rot]
        post_angle, iou_h, iou_v = report(f"egg_y(minor-side-asym) rot={rot}", eg)
        d_h = (1 - iou_h) - (1 - base[1])
        d_v = (1 - iou_v) - (1 - base[2])
        print(f"    delta vs symmetric baseline: d(asym_h)={d_h:+.4f}  d(asym_v)={d_v:+.4f}")
        results_side.append((d_h, d_v))

    print("\n=== Verdict (mean delta over rotations, relative to the symmetric-ellipse baseline "
          "at the same rotation) ===")
    tip_d_h = np.mean([d[0] for d in results_tip])
    tip_d_v = np.mean([d[1] for d in results_tip])
    side_d_h = np.mean([d[0] for d in results_side])
    side_d_v = np.mean([d[1] for d in results_side])
    print(f"MAJOR-axis-end asymmetry (egg_x)  -> mean d(asym_h)={tip_d_h:+.4f}  mean d(asym_v)={tip_d_v:+.4f}")
    print(f"MINOR-axis-side asymmetry (egg_y) -> mean d(asym_h)={side_d_h:+.4f}  mean d(asym_v)={side_d_v:+.4f}")
    if tip_d_h > tip_d_v and side_d_v > side_d_h:
        print("CONCLUSION: asymmetry_h (np.fliplr, array axis=1) rises specifically when asymmetry is "
              "placed between the two ends of the MAJOR axis => asymmetry_h == asymmetry_major_axis.")
        print("            asymmetry_v (np.flipud, array axis=0) rises specifically when asymmetry is "
              "placed between the two sides of the MINOR axis => asymmetry_v == asymmetry_minor_axis.")
    elif tip_d_v > tip_d_h and side_d_h > side_d_v:
        print("CONCLUSION: asymmetry_v (np.flipud, array axis=0) rises specifically when asymmetry is "
              "placed between the two ends of the MAJOR axis => asymmetry_v == asymmetry_major_axis.")
        print("            asymmetry_h (np.fliplr, array axis=1) rises specifically when asymmetry is "
              "placed between the two sides of the MINOR axis => asymmetry_h == asymmetry_minor_axis.")
    else:
        print("CONCLUSION: AMBIGUOUS -- the two folds did not cleanly separate by asymmetry type.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
