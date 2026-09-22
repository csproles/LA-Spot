"""Revised D: verifies detected 'hair' is actually hair-shaped (elongated)
before trusting it for physical calibration.

Does NOT modify Code/ComputerVisionStuff/hair.py. Reimplements the same
blackhat + distance-transform-skeleton procedure here (this reproduces the
existing hair_width_px values exactly when the elongation gate is disabled
— verified in investigation/investigate_diameter.py) and adds the one thing
the original never does: check whether each detected dark-blob component is
elongated (hair-shaped) or roughly round (a globule/pigment-dot/noise
artifact) via an ellipse-fit aspect ratio, and only use hair-LIKE components
for the width measurement.

Root cause this targets (see investigation/investigate_diameter.py): across
all 40 dataset images, ZERO had hair-shaped components as the majority of
what the blackhat filter detected — the "hair mask" is dominated by small
round blobs (pigment texture, surface detail), not hair shafts. The
existing "no reliable hair -> None" fallback never fires because its
thresholds (mask area < 50px, < 10 skeleton points) are cleared by this
non-hair texture in every image.

If no hair-LIKE evidence survives the gate, D_mm is None — never guessed —
but diameter_px is ALWAYS computed and retained regardless of calibration
availability. No field-of-view or other alternative calibration is used
here, per instruction; that remains a separate, not-yet-authorized decision.

The 70 micron constant (VELLUS_HAIR_UM) is left UNCHANGED. The investigation
also flagged that 70um is not actually a vellus-hair value (vellus hair is
generally reported under 30-40um; 70um sits in typical terminal-hair
territory) — that is a separate, distinct problem from the one fixed here
(whether we're measuring hair AT ALL) and is called out explicitly in the
run report as NOT corrected by this revision.
"""

import cv2
import numpy as np

VELLUS_HAIR_UM = 70.0  # UNCHANGED value from Code/main.py — see module docstring
DIAMETER_CONCERN_MM = 10.0  # UNCHANGED from Code/MelanomaDeterminingStuff/diameter.py

MIN_ASPECT_RATIO_HAIR_LIKE = 4.0  # elongation gate: major:minor axis ratio
MIN_HAIR_LIKE_SKELETON_POINTS = 10  # same minimum sample size as the original


def _measure_hair_width_px_verified(img, kernel_size=17, threshold=10):
    """Returns (hair_width_px_or_None, reason, n_hair_like, n_blob_like)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    _, hair_mask = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    if np.sum(hair_mask > 0) < 50:
        return None, "no_dark_structures_detected", 0, 0

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(hair_mask)
    hair_like_mask = np.zeros_like(hair_mask)
    n_hair_like = 0
    n_blob_like = 0
    for i in range(1, n_labels):
        comp = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours or len(contours[0]) < 5:
            n_blob_like += 1
            continue
        (_, _), (minor, major), _ = cv2.fitEllipse(contours[0])
        aspect = (major / minor) if minor > 1e-6 else 1.0
        if aspect >= MIN_ASPECT_RATIO_HAIR_LIKE:
            hair_like_mask[labels == i] = 255
            n_hair_like += 1
        else:
            n_blob_like += 1

    # Require hair-like components to be an actual MAJORITY of what was
    # detected, not merely present. This is the same criterion used to
    # diagnose the problem in investigation/investigate_diameter.py (which
    # found 0/40 images had a hair-like majority) — requiring only "at
    # least one" elongated component is far too weak given these images
    # commonly contain hundreds of small blob-like structures, some of
    # which clear an elongation cutoff by chance alone.
    if n_hair_like == 0 or n_hair_like <= n_blob_like:
        return None, f"hair_like_not_majority({n_hair_like}_vs_{n_blob_like}_blob_like)", n_hair_like, n_blob_like

    dist = cv2.distanceTransform(hair_like_mask, cv2.DIST_L2, 5)
    kernel_sk = np.ones((3, 3), np.uint8)
    dist_dilated = cv2.dilate(dist, kernel_sk)
    skeleton = (dist == dist_dilated) & (dist > 0)
    half_widths = dist[skeleton]
    half_widths = half_widths[(half_widths >= 0.5) & (half_widths <= 8.0)]

    if len(half_widths) < MIN_HAIR_LIKE_SKELETON_POINTS:
        return None, f"too_few_hair_like_points({len(half_widths)})", n_hair_like, n_blob_like

    hair_width_px = float(np.median(half_widths) * 2)
    return hair_width_px, "ok", n_hair_like, n_blob_like


def measure_calibration_revised(bilateral_img):
    """Returns (mm_per_px_or_None, reason, n_hair_like, n_blob_like)."""
    hair_width_px, reason, n_hair_like, n_blob_like = _measure_hair_width_px_verified(bilateral_img)
    if hair_width_px is None:
        return None, reason, n_hair_like, n_blob_like
    mm_per_px = (VELLUS_HAIR_UM / hair_width_px) / 1000.0
    return mm_per_px, reason, n_hair_like, n_blob_like


def score_diameter_revised(mask, mm_per_px):
    """diameter_px is ALWAYS populated when a contour exists. D_mm/concern
    only exist when a verified (hair-shape-gated) mm_per_px is available.
    Concern threshold (10.0mm) is UNCHANGED."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"diameter_px": None, "value": None, "concern": False,
                "label": "Diameter: N/A (no lesion contour)"}

    contour = max(contours, key=cv2.contourArea)
    (_, _), radius_px = cv2.minEnclosingCircle(contour)
    diameter_px = round(float(radius_px * 2), 1)

    if mm_per_px is None:
        return {"diameter_px": diameter_px, "value": None, "concern": False,
                "label": f"Diameter: {diameter_px}px (no verified hair-based calibration; mm N/A)"}

    diameter_mm = diameter_px * mm_per_px
    concern = diameter_mm > DIAMETER_CONCERN_MM
    return {
        "diameter_px": diameter_px,
        "value": round(diameter_mm, 2),
        "concern": bool(concern),
        "label": f"Est. diameter: {diameter_mm:.1f}mm (hair-calibrated, shape-verified) "
                 f"{'⚠ >10mm' if concern else '✔ <=10mm'}",
    }
