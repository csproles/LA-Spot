"""Per-instance segmentation quality flags (v2).

Unlike the v1 flags (used in RevisedABCDTest, computed on an already-unioned
mask), these operate on ONE lesion instance's own mask, and add a tiny-
artifact floor so a stray few-pixel rasterization speck (e.g. ISIC_0000006's
9-pixel fleck, confirmed in the segmentation investigation to come from a
SINGLE YOLO instance's raster mask, not a second detection) doesn't get
counted as a second "meaningful" component.

None of these are diagnostic/classification thresholds — they are
data-quality flags for review, exactly as before. TINY_ARTIFACT_PX and
LOW_CONFIDENCE_REPORT are round, generic engineering defaults, not fit to
any specific image's numbers; the 0.25 YOLO acceptance threshold is
untouched — LOW_CONFIDENCE_REPORT never removes or downweights a detection,
it only adds a note.
"""

import cv2
import numpy as np

TINY_ARTIFACT_PX = 20          # ignore components at/under this size as raster noise
LARGE_AREA_FRACTION = 0.5      # unchanged from the v1 flag
LOW_CONFIDENCE_REPORT = 0.5    # reporting-only marker; detections are still kept and scored


def segmentation_quality_flags_v2(mask, confidence):
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    component_areas = [int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)]
    meaningful_areas = [a for a in component_areas if a > TINY_ARTIFACT_PX]
    tiny_artifact_areas = [a for a in component_areas if a <= TINY_ARTIFACT_PX]

    lesion_fraction = float(np.mean(mask > 0))

    flags = []
    if len(meaningful_areas) > 1:
        flags.append(f"fragmented({len(meaningful_areas)}_meaningful_blobs)")
    if tiny_artifact_areas:
        flags.append(f"ignored_{len(tiny_artifact_areas)}_tiny_artifact(s)_le_{TINY_ARTIFACT_PX}px")
    if lesion_fraction > LARGE_AREA_FRACTION:
        flags.append(f"large_area_fraction({lesion_fraction:.0%})_possible_over_segmentation")
    if confidence < LOW_CONFIDENCE_REPORT:
        flags.append(f"low_confidence({confidence:.2f})_review_recommended")

    return {
        "num_total_components": n_labels - 1,
        "num_meaningful_components": len(meaningful_areas),
        "num_tiny_artifacts_ignored": len(tiny_artifact_areas),
        "lesion_fraction": round(lesion_fraction, 4),
        "quality_flags": ";".join(flags) if flags else "ok",
    }
