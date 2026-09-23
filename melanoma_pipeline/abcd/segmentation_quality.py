"""Per-instance segmentation quality flags. Consolidated, UNCHANGED, from
revised_abcd/segmentation_quality_v2.py.

None of these are diagnostic/classification thresholds — they are
data-quality flags for review only.
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
