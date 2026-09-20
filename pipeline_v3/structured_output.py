"""PHASE 8 — Final LASpot pipeline entry point.

Flow: input image -> YOLO segmentation -> per-instance raster mask ->
ABCD extraction (UNCHANGED from V2 — revised_abcd.pipeline_v2) -> frozen
V3 concern logic (pipeline_v3.decision) -> structured, application-facing
result.

This module NEVER outputs "melanoma", "benign", "cancer", or "not cancer".
Those words only ever appear in evaluation code that compares predictions
to labeled ISIC data (Evaluation_FinalTargeted/, Evaluation_3500/) — never
in anything reachable from a real user-facing call. The application-facing
concern level is always one of "LOWER VISUAL CONCERN" or
"ELEVATED VISUAL CONCERN", per project convention.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revised_abcd.pipeline_v2 import process_image  # UNCHANGED: segmentation + ABCD extraction
from pipeline_v3.decision import v3_predict_from_features, V3_CONFIG


CONCERN_LOWER = "LOWER VISUAL CONCERN"
CONCERN_ELEVATED = "ELEVATED VISUAL CONCERN"


def _instance_to_structured(image_id, inst, is_primary):
    """One detected lesion instance -> the structured, LLM-ready schema."""
    elevated = v3_predict_from_features(
        a_value=inst["A_value"], a_concern=inst["A_concern"],
        b_value=inst["B_circularity"], b_concern=inst["B_circularity_concern"],
        c_value=inst["C_value"], c_concern=inst["C_concern"],
    )
    overall = CONCERN_ELEVATED if elevated else CONCERN_LOWER

    return {
        "lesion_instance_id": inst["lesion_instance_id"],
        "is_primary_instance": is_primary,
        "yolo_confidence": inst["confidence"],
        "asymmetry": {"value": inst["A_value"], "concern": inst["A_concern"]},
        "border": {"value": inst["B_circularity"], "concern": inst["B_circularity_concern"]},
        "color": {"value": inst["C_value"], "concern": inst["C_concern"], "label": inst["C_label"]},
        "diameter_px": inst["D_px"],
        "mask_area_fraction": inst["lesion_fraction"],
        "overall_visual_concern": overall,
        "quality_flags": inst["quality_flags"].split(";") if inst["quality_flags"] not in ("", "ok") else [],
        "explanation_ready_features": {
            "asymmetry_raw": inst["A_value"],
            "border_raw": inst["B_circularity"],
            "border_experimental_raw": inst.get("B_experimental"),
            "color_raw": inst["C_value"],
            "diameter_px": inst["D_px"],
            "note": "diameter is in pixels only; no validated physical (mm) scale is available "
                    "for this pipeline (see limitations).",
        },
    }


def analyze_image(model, image_path, conf=0.25):
    """Full LASpot CV pipeline for one image. Returns the structured,
    application/LLM-facing result — never a diagnostic label."""
    try:
        rows, orig_shape = process_image(model, image_path, conf=conf)
    except Exception as exc:
        return {
            "segmentation_status": "FAILED",
            "error": str(exc),
            "lesion_instances": [],
            "overall_visual_concern": None,
        }

    if not rows:
        return {
            "segmentation_status": "NO_DETECTION",
            "lesion_instances": [],
            "overall_visual_concern": None,
            "note": "No lesion could be confidently located in this image. This does not mean "
                    "no lesion is present — retaking the photo (better lighting, centering, "
                    "focus) is recommended.",
        }

    instances = [_instance_to_structured(Path(image_path).stem, r, i == 0) for i, r in enumerate(rows)]
    segmentation_status = "MULTI_LESION_DETECTED" if len(rows) > 1 else "SINGLE_LESION_DETECTED"

    return {
        "segmentation_status": segmentation_status,
        "num_lesion_instances": len(rows),
        "lesion_instances": instances,
        # For a multi-lesion image the app should let the user pick which
        # spot they're tracking (or show all) — the pipeline does not guess
        # which one the user cares about (mirrors the evaluation policy:
        # MULTI_LESION_AMBIGUOUS is never silently collapsed to one answer).
        "primary_instance_overall_visual_concern":
            instances[0]["overall_visual_concern"] if len(instances) == 1 else None,
        "config_version": V3_CONFIG["version"],
    }
