"""FINAL LASpot pipeline entry point (V4 wins — see Evaluation_V4/FinalReport).

Flow: input image -> YOLO segmentation (UNCHANGED) -> per-instance raster
mask (UNCHANGED) -> ABCD extraction (UNCHANGED, revised_abcd.pipeline_v2) ->
frozen V4 logistic-regression decision model -> structured,
application-facing result.

Never outputs "melanoma", "benign", "cancer", or "not cancer" — those words
exist only in evaluation code that compares predictions to labeled ISIC
data, never in anything reachable from a real user-facing call. The
concern level is always "LOWER VISUAL CONCERN" or "ELEVATED VISUAL
CONCERN". The model's internal probability is exposed to the LLM layer
ONLY as a decision-support number, explicitly labeled as not a melanoma
probability (see pipeline_v4/decision_model.py and LLM_HANDOFF.md).
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from revised_abcd.pipeline_v2 import process_image  # UNCHANGED: segmentation + ABCD extraction
from pipeline_v4.decision_model import v4_predict_from_row, load_frozen_pipeline, V4_CONFIG

CONCERN_LOWER = "LOWER VISUAL CONCERN"
CONCERN_ELEVATED = "ELEVATED VISUAL CONCERN"

_PIPELINE_CACHE = None


def _get_pipeline():
    global _PIPELINE_CACHE
    if _PIPELINE_CACHE is None:
        _PIPELINE_CACHE = load_frozen_pipeline()
    return _PIPELINE_CACHE


def _instance_to_structured(inst, is_primary, fitted_pipeline):
    row_for_model = {
        "A_value": inst["A_value"], "B_circularity": inst["B_circularity"],
        "C_value": inst["C_value"], "D_px": inst["D_px"],
        "confidence": inst["confidence"], "lesion_fraction": inst["lesion_fraction"],
    }
    elevated, decision_score = v4_predict_from_row(row_for_model, fitted_pipeline)
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
        # Internal decision-support score only — NEVER a melanoma probability.
        # See pipeline_v4/LLM_HANDOFF.md for the exact framing constraint.
        "decision_score_internal": round(decision_score, 4),
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
    """Full LASpot CV pipeline for one image, V4 decision layer."""
    try:
        rows, orig_shape = process_image(model, image_path, conf=conf)
    except Exception as exc:
        return {"segmentation_status": "FAILED", "error": str(exc),
                "lesion_instances": [], "overall_visual_concern": None}

    if not rows:
        return {
            "segmentation_status": "NO_DETECTION", "lesion_instances": [],
            "overall_visual_concern": None,
            "note": "No lesion could be confidently located in this image. This does not mean "
                    "no lesion is present — retaking the photo (better lighting, centering, "
                    "focus) is recommended.",
        }

    fitted_pipeline = _get_pipeline()
    instances = [_instance_to_structured(r, i == 0, fitted_pipeline) for i, r in enumerate(rows)]
    segmentation_status = "MULTI_LESION_DETECTED" if len(rows) > 1 else "SINGLE_LESION_DETECTED"

    return {
        "segmentation_status": segmentation_status,
        "num_lesion_instances": len(rows),
        "lesion_instances": instances,
        "primary_instance_overall_visual_concern":
            instances[0]["overall_visual_concern"] if len(instances) == 1 else None,
        "config_version": V4_CONFIG["version"],
    }
