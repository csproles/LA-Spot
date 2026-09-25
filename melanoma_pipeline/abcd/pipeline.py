"""Shared per-image, per-instance ABCD scoring glue. Consolidated from
revised_abcd/pipeline_v2.py -- identical logic, imports rewritten to this
package's own sibling modules (relative imports) instead of the original
repository's `Code/` and `revised_abcd/` top-level directories, so this
package runs independently of them.

Per the original module's own documented decisions (all UNCHANGED here):
  - A: revised (principal-axis aligned) asymmetry, threshold unchanged (0.20)
  - B: EXISTING circularity (legacy_scoring.score_border, unmodified) drives
       the decision; B_experimental recorded for analysis only, no threshold
  - C: EXISTING score_color (legacy_scoring.score_color, unmodified)
  - D: diameter_px ALWAYS recorded; D_mm is ALWAYS None (no defensible
       physical calibration exists for this data source; D never
       contributes a concern flag; no pixel->mm conversion is attempted)
  - No np.max union across instances: this module scores ONE instance mask
    at a time. Multiple instances in one image are the caller's concern
    (each gets its own call, its own row).
"""
import cv2

from .preprocessing import remove_vignette, remove_salt_pepper_noise, apply_bilateral_filter, remove_hair
from .legacy_scoring import score_border, score_color
from .asymmetry import score_asymmetry_revised
from .border_experimental import score_border_experimental
from .diameter import score_diameter_revised
from .segmentation_quality import segmentation_quality_flags_v2

CRITICAL_COLOR_OVERRIDE = 0.50  # UNCHANGED, from the original classical scoring
CONCERNS_HIGH_THRESHOLD = 2     # UNCHANGED


def preprocess_image(image_bgr):
    """Reproduces the existing preprocessing chain UNCHANGED (vignette
    removal, denoise, bilateral filter, hair removal) up to the image that
    score_color consumes as "original"."""
    no_vignette, circle_info = remove_vignette(image_bgr)
    denoised = remove_salt_pepper_noise(no_vignette, kernel_size=3)
    bilateral = apply_bilateral_filter(denoised, diameter=9, sigma_color=75, sigma_space=75)
    no_hair = remove_hair(bilateral, kernel_size=17, threshold=10)
    return no_hair, circle_info


def score_instance(mask, no_hair, circle_info):
    """Full ABCD scoring for ONE lesion instance's mask. Returns a dict with
    every A/B/C/D field plus the combined concerns/risk_level, using
    UNCHANGED thresholds throughout."""
    a_result = score_asymmetry_revised(mask)              # revised (aligned), threshold 0.20 unchanged
    b_result = score_border(mask)                          # UNCHANGED, drives the decision
    b_experimental = score_border_experimental(mask)       # analysis only, no threshold
    c_result, pink_red_px, blue_gray_px, white_px, black_px = score_color(mask, no_hair, circle_info)
    d_result = score_diameter_revised(mask, mm_per_px=None)  # D_px always; D_mm ALWAYS None, no conversion attempted

    concerns = sum(1 for r in (a_result, b_result, c_result, d_result) if r["concern"])
    critical_color = max(pink_red_px, blue_gray_px, white_px, black_px)
    critical_override = critical_color > CRITICAL_COLOR_OVERRIDE
    risk_level = "HIGH" if critical_override else ("LOW" if concerns < CONCERNS_HIGH_THRESHOLD else "HIGH")

    return {
        "A_value": a_result["value"], "A_concern": a_result["concern"],
        "A_principal_axis_deg": a_result["principal_axis_deg"],
        "B_circularity": b_result["value"], "B_circularity_concern": b_result["concern"],
        "B_experimental": b_experimental["value"], "B_experimental_n_defects": b_experimental["n_significant_defects"],
        "C_value": c_result["value"], "C_concern": c_result["concern"], "C_label": c_result["label"],
        "D_px": d_result["diameter_px"], "D_mm": d_result["value"], "D_concern": d_result["concern"],
        "concerns": concerns, "risk_level": risk_level,
        "provisional_prediction": "positive" if risk_level == "HIGH" else "negative",
    }


def process_image(model, image_path, conf=0.25):
    """Full single-image pipeline: single-image YOLO inference (no
    batching), one row per detected instance, quality flags per instance,
    full ABCD per instance. Returns (list_of_instance_rows, image_shape).

    Raises on unexpected errors — callers are responsible for catching and
    logging.
    """
    from .yolo_inference import run_single_image_inference

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"could not read image: {image_path}")

    instances, orig_shape = run_single_image_inference(model, image_path, conf=conf)

    if not instances:
        return [], orig_shape

    no_hair, circle_info = preprocess_image(image_bgr)

    rows = []
    for inst in instances:
        mask = inst["mask"]
        quality = segmentation_quality_flags_v2(mask, inst["confidence"])
        scores = score_instance(mask, no_hair, circle_info)
        row = {
            "lesion_instance_id": inst["instance_id"],
            "confidence": round(inst["confidence"], 4),
            "class_id": inst["class_id"],
            "class_name": inst["class_name"],
            "mask": mask,
        }
        row.update(quality)
        row.update(scores)
        rows.append(row)

    return rows, orig_shape
