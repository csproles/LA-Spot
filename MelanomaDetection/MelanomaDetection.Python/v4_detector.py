"""V4Detector -- the ACTIVE CV path, replacing MelanomaDetector's classical
Otsu/hair-mm pipeline as main.py's `detector`.

Flow: image -> YOLO segmentation (revised_abcd.pipeline_v2, frozen) ->
ABCD extraction (unchanged Code/MelanomaDeterminingStuff scoring) -> frozen
V4 decision model (pipeline_v4/frozen_model.pkl, threshold 0.25) ->
LOWER/ELEVATED-equivalent result, packaged into the EXACT dict shape
main.py / llm_explainer.py / evolution.py already expect from a detector, so
none of that surrounding code needed to change.

Does NOT modify pipeline_v4/, revised_abcd/, or Code/ -- it only calls them.
Does NOT touch YOLO, ABCD formulas, the 6 V4 features, frozen_model.pkl, or
the 0.25 threshold. Never computes a hair-width mm calibration: diameter is
always reported as pixels only (mm_per_px is always None), per policy.

image_processor.MelanomaDetector (the old classical pipeline) is kept
UNMODIFIED and UNREMOVED -- this module reuses several of its *utility*
methods (resize/preprocessing/visualization) by composition, since those
are generic image-processing steps unrelated to which segmentation produced
the mask, rather than duplicating ~150 lines of already-working code. Only
_segment_lesion and the ABCDE scoring/risk-score methods are replaced, with
V4's real analysis.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from image_processor import MelanomaDetector
from yolo_config import get_yolo_weights_path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revised_abcd.pipeline_v2 import process_image as v4_process_image  # noqa: E402
from pipeline_v4.decision_model import v4_predict_from_row, load_frozen_pipeline  # noqa: E402

CONF = 0.25  # YOLO acceptance threshold -- UNCHANGED, matches every prior V4 evaluation


_PRIMARY_OUTLINE_COLOR = (0, 215, 255)   # BGR gold -- the instance V4 actually analyzed
_OTHER_OUTLINE_COLOR = (255, 200, 0)     # BGR cyan -- detected by YOLO, not analyzed


def _build_multi_instance_overlay(original, rows):
    """One image outlining EVERY YOLO-detected instance, so a multi-lesion
    photo doesn't look like only one spot was found. Outline only, never
    filled and never scored -- this never implies an instance other than
    `rows[0]` (the primary) was analyzed, only that YOLO found it. Only
    called when there's more than one instance; the single-instance and
    no-detection paths are untouched by this function entirely.
    """
    overlay = original.copy()
    w = original.shape[1]
    thickness_primary = max(3, w // 200)
    thickness_other = max(2, w // 350)
    for i, row in enumerate(rows):
        contours, _ = cv2.findContours(row["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if i == 0:
            cv2.drawContours(overlay, contours, -1, _PRIMARY_OUTLINE_COLOR, thickness_primary)
        else:
            cv2.drawContours(overlay, contours, -1, _OTHER_OUTLINE_COLOR, thickness_other)
    return overlay


def _empty_abcde_scores(reason: str) -> dict:
    """Mirrors MelanomaDetector._compute_abcde_scores's own no-lesion shape,
    so downstream code (evolution, storage, the LLM schema mapper) sees the
    same "nothing detected" contract it already handles."""
    return {
        "asymmetry": {"score": 0.0, "details": {}},
        "border": {"score": 0.0, "details": {}},
        "color": {"score": 0.0, "details": {}},
        "diameter": {"score": None, "details": {"reason": reason}},
        "evolving": {"score": None, "details": {"reason": "no prior check to compare against"}},
    }


def _scaled_0_10(raw_value, threshold):
    """Display-only rescaling so the UI's existing 0-10 sliders/labels still
    work: raw_value == threshold maps to 5.0. This number is NEVER used to
    make the elevated/lower decision -- that comes from the frozen logistic
    regression on the RAW A/B/C/D_px/confidence/lesion_fraction values,
    completely independently of this rescaling."""
    if raw_value is None or threshold in (None, 0):
        return 0.0
    return round(min(max(raw_value / threshold * 5.0, 0.0), 10.0), 2)


class V4Detector:
    """Drop-in replacement for MelanomaDetector as main.py's `detector`."""

    def __init__(self):
        # Reused purely for its generic preprocessing/visualization utility
        # methods (resize, vignette removal, denoise, hair removal, edge
        # detection, the ABCD visual-overlay renderer) -- never for its
        # _segment_lesion or scoring methods, which V4 replaces entirely.
        self._legacy = MelanomaDetector()
        self._yolo_model = YOLO(get_yolo_weights_path())
        self._decision_pipeline = load_frozen_pipeline()

    def process_image(self, image_path: str) -> dict:
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Same preprocessing utility calls the classical detector already
        # uses, reused as-is for the UI's pipeline-stage display images.
        # Deliberately NOT calling _resize_if_needed here: V4's own YOLO
        # inference always runs on the untouched original file (matching
        # every prior V4 evaluation, which never resizes), so the display
        # images are kept at that same resolution rather than introducing a
        # size mismatch between what's shown and what V4 actually scored.
        no_vignette, circle_info = self._legacy._remove_vignette(original)
        median_filtered = self._legacy._remove_salt_pepper_noise(no_vignette, kernel_size=3)
        bilateral_filtered = self._legacy._apply_bilateral_filter(median_filtered)
        hair_removed = self._legacy._remove_hair_and_artifacts(bilateral_filtered)

        # --- the actual V4 CV pipeline: YOLO segmentation + ABCD, unchanged ---
        rows, orig_shape = v4_process_image(self._yolo_model, image_path, conf=CONF)

        num_instances = len(rows)
        multi_lesion_detected = num_instances > 1

        if not rows:
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            # _detect_edges expects a background-zeroed (masked) image; with
            # no detection there's nothing to mask to, so it just sees the
            # (hair-removed) full frame -- harmless, it's a display-only image.
            edges = self._legacy._detect_edges(hair_removed)
            abcde_scores = _empty_abcde_scores("no lesion detected")
            risk_score = 0.0
            visuals = self._legacy._build_abcd_visuals(original, mask, abcde_scores)
            return self._package(
                original, bilateral_filtered, median_filtered, hair_removed, mask, edges,
                visuals, abcde_scores, risk_score, num_instances, multi_lesion_detected,
            )

        # Primary = highest-confidence instance (rows are pre-sorted by
        # confidence in revised_abcd.yolo_single_image). Multiple lesions are
        # never silently merged; the caller is told via multi_lesion_detected
        # / num_lesion_instances so the app can say "multiple spots found."
        primary = rows[0]
        mask = primary["mask"]
        masked_for_edges = cv2.bitwise_and(hair_removed, hair_removed, mask=mask)
        edges = self._legacy._detect_edges(masked_for_edges)

        elevated, decision_score = v4_predict_from_row(
            {"A_value": primary["A_value"], "B_circularity": primary["B_circularity"],
             "C_value": primary["C_value"], "D_px": primary["D_px"],
             "confidence": primary["confidence"], "lesion_fraction": primary["lesion_fraction"]},
            self._decision_pipeline,
        )

        abcde_scores = {
            "asymmetry": {
                "score": _scaled_0_10(primary["A_value"], 0.20),
                "details": {
                    "raw_asymmetry_ratio": primary["A_value"],
                    "concern": bool(primary["A_concern"]),
                },
            },
            "border": {
                "score": _scaled_0_10(primary["B_circularity"], 0.50),
                "details": {
                    "raw_border_irregularity": primary["B_circularity"],
                    "concern": bool(primary["B_circularity_concern"]),
                },
            },
            "color": {
                "score": _scaled_0_10(primary["C_value"], 0.35),
                "details": {
                    "color_cv": primary["C_value"],
                    "concern": bool(primary["C_concern"]),
                    # dangerous_colors_pct intentionally omitted: the unchanged
                    # Code/MelanomaDeterminingStuff/color.py scorer doesn't
                    # expose the individual pink/blue-gray/white/black
                    # fractions outside its own label string, and parsing
                    # that string back into numbers isn't worth the
                    # fragility -- _map_to_llm_schema already treats a
                    # missing dangerous_colors_pct as "none detected", which
                    # is honest (we simply don't surface that breakdown here,
                    # not that none exists).
                },
            },
            "diameter": {
                # No mm score, ever -- D_px only, per policy. Omitting
                # "diameter_mm" from details (never setting it, not even to
                # None) is what makes _map_to_llm_schema() skip diameter
                # entirely from the JSON payload, which correctly triggers
                # the LLM prompt's rule 4a ("say plainly size could not be
                # measured") instead of inventing or guessing a value.
                "score": None,
                "details": {
                    "reason": "no validated physical (mm) calibration is available for this "
                              "pipeline; diameter_px holds the raw pixel measurement instead",
                    "diameter_px": primary["D_px"],
                    "concern": False,
                },
            },
            "evolving": {"score": None, "details": {"reason": "no prior check to compare against"}},
        }

        # risk_score is V4's own decision_score (the frozen logistic
        # regression's output probability, 0-1) rescaled to the app's
        # existing 0-100 field for storage/evolution-trend compatibility.
        # It is NOT recomputed via the old weighted-sum-of-0-10-scores
        # formula, and it must never be presented to a user as a melanoma
        # probability -- it is a decision-support number only.
        risk_score = round(decision_score * 100, 1)

        visuals = self._legacy._build_abcd_visuals(original, mask, abcde_scores)
        multi_instance_overlay = (
            _build_multi_instance_overlay(original, rows) if multi_lesion_detected else None
        )

        result = self._package(
            original, bilateral_filtered, median_filtered, hair_removed, mask, edges,
            visuals, abcde_scores, risk_score, num_instances, multi_lesion_detected,
            multi_instance_overlay,
        )
        result["overall_visual_concern"] = "ELEVATED VISUAL CONCERN" if elevated else "LOWER VISUAL CONCERN"
        result["v4_decision_score"] = round(decision_score, 4)
        result["yolo_confidence"] = primary["confidence"]
        result["quality_flags"] = primary["quality_flags"].split(";") if primary["quality_flags"] not in ("", "ok") else []
        return result

    @staticmethod
    def _package(original, bilateral_filtered, noise_removed, hair_removed, mask, edges,
                 visuals, abcde_scores, risk_score, num_instances, multi_lesion_detected,
                 multi_instance_overlay=None):
        return {
            "original": original,
            "bilateral_filtered": bilateral_filtered,
            "noise_removed": noise_removed,
            "hair_removed": hair_removed,
            # The primary (highest-confidence) instance's mask ONLY -- unchanged
            # for single-instance and no-detection results. When more than one
            # instance was detected, multi_instance_overlay (below) additionally
            # shows every one of them; this field itself never changes shape or
            # meaning, so evolution.lesion_area_px and everything else that reads
            # it as "the analyzed lesion's mask" keeps working exactly as before.
            "segmentation": mask,
            "edges": edges,
            "asymmetry_visual": visuals["asymmetry"],
            "border_visual": visuals["border"],
            "color_visual": visuals["color"],
            "diameter_visual": visuals["diameter"],
            # Present only when more than one instance was detected -- every
            # YOLO-detected instance outlined on the original photo (primary
            # in one color, the rest in another), never filled, never scored.
            # None (never sent) for single-instance and no-detection results.
            "multi_instance_overlay": multi_instance_overlay,
            "abcde_scores": abcde_scores,
            "risk_score": risk_score,
            # Always None: no hair-width (or any other) physical calibration
            # is used by V4. Diameter is pixels-only; downstream code
            # (evolution.score_change) already tolerates mm_per_px=None.
            "mm_per_px": None,
            "num_lesion_instances": num_instances,
            "multi_lesion_detected": multi_lesion_detected,
        }
