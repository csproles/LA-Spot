"""V5Detector -- the ACTIVE CV path, replacing V4's decision layer as
main.py's `detector`. V4 (pipeline_v4/) is NOT used here and is left
untouched on disk for reference/reproducibility; it is simply never
instantiated by this application.

Flow: image -> YOLO segmentation (revised_abcd.pipeline_v2, frozen,
UNCHANGED) -> ABCD extraction (unchanged Code/MelanomaDeterminingStuff
scoring, UNCHANGED) -> V5's 11 additional features (pipeline_v5/
feature_extraction.py, UNCHANGED) -> frozen V5 decision model
(pipeline_v5/frozen_model.pkl, 17 features, threshold 0.25, UNCHANGED) ->
LOWER/ELEVATED-equivalent result, packaged into the EXACT dict shape
main.py / llm_explainer.py / evolution.py already expect from a detector.

Does NOT modify pipeline_v5/, revised_abcd/, or Code/ -- it only calls
them. Does NOT touch YOLO, ABCD formulas, the 17 V5 features,
frozen_model.pkl, or the 0.25 threshold. Never computes a hair-width mm
calibration: diameter is always reported as pixels only (mm_per_px is
always None), per policy.

image_processor.MelanomaDetector (the old classical pipeline) is kept
UNMODIFIED and UNREMOVED -- this module reuses several of its *utility*
methods (resize/preprocessing/visualization) by composition, exactly as
V4Detector did, since those are generic image-processing steps unrelated
to which segmentation/decision model produced the result.

One deliberate redundancy versus V4Detector: this module calls
revised_abcd.pipeline_v2.preprocess_image() a SECOND time (in addition to
process_image()'s own internal call) to obtain the same no_hair/circle_info
intermediates V5's new color features need. This is extra compute, not a
new implementation -- it calls the exact same, unmodified function
pipeline_v2.process_image() already calls internally for C, so the result
is guaranteed identical, not merely similar. Avoiding this second call
would require changing pipeline_v2.py's return signature, which this
integration deliberately does not do.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import policy
from image_processor import MelanomaDetector
from yolo_config import get_yolo_weights_path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from revised_abcd.pipeline_v2 import process_image as v5_process_image, preprocess_image as v5_preprocess_image  # noqa: E402
from pipeline_v5.decision_model import v5_predict_from_row, load_frozen_pipeline, V5_CONFIG  # noqa: E402
from pipeline_v5.feature_extraction import extract_v5_new_features  # noqa: E402

CONF = 0.25  # YOLO acceptance threshold -- UNCHANGED, matches every prior V4/V5 evaluation


_PRIMARY_OUTLINE_COLOR = (0, 215, 255)   # BGR gold -- the instance V5 actually analyzed
_OTHER_OUTLINE_COLOR = (255, 200, 0)     # BGR cyan -- detected by YOLO, not analyzed


def _build_multi_instance_overlay(original, rows):
    """One image outlining EVERY YOLO-detected instance, so a multi-lesion
    photo doesn't look like only one spot was found. Outline only, never
    filled and never scored. Only called when there's more than one
    instance."""
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
    regression on the RAW 17-feature vector, completely independently of
    this rescaling."""
    if raw_value is None or threshold in (None, 0):
        return 0.0
    return round(min(max(raw_value / threshold * 5.0, 0.0), 10.0), 2)


def _relative_lesion_size_pct(d_px, image_width_px):
    """User-facing, non-physical size proxy: the lesion's estimated diameter
    as a percentage of the photo's width. Deliberately NOT a millimeter
    measurement -- no physical scale reference exists in this pipeline (see
    pipeline_v5/v5_model_metadata.json's diameter_calibration section).
    Returns None if d_px or image_width_px is unavailable."""
    if d_px is None or not image_width_px:
        return None
    return round(100.0 * float(d_px) / float(image_width_px), 1)


class V5Detector:
    """Drop-in replacement for MelanomaDetector as main.py's `detector`.
    V4Detector-equivalent, but scores with the frozen V5 model instead."""

    def __init__(self):
        # Reused purely for its generic preprocessing/visualization utility
        # methods (resize/preprocessing/visualization) -- never for its
        # _segment_lesion or scoring methods, which V5 (like V4 before it)
        # replaces entirely.
        self._legacy = MelanomaDetector()
        self._yolo_model = YOLO(get_yolo_weights_path())
        self._decision_pipeline = load_frozen_pipeline()

    def process_image(self, image_path: str) -> dict:
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image_width_px = original.shape[1]

        # Same display-chain preprocessing the classical detector already
        # uses, reused as-is for the UI's pipeline-stage display images.
        # V5's own YOLO inference always runs on the untouched original
        # file, exactly as V4's did.
        no_vignette, circle_info = self._legacy._remove_vignette(original)
        median_filtered = self._legacy._remove_salt_pepper_noise(no_vignette, kernel_size=3)
        bilateral_filtered = self._legacy._apply_bilateral_filter(median_filtered)
        hair_removed = self._legacy._remove_hair_and_artifacts(bilateral_filtered)

        # --- the actual CV pipeline: YOLO segmentation + existing ABCD, unchanged ---
        rows, orig_shape = v5_process_image(self._yolo_model, image_path, conf=CONF)

        num_instances = len(rows)
        multi_lesion_detected = num_instances > 1

        if not rows:
            mask = np.zeros(original.shape[:2], dtype=np.uint8)
            edges = self._legacy._detect_edges(hair_removed)
            abcde_scores = _empty_abcde_scores("no lesion detected")
            risk_score = 0.0
            visuals = self._legacy._build_abcd_visuals(original, mask, abcde_scores)
            result = self._package(
                original, bilateral_filtered, median_filtered, hair_removed, mask, edges,
                visuals, abcde_scores, risk_score, num_instances, multi_lesion_detected,
            )
            result["no_detection"] = True
            result["overall_visual_concern"] = policy.CONCERN_NO_DETECTION
            return result

        # Primary = highest-confidence instance (rows are pre-sorted by
        # confidence in revised_abcd.yolo_single_image). Multiple lesions are
        # never silently merged.
        primary = rows[0]
        mask = primary["mask"]
        masked_for_edges = cv2.bitwise_and(hair_removed, hair_removed, mask=mask)
        edges = self._legacy._detect_edges(masked_for_edges)

        # V5's 11 new features, from the SAME unchanged preprocess_image()
        # pipeline_v2.process_image() already used internally for C -- see
        # module docstring for why this is called a second time here.
        no_hair_v5, circle_info_v5 = v5_preprocess_image(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
        new_features = extract_v5_new_features(mask, no_hair_v5, circle_info_v5, primary["D_px"])

        feature_row = {
            "A_value": primary["A_value"], "B_circularity": primary["B_circularity"],
            "C_value": primary["C_value"], "D_px": primary["D_px"],
            "confidence": primary["confidence"], "lesion_fraction": primary["lesion_fraction"],
            **new_features,
        }
        missing = [f for f in V5_CONFIG["features"] if feature_row.get(f) is None or feature_row.get(f) != feature_row.get(f)]
        if missing:
            # A degenerate mask produced an unusable feature (e.g. empty
            # contour) -- do not guess. Treat exactly like NO_DETECTION
            # rather than crashing the request or silently scoring garbage.
            edges = self._legacy._detect_edges(hair_removed)
            abcde_scores = _empty_abcde_scores(f"feature extraction incomplete ({', '.join(missing)})")
            risk_score = 0.0
            visuals = self._legacy._build_abcd_visuals(original, mask, abcde_scores)
            result = self._package(
                original, bilateral_filtered, median_filtered, hair_removed, mask, edges,
                visuals, abcde_scores, risk_score, num_instances, multi_lesion_detected,
            )
            result["no_detection"] = True
            result["overall_visual_concern"] = policy.CONCERN_NO_DETECTION
            return result

        elevated, decision_score = v5_predict_from_row(feature_row, self._decision_pipeline)

        relative_size_pct = _relative_lesion_size_pct(primary["D_px"], image_width_px)

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
                },
            },
            "diameter": {
                # No mm score, ever -- D_px only, per policy. lesion_size_px
                # and relative_size_pct are pixel/relative proxies only, never
                # labeled as millimeters.
                "score": None,
                "details": {
                    "reason": "no validated physical (mm) calibration is available for this "
                              "pipeline; lesion_size_px and relative_size_pct hold pixel-based "
                              "proxies instead",
                    "diameter_px": primary["D_px"],
                    "lesion_size_px": primary["D_px"],
                    "relative_size_pct": relative_size_pct,
                    "concern": False,
                },
            },
            "evolving": {"score": None, "details": {"reason": "no prior check to compare against"}},
        }

        # risk_score is V5's own decision_score (the frozen logistic
        # regression's output probability, 0-1) rescaled to the app's
        # existing 0-100 field. NOT a melanoma probability.
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
        result["v5_decision_score"] = round(decision_score, 4)
        result["yolo_confidence"] = primary["confidence"]
        result["quality_flags"] = primary["quality_flags"].split(";") if primary["quality_flags"] not in ("", "ok") else []
        result["no_detection"] = False
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
            "segmentation": mask,
            "edges": edges,
            "asymmetry_visual": visuals["asymmetry"],
            "border_visual": visuals["border"],
            "color_visual": visuals["color"],
            "diameter_visual": visuals["diameter"],
            "multi_instance_overlay": multi_instance_overlay,
            "abcde_scores": abcde_scores,
            "risk_score": risk_score,
            # Always None: no hair-width (or any other) physical calibration
            # is used by V5. Diameter is pixels/relative-only; downstream
            # code (evolution.score_change) already tolerates mm_per_px=None.
            "mm_per_px": None,
            "num_lesion_instances": num_instances,
            "multi_lesion_detected": multi_lesion_detected,
        }
