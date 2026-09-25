import copy
from pathlib import Path

import pytest

import abcd_scores
import llm_explainer
import validation

# What risk_model.predict returns as "abcd_features": the 17 raw measurements, None for a
# feature that could not be measured.
CALM = {
    "A_value": 0.10, "B_circularity": 0.25, "C_value": 0.175, "D_px": 300.0,
    "red_fraction": 0.01, "bluegray_fraction": 0.0, "dark_fraction": 0.02,
}


def features(**overrides):
    return {**CALM, **overrides}


class TestScaling:
    def test_the_concern_threshold_reads_five_out_of_ten(self):
        assert abcd_scores.scaled_0_10(0.20, 0.20) == 5.0
        assert abcd_scores.scaled_0_10(0.50, 0.50) == 5.0

    def test_it_is_clamped_to_the_bar(self):
        assert abcd_scores.scaled_0_10(5.0, 0.20) == 10.0
        assert abcd_scores.scaled_0_10(-1.0, 0.20) == 0.0

    def test_v5_uses_the_same_function(self):
        # One definition, so the two models' bars can never drift apart. Read as text on purpose:
        # importing v5_detector needs OpenCV and puts the repo's Code/ folder at the front of
        # sys.path, which would make a later `import main` pick up the wrong file.
        source = (Path(__file__).resolve().parents[1] / "v5_detector.py").read_text(encoding="utf-8")
        assert "from abcd_scores import scaled_0_10 as _scaled_0_10" in source
        assert "def _scaled_0_10" not in source


class TestFromRiskModelFeatures:
    def test_a_calm_spot_has_low_scores_and_no_flags(self):
        scores = abcd_scores.from_risk_model_features(features())
        assert scores["asymmetry"]["score"] == 2.5
        assert scores["border"]["score"] == 2.5
        assert scores["color"]["score"] == 2.5
        assert not any(scores[key]["details"]["concern"] for key in ("asymmetry", "border", "color"))

    @pytest.mark.parametrize("key, name, value", [
        ("asymmetry", "A_value", 0.21),
        ("border", "B_circularity", 0.51),
        ("color", "C_value", 0.36),
    ])
    def test_a_measurement_over_its_threshold_is_flagged(self, key, name, value):
        scores = abcd_scores.from_risk_model_features(features(**{name: value}))
        assert scores[key]["details"]["concern"] is True
        assert scores[key]["score"] > 5.0

    @pytest.mark.parametrize("key, name, threshold", [
        ("asymmetry", "A_value", 0.20), ("border", "B_circularity", 0.50), ("color", "C_value", 0.35),
    ])
    def test_a_measurement_exactly_at_the_threshold_is_flagged(self, key, name, threshold):
        # The measurements are rounded to 3 decimals, so a tie may have been just over the line.
        # It is flagged (the cautious side), and its bar reads exactly 5 out of 10.
        scores = abcd_scores.from_risk_model_features(features(**{name: threshold}))
        assert scores[key]["details"]["concern"] is True
        assert scores[key]["score"] == 5.0

    @pytest.mark.parametrize("key, name, threshold", [
        ("asymmetry", "A_value", 0.20), ("border", "B_circularity", 0.50), ("color", "C_value", 0.35),
    ])
    def test_a_measurement_just_under_the_threshold_is_not_flagged(self, key, name, threshold):
        scores = abcd_scores.from_risk_model_features(features(**{name: threshold - 0.001}))
        assert scores[key]["details"]["concern"] is False
        assert scores[key]["score"] < 5.0

    @pytest.mark.parametrize("name", ["red_fraction", "bluegray_fraction", "dark_fraction"])
    def test_an_unusual_colour_flags_colour_even_when_the_variation_is_low(self, name):
        # The legacy colour rule flags a spot with more than 8% pink/red, blue-grey or black,
        # however even its colour otherwise is, so the flag and the bar can differ.
        scores = abcd_scores.from_risk_model_features(features(**{name: 0.09}))
        assert scores["color"]["details"]["concern"] is True
        assert scores["color"]["score"] == 2.5

    def test_a_colour_share_exactly_at_the_limit_is_not_flagged(self):
        scores = abcd_scores.from_risk_model_features(features(red_fraction=0.08))
        assert scores["color"]["details"]["concern"] is False

    def test_an_unmeasured_feature_is_not_available_rather_than_zero(self):
        scores = abcd_scores.from_risk_model_features(features(A_value=None, C_value=float("nan")))
        assert scores["asymmetry"]["score"] is None
        assert scores["color"]["score"] is None
        assert scores["asymmetry"]["details"]["concern"] is False
        assert scores["color"]["details"]["color_cv"] is None
        assert scores["border"]["score"] == 2.5  # the others are unaffected

    def test_a_spot_with_nothing_measured_has_no_scores_and_no_flags(self):
        scores = abcd_scores.from_risk_model_features({})
        assert [scores[k]["score"] for k in ("asymmetry", "border", "color")] == [None, None, None]
        assert not any(scores[k]["details"]["concern"] for k in ("asymmetry", "border", "color"))

    def test_diameter_is_never_scored_and_evolving_is_left_for_the_comparison(self):
        scores = abcd_scores.from_risk_model_features(features())
        assert scores["diameter"]["score"] is None
        assert scores["diameter"]["details"]["diameter_px"] == 300.0
        assert scores["evolving"]["score"] is None

    def test_the_result_can_be_serialised(self):
        import json
        json.dumps(abcd_scores.from_risk_model_features(features(A_value=None)))

    def test_it_feeds_the_ai_explanation_even_with_nothing_measured(self):
        # A colour that could not be measured used to break the prompt payload (None > 0.35).
        payload = llm_explainer._map_to_llm_schema(abcd_scores.from_risk_model_features({}))
        assert payload["color"]["flagged"] is False
        assert payload["color"]["spread_high"] is False


def v5_record():
    """The shape of a stored V5 result, with V5's own numbers."""
    return {
        "user_id": "u1",
        "abcde_scores": {
            "asymmetry": {"score": 1.0, "details": {"raw_asymmetry_ratio": 0.04, "concern": False}},
            "border": {"score": 1.0, "details": {"raw_border_irregularity": 0.1, "concern": False}},
            "color": {"score": 1.0, "details": {"color_cv": 0.07, "concern": False}},
            "diameter": {"score": None, "details": {"diameter_px": 200, "relative_size_pct": 12.0}},
            "evolving": {"score": 4.0, "details": {"signals": ["growth"]}},
        },
        "risk_score": 21.0,
        "overall_visual_concern": "LOWER VISUAL CONCERN",
        "explanation": "written from the old numbers",
    }


class TestAdopt:
    def test_the_risk_models_a_b_and_c_replace_the_old_ones(self):
        record = v5_record()
        new = abcd_scores.from_risk_model_features(features(A_value=0.30, B_circularity=0.60, C_value=0.40))

        assert abcd_scores.adopt(record, new) is True

        for key in ("asymmetry", "border", "color"):
            assert record["abcde_scores"][key] == new[key]
        assert abcd_scores.is_from_risk_model(record)

    def test_diameter_evolving_and_the_v5_verdict_are_kept(self):
        record = v5_record()
        before = copy.deepcopy(record)
        abcd_scores.adopt(record, abcd_scores.from_risk_model_features(features(A_value=0.30)))

        assert record["abcde_scores"]["diameter"] == before["abcde_scores"]["diameter"]
        assert record["abcde_scores"]["evolving"] == before["abcde_scores"]["evolving"]
        assert record["risk_score"] == before["risk_score"]
        assert record["overall_visual_concern"] == before["overall_visual_concern"]

    def test_an_explanation_written_from_the_old_numbers_is_dropped(self):
        record = v5_record()
        abcd_scores.adopt(record, abcd_scores.from_risk_model_features(features()))
        assert "explanation" not in record

    @pytest.mark.parametrize("missing", ["A_value", "B_circularity", "C_value"])
    def test_nothing_changes_when_the_risk_model_could_not_measure_the_spot(self, missing):
        record = v5_record()
        before = copy.deepcopy(record)

        assert abcd_scores.adopt(record, abcd_scores.from_risk_model_features(features(**{missing: None}))) is False

        assert record == before
        assert not abcd_scores.is_from_risk_model(record)


class TestProcessingIdValidation:
    @pytest.mark.parametrize("value", ["proc_1a2b3c4d5e6f", "pred_abc123", "a-b_C9"])
    def test_ids_in_the_shape_this_service_issues_are_accepted(self, value):
        assert validation.clean_processing_id(value) == value

    @pytest.mark.parametrize("value", [None, ""])
    def test_absent_is_none(self, value):
        assert validation.clean_processing_id(value) is None

    @pytest.mark.parametrize("value", ["../etc/passwd", "a b", "x" * 65, "proc_1;drop", "é", 5])
    def test_anything_else_is_rejected(self, value):
        with pytest.raises(validation.ValidationError):
            validation.clean_processing_id(value)
