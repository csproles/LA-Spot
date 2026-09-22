"""Recheck cadence and risk banding.

These are the numbers the dashboard's "due for recheck" logic and the spot risk
badges are built on, and they are silently wrong-able -- an off-by-one band
boundary or a cadence table that ignores the risk profile would look completely
normal in the UI.
"""

import re
from pathlib import Path

import pytest

import policy


class TestRiskBand:
    def test_band_boundaries(self):
        assert policy.risk_band(0) == "low"
        assert policy.risk_band(34.9) == "low"
        assert policy.risk_band(35) == "moderate"
        assert policy.risk_band(64.9) == "moderate"
        assert policy.risk_band(65) == "high"
        assert policy.risk_band(100) == "high"

    def test_missing_score_is_not_treated_as_high_risk(self):
        assert policy.risk_band(None) == "low"


class TestElevatedRiskFactors:
    def test_no_profile_is_not_elevated(self):
        assert policy.has_elevated_risk_factors(None) is False
        assert policy.has_elevated_risk_factors({}) is False

    def test_fitzpatrick_one_and_two_are_elevated(self):
        assert policy.has_elevated_risk_factors({"fitzpatrick": 1}) is True
        assert policy.has_elevated_risk_factors({"fitzpatrick": 2}) is True

    def test_fitzpatrick_three_and_up_are_not(self):
        for skin_type in (3, 4, 5, 6):
            assert policy.has_elevated_risk_factors({"fitzpatrick": skin_type}) is False

    def test_any_single_history_factor_elevates(self):
        assert policy.has_elevated_risk_factors({"familyHistory": True}) is True
        assert policy.has_elevated_risk_factors({"blisteringSunburns": True}) is True
        assert policy.has_elevated_risk_factors({"manyMoles": True}) is True

    def test_high_sun_exposure_elevates(self):
        assert policy.has_elevated_risk_factors({"sunExposure": "high"}) is True

    def test_low_and_moderate_sun_exposure_are_not(self):
        assert policy.has_elevated_risk_factors({"sunExposure": "low"}) is False
        assert policy.has_elevated_risk_factors({"sunExposure": "moderate"}) is False


class TestCadence:
    def test_baseline_intervals(self):
        assert policy.cadence_days(10) == 90
        assert policy.cadence_days(50) == 30
        assert policy.cadence_days(90) == 0

    def test_elevated_profile_tightens_low_and_moderate(self):
        profile = {"familyHistory": True}
        assert policy.cadence_days(10, profile) == 60
        assert policy.cadence_days(50, profile) == 21

    def test_high_risk_stays_immediate_regardless_of_profile(self):
        assert policy.cadence_days(90, {"fitzpatrick": 1}) == 0


class TestNextDueAt:
    def test_adds_the_cadence_to_the_last_check(self):
        assert policy.next_due_at("2026-01-01T00:00:00+00:00", 10).startswith("2026-04-01")

    def test_high_risk_is_due_immediately(self):
        last = "2026-01-01T00:00:00+00:00"
        assert policy.next_due_at(last, 90) == last

    def test_never_checked_has_no_due_date(self):
        assert policy.next_due_at(None, 50) is None
        assert policy.next_due_at("", 50) is None

    def test_unparseable_timestamp_does_not_raise(self):
        assert policy.next_due_at("not-a-date", 50) is None


RISK_BANDS_CS = Path(__file__).resolve().parents[2] / "MelanomaDetection.Web" / "Services" / "RiskBands.cs"


def _cs_strings(source, method):
    """The three quoted strings (low, moderate, high) one RiskBands.cs method returns."""
    block = source[source.index(f"public static string {method}("):]
    block = block[: block.index("};")]
    found = dict(re.findall(r'"(low|moderate)" => "([^"]*)"', block))
    found["high"] = re.search(r'_ => "([^"]*)"', block).group(1)
    return found


class TestBandWording:
    def test_every_band_has_a_label_and_advice(self):
        assert set(policy.BAND_LABEL) == set(policy.BAND_ADVICE) == {"low", "moderate", "high"}

    @pytest.mark.parametrize("band", ["low", "moderate", "high"])
    def test_advice_always_points_to_a_skin_doctor(self, band):
        assert "skin doctor (a dermatologist)" in policy.BAND_ADVICE[band]

    def test_advice_gets_more_direct_as_the_band_rises(self):
        assert "once a year" in policy.BAND_ADVICE["low"]
        assert "next few months" in policy.BAND_ADVICE["moderate"]
        assert "as soon as you can" in policy.BAND_ADVICE["high"]

    def test_wording_matches_the_web_apps_RiskBands_cs(self):
        if not RISK_BANDS_CS.exists():
            pytest.skip("RiskBands.cs is not next to this test (running outside the repository)")
        source = RISK_BANDS_CS.read_text(encoding="utf-8")
        assert _cs_strings(source, "Label") == policy.BAND_LABEL
        assert _cs_strings(source, "Recommendation") == policy.BAND_ADVICE


class TestRecheckAdvice:
    def test_low_and_moderate_get_a_recheck_reminder(self):
        assert "about 90 days" in policy.recheck_advice(20)
        assert "about 30 days" in policy.recheck_advice(50)

    def test_high_risk_is_told_to_see_a_doctor_not_to_wait_for_another_photo(self):
        assert policy.recheck_advice(80) is None

    def test_risk_factors_shorten_the_interval_and_say_why(self):
        line = policy.recheck_advice(20, {"familyHistory": True})
        assert "about 60 days" in line
        assert "sooner than usual" in line

    def test_no_risk_factor_note_without_risk_factors(self):
        assert "sooner" not in policy.recheck_advice(20, {"familyHistory": False})
