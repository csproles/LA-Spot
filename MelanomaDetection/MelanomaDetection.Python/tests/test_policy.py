"""Recheck cadence and risk banding.

These are the numbers the dashboard's "due for recheck" logic and the spot risk
badges are built on, and they are silently wrong-able -- an off-by-one band
boundary or a cadence table that ignores the risk profile would look completely
normal in the UI.
"""

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
