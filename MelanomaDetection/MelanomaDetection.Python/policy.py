"""Risk bands and recheck cadence -- the single source of truth for both.

Why this exists:
    The web UI previously defined risk bands three separate times, with three
    different sets of thresholds, so the same score could render as a different
    band depending on which page you were looking at. Thresholds and cadence now
    live here.

Kept in sync by hand with:
    MelanomaDetection.Web/Services/RiskBands.cs -- the client needs the same
    bands to colour and label a score it already holds, without a round trip.
    If you change a threshold here, change it there too.

Cadence baseline:
    A low-risk spot is rechecked every 3 months, matching the self-check
    interval SkinVision advises its low-risk users. Moderate tightens to a
    month; high is "don't wait, see a clinician".

V4 note (2026-09-20):
    BAND_LOW_MAX/BAND_MODERATE_MAX and risk_band() below were calibrated for
    the old classical detector's weighted-sum 0-100 score. V4's risk_score is
    a different thing -- its frozen decision-model probability times 100 --
    with its own separate 0.25 operating threshold (pipeline_v4/decision_model.py),
    which does not line up with these bands (25 falls inside "low"'s 0-35).
    cadence_days/next_due_at below take an explicit overall_visual_concern
    ("LOWER VISUAL CONCERN" / "ELEVATED VISUAL CONCERN", V4's own real result)
    and use it whenever it's known, instead of deriving medical-action cadence
    from these legacy score bands. risk_band()/BAND_* are kept only for
    display code that still colours a bare historical risk_score (e.g. a
    trend sparkline); they are never used to decide cadence for a check that
    has a recorded overall_visual_concern.
"""

import datetime

BAND_LOW_MAX = 35.0
BAND_MODERATE_MAX = 65.0

# Self-reported average time spent outdoors with skin exposed, low to high.
SUN_EXPOSURE_LEVELS = ("low", "moderate", "high")

CADENCE_DAYS = {"low": 90, "moderate": 30, "high": 0}

# Applied when the risk profile shows any elevated personal risk factor.
TIGHTENED_CADENCE_DAYS = {"low": 60, "moderate": 21, "high": 0}

CONCERN_LOWER = "LOWER VISUAL CONCERN"
CONCERN_ELEVATED = "ELEVATED VISUAL CONCERN"

# V4 only ever produces two concern levels (or none, on no detection) -- there
# is no "moderate" here. Same cadence values as the legacy low/high bands,
# reused for continuity, but selected by V4's own real result, not a risk_score cut.
CADENCE_DAYS_BY_CONCERN = {CONCERN_LOWER: 90, CONCERN_ELEVATED: 0}
TIGHTENED_CADENCE_DAYS_BY_CONCERN = {CONCERN_LOWER: 60, CONCERN_ELEVATED: 0}


def risk_band(risk_score) -> str:
    """Map a 0-100 risk score to "low", "moderate" or "high"."""
    if risk_score is None:
        return "low"
    if risk_score < BAND_LOW_MAX:
        return "low"
    if risk_score < BAND_MODERATE_MAX:
        return "moderate"
    return "high"


def has_elevated_risk_factors(profile) -> bool:
    """Whether a risk profile warrants a tighter recheck interval.

    Fitzpatrick I-II (skin that burns rather than tans), a family history of
    skin cancer, a history of blistering sunburns, a high mole count, or high
    average sun exposure are the standard personal risk factors dermatologists
    screen on, so any one of them shortens the interval.
    """
    if not profile:
        return False

    fitzpatrick = profile.get("fitzpatrick")
    return bool(
        (fitzpatrick is not None and fitzpatrick <= 2)
        or profile.get("familyHistory")
        or profile.get("blisteringSunburns")
        or profile.get("manyMoles")
        or profile.get("sunExposure") == "high"
    )


def cadence_days(risk_score, profile=None, overall_visual_concern=None) -> int:
    """How many days until a spot should be rechecked.

    Uses overall_visual_concern (V4's own LOWER/ELEVATED result) when it's
    given -- this is the authoritative path for any check V4 produced.
    Falls back to the legacy risk_score band only when the concern is
    unknown (no lesion was detected, or the check predates this field).

    Returns 0 for "elevated"/"high", meaning "now" rather than "on a schedule".
    """
    tightened = has_elevated_risk_factors(profile)
    if overall_visual_concern in (CONCERN_LOWER, CONCERN_ELEVATED):
        table = TIGHTENED_CADENCE_DAYS_BY_CONCERN if tightened else CADENCE_DAYS_BY_CONCERN
        return table[overall_visual_concern]

    band = risk_band(risk_score)
    table = TIGHTENED_CADENCE_DAYS if tightened else CADENCE_DAYS
    return table[band]


def next_due_at(last_checked_at: str, risk_score, profile=None, overall_visual_concern=None):
    """ISO timestamp for a spot's next recheck, or None if it's never been checked.

    Args:
        last_checked_at: ISO timestamp of the most recent check.
        risk_score: That check's 0-100 risk score (fallback only -- see cadence_days).
        profile: The risk profile dict from store.get_profile(), or None.
        overall_visual_concern: That check's V4 result ("LOWER VISUAL CONCERN" /
            "ELEVATED VISUAL CONCERN"), or None if unknown. Authoritative
            whenever it's known -- see cadence_days.

    Returns:
        An ISO 8601 string. For an elevated-concern (or legacy "high") spot
        this is the last-checked timestamp itself (i.e. already due), so
        "overdue" sorting needs no special-casing for the 0-day cadence.
    """
    if not last_checked_at:
        return None

    try:
        last = datetime.datetime.fromisoformat(last_checked_at)
    except ValueError:
        return None

    days = cadence_days(risk_score, profile, overall_visual_concern)
    return (last + datetime.timedelta(days=days)).isoformat()
