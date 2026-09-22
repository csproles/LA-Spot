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
"""

import datetime

BAND_LOW_MAX = 35.0
BAND_MODERATE_MAX = 65.0

# Self-reported average time spent outdoors with skin exposed, low to high.
SUN_EXPOSURE_LEVELS = ("low", "moderate", "high")

CADENCE_DAYS = {"low": 90, "moderate": 30, "high": 0}

# Applied when the risk profile shows any elevated personal risk factor.
TIGHTENED_CADENCE_DAYS = {"low": 60, "moderate": 21, "high": 0}


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


def cadence_days(risk_score, profile=None) -> int:
    """How many days until a spot at this risk score should be rechecked.

    Returns 0 for high risk, meaning "now" rather than "on a schedule".
    """
    band = risk_band(risk_score)
    table = TIGHTENED_CADENCE_DAYS if has_elevated_risk_factors(profile) else CADENCE_DAYS
    return table[band]


def next_due_at(last_checked_at: str, risk_score, profile=None):
    """ISO timestamp for a spot's next recheck, or None if it's never been checked.

    Args:
        last_checked_at: ISO timestamp of the most recent check.
        risk_score: That check's 0-100 risk score.
        profile: The risk profile dict from store.get_profile(), or None.

    Returns:
        An ISO 8601 string. For a high-risk spot this is the last-checked
        timestamp itself (i.e. already due), so "overdue" sorting needs no
        special-casing for the 0-day cadence.
    """
    if not last_checked_at:
        return None

    try:
        last = datetime.datetime.fromisoformat(last_checked_at)
    except ValueError:
        return None

    return (last + datetime.timedelta(days=cadence_days(risk_score, profile))).isoformat()


# What a person is told about each band. The web app shows the same words
# (MelanomaDetection.Web/Services/RiskBands.cs) beside the score, and the AI
# explanation opens with them, so all three read as one voice. They live in two
# languages, so tests/test_policy.py compares the two files and fails if they drift.
BAND_LABEL = {
    "low": "Low risk signs",
    "moderate": "Some risk signs",
    "high": "High risk signs",
}

BAND_ADVICE = {
    "low": (
        "You don't need to do anything right now. Keep checking your skin every so often, "
        "and see a skin doctor (a dermatologist) once a year."
    ),
    "moderate": (
        "Think about seeing a skin doctor (a dermatologist) in the next few months "
        "to have this spot looked at."
    ),
    "high": (
        "Please see a skin doctor (a dermatologist) as soon as you can "
        "to have this spot looked at."
    ),
}


def recheck_advice(risk_score, profile=None):
    """The line telling someone when to photograph the spot again, or None when it is already due.

    A high-risk spot has a cadence of 0 (already due): the advice there is to
    see a doctor, not to wait for another photo, so it says nothing extra.
    """
    days = cadence_days(risk_score, profile)
    if days <= 0:
        return None

    line = f"Take a new photo of this spot in about {days} days, so the app can compare it with this one."
    if has_elevated_risk_factors(profile):
        line += " That is sooner than usual because of your personal risk factors."
    return line
