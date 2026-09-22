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

V4/V5 note (2026-09-20, updated 2026-09-21 for V5):
    BAND_LOW_MAX/BAND_MODERATE_MAX and risk_band() below were calibrated for
    the old classical detector's weighted-sum 0-100 score. V4's and V5's risk_score is
    a different thing -- its frozen decision-model probability times 100 --
    with its own separate 0.25 operating threshold (pipeline_v5/decision_model.py; V4's
    pipeline_v4/decision_model.py used the same threshold value but is no longer active),
    which does not line up with these bands (25 falls inside "low"'s 0-35).
    cadence_days/next_due_at below take an explicit overall_visual_concern
    ("LOWER VISUAL CONCERN" / "ELEVATED VISUAL CONCERN", V5's own real result)
    and use it whenever it's known, instead of deriving medical-action cadence
    from these legacy score bands. risk_band()/BAND_* are kept only for
    display code that still colours a bare historical risk_score (e.g. a
    trend sparkline); they are never used to decide cadence -- or the AI
    explanation's opening line (see CONCERN_LABEL/CONCERN_ADVICE) -- for a
    check that has a recorded overall_visual_concern.
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
# Distinct sentinel for "YOLO found no lesion in this photo" -- NOT a third
# concern level and NOT treated as "low"/clean. Previously this case left
# overall_visual_concern as NULL, which was indistinguishable from "this
# check predates the field entirely" (a real collision -- see
# Services/VisualConcern.cs's docstring and store.py's checks-table comment).
# A NULL concern is now reserved exclusively for that legacy case.
CONCERN_NO_DETECTION = "NO_DETECTION"

# V4/V5 only ever produce two real concern levels (plus NO_DETECTION, which
# is not a risk assessment at all) -- there is no "moderate" here. Same
# cadence values as the legacy low/high bands, reused for continuity, but
# selected by V5's own real result, not a risk_score cut.
CADENCE_DAYS_BY_CONCERN = {CONCERN_LOWER: 90, CONCERN_ELEVATED: 0}
TIGHTENED_CADENCE_DAYS_BY_CONCERN = {CONCERN_LOWER: 60, CONCERN_ELEVATED: 0}
# A failed-to-locate photo isn't a clean result -- schedule a prompt retake
# rather than the 90-day "all clear" cadence a risk_score of 0 would
# otherwise imply. Not tightened further by risk-profile factors: the retake
# is about image quality, not the person's personal risk level.
NO_DETECTION_RETAKE_DAYS = 7


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

    Uses overall_visual_concern (V5's own LOWER/ELEVATED/NO_DETECTION result)
    when it's given -- this is the authoritative path for any check V5
    produced. Falls back to the legacy risk_score band only when the concern
    is genuinely unknown (a check saved before this field existed).

    Returns 0 for "elevated"/"high", meaning "now" rather than "on a schedule".
    """
    tightened = has_elevated_risk_factors(profile)
    if overall_visual_concern in (CONCERN_LOWER, CONCERN_ELEVATED):
        table = TIGHTENED_CADENCE_DAYS_BY_CONCERN if tightened else CADENCE_DAYS_BY_CONCERN
        return table[overall_visual_concern]
    if overall_visual_concern == CONCERN_NO_DETECTION:
        return NO_DETECTION_RETAKE_DAYS

    band = risk_band(risk_score)
    table = TIGHTENED_CADENCE_DAYS if tightened else CADENCE_DAYS
    return table[band]


def next_due_at(last_checked_at: str, risk_score, profile=None, overall_visual_concern=None):
    """ISO timestamp for a spot's next recheck, or None if it's never been checked.

    Args:
        last_checked_at: ISO timestamp of the most recent check.
        risk_score: That check's 0-100 risk score (fallback only -- see cadence_days).
        profile: The risk profile dict from store.get_profile(), or None.
        overall_visual_concern: That check's V5 result ("LOWER VISUAL CONCERN" /
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


# What a person is told about each *legacy* band (a pre-V5 check with no
# recorded overall_visual_concern). The web app shows the same words
# (MelanomaDetection.Web/Services/RiskBands.cs) beside the score, and the AI
# explanation opens with them for a check this old, so all three read as one
# voice. They live in two languages, so tests/test_policy.py compares the two
# files and fails if they drift. Never used for a check that has a recorded
# overall_visual_concern -- see CONCERN_LABEL/CONCERN_ADVICE below.
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

# What a person is told about V5's own overall_visual_concern -- the
# authoritative path for any check V5 produced (see cadence_days). Word-for-
# word the same as MelanomaDetection.Web/Services/VisualConcern.cs's
# Label/Guidance, so the AI explanation's opening line never disagrees with
# what the results page itself shows. Deliberately never says "no immediate
# action needed" or otherwise tells someone what to do medically beyond "see
# a dermatologist" -- this is a screening prototype's threshold result, not a
# clearance. NO_DETECTION has no entry here: "a lesion couldn't be located"
# isn't a verdict to open an explanation with, so explain_findings treats it
# the same as "not available" rather than pulling a line from this dict.
CONCERN_LABEL = {
    CONCERN_LOWER: "Lower visual concern",
    CONCERN_ELEVATED: "Elevated visual concern",
}

CONCERN_ADVICE = {
    CONCERN_LOWER: (
        "This photo's visual features did not cross this tool's screening threshold. "
        "This is not a clearance -- keep up with routine skin self-exams, and see a "
        "dermatologist if this spot changes or concerns you."
    ),
    CONCERN_ELEVATED: (
        "This photo's visual features crossed this tool's screening threshold. "
        "Consider having this spot evaluated by a licensed dermatologist."
    ),
}


def recheck_advice(risk_score, profile=None, overall_visual_concern=None) -> str | None:
    """The line telling someone when to photograph the spot again, or None when it is already due.

    Mirrors cadence_days' own fallback exactly: overall_visual_concern is
    authoritative when known, risk_score's legacy band otherwise. An
    elevated (or legacy "high") spot has a cadence of 0 (already due): the
    advice there is to see a doctor, not to wait for another photo, so this
    says nothing extra.
    """
    days = cadence_days(risk_score, profile, overall_visual_concern)
    if days <= 0:
        return None

    line = f"Take a new photo of this spot in about {days} days, so the app can compare it with this one."
    if has_elevated_risk_factors(profile):
        line += " That is sooner than usual because of your personal risk factors."
    return line
