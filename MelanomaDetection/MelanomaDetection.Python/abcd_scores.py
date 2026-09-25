"""The A, B and C scores and flags shown to a person, built from the risk model's own
measurements.

The CatBoost risk model (risk_model.py) is fed 17 ABCD-style measurements taken from its
own outline of the spot. Showing bars and explanations based on a different outline (the
older V5 detector's) made the numbers a person sees disagree with the score beside them, so
once the risk model has measured the spot these are the numbers used everywhere the check's
A, B and C appear: the bars, the "how we got this" numbers, the AI explanation, and the saved
check.

These are display numbers. The model has no per-letter output: it takes the raw measurements
as inputs, and each bar is one measurement rescaled so its concern threshold reads 5 out of 10.
No number here feeds the risk score.

Kept free of cv2, torch and Flask so it can be tested on its own.
"""

# Concern thresholds for the raw measurements, the same ones V5 has always flagged at.
# V5 decides each flag on the unrounded measurement but reports it rounded to 3 decimals, and
# the model is given the rounded value. A value that reads exactly the threshold could have been
# just under it or just over, and cannot be told apart here. It is flagged, the cautious side for
# a screening tool: on 320 real photos that matched V5's own flag better than not flagging it.
ASYMMETRY_THRESHOLD = 0.20
BORDER_THRESHOLD = 0.50
COLOR_CV_THRESHOLD = 0.35

# The legacy colour rule also flags a spot when more than this share of it is pink/red,
# blue-grey, white or black. Three of those four shares are among the model's own inputs
# (red_fraction, bluegray_fraction, dark_fraction, the very values the rule uses); the white
# share is not, so a spot whose only unusual colour is white cannot be flagged from here.
DANGEROUS_COLOR_FRACTION = 0.08

RISK_MODEL_SOURCE = "risk_model"


def scaled_0_10(raw_value, threshold):
    """Display-only rescaling so the UI's 0-10 bars work: raw_value == threshold maps to 5.0.
    This number is NEVER used to make a decision -- see the module docstring."""
    if raw_value is None or threshold in (None, 0):
        return 0.0
    return round(min(max(raw_value / threshold * 5.0, 0.0), 10.0), 2)


def _number(value):
    """The value as a float, or None when it is missing or NaN (an unmeasured feature)."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _at_or_over(value, threshold):
    """True when a rounded measurement reaches its concern threshold (a tie counts; see above)."""
    return value is not None and value >= threshold


def _score(raw, threshold):
    """0-10 score, or None (shown as N/A) when the feature could not be measured. An
    unmeasured feature is not the same as a measured zero."""
    return None if raw is None else scaled_0_10(raw, threshold)


def from_risk_model_features(abcd: dict) -> dict:
    """The {"asymmetry": {"score", "details"}, ...} shape the UI panels and the AI explanation
    expect, from the risk model's raw ABCD feature values (feature name -> number or None)."""
    a = _number(abcd.get("A_value"))
    b = _number(abcd.get("B_circularity"))
    c = _number(abcd.get("C_value"))
    dangerous = [_number(abcd.get(name)) for name in ("red_fraction", "bluegray_fraction", "dark_fraction")]

    color_concern = _at_or_over(c, COLOR_CV_THRESHOLD) or any(
        share is not None and share > DANGEROUS_COLOR_FRACTION for share in dangerous
    )

    return {
        "asymmetry": {
            "score": _score(a, ASYMMETRY_THRESHOLD),
            "details": {"raw_asymmetry_ratio": a, "concern": _at_or_over(a, ASYMMETRY_THRESHOLD)},
        },
        "border": {
            "score": _score(b, BORDER_THRESHOLD),
            "details": {"raw_border_irregularity": b, "concern": _at_or_over(b, BORDER_THRESHOLD)},
        },
        "color": {
            "score": _score(c, COLOR_CV_THRESHOLD),
            "details": {"color_cv": c, "concern": bool(color_concern)},
        },
        "diameter": {
            "score": None,
            "details": {
                "reason": "no validated physical (mm) calibration is available for this pipeline",
                "diameter_px": _number(abcd.get("D_px")),
                "concern": False,
            },
        },
        "evolving": {"score": None, "details": {"reason": "no prior check to compare against"}},
    }


def adopt(record: dict, scores: dict) -> bool:
    """Replaces A, B and C in a saved check (a V5 result) with the risk model's, in place.

    Returns False and changes nothing when the risk model could not measure the spot (any of
    A, B or C is missing), so a check never goes from measured to blank. D and E are left
    alone: neither model scores diameter, and E is the comparison between two checks.
    An explanation written earlier from the old numbers is dropped so it is written again
    from the new ones.
    """
    if any(scores[key]["score"] is None for key in ("asymmetry", "border", "color")):
        return False

    for key in ("asymmetry", "border", "color"):
        record["abcde_scores"][key] = scores[key]
    record["abcde_source"] = RISK_MODEL_SOURCE
    record.pop("explanation", None)
    return True


def is_from_risk_model(record: dict) -> bool:
    return record.get("abcde_source") == RISK_MODEL_SOURCE
