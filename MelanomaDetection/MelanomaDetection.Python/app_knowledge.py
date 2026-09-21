"""Curated passages describing the app's own concepts (risk bands, recheck
cadence, what "flagged" means) for the textbook chat feature (see
textbook_chat.py).

The risk-band and cadence numbers are generated from policy.py's own
constants -- the same numbers the app itself uses to score and schedule
rechecks -- rather than typed out by hand, so this can never state a
different threshold than the app actually applies. Kept deliberately small:
a handful of entries, offered to the chat model in full rather than
retrieved by embedding search, unlike the (much larger) textbook.
"""

import functools

import policy

_STATIC_PASSAGES = (
    {
        "id": "app-abcde-overview",
        "text": (
            "ABCDE is shorthand for the five features this app's image analysis checks on a "
            "photographed spot: Asymmetry, Border irregularity, Color variation, Diameter, and "
            "Evolution (change over time). A feature is \"flagged\" when the analysis measures it "
            "outside the range it treats as typical for that feature -- flagging is a pattern "
            "match, not a diagnosis."
        ),
    },
    {
        "id": "app-evolution-note",
        "text": (
            "The Evolution (E) feature compares a spot against its own earlier photos over time; "
            "it has no result until at least two checks of the same tracked spot exist."
        ),
    },
)


def _risk_band_passage() -> dict:
    return {
        "id": "app-risk-bands",
        "text": (
            f"This app maps a 0-100 risk score to one of three bands: \"low\" below "
            f"{policy.BAND_LOW_MAX:g}, \"moderate\" from {policy.BAND_LOW_MAX:g} up to "
            f"{policy.BAND_MODERATE_MAX:g}, and \"high\" at {policy.BAND_MODERATE_MAX:g} or above. "
            f"The band affects only how soon a recheck is suggested -- it is not a medical diagnosis."
        ),
    }


def _cadence_passage() -> dict:
    low, moderate = policy.CADENCE_DAYS["low"], policy.CADENCE_DAYS["moderate"]
    tightened_low = policy.TIGHTENED_CADENCE_DAYS["low"]
    tightened_moderate = policy.TIGHTENED_CADENCE_DAYS["moderate"]
    return {
        "id": "app-recheck-cadence",
        "text": (
            f"By default this app suggests rechecking a low-risk spot every {low} days and a "
            f"moderate-risk spot every {moderate} days; a high-risk spot is flagged to see a "
            f"clinician now rather than on a schedule. If the risk profile shows an elevated "
            f"personal risk factor (Fitzpatrick I-II skin, a family history of skin cancer, a "
            f"history of blistering sunburns, a high mole count, or high sun exposure), those "
            f"intervals tighten to {tightened_low} days and {tightened_moderate} days."
        ),
    }


@functools.lru_cache(maxsize=1)
def passages() -> tuple:
    """All app-concept passages as {"id", "text"} dicts, generated once and cached."""
    return _STATIC_PASSAGES + (_risk_band_passage(), _cadence_passage())
