"""LLM explanation layer -- turns MelanomaDetector's ABCDE output into a plain-
language explanation using OpenAI's API.

This is a port of the repository's original `llm_explainer.py` prototype: the
safety-constrained SYSTEM_PROMPT below is copied verbatim (it encodes careful,
deliberate rules -- e.g. never state a diagnosis, never mention an unflagged
criterion, always recommend a dermatologist -- that shouldn't be casually
rewritten). What's new here is `_map_to_llm_schema()`, which adapts
MelanomaDetector's actual output shape (0-10 scores + rich "details" dicts) into
the flagged/score JSON shape this prompt was originally designed around, since
this pipeline's schema evolved independently of the original prototype's.

Requires an OPENAI_API_KEY in a .env file. This repo keeps that .env at the
repository root (see load_dotenv() call below) rather than duplicating it here.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_ANCESTORS = Path(__file__).resolve().parents
_REPO_ROOT_ENV = _ANCESTORS[2] / ".env" if len(_ANCESTORS) > 2 else None
if _REPO_ROOT_ENV and _REPO_ROOT_ENV.exists():
    load_dotenv(dotenv_path=_REPO_ROOT_ENV)
else:
    # No repo-root .env to find (e.g. running inside a container where this
    # file sits at /app) -- OPENAI_API_KEY is expected to already be in the
    # environment in that case.
    load_dotenv()

MODEL = "gpt-4o"

SYSTEM_PROMPT = """You are a plain-language explainer for a skin lesion screening
tool used in a hackathon prototype (not a diagnostic medical device).

You will receive structured ABCDE analysis output (Asymmetry, Border, Color,
Diameter, Evolution) from an image analysis pipeline, expressed as scores or
flags. Your job is to translate that into a calm, clear explanation a
non-expert can understand, plus concrete next steps.

STRICT RULES -- follow every one of these, no exceptions:

1. NEVER say or imply "you have [condition]", "this is/isn't cancer", "this is
   benign", or any diagnostic conclusion. You are describing what the ABCDE
   FEATURES show, not what they mean medically.
   - Wrong: "You have an irregular mole that could be melanoma."
   - Right: "The analysis flagged some border irregularity and color
     variation in this spot -- these are features a dermatologist typically
     looks at closely."

2. Tone is calm and matter-of-fact. No alarming language ("dangerous",
   "urgent warning", "high risk of cancer"). No false reassurance either
   ("don't worry, it's probably fine"). Just state what was observed.

3. ALWAYS include a recommendation to see a licensed dermatologist or
   healthcare provider for any actual evaluation -- regardless of whether the
   scores look "low" or "high". This tool flags patterns; it does not
   evaluate them.

4. Explain each ABCDE component that was flagged in one plain-language
   sentence (what it measures, why it's tracked). Do NOT mention components
   where "flagged" is false -- not even briefly, not even as a reassuring
   aside. If a component wasn't flagged, act as if it wasn't mentioned in
   the input at all.

4a. The "D" (diameter) component may be entirely absent from the input if
    the pipeline couldn't detect hair to calibrate its pixel-to-millimeter
    scale. If diameter is missing, say plainly that size could not be
    measured for this image rather than guessing or omitting the gap
    silently.

4b. Component C (color) is calibrated to the person's own skin tone from
    the image itself, not a generic skin-tone default -- if useful, this can
    be mentioned as a reason the color assessment is specific to this photo.

5. End with 2-4 concrete, doable next steps (e.g., "photograph the spot
   monthly to track changes", "bring this analysis to a dermatology
   appointment", "note if it itches, bleeds, or changes size").

6. Never mention specific probabilities, percentages, or risk levels unless
   they are provided directly in the input data -- do not invent confidence
   numbers.

7. If the input data is incomplete or a component is missing/unclear, say so
   plainly rather than filling in a guess.

Output format: return plain text with exactly two labeled sections, in this
exact order, using these exact headers on their own line:

What the analysis noticed:
[your explanation here]

Suggested next steps:
[your next steps here]

No markdown symbols (no #, no **, no bullet characters) -- this is a
screening prototype, not a report from a clinician, and needs to render as
plain text.

EXAMPLE -- showing rule 4 in action:

Given this input:
{
  "asymmetry": {"score": 0.34, "flagged": true},
  "border": {"score": 0.62, "flagged": true},
  "diameter_mm": {"value": 8.4, "flagged": false, "measured": true}
}

A CORRECT response mentions only asymmetry and border, and says nothing
about diameter at all -- not even "diameter was measured but not flagged."
Silence on an unflagged field is correct; describing it, even neutrally,
is not.
"""

USER_PROMPT_TEMPLATE = """Here is the ABCDE analysis output from the image pipeline:

{abcde_json}

Explain this to the person who uploaded the photo, following your system
instructions exactly.
"""


def _map_to_llm_schema(abcde_scores: dict) -> dict:
    """Adapt MelanomaDetector's abcde_scores dict into this prompt's expected shape.

    MelanomaDetector reports each letter as {"score": 0-10, "details": {...}},
    with a "concern" bool and raw (pre-scaled) measurement inside "details".
    The prompt above expects a simpler {"score": raw 0-1 value, "flagged": bool}
    shape per letter (matching the original prototype's own output format), plus
    a couple of color-specific fields. This function bridges the two without
    changing the prompt itself.
    """
    asymmetry = abcde_scores["asymmetry"]["details"]
    border = abcde_scores["border"]["details"]
    color = abcde_scores["color"]["details"]
    diameter = abcde_scores["diameter"]["details"]

    payload = {
        "asymmetry": {
            "score": asymmetry.get("raw_asymmetry_ratio", 0.0),
            "flagged": asymmetry.get("concern", False),
        },
        "border": {
            "score": border.get("raw_border_irregularity", 0.0),
            "flagged": border.get("concern", False),
        },
        "color": {
            "spread_high": color.get("color_cv", 0.0) > 0.35,
            "dangerous_color_detected": None,
            "dangerous_color_coverage_pct": 0,
            "flagged": color.get("concern", False),
        },
        "evolution": {
            "flagged": False,
            "note": "not yet implemented -- no prior-image comparison available",
        },
    }

    dangerous_colors_pct = color.get("dangerous_colors_pct", {})
    if dangerous_colors_pct:
        top_color, top_pct = max(dangerous_colors_pct.items(), key=lambda item: item[1])
        payload["color"]["dangerous_color_detected"] = top_color.replace("_", "-")
        payload["color"]["dangerous_color_coverage_pct"] = round(top_pct * 100)

    diameter_mm = diameter.get("diameter_mm")
    if diameter_mm is not None:
        payload["diameter_mm"] = {
            "value": diameter_mm,
            "flagged": diameter.get("concern", False),
            "measured": True,
        }
    # else: omit diameter_mm entirely, per rule 4a -- absence, not a null value.

    return payload


def explain_findings(abcde_scores: dict) -> str:
    """Send ABCDE JSON to the LLM layer and return a plain-language explanation."""
    import json

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    payload = _map_to_llm_schema(abcde_scores)

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(abcde_json=json.dumps(payload, indent=2)),
            },
        ],
    )
    return response.choices[0].message.content
