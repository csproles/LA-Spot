"""LLM explanation layer -- turns the detector's ABCDE output into a plain-
language explanation using OpenAI's API.

This is a port of the repository's original `llm_explainer.py` prototype: the
safety-constrained SYSTEM_PROMPT below is copied verbatim (it encodes careful,
deliberate rules -- e.g. never state a diagnosis, never mention an unflagged
criterion, always recommend a dermatologist -- that shouldn't be casually
rewritten). What's new here is `_map_to_llm_schema()`, which adapts the
detector's actual output shape (0-10 scores + rich "details" dicts, plus V5's
own overall_visual_concern verdict) into the flagged/score JSON shape this
prompt was originally designed around, since this pipeline's schema evolved
independently of the original prototype's.

Grounding: the reply is constrained to what the analysis data shows and to a
small set of National Cancer Institute passages (see knowledge.py). The model
must cite passage ids and quote them exactly; the reply is checked in code, and
if it can't pass that check after one retry, an explanation is built straight
from the passages instead (knowledge.fallback_data). SYSTEM_PROMPT itself is
left untouched; GROUNDING_PROMPT is added after it and only narrows what the
model may say. STYLE_PROMPT follows it and only changes how the model writes
(plain, direct sentences, no dashes; see also the dash check in
knowledge.parse_and_validate). Each sentence is shown as a bullet point, added
by knowledge.render_explanation rather than asked of the model, and the app groups
the explanation under A to E headings (Asymmetry, Border, Color, Diameter, Evolving):
each lettered item reads "We noticed ..., which means ...", and which letters must
appear is fixed by the analysis data (knowledge._expected_letters), so a feature that
was not flagged is still never mentioned.

Coherence with the rest of the app: explain_findings can also be given V5's
own overall_visual_concern verdict, the change-since-last-photo comparison,
and the symptoms the person selected (all optional -- the plain per-letter
explanation this module has always produced still works with none of them).
When the concern is supplied, the summary line and the first "see a doctor"
step are written by this module from policy.py's own CONCERN_LABEL/
CONCERN_ADVICE wording, never by the model, so the explanation can never name
a different result than the one the results page shows (see
Services/VisualConcern.cs, which uses the same wording). knowledge.
parse_and_validate rejects a reply that claims a change the comparison did
not find, or names the wrong result, the same way it already rejects an
invented number or an unflagged feature.

Requires an OPENAI_API_KEY in a .env file. This repo keeps that .env at the
repository root (see load_dotenv() call below) rather than duplicating it here.
"""

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

import knowledge
import policy

logger = logging.getLogger(__name__)

_ANCESTORS = Path(__file__).resolve().parents
_REPO_ROOT_ENV = _ANCESTORS[2] / ".env" if len(_ANCESTORS) > 2 else None
if _REPO_ROOT_ENV and _REPO_ROOT_ENV.exists():
    load_dotenv(dotenv_path=_REPO_ROOT_ENV)
else:
    # No repo-root .env to find (e.g. running inside a container where this
    # file sits at /app) -- OPENAI_API_KEY is expected to already be in the
    # environment in that case.
    load_dotenv()

MODEL = "gpt-5.6-terra"

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

4a. The "D" (diameter) component may be entirely absent from the input
    because this pipeline has no physical scale reference (no ruler or
    similar object in the photo), so it reports pixel-based/relative-to-photo
    size only, never millimeters. If diameter is missing, say plainly that a
    physical size measurement is not available for this image (never invent
    or imply a millimeter figure) rather than guessing or omitting the gap
    silently.

4b. Component C (color) is calibrated to the person's own skin tone from
    the image itself, not a generic skin-tone default -- if useful, this can
    be mentioned as a reason the color assessment is specific to this photo.

4c. The input may include an "overall_result" field: "elevated", "lower", or
    "not_available", V5's own authoritative screening result for this photo,
    arrived at from many measurements considered together, not a simple
    count of the flagged components below. It also may include a CHANGE
    SINCE LAST PHOTO comparison and SYMPTOMS REPORTED by the person. Use
    whichever of these are present:
    - If "overall_result" is "elevated" or "lower", the summary line
      matching it is written for you and given first. It has already told
      the person the result and what to do next, so do not write any
      sentence in your own reply that states, restates, or refers to there
      being an overall result -- not even to note that one was given.
      Simply move straight to what the analysis noticed. Treat
      "overall_result" as the headline fact and the flagged components
      below as supporting detail, not the reverse:
        - If "elevated" but no individual component below is flagged, add
          one sentence noting that this photo's overall visual pattern
          crossed this tool's screening threshold even though no single
          feature was individually flagged -- do not guess or invent which
          feature caused it.
        - If "lower", do not describe the photo in alarming terms even if
          one individual component happens to be flagged; you may still
          name that flagged component per rule 4, just without implying it
          changes the overall result.
      Separately: never use a different overall-result word ("elevated" /
      "lower") than the one already given, and never state or imply a
      percentage, probability, or numeric score for it -- it is a category,
      not a number (rule 6 already forbids inventing numbers not present in
      the input).
    - If "overall_result" is "not_available" or absent, say nothing about an
      overall result at all -- describe only the individual components.
    - If CHANGE SINCE LAST PHOTO is present and available is true, describe
      only the changes it names (shape, color, and/or size) -- never say a
      spot grew, changed, or differs from before unless this comparison
      says so. If available is false, say plainly that there was no earlier
      photo to compare against; do not guess whether it has changed.
    - If SYMPTOMS REPORTED is present, you may mention what the person
      reported noticing (e.g. itching, bleeding) as something they told the
      app, not as something the analysis measured.
    Say nothing about change or symptoms if they are absent from the input.

5. End with 2-4 concrete, doable next steps (e.g., "photograph the spot
   monthly to track changes", "bring this analysis to a dermatology
   appointment", "note if it itches, bleeds, or changes size"). If the input
   already includes a next step (because the app supplied one from the
   overall result), add 1-3 more of your own instead of repeating it or
   contradicting its timing.

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

GROUNDING_PROMPT = """GROUNDING RULES -- these are added to the STRICT RULES above and never relax
any of them.

G1. You may state only two kinds of things: (a) what the ANALYSIS DATA in the
    user message shows (basis "analysis"), and (b) what one of the REFERENCE
    PASSAGES in the user message says (basis "booklet"). Nothing else. Do not
    use outside medical knowledge, statistics, causes, survival or treatment
    information, or comparisons, even if you are sure they are true.

G2. Every "booklet" statement must list the ids of the passages it relies on
    in "source_ids", and copy into "quote" one exact, contiguous excerpt (at
    least a full clause) from one of those passages, word for word. Never cite
    an id that was not provided.

G3. Every "analysis" statement must have "source_ids": [] and "quote": null,
    and may only describe the flagged features in the analysis data.

G4. Only write a number if it appears in the analysis data or in a passage you
    cite.

G5. If the data and passages do not support a statement, leave it out. A
    shorter answer is always acceptable; an unsupported one never is.

OUTPUT FORMAT -- this REPLACES the plain-text "Output format" section above.
Return only a JSON object, with no markdown and no other keys:

{
  "noticed": [
    {"letter": "A", "text": "We noticed ..., which means ...", "basis": "booklet", "source_ids": ["<id>"], "quote": "<exact excerpt>"},
    {"letter": null, "text": "one plain sentence", "basis": "analysis", "source_ids": [], "quote": null}
  ],
  "next_steps": [
    {"text": "one plain sentence", "basis": "booklet", "source_ids": ["<id>"], "quote": "<exact excerpt>"}
  ]
}

"noticed" holds the lettered items described below, plus at most two more items
with "letter": null. "next_steps" holds 2 to 4 items. Each "text" is one plain
sentence with no markdown. At least one next step must recommend seeing a
licensed dermatologist or healthcare provider.

THE ABCDE SECTIONS. The explanation is shown as sections headed A, B, C, D and
E. Write one lettered item for each letter below that applies, in order A to E,
and none for a letter that does not apply. Never mention a letter that does not
apply, not even to say it was fine.

  A (Asymmetry): only when "asymmetry" is flagged.
  B (Border): only when "border" is flagged.
  C (Color): only when "color" is flagged.
  D (Diameter, meaning size): when the data has no "diameter_mm" entry, always,
     and say the size could not be measured because the photo has no ruler or
     scale. When it has one, only if it is flagged.
  E (Evolving, meaning change over time): only when the data has
     "change_since_last_photo". Say only what it says. When "available" is
     false, say there is no earlier photo to compare.

Every lettered item is ONE sentence in the form "We noticed <what the analysis
found>, which means <what that tells us>." Keep it under 30 words. For A, B, C
(and D when diameter is flagged) the "which means" part must come from that
feature's reference passage: use "basis": "booklet", cite the passage id, and
copy an exact quote. D when size could not be measured, and E, are about this
analysis itself, so use "basis": "analysis", "source_ids": [] and "quote": null.

Anything else you are allowed to say (the one sentence rule 4c allows when the
overall result is "elevated" but nothing is flagged, or what the person reported
in SYMPTOMS REPORTED) goes in an item with "letter": null.
"""

STYLE_PROMPT = """STYLE RULES -- these change how you write, never what you may say. They are
added to the STRICT RULES and GROUNDING RULES above and never relax any of them.

S1. Every "text" is shown as its own bullet point. Make it one short sentence
    that makes sense on its own. Do not write bullet characters, numbers,
    labels, letters, or bold text yourself: the app adds the A to E headings
    and the bullets.

S2. State each point directly. Do not use "not X but Y" or "not just X, but Y"
    contrasts, and do not answer an objection nobody raised ("this does not
    mean...", "to be clear", "you might think...").

S3. Never use an em dash, an en dash, or a double hyphen. Use a comma, a
    period, a colon, or parentheses, or rewrite the sentence. A word for word
    quotation from a passage keeps its own punctuation.

S4. Do not group things in threes for rhythm. List only the items the analysis
    data actually contains.

S5. No stagecraft or filler: no "let's dive in", "here's what you need to
    know", "honestly", no rhetorical questions, no closing line that repeats a
    point, and no sayings that sound deep ("at its core", "the real question").

S6. Use plain words. Avoid: additionally, crucial, pivotal, key (as an
    adjective), robust, delve, highlight, underscore, showcase, landscape,
    tapestry, testament, vibrant, valuable, enhance, foster, meticulous. Write
    "is", "are", and "has" instead of "serves as", "stands as", "features", or
    "boasts". Do not call anything important, significant, or a turning point
    unless the analysis data or a cited passage says so.

S7. No praise, sales language, greeting, sign-off, or offer of more help. Do
    not write "I hope this helps" or "let me know".

S8. Prefer active voice and name who acts: "The analysis flagged...", "A
    dermatologist can...".

S9. Do not stack qualifiers ("could potentially perhaps"). Use at most one
    hedge in a sentence, and only when the data is genuinely uncertain.

S10. Vary sentence length, and do not begin consecutive sentences with the
     same words.

S11. Write for someone with no medical training, at about a sixth grade reading
     level. Use short, everyday words and keep every sentence under 25 words.
     Say "uneven edges", not "border irregularity". If a medical word cannot be
     avoided, explain it in the same sentence. Be brief: say each thing once.

S12. Begin each next step with an action word ("Book", "Take", "Check", "Write"),
     and keep it to 15 words or fewer.

S13. In a lettered item, the part after "which means" must add something the part
     before it does not say. Never repeat the observation in other words. Use the
     reference passage to say what the feature is, or what it tells a dermatologist.
       Wrong: "We noticed uneven color, which means the color is uneven."
       Right: "We noticed the color varies across the spot, which means shades of
              black, brown, or tan may be present."
"""

USER_PROMPT_TEMPLATE = """ANALYSIS DATA from the image pipeline (may include "overall_result",
"change_since_last_photo" and "symptoms_reported" -- see rule 4c):

{abcde_json}

REFERENCE PASSAGES (the only outside information you may use), each shown as
[id] (page) "text":

{passages}

Explain this to the person who uploaded the photo, following your system
instructions and the grounding rules exactly.
"""

MAX_ATTEMPTS = 2


def _map_to_llm_schema(abcde_scores: dict, overall_visual_concern=None) -> dict:
    """Adapt the detector's abcde_scores dict (plus V5's own overall verdict)
    into this prompt's expected shape.

    The detector reports each letter as {"score": 0-10, "details": {...}},
    with a "concern" bool and raw (pre-scaled) measurement inside "details" --
    these are per-feature reference thresholds, NOT V5's actual decision (see
    pipeline_v5/decision_model.py). The prompt above expects a simpler
    {"score": raw 0-1 value, "flagged": bool} shape per letter (matching the
    original prototype's own output format), plus a couple of color-specific
    fields and an "overall_result" field carrying V5's real verdict -- see
    rule 4c. This function bridges the two without changing the prompt itself.
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
            "spread_high": (color.get("color_cv") or 0.0) > 0.35,
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
        # Set when the person lined a circle up with a coin in the photo
        # (v5_detector.diameter_details), and on pre-V4 saved checks.
        payload["diameter_mm"] = {
            "value": diameter_mm,
            "flagged": diameter.get("concern", False),
            "measured": True,
        }
    # else: omit diameter_mm entirely, per rule 4a -- absence, not a null value.
    # V5 takes this branch whenever no coin was lined up; the pixel/
    # relative-size figures shown in the UI (AbcdeDetails.cs) are deliberately
    # NOT sent here, since rule 4a's job is simply "say size isn't available
    # as mm," not to describe the pixel proxy.

    concern_map = {"ELEVATED VISUAL CONCERN": "elevated", "LOWER VISUAL CONCERN": "lower"}
    payload["overall_result"] = concern_map.get(overall_visual_concern, "not_available")

    return payload


# Evolution's own "signals" name (see evolution.score_change) mapped to what this schema
# calls the same thing, so a growth signal reads as a change in "size", not "growth".
_CHANGE_SIGNAL_LABEL = {"shape": "shape", "color": "color", "growth": "size"}


def _map_change_to_llm_schema(evolving: dict) -> dict:
    """Adapt the detector's "evolving" entry (evolution.score_change's own shape) into
    the small, code-checkable summary the prompt and the validator use.

    Kept separate from the ABCD criteria: unlike them, "changed" is a claim about two
    photos, not one, so it gets its own coherence check (knowledge._check_coherence)
    instead of the flagged/unflagged pattern check the ABCD letters use.
    """
    details = evolving.get("details") or {}
    if evolving.get("score") is None:
        return {"available": False, "reason": details.get("reason", "no prior check to compare against")}

    changes_noticed = sorted({_CHANGE_SIGNAL_LABEL[s] for s in details.get("signals", []) if s in _CHANGE_SIGNAL_LABEL})
    result = {"available": True, "changes_noticed": changes_noticed}
    growth = details.get("area_growth_ratio")
    if growth is not None:
        result["area_change_percent"] = round(growth * 100)
    return result


def _map_symptoms(symptoms) -> list:
    """Free text never reaches the model (it is not grounded in anything checkable);
    only the fixed checkbox choices SymptomChipRow offers do."""
    allowed = {"Itchy", "Growing", "Bleeding", "Painful", "Changed color", "New"}
    return [s for s in (symptoms or []) if s in allowed]


def explain_findings(
    abcde_scores: dict,
    overall_visual_concern: str = None,
    risk_score: float = None,
    profile: dict = None,
    evolving: dict = None,
    symptoms=None,
    client=None,
) -> str:
    """Return a plain-language, source-grounded explanation of the ABCDE output.

    `overall_visual_concern` is V5's own authoritative "LOWER VISUAL CONCERN"
    / "ELEVATED VISUAL CONCERN" result for this check, when known -- it is
    the authoritative source for the summary line and next step (mirroring
    policy.cadence_days' own fallback), and is never itself presented as a
    number or probability. `risk_score`/`profile` still matter even when
    `overall_visual_concern` is given: they drive the recheck timing (see
    policy.recheck_advice). `evolving` (the "evolving" entry of abcde_scores)
    and `symptoms` are independent of the above and add the change-since-last-
    photo comparison and reported symptoms to the explanation. All of these
    are optional -- omitting every one of them gives the original, letters-
    only explanation with no behavior change.

    Model failures (a missing key, a network error) propagate to the caller. A
    reply that merely fails validation does not: it is retried once with the
    problems listed, then replaced by knowledge.fallback_data.

    `client` is injectable so tests can supply a fake OpenAI client.
    """
    client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    payload = _map_to_llm_schema(abcde_scores, overall_visual_concern)

    # The summary line and first next step are written here, from policy.py's
    # own wording, never by the model -- so the explanation can never name a
    # result other than the one the results page shows. Only for a genuinely
    # known verdict: CONCERN_NO_DETECTION ("a lesion couldn't be located")
    # isn't a result to open an explanation with, so it's left to read as
    # "not_available" (see _map_to_llm_schema) with no app-authored lead,
    # exactly like a check with no recorded concern at all.
    lead = opening_steps = None
    if overall_visual_concern in (policy.CONCERN_LOWER, policy.CONCERN_ELEVATED):
        lead = f"{policy.CONCERN_LABEL[overall_visual_concern]}. {policy.CONCERN_ADVICE[overall_visual_concern]}"
        advice = policy.recheck_advice(risk_score, profile, overall_visual_concern)
        opening_steps = (advice,) if advice else ()
    if evolving is not None:
        payload["change_since_last_photo"] = _map_change_to_llm_schema(evolving)
    reported = _map_symptoms(symptoms)
    if reported:
        payload["symptoms_reported"] = reported

    source, _ = knowledge.load()
    passages = knowledge.select_passages(payload)

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": GROUNDING_PROMPT},
        {"role": "system", "content": STYLE_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                abcde_json=json.dumps(payload, indent=2),
                passages=knowledge.format_passages_for_prompt(passages),
            ),
        },
    ]
    messages = list(base_messages)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        # GPT-5.6 rejects max_tokens, and only accepts temperature=0 when reasoning
        # is off. Off is what we want anyway: the reply is a short, checked JSON
        # object, so there's nothing to gain from paying for hidden reasoning tokens.
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=1600,
            reasoning_effort="none",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        data, problems = knowledge.parse_and_validate(content, payload, passages)
        if data is not None:
            return knowledge.render_explanation(data, source, passages, lead=lead, opening_steps=opening_steps)

        logger.warning("Explanation attempt %d rejected: %s", attempt, "; ".join(problems))
        messages = base_messages + [
            {"role": "assistant", "content": content or ""},
            {
                "role": "user",
                "content": "Your reply was rejected for these reasons:\n- "
                + "\n- ".join(problems)
                + "\nReturn a corrected JSON object that fixes every one of them.",
            },
        ]

    logger.warning("Falling back to a passage-only explanation after %d rejected attempts.", MAX_ATTEMPTS)
    return knowledge.render_explanation(
        knowledge.fallback_data(payload, passages), source, passages, lead=lead, opening_steps=opening_steps
    )
