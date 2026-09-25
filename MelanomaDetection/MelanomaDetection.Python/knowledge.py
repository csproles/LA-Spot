"""Grounding for the AI explanation: which reference text it may cite, and a
checker for what it says.

The risk score never comes from here. It is computed by image_processor.py; the
language model only *explains* it (llm_explainer.py). This module limits what
that explanation may claim to two things: what the analysis data shows, and
what a small, curated set of National Cancer Institute passages says.

No language model can be made incapable of hallucinating, so the guarantee here
is different and checkable:

  * the model only ever sees passages picked by fixed rules (select_passages),
    never the whole booklet, and never a passage about a feature that wasn't
    flagged;
  * every statement it returns must cite passage ids and copy an exact quote
    from one of them, which parse_and_validate verifies in code;
  * numbers, diagnosis wording and mentions of unflagged features are rejected
    in code, not just discouraged in the prompt;
  * when the app supplies the overall score, a reply that names a different risk
    band than the one shown beside it, or describes a comparison with an earlier
    photo that never happened, is rejected the same way;
  * if the model can't produce a valid answer, fallback_data builds one from the
    passages directly, with no model involved.

The passages live in knowledge/nci_melanoma_booklet.json as verbatim excerpts.
The scanned PDF is not read at runtime: it is a low-quality scan whose text
layer has OCR errors, and its photographs and artwork may be copyrighted.
"""

import functools
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

KNOWLEDGE_PATH = Path(__file__).resolve().parent / "knowledge" / "nci_melanoma_booklet.json"

# Payload keys (see llm_explainer._map_to_llm_schema) mapped to the one passage
# that describes that feature.
CRITERION_PASSAGE = {
    "asymmetry": "nci-abcd-asymmetry",
    "border": "nci-abcd-border",
    "color": "nci-abcd-color",
    "diameter_mm": "nci-abcd-diameter",
}

CRITERION_LABEL = {
    "asymmetry": "asymmetry",
    "border": "border irregularity",
    "color": "color variation",
    "diameter_mm": "diameter",
}

# Passages that never name a specific ABCD feature, so they are always safe to
# offer for next steps without risking a mention of an unflagged criterion.
GENERAL_PASSAGES = (
    "nci-report-changes",
    "nci-early-changes",
    "nci-skin-exam-checkup",
    "nci-self-exam-how",
    "nci-self-exam-record",
    "nci-uv-midday",
    "nci-biopsy",
)

# Words that mean a feature is being discussed. If a feature was NOT flagged, a
# statement matching its pattern is rejected (rule 4 of the system prompt). Each
# passage may match only its own feature's pattern -- tests enforce that.
UNFLAGGED_PATTERNS = {
    "asymmetry": re.compile(r"asymmetr|symmetr", re.IGNORECASE),
    "border": re.compile(r"\bborders?\b|\bedges?\b|ragged|notched|blurred|outline", re.IGNORECASE),
    "color": re.compile(r"\bcolou?rs?\b|\bshades?\b|\bhues?\b", re.IGNORECASE),
    "diameter_mm": re.compile(r"diameter|millimet|\bmm\b|pencil|eraser", re.IGNORECASE),
}

_FORBIDDEN = (
    (
        re.compile(
            r"\byou (?:likely |probably |may |might |do )?have (?:a |an )?"
            r"(?:melanoma|skin cancer|cancer|malignan\w*|tumou?r)",
            re.IGNORECASE,
        ),
        "states a diagnosis",
    ),
    (
        re.compile(
            r"\b(?:this|it|the spot|the lesion|your spot|the mole) "
            r"(?:is|was|looks like|appears to be|could be|may be|might be|is likely) "
            r"(?:a |an )?(?:melanoma|cancer)",
            re.IGNORECASE,
        ),
        "states a diagnosis",
    ),
    (re.compile(r"\b(?:benign|malignant|cancerous)\b", re.IGNORECASE), "makes a diagnostic claim"),
    (
        re.compile(
            r"\bdon'?t worry\b|\bnothing to worry\b|\bprobably (?:fine|nothing|harmless)\b"
            r"|\bno cause for concern\b",
            re.IGNORECASE,
        ),
        "gives false reassurance",
    ),
    (
        re.compile(r"\b(?:dangerous|urgent|emergency|deadly|life-threatening)\b", re.IGNORECASE),
        "uses alarming language",
    ),
)

# STYLE_PROMPT S3: no em dash, en dash, or double hyphen written as a dash. Text inside
# quotation marks is skipped, since a quoted passage keeps its own punctuation (the
# booklet itself uses dashes, and the passage-only fallback quotes it word for word).
_DASH = re.compile("[—–]|--")
_QUOTED = re.compile('"[^"]*"|“[^”]*”')

# A reply may not name a risk band other than the one the app shows beside it. "higher risk"
# (as in a risk factor) is deliberately not a band name, so it is not matched.
# V5's two real verdicts, always paired with "visual concern" in policy.py's own
# CONCERN_LABEL/CONCERN_ADVICE wording -- requiring that phrase (not a bare "elevated"
# or "lower") avoids false positives from an unrelated use of either word (e.g. "elevated
# sun exposure" is a risk-profile term, not this result).
_CONCERN_MENTION = re.compile(r"\b(elevated|lower)\s+visual\s+concern\b", re.IGNORECASE)

# Claims about how a spot changed. They are only allowed when the comparison with the
# previous photo actually found that change; the cue words keep ordinary sentences
# (a pencil eraser "larger than...") from being mistaken for one.
_GROWTH_CLAIM = re.compile(r"\b(?:grown|grew|larger|bigger|increase[ds]?|differ(?:s|ent|ence)?)\b", re.IGNORECASE)
_HAS_CHANGED_CLAIM = re.compile(
    r"\b(?:has|have|had) (?:changed|grown|got (?:bigger|larger)|become (?:bigger|larger))\b", re.IGNORECASE
)
_COMPARISON_CUE = re.compile(r"\b(?:since|compared|earlier|previous|last|before)\b", re.IGNORECASE)

_SEES_A_PROFESSIONAL = re.compile(
    r"dermatolog|health ?care provider|doctor|physician", re.IGNORECASE
)
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_MIN_QUOTE_CHARS = 12
_MAX_STATEMENT_CHARS = 500
_STATEMENT_KEYS = {"text", "basis", "source_ids", "quote"}


@dataclass(frozen=True)
class Passage:
    id: str
    page: int
    text: str


@dataclass(frozen=True)
class Source:
    title: str
    publisher: str
    publication_number: str
    edition: str
    caveat: str


@functools.lru_cache(maxsize=1)
def load(path: str = str(KNOWLEDGE_PATH)):
    """Read the knowledge file. Returns (Source, {passage id: Passage})."""
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    source = Source(**raw["source"])
    passages = {}
    for entry in raw["passages"]:
        passage = Passage(id=entry["id"], page=entry["page"], text=entry["text"])
        if passage.id in passages:
            raise ValueError(f"Duplicate passage id in knowledge file: {passage.id}")
        passages[passage.id] = passage
    return source, passages


def select_passages(payload: dict) -> list:
    """The passages the model may cite for this analysis, in a fixed order.

    A feature's passage is offered only when that feature was flagged, so the
    model has nothing to say about an unflagged one.
    """
    _, everything = load()
    flagged = [key for key in CRITERION_PASSAGE if payload.get(key, {}).get("flagged")]
    ids = [CRITERION_PASSAGE[key] for key in flagged]
    if flagged:
        ids.append("nci-abcd-variation")
    ids.extend(GENERAL_PASSAGES)

    selected = []
    for passage_id in dict.fromkeys(ids):
        if passage_id in everything:
            selected.append(everything[passage_id])
    return selected


def format_passages_for_prompt(passages: list) -> str:
    return "\n".join(f'[{p.id}] (page {p.page}) "{p.text}"' for p in passages)


def _normalize(text: str) -> str:
    """Fold typographic dashes/quotes and whitespace so a quote can be compared by content."""
    translation = {ord(c): "-" for c in "‐‑‒–—−"}
    translation.update({ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'})
    folded = unicodedata.normalize("NFKC", text).translate(translation)
    return " ".join(folded.split()).casefold()


def _allowed_numbers(payload: dict, cited: list) -> set:
    allowed = set()

    def add(token):
        allowed.add(token)
        value = float(token)
        allowed.update({str(round(value)), f"{value:.1f}", f"{value:g}"})

    for token in _NUMBER.findall(json.dumps(payload)):
        add(token)
    for passage in cited:
        allowed.update(_NUMBER.findall(passage.text))
    return allowed


def _mentionable_when_unflagged(payload: dict) -> set:
    """Features a reply may name even though the colour criterion itself was not flagged.

    "The colour changed since your last photo" and "you noted it changed colour" are
    facts about change and about what the person reported, not claims about colour
    variation, so they must not trip the unflagged-feature check.
    """
    allowed = set()
    if "color" in (payload.get("change_since_last_photo") or {}).get("changes_noticed", []):
        allowed.add("color")
    if "changed color" in payload.get("symptoms_reported", []):
        allowed.add("color")
    return allowed


def _check_coherence(where: str, text: str, payload: dict) -> list:
    """Checks that only apply when the app supplied the overall result and the change data."""
    problems = []

    overall_result = payload.get("overall_result")
    if overall_result in ("elevated", "lower"):
        for match in _CONCERN_MENTION.finditer(text):
            named = match.group(1).lower()
            if named != overall_result:
                problems.append(f"{where}: names {named} visual concern, but this result is {overall_result}.")

    change = payload.get("change_since_last_photo")
    if change is not None and _COMPARISON_CUE.search(text):
        claims_change = _GROWTH_CLAIM.search(text) or _HAS_CHANGED_CLAIM.search(text)
        if not change.get("available") and claims_change:
            problems.append(f"{where}: describes a change since an earlier photo, but no comparison was possible.")
        elif change.get("available") and "size" not in change.get("changes_noticed", []) and _GROWTH_CLAIM.search(text):
            problems.append(f"{where}: says the spot grew or differs, but the comparison did not find that.")

    return problems


def _check_statement(where: str, item, payload: dict, provided: dict) -> list:
    if not isinstance(item, dict) or set(item) != _STATEMENT_KEYS:
        return [f'{where}: must be an object with exactly the keys "text", "basis", "source_ids", "quote".']

    text, basis, source_ids, quote = item["text"], item["basis"], item["source_ids"], item["quote"]
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_STATEMENT_CHARS:
        return [f'{where}: "text" must be one non-empty sentence under {_MAX_STATEMENT_CHARS} characters.']
    if not isinstance(source_ids, list) or not all(isinstance(i, str) for i in source_ids):
        return [f'{where}: "source_ids" must be a list of strings.']

    problems = []
    cited = []
    if basis == "booklet":
        unknown = [i for i in source_ids if i not in provided]
        if not source_ids or unknown:
            problems.append(
                f"{where}: cites {unknown or 'no passage'}; only cite these ids: {sorted(provided)}."
            )
        cited = [provided[i] for i in source_ids if i in provided]
        if not isinstance(quote, str) or len(_normalize(quote)) < _MIN_QUOTE_CHARS:
            problems.append(f'{where}: "quote" must be an exact excerpt of a cited passage.')
        elif cited and not any(_normalize(quote) in _normalize(p.text) for p in cited):
            problems.append(f'{where}: "quote" does not appear verbatim in the cited passage(s).')
    elif basis == "analysis":
        if source_ids or quote is not None:
            problems.append(f'{where}: an "analysis" statement must have "source_ids": [] and "quote": null.')
    else:
        problems.append(f'{where}: "basis" must be "analysis" or "booklet".')

    allowed = _allowed_numbers(payload, cited)
    stray = sorted({n for n in _NUMBER.findall(text) if n not in allowed})
    if stray:
        problems.append(f"{where}: uses number(s) {stray} that are in neither the analysis data nor the cited passage.")

    for pattern, reason in _FORBIDDEN:
        if pattern.search(text):
            problems.append(f"{where}: {reason}.")

    if _DASH.search(_QUOTED.sub("", text)):
        problems.append(f"{where}: uses a dash; use a comma, period, colon, or parentheses, or rewrite the sentence.")

    mentionable = _mentionable_when_unflagged(payload)
    for key, pattern in UNFLAGGED_PATTERNS.items():
        entry = payload.get(key)
        if entry is not None and not entry.get("flagged") and key not in mentionable and pattern.search(text):
            problems.append(f"{where}: mentions {CRITERION_LABEL[key]}, which was not flagged.")

    problems.extend(_check_coherence(where, text, payload))
    return problems


def parse_and_validate(content, payload: dict, passages: list):
    """Check a model reply. Returns (data, problems); data is None unless problems is empty."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, ["The reply was not valid JSON."]
    if not isinstance(data, dict) or set(data) != {"noticed", "next_steps"}:
        return None, ['The JSON must be an object with exactly the keys "noticed" and "next_steps".']

    provided = {p.id: p for p in passages}
    problems = []
    # When the app supplies a known overall result (elevated/lower) it writes the first
    # next step itself (the timeframe for seeing a doctor), so the model adds one to
    # three more, not two to four. "overall_result" is always present in the payload
    # (see llm_explainer._map_to_llm_schema), so check its value, not just its presence.
    has_lead_step = payload.get("overall_result") in ("elevated", "lower")
    step_range = (1, 3) if has_lead_step else (2, 4)
    for section, low, high in (("noticed", 1, 6), ("next_steps", *step_range)):
        items = data[section]
        if not isinstance(items, list) or not low <= len(items) <= high:
            problems.append(f'"{section}" must be a list of {low} to {high} items.')
            continue
        for index, item in enumerate(items, start=1):
            problems.extend(_check_statement(f"{section}[{index}]", item, payload, provided))

    # When the app supplies a known overall result, its own CONCERN_ADVICE line (see
    # llm_explainer's "lead") always recommends seeing a professional, so the model's
    # own next_steps aren't required to repeat it.
    if not problems and not has_lead_step and not any(
        _SEES_A_PROFESSIONAL.search(step["text"]) for step in data["next_steps"]
    ):
        problems.append("At least one next step must recommend seeing a dermatologist or healthcare provider.")

    return (None, problems) if problems else (data, [])


def render_explanation(data: dict, source: Source, passages: list, lead=None, opening_steps=()) -> str:
    """The plain-text shape the web app displays, plus a Sources block for what was cited.

    `lead` (the score-and-band summary) and `opening_steps` (the first next step)
    are written by the app from the same wording the screen shows, not by the model,
    so the explanation can never contradict the verdict beside it. Everything else
    was checked in parse_and_validate.
    """
    by_id = {p.id: p for p in passages}
    pages = sorted(
        {by_id[i].page for section in ("noticed", "next_steps") for item in data[section] for i in item["source_ids"] if i in by_id}
    )

    # Every point is its own bullet. The bullets are added here, not asked of the model
    # (its prompt forbids markdown); the page shows line breaks as written.
    lines = []
    if lead:
        lines.extend(["In short:", _bullet(lead), ""])
    lines.append("What the analysis noticed:")
    lines.extend(_bullet(item["text"]) for item in data["noticed"])
    lines.extend(["", "Suggested next steps:"])
    lines.extend(_bullet(step) for step in (opening_steps or ()) if step)
    lines.extend(_bullet(item["text"]) for item in data["next_steps"])

    if pages:
        label = "page" if len(pages) == 1 else "pages"
        lines.extend(
            [
                "",
                "Sources:",
                f"{source.publisher}, {source.title} ({source.publication_number}, {source.edition}), "
                f"{label} {', '.join(str(p) for p in pages)}. {source.caveat}",
            ]
        )
    return "\n".join(lines)


def _bullet(text: str) -> str:
    return "• " + text.strip()


def _statement(text: str, basis: str = "analysis", passage_ids=(), quote=None) -> dict:
    return {"text": text, "basis": basis, "source_ids": list(passage_ids), "quote": quote}


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _change_sentence(change: dict) -> str:
    found = change.get("changes_noticed", [])
    if not found:
        return "Compared with the last photo of this spot, the analysis did not find a notable change."

    sentence = f"Compared with the last photo of this spot, the analysis found a change in {_join(found)}."
    percent = change.get("area_change_percent")
    if "size" in found and percent is not None:
        sentence += f" The spot's area is about {abs(percent)}% {'larger' if percent > 0 else 'smaller'}."
    return sentence


def fallback_data(payload: dict, passages: list) -> dict:
    """An explanation built from the passages with no model involved.

    Used when the model can't produce a reply that passes parse_and_validate.
    Plainer than a model's, but it reads in the same order (what was found, what
    changed, what the person reported, what the booklet says) and every sentence
    is either fixed wording or a verbatim excerpt.
    """
    by_id = {p.id: p for p in passages}
    flagged = [key for key in CRITERION_PASSAGE if payload.get(key, {}).get("flagged")]
    change = payload.get("change_since_last_photo")
    symptoms = payload.get("symptoms_reported")

    noticed = []
    if flagged:
        measured = payload.get("diameter_mm", {}).get("value") if "diameter_mm" in flagged else None
        detail = f" The spot measures about {measured} mm across." if measured is not None else ""
        noticed.append(_statement(f"The analysis flagged {_join(CRITERION_LABEL[k] for k in flagged)} in this spot.{detail}"))
    else:
        noticed.append(_statement("The analysis did not flag any of the features it checks in this photo. That does not rule anything out."))

    if change and change.get("available"):
        noticed.append(_statement(_change_sentence(change)))
    elif change:
        noticed.append(_statement("There was no earlier photo of this spot that could be compared, so change over time was not checked."))

    if symptoms:
        noticed.append(_statement(f"You reported these about the spot: {', '.join(symptoms)}."))
    if "diameter_mm" not in payload:
        noticed.append(_statement("Size could not be measured for this image."))

    for key in flagged:
        passage = by_id[CRITERION_PASSAGE[key]]
        noticed.append(
            _statement(
                f'The National Cancer Institute booklet describes {CRITERION_LABEL[key]}: "{passage.text}"',
                "booklet",
                [passage.id],
                passage.text,
            )
        )

    def with_quote(lead: str, passage_id: str) -> dict:
        passage = by_id[passage_id]
        return _statement(f'{lead} The booklet says: "{passage.text}"', "booklet", [passage.id], passage.text)

    next_steps = [
        with_quote("See a licensed dermatologist or healthcare provider for an actual evaluation.", "nci-report-changes"),
        with_quote("Check your skin regularly and keep notes on this spot so you can notice changes.", "nci-self-exam-record"),
        with_quote("Protect your skin from the sun.", "nci-uv-midday"),
    ]
    return {"noticed": noticed[:6], "next_steps": next_steps}
