"""Grounding for the AI explanation: which reference text it may cite, and a
checker for what it says.

The risk score never comes from here. It is computed by the detectors (v5_detector.py, risk_model.py); the
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

# The ABCDE sections of an explanation. A lettered item in "noticed" is one sentence, "We noticed
# ..., which means ...", and render_explanation groups it under its letter's heading. Which
# letters a reply may and must use is fixed by the analysis data (see _expected_letters).
LETTER_TITLES = {
    "A": "Asymmetry (shape)",
    "B": "Border (edges)",
    "C": "Color",
    "D": "Diameter (size)",
    "E": "Evolving (change over time)",
}
_MAX_LETTERED_CHARS = 230

# Words that carry no content when comparing the two halves of a lettered item.
_FILLER_WORDS = frozenset(
    "a an the this that these those it its is are was were be been being of in on at to for with as by from "
    "and or but so which what who we you your our can could may might will would should does do did has have had "
    "there here more most some any one".split()
)
_WORD = re.compile(r"[a-z']+")


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


def _check_statement(where: str, item, payload: dict, provided: dict, lettered: bool = False) -> list:
    if not isinstance(item, dict):
        keys = None
    elif lettered:
        keys = set(item) - {"letter"}
    else:
        keys = set(item)
    if keys != _STATEMENT_KEYS:
        extra = ' (plus "letter" for a lettered item)' if lettered else ""
        return [f'{where}: must be an object with exactly the keys "text", "basis", "source_ids", "quote"{extra}.']

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


def _expected_letters(payload: dict) -> list:
    """The ABCDE letters a reply must give a section to, and the only ones it may.

    A, B and C only when that feature was flagged (an unflagged feature is never mentioned).
    D when diameter was flagged, or when it can't be measured at all, which the reply has to
    say plainly. E when there is a comparison with an earlier photo to describe.
    """
    letters = [
        letter
        for letter, key in (("A", "asymmetry"), ("B", "border"), ("C", "color"))
        if payload.get(key, {}).get("flagged")
    ]
    diameter = payload.get("diameter_mm")
    if diameter is None or diameter.get("flagged"):
        letters.append("D")
    if payload.get("change_since_last_photo") is not None:
        letters.append("E")
    return letters


def _content_words(text: str) -> set:
    """The meaningful words of a phrase, lightly normalised (plural s dropped)."""
    words = set()
    for word in _WORD.findall(text.lower()):
        if word in _FILLER_WORDS:
            continue
        words.add(word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word)
    return words


def _repeats_the_observation(text: str) -> bool:
    """True when what follows "which means" says nothing the part before it did not.

    "We noticed uneven color, which means the color is uneven." adds no information, so a
    reader learns nothing from it. It is only flagged when every meaningful word after
    "which means" already appears before it; one new word (a shade, a shape, a consequence)
    is enough to pass.
    """
    lowered = text.lower()
    split = lowered.find("which means")
    if split < 0:
        return False
    noticed = _content_words(lowered[len("we noticed"):split])
    means = _content_words(lowered[split + len("which means"):])
    return bool(means) and means <= noticed


def _check_letters(items: list, payload: dict) -> list:
    """Checks the lettered items of "noticed": right letters, in order, each in the plain form."""
    expected = _expected_letters(payload)
    problems = []
    seen = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict) or item.get("letter") is None:
            continue
        where = f"noticed[{index}]"
        letter = item["letter"]
        if letter not in LETTER_TITLES:
            problems.append(f'{where}: "letter" must be A, B, C, D, E, or null for a point that belongs to no letter.')
            continue
        if letter not in expected:
            problems.append(
                f"{where}: letter {letter} ({LETTER_TITLES[letter]}) does not apply to this analysis, "
                "so leave it out and say nothing about it."
            )
            continue
        if letter in seen:
            problems.append(f"{where}: letter {letter} appears more than once; give each letter one item.")
            continue
        seen.append(letter)

        text = item.get("text")
        if not isinstance(text, str):
            continue
        if not text.startswith("We noticed") or "which means" not in text.lower():
            problems.append(f'{where}: a lettered item must be one sentence in the form "We noticed ..., which means ...".')
        elif _repeats_the_observation(text):
            problems.append(
                f'{where}: the part after "which means" only repeats what was noticed; use it to say what the feature '
                "is or what it tells a dermatologist, in words the first part did not use."
            )
        if len(text) > _MAX_LETTERED_CHARS:
            problems.append(f"{where}: keep a lettered item under {_MAX_LETTERED_CHARS} characters.")
        # What a feature means comes from the booklet, quoted exactly. D "not measurable" and E
        # (comparison with the last photo) are about this analysis, so they can rest on the data.
        needs_booklet = letter in ("A", "B", "C") or (letter == "D" and "diameter_mm" in payload)
        if needs_booklet and item.get("basis") != "booklet":
            problems.append(
                f'{where}: the meaning of {LETTER_TITLES[letter]} must come from a reference passage '
                '("basis": "booklet" with an exact quote).'
            )

    missing = [letter for letter in expected if letter not in seen]
    if missing:
        problems.append("noticed: missing a lettered item for " + ", ".join(f"{l} ({LETTER_TITLES[l]})" for l in missing) + ".")
    if seen != sorted(seen):
        problems.append("noticed: put the lettered items in order, A then B then C then D then E.")
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
    for section, low, high in (("noticed", 1, 9), ("next_steps", *step_range)):
        items = data[section]
        if not isinstance(items, list) or not low <= len(items) <= high:
            problems.append(f'"{section}" must be a list of {low} to {high} items.')
            continue
        for index, item in enumerate(items, start=1):
            problems.extend(
                _check_statement(f"{section}[{index}]", item, payload, provided, lettered=section == "noticed")
            )
    if isinstance(data["noticed"], list):
        problems.extend(_check_letters(data["noticed"], payload))

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
    lines.extend(_bullet(item["text"]) for item in data["noticed"] if not item.get("letter"))
    # One heading per letter, A to E, each with its one "We noticed ..., which means ..." line.
    for letter in LETTER_TITLES:
        for item in data["noticed"]:
            if item.get("letter") == letter:
                lines.extend(["", f"{letter}: {LETTER_TITLES[letter]}", _bullet(item["text"])])
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


def _statement(text: str, basis: str = "analysis", passage_ids=(), quote=None, letter=None) -> dict:
    item = {"text": text, "basis": basis, "source_ids": list(passage_ids), "quote": quote}
    if letter is not None:
        item["letter"] = letter
    return item


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _definition(passage: Passage) -> str:
    """A criterion passage without its leading heading word ("Border" and a dash), which
    is still a contiguous excerpt of the passage and so a valid exact quote."""
    return re.sub(r"^\w+\s*[\u2010-\u2015\u2212-]\s*", "", passage.text)


def fallback_data(payload: dict, passages: list) -> dict:
    """An explanation built from the passages with no model involved.

    Used when the model can't produce a reply that passes parse_and_validate. Plainer than a
    model's, but it has the same shape: one "We noticed ..., which means ..." line for each
    letter that applies (see _expected_letters), then any other points. Every sentence is
    fixed wording or a verbatim excerpt.
    """
    by_id = {p.id: p for p in passages}
    letters = _expected_letters(payload)
    change = payload.get("change_since_last_photo")
    symptoms = payload.get("symptoms_reported")

    noticed = []
    if not any(letter in letters for letter in ("A", "B", "C")):
        noticed.append(_statement("The analysis did not flag any of the features it checks in this photo. That does not rule anything out."))
    if symptoms:
        noticed.append(_statement(f"You reported these about the spot: {', '.join(symptoms)}."))

    for letter, key in (("A", "asymmetry"), ("B", "border"), ("C", "color")):
        if letter in letters:
            passage = by_id[CRITERION_PASSAGE[key]]
            definition = _definition(passage)
            noticed.append(
                _statement(
                    f'We noticed {CRITERION_LABEL[key]} in this spot, which means the booklet says: "{definition}"',
                    "booklet", [passage.id], definition, letter,
                )
            )

    if "D" in letters:
        diameter = payload.get("diameter_mm")
        if diameter is None:
            noticed.append(
                _statement(
                    "We noticed this photo has no ruler or scale, which means we cannot measure the size of the spot in millimeters.",
                    letter="D",
                )
            )
        else:
            passage = by_id[CRITERION_PASSAGE["diameter_mm"]]
            definition = _definition(passage)
            noticed.append(
                _statement(
                    f'We noticed the spot measures about {diameter.get("value")} mm across, which means the booklet says: "{definition}"',
                    "booklet", [passage.id], definition, "D",
                )
            )

    if "E" in letters:
        found = change.get("changes_noticed", []) if change.get("available") else None
        if found:
            text = f"We noticed a change in {_join(found)} compared with the last photo, which means it is worth showing to a dermatologist."
        elif found is not None:
            text = "We noticed no notable change compared with the last photo, which means the two photos look alike to the analysis."
        else:
            text = "We noticed there is no earlier photo of this spot, which means we cannot check for changes over time."
        noticed.append(_statement(text, letter="E"))

    def with_quote(lead: str, passage_id: str) -> dict:
        passage = by_id[passage_id]
        # The quote stays attached, and its page is listed under Sources, but is not repeated in
        # the sentence itself: the explanation is meant to be short.
        return _statement(lead, "booklet", [passage.id], passage.text)

    next_steps = [
        with_quote("See a licensed dermatologist or healthcare provider for an actual evaluation.", "nci-report-changes"),
        with_quote("Check your skin regularly and keep notes on this spot so you can notice changes.", "nci-self-exam-record"),
        with_quote("Protect your skin from the sun.", "nci-uv-midday"),
    ]
    return {"noticed": noticed, "next_steps": next_steps}
