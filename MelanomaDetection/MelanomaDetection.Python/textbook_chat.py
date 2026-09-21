"""Retrieval and grounded chat for the textbook Q&A feature (the chat widget
shown on every page). Structured like knowledge.py + llm_explainer.py
combined, but for open-ended questions instead of ABCDE-flag-driven passage
selection.

Three grounded sources, and nothing else:
  * textbook passages -- retrieved by embedding similarity across every book
    in knowledge/textbooks/ (see scripts/ocr_extract_textbook.py and
    scripts/embed_textbook.py for how a book's {slug}.json and
    {slug}_embeddings.npz pair is made; drop in as many books as needed, each
    its own pair -- there is no fixed limit of one);
  * app-concept passages -- a small, fixed set from app_knowledge.py, always
    offered in full;
  * the caller's current-screen data (page_context) -- specific values the
    reply may state numerically, but never interpret beyond what a cited
    passage supports.

As in knowledge.py, no language model can be made incapable of hallucinating,
so every claim is checked in code: it must cite a real passage id and copy an
exact quote from it, or it must be a bare statement of a page_context value.
If the model can't produce a reply that passes that check after one retry,
the safest fallback here is a refusal, not a guess -- unlike knowledge.py's
ABCDE-flag-constrained fallback, an open-ended question has no fixed shape to
safely fall back to.

The knowledge/textbooks/ folder may be empty (before any textbook is
ingested); load_textbook_passages/load_embeddings treat that as "no textbook
passages available" rather than an error, so the feature still answers
app-concept questions in the meantime.
"""

import functools
import json
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

import app_knowledge

logger = logging.getLogger(__name__)

_ANCESTORS = Path(__file__).resolve().parents
_REPO_ROOT_ENV = _ANCESTORS[2] / ".env" if len(_ANCESTORS) > 2 else None
if _REPO_ROOT_ENV and _REPO_ROOT_ENV.exists():
    load_dotenv(dotenv_path=_REPO_ROOT_ENV)
else:
    # No repo-root .env to find (e.g. running inside a container) -- OPENAI_API_KEY
    # is expected to already be in the environment in that case.
    load_dotenv()

CHAT_MODEL = "gpt-5.6-terra"
EMBEDDING_MODEL = "text-embedding-3-small"

TEXTBOOKS_DIR = Path(__file__).resolve().parent / "knowledge" / "textbooks"

MIN_SIMILARITY = 0.25  # below this, a retrieved textbook passage is treated as unrelated
TOP_K = 5
MAX_ATTEMPTS = 2
MAX_HISTORY_TURNS = 6

_MIN_QUOTE_CHARS = 12
_MAX_STATEMENT_CHARS = 500
_STATEMENT_KEYS = {"text", "basis", "source_ids", "quote"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")

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


@dataclass(frozen=True)
class Passage:
    id: str
    source: str  # display name: a book/article's title, or "App Help"
    # A page number for a paginated source (a scanned/PDF book), a section name
    # for one that isn't (a web article -- see scripts/extract_html_article.py),
    # or None for app-concept passages, which have neither.
    page: int | str | None
    text: str

    @property
    def label(self) -> str:
        """How this passage is shown to the model, e.g. "Some Book, page 12", "Some Article, Risk factors", or "App Help"."""
        if self.page is None:
            return self.source
        location = f"page {self.page}" if isinstance(self.page, int) else str(self.page)
        return f"{self.source}, {location}"


@functools.lru_cache(maxsize=1)
def load_textbook_passages() -> dict:
    """{"id": Passage} across every book in knowledge/textbooks/. Empty if none are ingested yet."""
    if not TEXTBOOKS_DIR.exists():
        return {}
    passages: dict = {}
    for json_path in sorted(TEXTBOOKS_DIR.glob("*.json")):
        with open(json_path, encoding="utf-8") as handle:
            raw = json.load(handle)
        title = raw.get("source", {}).get("title") or json_path.stem
        for entry in raw.get("passages", []):
            passage = Passage(id=entry["id"], source=title, page=entry.get("page"), text=entry["text"])
            if passage.id in passages:
                raise ValueError(
                    f"Duplicate passage id {passage.id!r} in {json_path.name} -- passage ids must be "
                    "unique across every book in knowledge/textbooks/."
                )
            passages[passage.id] = passage
    return passages


@functools.lru_cache(maxsize=1)
def load_app_passages() -> dict:
    """{"id": Passage} from app_knowledge.py's generated entries."""
    return {
        entry["id"]: Passage(id=entry["id"], source="App Help", page=None, text=entry["text"])
        for entry in app_knowledge.passages()
    }


@functools.lru_cache(maxsize=1)
def load_embeddings():
    """(ids, vectors) across every book's *_embeddings.npz in knowledge/textbooks/, or ([], empty array)."""
    if not TEXTBOOKS_DIR.exists():
        return [], np.zeros((0, 0), dtype=np.float32)

    all_ids: list = []
    all_vectors: list = []
    for npz_path in sorted(TEXTBOOKS_DIR.glob("*_embeddings.npz")):
        with np.load(npz_path, allow_pickle=False) as data:
            all_ids.extend(data["ids"].tolist())
            all_vectors.append(data["vectors"].astype(np.float32))

    if not all_ids:
        return [], np.zeros((0, 0), dtype=np.float32)

    known_ids = set(load_textbook_passages())
    if set(all_ids) != known_ids:
        raise ValueError(
            "One or more knowledge/textbooks/*_embeddings.npz files are out of sync with their "
            ".json passages -- re-run scripts/embed_textbook.py for the book(s) you edited."
        )
    return all_ids, np.concatenate(all_vectors, axis=0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def embed_query(question: str, client) -> np.ndarray:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[question])
    return _normalize_vector(np.array(response.data[0].embedding, dtype=np.float32))


def retrieve(question: str, client, k: int = TOP_K) -> list:
    """Up to k (Passage, similarity) pairs from the textbook, best first. [] if not yet ingested."""
    ids, vectors = load_embeddings()
    if not ids:
        return []
    passages = load_textbook_passages()
    query_vector = embed_query(question, client)
    scores = vectors @ query_vector
    order = np.argsort(scores)[::-1][:k]
    return [(passages[ids[i]], float(scores[i])) for i in order]


def format_passages_for_prompt(passages: list) -> str:
    return "\n".join(f'[{p.id}] ({p.label}) "{p.text}"' for p in passages)


def _normalize(text: str) -> str:
    """Fold typographic dashes/quotes and whitespace so a quote can be compared by content."""
    translation = {ord(c): "-" for c in "‐‑‒–—−"}
    translation.update({ord("‘"): "'", ord("’"): "'", ord("“"): '"', ord("”"): '"'})
    folded = unicodedata.normalize("NFKC", text).translate(translation)
    return " ".join(folded.split()).casefold()


def _allowed_numbers(screen_data: dict, cited: list) -> set:
    allowed = set()

    def add(token):
        allowed.add(token)
        try:
            value = float(token)
        except ValueError:
            return
        allowed.update({str(round(value)), f"{value:.1f}", f"{value:g}"})

    if screen_data:
        for token in _NUMBER.findall(json.dumps(screen_data)):
            add(token)
    for passage in cited:
        for token in _NUMBER.findall(passage.text):
            add(token)
    return allowed


def _check_statement(where: str, item, screen_data: dict, provided: dict) -> list:
    if not isinstance(item, dict) or set(item) != _STATEMENT_KEYS:
        return [f'{where}: must be an object with exactly the keys "text", "basis", "source_ids", "quote".']

    text, basis, source_ids, quote = item["text"], item["basis"], item["source_ids"], item["quote"]
    if not isinstance(text, str) or not text.strip() or len(text) > _MAX_STATEMENT_CHARS:
        return [f'{where}: "text" must be one non-empty sentence under {_MAX_STATEMENT_CHARS} characters.']
    if not isinstance(source_ids, list) or not all(isinstance(i, str) for i in source_ids):
        return [f'{where}: "source_ids" must be a list of strings.']

    problems = []
    cited = []
    if basis == "passage":
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
    elif basis == "screen":
        if source_ids or quote is not None:
            problems.append(f'{where}: a "screen" statement must have "source_ids": [] and "quote": null.')
    else:
        problems.append(f'{where}: "basis" must be "passage" or "screen".')

    allowed = _allowed_numbers(screen_data, cited)
    stray = sorted({n for n in _NUMBER.findall(text) if n not in allowed})
    if stray:
        problems.append(f"{where}: uses number(s) {stray} that are in neither the screen data nor the cited passage(s).")

    for pattern, reason in _FORBIDDEN:
        if pattern.search(text):
            problems.append(f"{where}: {reason}.")

    return problems


def parse_and_validate(content, screen_data: dict, passages: list):
    """Check a model reply. Returns (data, problems); data is None unless problems is empty."""
    try:
        data = json.loads(content)
    except (TypeError, ValueError):
        return None, ["The reply was not valid JSON."]
    if not isinstance(data, dict) or set(data) != {"refused", "refusal_reason", "answer"}:
        return None, ['The JSON must be an object with exactly the keys "refused", "refusal_reason", "answer".']

    if data["refused"]:
        if data["answer"] != []:
            return None, ['A refused reply must have "answer": [].']
        if not isinstance(data["refusal_reason"], str) or not data["refusal_reason"].strip():
            return None, ['A refused reply must have a non-empty "refusal_reason".']
        return data, []

    items = data["answer"]
    if not isinstance(items, list) or not 1 <= len(items) <= 6:
        return None, ['"answer" must be a list of 1 to 6 items when not refused.']

    provided = {p.id: p for p in passages}
    problems = []
    for index, item in enumerate(items, start=1):
        problems.extend(_check_statement(f"answer[{index}]", item, screen_data, provided))

    return (None, problems) if problems else (data, [])


def render_answer(data: dict, passages: list) -> dict:
    """The shape the web app receives: plain text (or a refusal) plus cited sources.

    Sources are grouped by book/article (or "App Help"), each followed by every
    page or section cited from it: "Some Book: page 5, page 12" or "Some
    Article: Risk factors, Symptoms" rather than one entry per passage, so
    citing several passages from the same source doesn't repeat its title.
    """
    if data["refused"]:
        return {"refused": True, "answer": None, "refusalReason": data["refusal_reason"], "sources": []}

    by_id = {p.id: p for p in passages}
    cited = [by_id[i] for item in data["answer"] for i in item["source_ids"] if i in by_id]

    pages_by_source: dict = {}
    for passage in cited:
        pages = pages_by_source.setdefault(passage.source, set())
        if passage.page is not None:
            pages.add(passage.page)

    def sort_key(page):
        # Numeric pages sort numerically before any section-name strings, which
        # sort alphabetically -- only matters if one source somehow mixed both.
        return (0, page) if isinstance(page, int) else (1, str(page))

    def format_page(page):
        return f"page {page}" if isinstance(page, int) else str(page)

    sources = []
    for source in sorted(pages_by_source):
        pages = sorted(pages_by_source[source], key=sort_key)
        if pages:
            sources.append(f"{source}: " + ", ".join(format_page(p) for p in pages))
        else:
            sources.append(source)

    text = " ".join(item["text"].strip() for item in data["answer"])
    return {"refused": False, "answer": text, "refusalReason": None, "sources": sources}


def _refusal(reason: str) -> dict:
    return {"refused": True, "refusal_reason": reason, "answer": []}


def _format_history(history) -> str:
    if not history:
        return ""
    trimmed = history[-MAX_HISTORY_TURNS:]
    lines = [f'{"You" if turn["role"] == "user" else "Assistant"}: {turn["text"]}' for turn in trimmed]
    return (
        "Recent conversation so far (context only -- do not cite this as a source):\n"
        + "\n".join(lines)
        + "\n\n"
    )


SYSTEM_PROMPT = """You are a question-answering assistant embedded in a skin-lesion
screening app (a hackathon prototype, not a diagnostic medical device). People
ask you general melanoma questions from inside the app, sometimes while looking
at their own analysis results.

STRICT RULES -- follow every one of these, no exceptions:

1. You may answer ONLY from the REFERENCE PASSAGES and, when given, the
   CURRENT SCREEN DATA in the user message. Never use outside medical
   knowledge, even if you are confident it is correct.

2. If the passages and screen data do not support an answer, refuse. Do not
   guess, hedge with outside knowledge, or answer a related-but-different
   question instead.

3. NEVER say or imply "you have [condition]", "this is/isn't cancer", or any
   diagnostic conclusion about the person's own spot, even indirectly.

4. Tone is calm and matter-of-fact. No alarming language ("dangerous",
   "urgent", "emergency"). No false reassurance ("don't worry", "probably
   fine") either.

5. If a question is about the person's own health situation beyond what a
   cited passage or their screen data supports, say so and suggest a
   dermatologist or healthcare provider rather than guessing.
"""

GROUNDING_PROMPT = """GROUNDING RULES -- these are added to the STRICT RULES above and never relax
any of them.

G1. Every statement must have "basis" set to either "passage" or "screen".

G2. A "passage" statement must list the ids it relies on in "source_ids", and
    copy into "quote" one exact, contiguous excerpt (at least a full clause)
    from one of those passages, word for word. Never cite an id that was not
    provided.

G3. A "screen" statement must have "source_ids": [] and "quote": null, and
    may only state values that appear in CURRENT SCREEN DATA -- never
    interpret them beyond what a cited passage also supports.

G4. Only write a number if it appears in a passage you cite or in CURRENT
    SCREEN DATA.

G5. If nothing supports an answer, return "refused": true with a short,
    plain "refusal_reason" and an empty "answer" list. A refusal is always
    acceptable; an unsupported statement never is.

OUTPUT FORMAT -- return only a JSON object, with no markdown and no other keys:

{
  "refused": false,
  "refusal_reason": null,
  "answer": [
    {"text": "one plain sentence", "basis": "passage", "source_ids": ["<id>"], "quote": "<exact excerpt>"}
  ]
}

"answer" holds 1 to 6 items when "refused" is false, and must be [] when
"refused" is true.
"""

USER_PROMPT_TEMPLATE = """REFERENCE PASSAGES (the only outside information you may use), each shown as
[id] (location) "text":

{passages}

{screen_block}Question: {question}

Answer following your system instructions and the grounding rules exactly.
"""

# Internal page identifiers (ChatPageContext.Set's first argument, one per page that
# wires it up -- see Pages/*.razor) mapped to a human-readable name, since a raw slug
# like "findCare" reads oddly in an answer. A page not in this map (or page_context
# omitted entirely) just isn't named -- the chat still works from its own data alone.
_PAGE_DISPLAY_NAMES = {
    "home": "the Home dashboard",
    "spots": "the Spots list",
    "spot": "a tracked spot's detail page",
    "results": "a saved check's results page",
    "history": "the check History page",
    "profile": "the Profile / risk settings page",
    "findCare": "the Find Care page",
}


def answer_question(question: str, page_context: dict = None, history: list = None, client=None) -> dict:
    """Return a grounded answer or a refusal for `question`.

    Model failures (a missing key, a network error) propagate to the caller. A
    reply that merely fails validation does not: it is retried once with the
    problems listed, then replaced by a deterministic refusal -- unlike
    knowledge.py's ABCDE-flag-constrained fallback, an open-ended question has
    no fixed shape a fallback could safely synthesize an answer from.

    `client` is injectable so tests can supply a fake OpenAI client.
    """
    client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    page_name = (page_context or {}).get("page")
    screen_data = (page_context or {}).get("data") or {}

    retrieved = retrieve(question, client)
    textbook_passages = [p for p, score in retrieved if score >= MIN_SIMILARITY]
    app_passages = list(load_app_passages().values())
    passages = textbook_passages + app_passages

    if not passages:
        return render_answer(_refusal("The textbook hasn't been loaded yet, so I can't answer that."), passages)

    # The page name is included even when there's no data on it yet (e.g. an
    # empty list) -- otherwise "what screen am I on" has nothing to answer from,
    # even though the caller did say which page this is.
    screen_parts = []
    if page_name:
        screen_parts.append(f"You are looking at {_PAGE_DISPLAY_NAMES.get(page_name, page_name)}.")
    if screen_data:
        screen_parts.append(f"Its current data: {json.dumps(screen_data)}")
    screen_block = (
        "CURRENT SCREEN DATA (the app's own state; safe to state as-is): " + " ".join(screen_parts) + "\n\n"
        if screen_parts
        else ""
    )

    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "system", "content": GROUNDING_PROMPT},
        {
            "role": "user",
            "content": _format_history(history)
            + USER_PROMPT_TEMPLATE.format(
                passages=format_passages_for_prompt(passages),
                screen_block=screen_block,
                question=question,
            ),
        },
    ]
    messages = list(base_messages)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            max_completion_tokens=700,
            reasoning_effort="none",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages,
        )
        content = response.choices[0].message.content
        data, problems = parse_and_validate(content, screen_data, passages)
        if data is not None:
            return render_answer(data, passages)

        logger.warning("Chat answer attempt %d rejected: %s", attempt, "; ".join(problems))
        messages = base_messages + [
            {"role": "assistant", "content": content or ""},
            {
                "role": "user",
                "content": "Your reply was rejected for these reasons:\n- "
                + "\n- ".join(problems)
                + "\nReturn a corrected JSON object that fixes every one of them.",
            },
        ]

    logger.warning("Falling back to a refusal after %d rejected attempts.", MAX_ATTEMPTS)
    return render_answer(
        _refusal("I couldn't put together a reliably grounded answer to that -- try rephrasing, or ask something else."),
        passages,
    )
