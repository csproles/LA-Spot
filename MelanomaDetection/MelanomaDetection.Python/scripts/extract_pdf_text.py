#!/usr/bin/env python
"""One-time, offline extraction of text from a textbook PDF that already has a
text layer -- an alternative to ocr_extract_textbook.py for a born-digital or
previously-digitized PDF (rather than raw page-scan images).

Some digitized PDFs (this repo's textbook among them) have a text layer where
the words are correct but the *spacing* is lost -- e.g. "Thisbookletisabout
melanoma" -- an artifact of how the original scan positioned characters,
rather than a word-boundary problem OCR would introduce. Because word order
and spelling are already right, an LLM reliably reinserts the missing spaces
without guessing at content, which is a materially safer operation than
running image OCR (which can misread individual characters).

Not every PDF has this problem -- a born-digital PDF (e.g. one produced
straight from a word processor or publishing tool, rather than scanned) is
often already properly spaced. Pass --no-respace to skip the OpenAI call
entirely for one of those; it costs nothing and there's nothing for the LLM
to fix. A PDF with permissions-only encryption (no real password, just a
restriction flag) is decrypted with an empty password automatically; a PDF
that needs an actual password isn't handled here.

Run manually, never at runtime and never inside the Docker image, same as
ocr_extract_textbook.py. Requires OPENAI_API_KEY (loaded the same way
llm_explainer.py does, from the repo-root .env) unless --no-respace is given.

Usage:
    python scripts/extract_pdf_text.py "scripts/textbook_pages/My Book.pdf" \\
        --out scripts/textbook_ocr_raw/ --start 6 --end 39
    python scripts/extract_pdf_text.py "scripts/textbook_pages/Clean Book.pdf" \\
        --out scripts/textbook_ocr_raw/ --no-respace
"""

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader

_ANCESTORS = Path(__file__).resolve().parents
_REPO_ROOT_ENV = _ANCESTORS[2] / ".env" if len(_ANCESTORS) > 2 else None


def _load_env() -> None:
    if _REPO_ROOT_ENV and _REPO_ROOT_ENV.exists():
        load_dotenv(dotenv_path=_REPO_ROOT_ENV)
    else:
        load_dotenv()


MODEL = "gpt-5.6-terra"

RESPACE_PROMPT = """The following text was extracted from a digitized PDF and lost its word
spacing in the process (letters and punctuation are correct; spaces between
words were dropped). Reinsert the missing spaces so it reads as normal English.

Rules:
- Do not add, remove, reorder, or reword anything -- only insert spaces (and
  restore obvious paragraph breaks).
- Keep every word, number, and punctuation mark exactly as given.
- Return only the corrected text, nothing else -- no commentary, no markdown.

TEXT:
{text}
"""


def respace(client, text: str) -> str:
    if not text.strip():
        return text
    response = client.chat.completions.create(
        model=MODEL,
        max_completion_tokens=2000,
        reasoning_effort="none",
        temperature=0,
        messages=[{"role": "user", "content": RESPACE_PROMPT.format(text=text)}],
    )
    return response.choices[0].message.content or text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pdf_path", type=Path, help="The source PDF")
    parser.add_argument("--out", type=Path, required=True, help="Folder to write cleaned per-page .txt files to")
    parser.add_argument("--start", type=int, default=1, help="First PDF page (1-based) to extract")
    parser.add_argument("--end", type=int, default=None, help="Last PDF page (1-based, inclusive); default: last page")
    parser.add_argument(
        "--no-respace",
        action="store_true",
        help="Skip the OpenAI respacing pass -- use for a PDF whose text layer is already properly spaced",
    )
    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"{args.pdf_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    reader = PdfReader(str(args.pdf_path))
    if reader.is_encrypted:
        # Permissions-only encryption (no real password) is the common case for a
        # PDF that restricts printing/copying in a viewer but isn't actually locked;
        # decrypt("") succeeds for that and fails (returns 0) for a real password.
        if not reader.decrypt(""):
            print(f"{args.pdf_path} needs a real password -- this script doesn't handle that.", file=sys.stderr)
            sys.exit(1)

    end = args.end or len(reader.pages)
    args.out.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.no_respace:
        _load_env()
        from openai import OpenAI

        client = OpenAI()

    for i in range(args.start, end + 1):
        raw = reader.pages[i - 1].extract_text() or ""
        cleaned = respace(client, raw) if client else raw
        out_path = args.out / f"pdf_page_{i:03d}.txt"
        out_path.write_text(cleaned, encoding="utf-8")
        print(f"PDF page {i} -> {out_path} ({len(cleaned)} chars)")

    print(
        f"\nWrote {end - args.start + 1} cleaned text file(s) to {args.out}.\n"
        "Next: read each one, verify against the original PDF page, and hand-curate the\n"
        "excerpts into a new knowledge/textbooks/{slug}.json (same shape as an existing\n"
        "file there). Then run scripts/embed_textbook.py."
    )


if __name__ == "__main__":
    main()
