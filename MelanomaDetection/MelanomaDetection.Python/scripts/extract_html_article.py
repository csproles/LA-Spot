#!/usr/bin/env python
"""One-time, offline extraction of clean text from an HTML article (e.g. a
reference-database article saved as .html) -- an alternative to
ocr_extract_textbook.py / extract_pdf_text.py for a source that's already
well-formed markup rather than a scanned page image or a PDF's text layer.

Strips tags and script/style/figure/footer content with the standard
library's html.parser -- no OCR, no LLM cleanup, since well-formed HTML has
no spacing or character-recognition problems to fix. Paragraph and list-item
breaks are kept so the output reads the same way the article does. There is
no page pagination, so unlike the PDF/OCR scripts this writes one text file
for the whole article, not one per page -- when curating, use a section name
as each passage's "page" value instead of a page number (see Passage in
textbook_chat.py, which accepts either).

Run manually, never at runtime.

Usage:
    python scripts/extract_html_article.py "scripts/textbook_pages/Some Article.html" \\
        --out scripts/textbook_ocr_raw/
"""

import argparse
import sys
from html.parser import HTMLParser
from pathlib import Path

_SKIP_TAGS = {"script", "style", "figure", "footer", "head"}
_BREAK_TAGS = {"p", "li", "h1", "h2", "h3", "h4"}


class ArticleTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self._skip_depth = 0
        self._paragraphs: list[str] = []
        self._current: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BREAK_TAGS and self._skip_depth == 0:
            self._flush()

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BREAK_TAGS and self._skip_depth == 0:
            self._flush()

    def handle_data(self, data):
        if self._skip_depth == 0:
            self._current.append(data)

    def _flush(self) -> None:
        text = " ".join("".join(self._current).split())
        if text:
            self._paragraphs.append(text)
        self._current = []

    def get_text(self) -> str:
        self._flush()
        return "\n\n".join(self._paragraphs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("html_path", type=Path, help="The source .html file")
    parser.add_argument("--out", type=Path, required=True, help="Folder to write the cleaned text file to")
    args = parser.parse_args()

    if not args.html_path.exists():
        print(f"{args.html_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    html = args.html_path.read_text(encoding="utf-8", errors="replace")
    extractor = ArticleTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{args.html_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")

    print(f"{args.html_path.name} -> {out_path} ({len(text)} chars)")
    print(
        "\nNext: read it, hand-curate excerpts into a new knowledge/textbooks/{slug}.json\n"
        '(same shape as an existing file there) -- this source has no page numbers, so use\n'
        'a section name as each passage\'s "page" value instead (e.g. "Risk factors").\n'
        "Then run scripts/embed_textbook.py."
    )


if __name__ == "__main__":
    main()
