#!/usr/bin/env python
"""One-time, offline OCR pass over textbook page scans.

Run manually, never at runtime and never inside the Docker image (see
requirements.txt for why pytesseract is listed as an offline-script-only
dependency, and note the production Dockerfile deliberately does not install
the system tesseract binary). Requires that binary installed locally --
`apt install tesseract-ocr`, `brew install tesseract`, or the Windows
installer at https://github.com/UB-Mannheim/tesseract/wiki -- plus
`pip install pytesseract` from requirements.txt.

This only produces raw, uncorrected text. OCR on a scanned textbook page
(columns, footnotes, italics, medical terms) is noisy, so a person must read
and correct each page's output before any of it becomes a passage in a new
knowledge/textbooks/{slug}.json (see any existing file there for the shape).
Adding a second, third, etc. book is the same process again with a new slug
and output folder -- there is no limit of one.

Usage:
    python scripts/ocr_extract_textbook.py scripts/textbook_pages/ --out scripts/textbook_ocr_raw/
"""

import argparse
import sys
from pathlib import Path

import cv2

try:
    import pytesseract
except ImportError:
    print("pytesseract is not installed. Run: pip install pytesseract", file=sys.stderr)
    print("You also need the system tesseract binary -- see this file's module docstring.", file=sys.stderr)
    raise


def extract_page(image_path: Path) -> str:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return pytesseract.image_to_string(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pages_dir", type=Path, help="Folder of page scan images (page_001.png, page_002.png, ...)")
    parser.add_argument("--out", type=Path, required=True, help="Folder to write raw per-page .txt files to")
    args = parser.parse_args()

    pages = sorted(args.pages_dir.glob("*.png")) + sorted(args.pages_dir.glob("*.jpg"))
    if not pages:
        print(f"No .png/.jpg files found in {args.pages_dir}", file=sys.stderr)
        sys.exit(1)

    args.out.mkdir(parents=True, exist_ok=True)
    for page_path in pages:
        text = extract_page(page_path)
        out_path = args.out / f"{page_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8")
        print(f"{page_path.name} -> {out_path} ({len(text)} chars)")

    print(
        f"\nWrote {len(pages)} raw text file(s) to {args.out}.\n"
        "Next: read each one, correct OCR errors, and hand-curate the corrected excerpts into\n"
        "a new knowledge/textbooks/{slug}.json (same shape as an existing file there).\n"
        "Then run scripts/embed_textbook.py."
    )


if __name__ == "__main__":
    main()
