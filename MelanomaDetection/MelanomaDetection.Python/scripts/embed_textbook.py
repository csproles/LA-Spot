#!/usr/bin/env python
"""One-time (re-)embedding of a curated textbook's passages.

Run after a book's knowledge/textbooks/{slug}.json is created or edited (each
book is its own file -- there is no limit of one; drop in as many as needed).
Calls the OpenAI embeddings API once per passage and writes a matching
knowledge/textbooks/{slug}_embeddings.npz, which textbook_chat.py loads at
runtime instead of re-embedding anything on every request. Requires
OPENAI_API_KEY (the repo-root .env is picked up automatically, same as
llm_explainer.py).

Usage:
    python scripts/embed_textbook.py                                  # (re-)embed every book
    python scripts/embed_textbook.py knowledge/textbooks/some-book.json  # just one book
"""

import json
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

_ANCESTORS = Path(__file__).resolve().parents
_REPO_ROOT_ENV = _ANCESTORS[2] / ".env" if len(_ANCESTORS) > 2 else None
if _REPO_ROOT_ENV and _REPO_ROOT_ENV.exists():
    load_dotenv(dotenv_path=_REPO_ROOT_ENV)
else:
    load_dotenv()

TEXTBOOKS_DIR = Path(__file__).resolve().parent.parent / "knowledge" / "textbooks"
EMBEDDING_MODEL = "text-embedding-3-small"


def embed_one(client, json_path: Path) -> None:
    with open(json_path, encoding="utf-8") as handle:
        raw = json.load(handle)
    passages = raw.get("passages", [])
    if not passages:
        print(f"{json_path} has no passages yet -- skipping.", file=sys.stderr)
        return

    ids = [p["id"] for p in passages]
    texts = [p["text"] for p in passages]

    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    vectors = np.array([item.embedding for item in response.data], dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    out_path = json_path.with_name(json_path.stem + "_embeddings.npz")
    np.savez(out_path, ids=np.array(ids), vectors=vectors, model=np.array([EMBEDDING_MODEL]))
    print(f"Wrote {len(ids)} embedding(s) to {out_path}")


def main() -> None:
    if len(sys.argv) > 1:
        targets = [Path(sys.argv[1])]
    else:
        if not TEXTBOOKS_DIR.exists():
            print(f"{TEXTBOOKS_DIR} does not exist -- curate a book there first.", file=sys.stderr)
            sys.exit(1)
        targets = sorted(p for p in TEXTBOOKS_DIR.glob("*.json"))

    if not targets:
        print(f"No book .json files found in {TEXTBOOKS_DIR}.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI()
    for json_path in targets:
        if not json_path.exists():
            print(f"{json_path} does not exist.", file=sys.stderr)
            sys.exit(1)
        embed_one(client, json_path)


if __name__ == "__main__":
    main()
