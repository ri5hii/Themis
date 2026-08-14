"""Build dense embeddings over unique index sections -> FAISS.

Embeds data/indexes/sections.jsonl (the deduped sections from the index step)
with a transformer and writes a FAISS IndexFlatIP plus an id/text/sources map
into the embedding dir, ready for semantic queries.

Usage:
    python scripts/build_embeddings.py [--sections data/indexes/sections.jsonl] \
        [--out data/indexes/embeddings] [--model nlpaueb/legal-bert-base-uncased]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.retrieve import buildEmbeddings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", default="data/indexes/sections.jsonl")
    ap.add_argument("--out", default="data/indexes/embeddings")
    ap.add_argument("--model", default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    sections_path = Path(args.sections)
    if not sections_path.is_file():
        print(f"[embed] missing sections index: {sections_path}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    meta = buildEmbeddings(sections_path, out_dir, args.model, args.batch_size)
    print(f"[embed] {meta['n_sections']} sections, dim {meta['dim']} -> {out_dir}")
    print(f"[embed] summary: {json.dumps(meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())