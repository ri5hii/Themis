"""Semantic query over the built embedding index.

Searches data/indexes/embeddings (built by build_embeddings.py) for sections
semantically similar to the query, returning top-k hits with scores/sources.

Usage:
    python scripts/query_documents.py "subletting clause"
        [--embeddings data/indexes/embeddings] [--model nlpaueb/legal-bert-base-uncased]
        [--k 5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.retrieve import queryEmbeddings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("query", help="natural-language query text")
    ap.add_argument("--embeddings", default="data/indexes/embeddings")
    ap.add_argument("--model", default="nlpaueb/legal-bert-base-uncased")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    emb_dir = Path(args.embeddings)
    if not (emb_dir / "sections.faiss").is_file():
        print(f"[query] missing embedding index in {emb_dir}; run build_embeddings.py first", file=sys.stderr)
        return 1

    hits = queryEmbeddings(args.query, emb_dir, args.model, args.k)
    if not hits:
        print("[query] no hits", file=sys.stderr)
        return 1

    for h in hits:
        print(f"#{h['rank']} score={h['score']:.3f} sources={','.join(h['sources']) or '-'}")
        print(f"   {h['text'][:180]}{'...' if len(h['text']) > 180 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())