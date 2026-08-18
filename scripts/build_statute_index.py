"""Build the statute grounding index from data/statutes/*.md sources.

Chunks each act by section heading with unique ids (mta_2021#s.11,
delhi_rent_control_1958#s.14), drops title/TOC/background-note noise and
mojibake pages, then rebuilds the FAISS index + id map consumed by grounding.

Usage:
    python scripts/build_statute_index.py [--src data/statutes] [--out data/indexes/statutes]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.retrieve import buildEmbeddings
from legalrag.retrieve.statutes import chunkAct, cleanChunks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="data/statutes", help="statute md sources (default data/statutes)")
    ap.add_argument("--out", default="data/indexes/statutes", help="index output dir (default data/indexes/statutes)")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    if not src.is_dir():
        print(f"[statute-index] missing source dir: {src}", file=sys.stderr)
        return 1

    chunks: list[dict] = []
    for md in sorted(src.glob("*.md")):
        act = md.stem
        act_chunks = cleanChunks(chunkAct(md, act))
        chunks.extend(act_chunks)
        print(f"[statute-index] {act}: {len(act_chunks)} chunks")

    if not chunks:
        print("[statute-index] no chunks produced", file=sys.stderr)
        return 1

    out.mkdir(parents=True, exist_ok=True)
    sections_path = out / "sections.jsonl"
    with sections_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(
                json.dumps(
                    {
                        "id": chunk["id"],
                        "text": chunk["text"],
                        "n_occurrences": 1,
                        "sources": [chunk["id"].split("#")[0]],
                    }
                )
                + "\n"
            )

    meta = buildEmbeddings(sections_path, out)
    print(f"[statute-index] {meta['n_sections']} chunks embedded ({meta['model']}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())