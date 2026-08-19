"""Build the document index from ingested raw outputs.

Scans data/raw/ for ingest output dirs (manifest.json + sections.jsonl +
sentences.jsonl), dedups section/sentence units by content hash, and writes
hash-keyed indexes into data/indexes/ ready for retrieval.

Usage:
    python scripts/build_index.py [--raw data/raw] [--out data/indexes]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.ingest.index import buildIndex, discoverOutputs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", default="data/raw", help="ingest output root (default data/raw)")
    ap.add_argument("--out", default="data/indexes", help="index output dir (default data/indexes)")
    args = ap.parse_args()

    raw_root = Path(args.raw)
    if not raw_root.is_dir():
        print(f"[index] missing raw root: {raw_root}", file=sys.stderr)
        return 1

    found = discoverOutputs(raw_root)
    if not found:
        print(f"[index] no ingest outputs found under {raw_root}", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    stats = buildIndex(raw_root, out_dir)

    print(f"[index] indexed {stats['n_docs']} docs "
          f"({stats['n_sections_unique']}/{stats['n_sections_total']} unique sections, "
          f"{stats['n_sentences_unique']}/{stats['n_sentences_total']} unique sentences) "
          f"-> {out_dir}")
    print(f"[index] summary: {json.dumps(stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())