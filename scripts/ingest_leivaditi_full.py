"""Ingest the full Leivaditi benchmark CSVs into canonical cleaned JSONL.

Usage:
    python scripts/ingest_leivaditi_full.py [--dir DIR]

Reads the consolidated CSVs from data/annotated/leivaditi_full and writes
canonical per-corpus JSONL into data/cleaned.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.dataset import ingest_leivaditi as ing

CORPORA = {
    "redflags": ("redflags.csv", ing.ingestRedflags),
    "easy_redflags": ("easy_redflag.csv", ing.ingestEasyRedflags),
    "docs": ("docclass.csv", ing.ingestDocs),
    "entities": ("entities.csv", ing.ingestEntities),
    "clauses": ("clauses.csv", ing.ingestClauses),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/annotated/leivaditi_full", help="benchmark CSV dir")
    parser.add_argument("--out", default="data/cleaned", help="output dir for JSONL")
    args = parser.parse_args()

    src = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, (csv_file, ingest_fn) in CORPORA.items():
        path = src / csv_file
        if not path.exists():
            print(f"[ingest] skip {name}: missing {path}", file=sys.stderr)
            continue
        rows = ingest_fn(ing.parseCsv(str(path)))
        dst = out / f"leivaditi_full_{name}.jsonl"
        with open(dst, "w") as f:
            f.writelines(json.dumps(row) + "\n" for row in rows)
        print(f"[ingest] {name}: {len(rows)} rows -> {dst}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())