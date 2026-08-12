"""Ingest the LEXDEMOD annotated CSVs into canonical cleaned JSONL.

Usage:
    python scripts/ingest_lexdemod.py [--dir DIR]

Reads train_eval_annotated_data.csv + test_annotated_data.csv from
data/annotated/lexdemod and writes data/cleaned/lexdemod_annotated.jsonl.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.dataset import ingest_lexdemod as ing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="data/annotated/lexdemod", help="LEXDEMOD CSV dir")
    parser.add_argument("--out", default="data/cleaned", help="output dir")
    args = parser.parse_args()

    src = Path(args.dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = ["train_eval_annotated_data.csv", "test_annotated_data.csv"]
    rows: list[dict] = []
    for fname in files:
        path = src / fname
        if not path.exists():
            print(f"[ingest] skip {fname}: missing {path}", file=sys.stderr)
            continue
        rows.extend(ing.parseCsv(str(path)))
    if not rows:
        print("[ingest] no input CSVs found", file=sys.stderr)
        return 1

    cleaned = ing.ingestAnnotated(rows)
    dst = out / "lexdemod_annotated.jsonl"
    with open(dst, "w") as f:
        f.writelines(json.dumps(row) + "\n" for row in cleaned)
    print(f"[ingest] lexdemod: {len(cleaned)} rows -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())