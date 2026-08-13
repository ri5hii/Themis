"""Build deterministic train/val/test splits for the supervised tasks.

Usage:
    python scripts/task_splits.py [--seed 42] [--out data/splits]

Reads cleaned corpora and writes per-task split JSONL into data/splits.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks

CLEAN = Path("data/cleaned")

TASKS = {
    "redflag_paragraph": ("leivaditi_full_redflags.jsonl",),
    "deontic_span": ("leivaditi_full_easy_redflags.jsonl",),
    "deontic_multilabel": ("lexdemod_annotated.jsonl",),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=tasks.DEFAULT_SEED)
    parser.add_argument("--out", default=str(CLEAN.parent / "splits"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, sources in TASKS.items():
        rows: list[dict] = []
        for fname in sources:
            path = CLEAN / fname
            if not path.exists():
                print(f"[splits] skip {name}: missing {path}", file=sys.stderr)
                continue
            rows.extend(tasks.loadJsonl(str(path)))
        if not rows:
            print(f"[splits] skip {name}: no rows", file=sys.stderr)
            continue
        split = tasks.splitRows(rows, seed=args.seed)
        for s, items in split.items():
            dst = out / f"{name}.{s}.jsonl"
            with open(dst, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in items)
        sizes = {s: len(items) for s, items in split.items()}
        print(f"[splits] {name}: {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())