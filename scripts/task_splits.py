"""Build deterministic train/val/test splits for the supervised tasks.

Usage:
    python scripts/task_splits.py [--seed 42] [--out data/splits]

Reads cleaned corpora and writes per-task split JSONL into data/splits.

redflag_paragraph uses the benchmark's official split (rf_train.csv /
rf_val.csv) so every class has training support; the official validation
set is split deterministically into our val/test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag import tasks
from legalrag.dataset import ingest_leivaditi as ing

CLEAN = Path("data/cleaned")
ANNOTATED = Path("data/annotated/leivaditi_full")
OFFICIAL_REDFLAG = ("rf_train.csv", "rf_val.csv")

TASKS = {
    "redflag_paragraph": ("leivaditi_full_redflags.jsonl",),
    "deontic_span": ("leivaditi_full_easy_redflags.jsonl",),
    "deontic_multilabel": ("lexdemod_annotated.jsonl",),
}


def _official_redflag_split(out: Path, seed: int) -> None:
    import random

    train_csv, val_csv = (ANNOTATED / name for name in OFFICIAL_REDFLAG)
    if not (train_csv.exists() and val_csv.exists()):
        print("[splits] skip redflag_paragraph: missing official CSVs", file=sys.stderr)
        return
    train = ing.ingestRedflags(ing.parseCsv(str(train_csv)))
    val_rows = ing.ingestRedflags(ing.parseCsv(str(val_csv)))
    rng = random.Random(seed)
    rng.shuffle(val_rows)
    mid = len(val_rows) // 2
    split = {"train": train, "val": val_rows[:mid], "test": val_rows[mid:]}
    for s, items in split.items():
        dst = out / f"redflag_paragraph.{s}.jsonl"
        with open(dst, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in items)
    sizes = {s: len(items) for s, items in split.items()}
    print(f"[splits] redflag_paragraph (official): {sizes}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=tasks.DEFAULT_SEED)
    parser.add_argument("--out", default=str(CLEAN.parent / "splits"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name, sources in TASKS.items():
        if name == "redflag_paragraph":
            _official_redflag_split(out, args.seed)
            continue
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