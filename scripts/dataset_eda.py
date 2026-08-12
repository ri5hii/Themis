"""EDA over the Leivaditi lease + redflag corpora. Prints a summary and writes
an optional JSON report. Run from the repo root:

    PYTHONUNBUFFERED=1 python scripts/dataset_eda.py [--json eval/eda_report.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.dataset import eda

DEFAULTS = {
    "leases": "leivaditi_leases.jsonl",
    "redflags": "leivaditi_redflags.jsonl",
}


def _printDist(title: str, dist: dict) -> None:
    print(f"  {title}")
    for key, count in dist.items():
        print(f"    {key}: {count}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="EDA over Leivaditi datasets")
    parser.add_argument("--in-dir", default=str(Path("data/annotated")))
    parser.add_argument("--json", default="", help="optional path for JSON report")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    corpora: dict[str, list[dict]] = {}
    for name, fname in DEFAULTS.items():
        corpora[name] = eda.loadJsonl(in_dir / fname)

    for name, rows in corpora.items():
        s = eda.summarizeRows(rows)
        print(f"== {name} ==")
        print(f"  rows={s['rows']} sources={s['unique_sources']}")
        print(f"  rows_per_source: {s['rows_per_source']}")
        print(f"  text_chars: {s['text']}")
        print(f"  truncated_at_{eda.TRUNCATION_LIMIT}: {s['truncated_at_limit']} ({s['truncated_pct']}%)")
        print(f"  empty_text: {s['empty_text']}")

    report = eda.buildReport(corpora["leases"], corpora["redflags"])
    _printDist("type_fast_lane", report["leases"]["type_fast_lane"])
    _printDist("redflags.type", report["redflags"]["type"])
    _printDist("redflags.redflag_type", report["redflags"]["redflag_type"])

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print(f"[eda] report -> {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())