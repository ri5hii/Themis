"""Clean the Leivaditi lease + redflag corpora and write cleaned jsonl + a
cleaning report. Run from the repo root:

    PYTHONUNBUFFERED=1 python scripts/clean_dataset.py

Usage:
    python scripts/clean_dataset.py [--in-dir data/annotated] [--out-dir data/cleaned]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalrag.dataset import clean, eda

DEFAULTS = {
    "leases": "leivaditi_leases.jsonl",
    "redflags": "leivaditi_redflags.jsonl",
}


def _writeJsonl(rows: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean Leivaditi datasets")
    parser.add_argument("--in-dir", default=str(Path("data/annotated")))
    parser.add_argument("--out-dir", default=str(Path("data/cleaned")))
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"inputs": {}, "drops": {}, "fixed_labels": 0}

    for name, fname in DEFAULTS.items():
        path = in_dir / fname
        rows = eda.loadJsonl(path)
        validate = clean.validateLease if name == "leases" else clean.validateRedflag
        clean_row = clean.cleanLease if name == "leases" else clean.cleanRedflag

        invalid: list[str] = []
        cleaned: list[dict] = []
        for row in rows:
            errs = validate(row)
            if errs:
                invalid.append("; ".join(errs))
                continue
            cleaned.append(clean_row(row))

        before = len(cleaned)
        unique_keys = ("source", "section_idx") if name == "leases" else ("source", "text")
        cleaned = clean.dedupe(cleaned, unique_keys)
        drops = before - len(cleaned)

        fixed = 0
        if name == "redflags":
            fixed = sum(1 for r in rows if r.get("redflag_type") in clean.REDFLAG_TYPE_FIXES)

        out_path = out_dir / f"cleaned_{fname}"
        _writeJsonl(cleaned, out_path)
        report["inputs"][name] = {"path": str(path), "rows": len(rows)}
        report["drops"][name] = {"schema_invalid": len(invalid), "duplicates": drops, "kept": len(cleaned)}
        if fixed:
            report["fixed_labels"] += fixed

        print(f"[clean] {name}: {len(rows)} -> {len(cleaned)} "
              f"(schema_invalid={len(invalid)}, dupes={drops}, label_fixes={fixed})")
        print(f"[clean] wrote {out_path}")

    report_path = out_dir / "cleaning_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"[clean] report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
