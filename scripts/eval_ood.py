#!/usr/bin/env python3
"""Out-of-distribution section-classification eval for the Extract stage.

Scores the fast-lane and the hybrid engine (fast-lane + trained classifier
fallback) against hand-labeled gold for the real scanned/text corpus in
data/raw/external/ (docs §6.9). Gold labels live in
data/annotated/ood_sections_gold.jsonl (one row per section: source,
section_idx, type).

Metrics (docs M4/M5):
  - overall section classification accuracy (all sections, incl. UNKNOWN gold)
  - clause-only accuracy (gold != UNKNOWN)
  - per-class and macro precision/recall/F1 via legalrag.eval.metrics

Usage:
    python scripts/eval_ood.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from legalrag.eval.metrics import multiclassStats
from legalrag.extract.analyze import analyzeSections

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data" / "annotated" / "ood_sections_gold.jsonl"
CLASSIFIER = ROOT / "models" / "classifier.npz"

SECTIONS = {
    "yates_scan": ROOT / "data" / "raw" / "external" / "scanned-01-yates" / "yates_scan" / "sections.jsonl",
    "cia_lease_scan": ROOT / "data" / "raw" / "external" / "scanned-02-cia" / "cia_lease_scan" / "sections.jsonl",
    "commercial_lease_template": ROOT / "data" / "raw" / "external" / "text-01-opendocs" / "commercial_lease_template" / "sections.jsonl",
    "everest_36mo_lease": ROOT / "data" / "raw" / "external" / "text-02-archive" / "everest_36mo_lease" / "sections.jsonl",
}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _rows_for_source(gold: list[dict], source: str) -> list[dict]:
    return [r for r in gold if r["source"] == source]


def _report(gold: list[dict], preds: list[str], tag: str) -> dict:
    y_true = [r["type"] for r in gold]
    acc = sum(1 for t, p in zip(y_true, preds) if t == p) / len(gold)
    stats = multiclassStats(y_true, preds)
    clause = [(t, p) for t, p in zip(y_true, preds) if t != "unknown"]
    clause_acc = sum(1 for t, p in clause if t == p) / len(clause) if clause else None
    report = {
        "tag": tag,
        "rows": len(gold),
        "accuracy": round(acc, 4),
        "clause_rows": len(clause),
        "clause_accuracy": round(clause_acc, 4) if clause_acc is not None else None,
        "macro": stats["macro"],
        "unknown_gold_unknown_pred": sum(
            1 for t, p in zip(y_true, preds) if t == "unknown" and p == "unknown"
        ),
        "unknown_gold_other_pred": sum(
            1 for t, p in zip(y_true, preds) if t == "unknown" and p != "unknown"
        ),
    }
    print(f"[eval_ood] {tag}: {json.dumps(report)}")
    return report


def main() -> int:
    if not GOLD.is_file():
        print(f"[eval_ood] missing gold labels: {GOLD}", file=sys.stderr)
        return 1

    gold = _load(GOLD)
    fast_report = _report(gold, _predict(gold, use_classifier=False), "fast-lane")
    hybrid_report = None
    if CLASSIFIER.is_file():
        hybrid_report = _report(gold, _predict(gold, use_classifier=True), "hybrid")

    summary = {"fast_lane": fast_report, "hybrid": hybrid_report}
    out = ROOT / "eval" / "ood_section_classification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[eval_ood] wrote {out}")
    return 0


def _predict(gold: list[dict], use_classifier: bool) -> list[str]:
    """Run analyzeSections per source in gold order; return aligned preds."""
    from legalrag.extract.classifier import TrainedClassifier

    fallback = TrainedClassifier.load(CLASSIFIER) if use_classifier and CLASSIFIER.is_file() else None
    preds: list[str] = []
    sources = list(dict.fromkeys(r["source"] for r in gold))
    for source in sources:
        g = _rows_for_source(gold, source)
        if not g:
            continue
        rows = _load(SECTIONS[source])
        analyzed = analyzeSections(rows, fallback=fallback)
        preds.extend(r["clause_type"] for r in analyzed)
    return preds


if __name__ == "__main__":
    raise SystemExit(main())
