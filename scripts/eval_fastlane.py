#!/usr/bin/env python3
"""Fast-lane (and hybrid fast-lane + classifier) clause classifier evaluation.

Scores the fast-lane against:
  1. data/annotated/leivaditi_redflags.jsonl  -- 738 hand-labeled sentences
     (7 gold taxonomy types); reports per-class P/R/F1 + unknown rate. If
     models/classifier.npz exists, also reports the hybrid (fast-lane first,
     classifier fallback on UNKNOWN).
  2. data/annotated/leivaditi_leases.jsonl    -- 8,659 auto-labeled sections
     (candidate labels, not gold); reports exact-accuracy + unknown rate.

Usage:
    python scripts/eval_fastlane.py
"""
from __future__ import annotations

import json
from pathlib import Path

from legalrag.extract.fast_lane import UNKNOWN, classifyClause

ROOT = Path(__file__).resolve().parent.parent
REDFLAGS = ROOT / "data" / "annotated" / "leivaditi_redflags.jsonl"
LEASES = ROOT / "data" / "annotated" / "leivaditi_leases.jsonl"
CLASSIFIER = ROOT / "models" / "classifier.npz"


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def _per_class(gold: list[str], pred: list[str]) -> dict:
    types = sorted(set(gold))
    out: dict = {}
    tot_tp = tot_p = tot_r = 0
    for t in types:
        tp = sum(1 for g, p in zip(gold, pred) if g == t and p == t)
        fp = sum(1 for g, p in zip(gold, pred) if g != t and p == t)
        fn = sum(1 for g, p in zip(gold, pred) if g == t and p != t)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[t] = {"n": tp + fn, "precision": round(prec, 3), "recall": round(rec, 3), "f1": round(f1, 3)}
        tot_tp += tp
        tot_p += tp + fp
        tot_r += tp + fn
    p = tot_tp / tot_p if tot_p else 0.0
    r = tot_tp / tot_r if tot_r else 0.0
    out["macro"] = {
        "precision": round(p, 3),
        "recall": round(r, 3),
        "f1": round(2 * p * r / (p + r) if p + r else 0.0, 3),
        "tp": tot_tp,
    }
    return out


def _hybrid_predict(rows: list[dict], classifier_path: Path) -> list[str]:
    """Fast-lane first; batch-embed UNKNOWN sections through the classifier."""
    from legalrag.extract.classifier import TrainedClassifier, encodeTexts

    tc = TrainedClassifier.load(classifier_path)
    fast = [classifyClause(r["text"])[0] for r in rows]
    unk_idx = [i for i, p in enumerate(fast) if p == UNKNOWN]
    if not unk_idx:
        return fast
    vecs = encodeTexts([rows[i]["text"] for i in unk_idx], tc.model_name)
    preds = tc.predict(vecs)
    out = list(fast)
    for i, p in zip(unk_idx, preds):
        out[i] = p
    return out


def _gold_report(rows: list[dict], pred: list[str]) -> dict:
    gold = [r["type"] for r in rows]
    report = {
        "rows": len(rows),
        "unknown": sum(1 for p in pred if p == "unknown"),
        "per_class": _per_class(gold, pred),
    }
    return report


def main() -> int:
    if REDFLAGS.is_file():
        rows = _load(REDFLAGS)
        pred = [classifyClause(r["text"])[0] for r in rows]
        print("[eval_fastlane] gold redflags (fast-lane):", json.dumps(_gold_report(rows, pred)))
        if CLASSIFIER.is_file():
            hy = _hybrid_predict(rows, CLASSIFIER)
            print("[eval_fastlane] gold redflags (hybrid):", json.dumps(_gold_report(rows, hy)))

    if LEASES.is_file():
        rows = _load(LEASES)
        gold = [r.get("type_fast_lane", "?") for r in rows]
        pred = [classifyClause(r["text"])[0] for r in rows]
        acc = sum(1 for g, p in zip(gold, pred) if g == p) / len(rows)
        print(
            "[eval_fastlane] auto-labeled sections: rows={} exact_acc={:.3f} unknown={}".format(
                len(rows), acc, sum(1 for p in pred if p == "unknown")
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
