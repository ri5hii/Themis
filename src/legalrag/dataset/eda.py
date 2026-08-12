# Dataset EDA: load jsonl corpora and produce summary statistics + a JSON
# report. Pure logic (stdlib only), shared by scripts/dataset_eda.py.
from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from .clean import TRUNCATION_LIMIT


def loadJsonl(path: Path) -> list[dict[str, Any]]:
    """Load a jsonl file of objects; skips blank lines, raises on bad JSON."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}: non-object row")
            rows.append(row)
    return rows


def textStats(texts: list[str]) -> dict[str, Any]:
    if not texts:
        return {"rows": 0}
    lens = sorted(len(t) for t in texts)
    n = len(lens)
    p = lambda q: lens[min(n - 1, int(q * (n - 1)))]
    return {
        "rows": n,
        "mean": round(statistics.mean(lens), 1),
        "min": min(lens),
        "p50": p(0.5),
        "p90": p(0.9),
        "p95": p(0.95),
        "max": max(lens),
    }


def _pct(count: int, total: int) -> float:
    return round(count / total * 100, 1) if total else 0.0


def summarizeRows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    sources = [r.get("source") for r in rows]
    per_source = Counter(sources)
    texts = [str(r.get("text", "")) for r in rows]
    truncated = sum(1 for t in texts if len(t) >= TRUNCATION_LIMIT)
    return {
        "rows": total,
        "unique_sources": len(per_source),
        "rows_per_source": {
            "min": min(per_source.values()) if per_source else 0,
            "p50": sorted(per_source.values())[len(per_source) // 2] if per_source else 0,
            "max": max(per_source.values()) if per_source else 0,
        },
        "text": textStats(texts),
        "truncated_at_limit": truncated,
        "truncated_pct": _pct(truncated, total),
        "empty_text": sum(1 for t in texts if not t.strip()),
    }


def columnDistribution(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    return dict(Counter(r.get(column) for r in rows).most_common())


def buildReport(leases: list[dict[str, Any]], redflags: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "leases": {
            **summarizeRows(leases),
            "type_fast_lane": columnDistribution(leases, "type_fast_lane"),
        },
        "redflags": {
            **summarizeRows(redflags),
            "type": columnDistribution(redflags, "type"),
            "redflag_type": columnDistribution(redflags, "redflag_type"),
        },
    }