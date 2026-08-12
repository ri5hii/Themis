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


def truncationByColumn(rows: list[dict[str, Any]], column: str) -> dict[str, Any]:
    """Truncation rate (share of rows hitting the 600-char cap) per column value."""
    out: dict[str, Any] = {}
    per_value: dict[Any, list[bool]] = {}
    for row in rows:
        key = row.get(column)
        per_value.setdefault(key, []).append(len(str(row.get("text", ""))) >= TRUNCATION_LIMIT)
    for key, flags in sorted(per_value.items(), key=lambda kv: -sum(kv[1])):
        n = len(flags)
        out[str(key)] = {
            "n": n,
            "truncated": sum(flags),
            "truncated_pct": round(sum(flags) / n * 100, 1),
        }
    return out


def sourceOverlap(leases: list[dict[str, Any]], redflags: list[dict[str, Any]]) -> dict[str, Any]:
    """Overlap between the lease-section corpus and the redflag sentence corpus."""
    lease_srcs = {r.get("source") for r in leases}
    red_srcs = {r.get("source") for r in redflags}
    return {
        "leases_only": len(lease_srcs - red_srcs),
        "redflags_only": len(red_srcs - lease_srcs),
        "shared": len(lease_srcs & red_srcs),
    }


def crossTab(rows: list[dict[str, Any]], col_a: str, col_b: str) -> dict[str, dict[str, int]]:
    """Count co-occurrences of two columns as {a_value: {b_value: count}}."""
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        a, b = str(row.get(col_a)), str(row.get(col_b))
        out.setdefault(a, {}).setdefault(b, 0)
        out[a][b] += 1
    return out


def headingFrequency(rows: list[dict[str, Any]], limit: int = 25) -> dict[str, int]:
    return dict(Counter(r.get("heading") for r in rows).most_common(limit))


def buildFullReport(
    docs: list[dict[str, Any]],
    redflags: list[dict[str, Any]],
    easy_redflags: list[dict[str, Any]],
    entities: list[dict[str, Any]],
    clauses: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summary stats for the full Leivaditi benchmark corpora."""
    doc_texts = {r["source"]: str(r.get("text", "")) for r in docs}
    doc_len = [len(t) for t in doc_texts.values()]
    pos = [r for r in redflags if str(r.get("type", "")) != "none"]

    def _docCount(rows: list[dict[str, Any]]) -> int:
        return len({r["source"] for r in rows})

    return {
        "docs": {
            **summarizeRows(docs),
            "document_class": columnDistribution(docs, "document_class"),
            "len_chars": {
                "min": min(doc_len),
                "p50": sorted(doc_len)[len(doc_len) // 2],
                "max": max(doc_len),
            },
        },
        "redflags": {
            "rows": len(redflags),
            "docs": _docCount(redflags),
            "positive": len(pos),
            "negative_none": len(redflags) - len(pos),
            "positive_types": len({r.get("type") for r in pos}),
            "docs_with_positive": _docCount(pos),
            "type": columnDistribution(redflags, "type"),
        },
        "easy_redflags": {
            "rows": len(easy_redflags),
            "docs": _docCount(easy_redflags),
            "types": len({r.get("type") for r in easy_redflags}),
            "type": columnDistribution(easy_redflags, "type"),
        },
        "entities": {
            "rows": len(entities),
            "docs": _docCount(entities),
            "class_id": columnDistribution(entities, "class_id"),
        },
        "clauses": {
            "rows": len(clauses),
            "docs": _docCount(clauses),
            "clause_begin_true": sum(1 for r in clauses if r.get("clause_begin")),
            "clause_type": columnDistribution(clauses, "clause_type"),
        },
    }


def buildReport(leases: list[dict[str, Any]], redflags: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "leases": {
            **summarizeRows(leases),
            "type_fast_lane": columnDistribution(leases, "type_fast_lane"),
            "truncation_by_type_fast_lane": truncationByColumn(leases, "type_fast_lane"),
            "top_headings": headingFrequency(leases),
        },
        "redflags": {
            **summarizeRows(redflags),
            "type": columnDistribution(redflags, "type"),
            "redflag_type": columnDistribution(redflags, "redflag_type"),
            "truncation_by_type": truncationByColumn(redflags, "type"),
            "type_x_redflag_type": crossTab(redflags, "type", "redflag_type"),
        },
        "cross_corpus": {
            "source_overlap": sourceOverlap(leases, redflags),
        },
    }