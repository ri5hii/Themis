# Lease clause taxonomy: the 15 clause types Extract classifies into.
#
# The order of TAXONOMY is a spec: `classifyClause` breaks evidence-count ties
# in this order (earlier-declared type wins, per docs/progress.md §5.14), so it
# must stay stable. `unknown` is not in the taxonomy — it is the "no evidence"
# output of the fast lane.
from __future__ import annotations

TAXONOMY: tuple[str, ...] = (
    "term",
    "rent",
    "deposit",
    "maintenance",
    "utilities",
    "termination",
    "holdover",
    "subletting",
    "access",
    "late_fee",
    "registration",
    "dispute_resolution",
    "premises",
    "pets",
    "entire_agreement",
)

UNKNOWN = "unknown"

# Types for which we have hand-labeled ground truth in
# data/annotated/leivaditi_redflags.jsonl (7 of 15).
GOLD_TYPES: frozenset[str] = frozenset(
    ("maintenance", "utilities", "subletting", "termination", "holdover", "deposit", "term")
)


def taxonomyIndex(clause_type: str) -> int:
    """0-based position of a type in TAXONOMY (unknown sorts last)."""
    try:
        return TAXONOMY.index(clause_type)
    except ValueError:
        return len(TAXONOMY)


def types() -> tuple[str, ...]:
    return TAXONOMY
