# Fast-lane regex clause classifier.
#
# `classifyClause` scores a section against a per-type trigger lexicon and
# returns the best type by (evidence count desc, taxonomy index asc). It is
# authoritative when it fires; sections with no evidence return UNKNOWN and are
# handed to the trained classifier fallback (classifier.py).
#
# Design notes from docs/progress.md:
#   §5.8   a generic `\bpremises?\b` trigger fired on every lease and was
#          removed; premises uses distinctive phrases (`leased to`, `flat no.`,
#          `described as`).
#   §5.14  a bare `\brefund\w*\b` deposit trigger over-matched; deposit uses
#          `security deposit` / `deposit of Rs.` only.
#   §5.15  trigger inflection gaps (OOD): access must match "entering the
#          premises", "repairs and inspections"; term must match "term of this
#          license"; holdover must match "twice the monthly rent".
#   §5.14  ties resolve in taxonomy order (evidence count desc, index asc).
from __future__ import annotations

import re

from .taxonomy import TAXONOMY, UNKNOWN

# Per-type trigger patterns. Word-boundary aware, case-insensitive, inflection
# tolerant. Every pattern is anchored to distinctive legal phrasing to avoid
# the premises/refund over-match failure mode.
_TRIGGERS: dict[str, tuple[re.Pattern[str], ...]] = {
    "term": (
        re.compile(r"\bterm\s+of\s+this\s+(?:lease|tenancy|licen[cs]e)\b", re.IGNORECASE),
        re.compile(r"\b(?:initial|renewal|extension|additional)\s+term\b", re.IGNORECASE),
        re.compile(r"\bcommence\w*\s+date\b", re.IGNORECASE),
        re.compile(r"\b(?:expires?|expiration)\s+(?:on|after|of)?\s*", re.IGNORECASE),
        re.compile(r"\bfor a period of\b", re.IGNORECASE),
        re.compile(r"\blease\s+term\b", re.IGNORECASE),
        re.compile(r"\bexpiration\s+(?:date|of the term)\b", re.IGNORECASE),
    ),
    "rent": (
        re.compile(r"\bbase\s+rent\b", re.IGNORECASE),
        re.compile(r"\bmonthly\s+rent\b", re.IGNORECASE),
        re.compile(r"\brent(?:al)?\s+(?:rate|payments?|amount|due)\b", re.IGNORECASE),
        re.compile(r"\badditional\s+rent\b", re.IGNORECASE),
        re.compile(r"\brent\s+shall\s+be\b", re.IGNORECASE),
        re.compile(r"\brent(?:al)?\s+(?:for|of)\s+the\s+premises\b", re.IGNORECASE),
        re.compile(r"\bannual\s+rent(?:al)?\b", re.IGNORECASE),
    ),
    "deposit": (
        re.compile(r"\bsecurity\s+deposit\b", re.IGNORECASE),
        re.compile(r"\blease\s+deposit\b", re.IGNORECASE),
        re.compile(r"\bdeposit\s+of\s+(?:Rs\.?|INR|\$|₹)", re.IGNORECASE),
        re.compile(r"\bdeposit\s+(?:equal\s+to|shall\s+be)\b", re.IGNORECASE),
        re.compile(r"\b(?:refund|return|repay)\w*\s+.{0,40}\bdeposit\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\b(?:the\s+)?deposit\s+amount\b", re.IGNORECASE),
    ),
    "maintenance": (
        re.compile(r"\brepair\w*\s+and\s+maintenance\b", re.IGNORECASE),
        re.compile(r"\bmainten\w*\s+(?:of|obligation|dut)", re.IGNORECASE),
        re.compile(r"\bresponsible\s+for\s+the\s+maintenance\b", re.IGNORECASE),
        re.compile(r"\bkeep\w*\s+.{0,20}\bin\s+good\s+(?:repair|condition)\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bmaintenance\s+(?:obligations?|duties?|charges?)\b", re.IGNORECASE),
    ),
    "utilities": (
        re.compile(r"\butilit\w*\b", re.IGNORECASE),
        re.compile(r"\belectricit\w*\b", re.IGNORECASE),
        re.compile(r"\bgas\s+(?:charges?|bills?|costs?)?\b", re.IGNORECASE),
        re.compile(r"\bwater\s+(?:charges?|bills?|rates?)?\b", re.IGNORECASE),
        re.compile(r"\bsewer\b", re.IGNORECASE),
        re.compile(r"\bhvac\b", re.IGNORECASE),
        re.compile(r"\bair[- ]conditioning\b", re.IGNORECASE),
    ),
    "termination": (
        re.compile(r"\bterminat\w*\b", re.IGNORECASE),
        re.compile(r"\bcancel\w*\s+(?:this\s+)?(?:lease|contract)\b", re.IGNORECASE),
        re.compile(r"\b(?:right|option)\s+to\s+terminate\b", re.IGNORECASE),
        re.compile(r"\bterminate\s+this\s+(?:lease|contract|agreement)\b", re.IGNORECASE),
        re.compile(r"\bnot\s+renew\w*\b", re.IGNORECASE),
    ),
    "holdover": (
        re.compile(r"\bhold\w*\s+over\b", re.IGNORECASE),
        re.compile(r"\btenant\s+at\s+sufferance\b", re.IGNORECASE),
        re.compile(r"\b(?:double|twice)\s+(?:the\s+)?(?:monthly\s+)?rent\b", re.IGNORECASE),
        re.compile(r"\bdouble\s+rent\b", re.IGNORECASE),
        re.compile(r"\b(?:remain|retain|continue)\w*\s+in\s+possession\b", re.IGNORECASE),
        re.compile(r"\bmove\s+out\s+of\s+(?:the\s+)?(?:leasehold|premises)\b", re.IGNORECASE),
        re.compile(r"\b(?:vacate|surrender)\w*\s+possession\b", re.IGNORECASE),
        re.compile(r"\bfail(?:s|ed|ing|ure)?\w*\s+to\s+return\w*\s+.{0,30}\bpremises\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bafter\s+the\s+expiration\b.{0,40}\bpossession\b", re.IGNORECASE | re.DOTALL),
    ),
    "subletting": (
        re.compile(r"\bsublet\w*\b", re.IGNORECASE),
        re.compile(r"\bsubleas\w*\b", re.IGNORECASE),
        re.compile(r"\bassign\w*\s+this\s+(?:lease|contract|agreement)\b", re.IGNORECASE),
        re.compile(r"\bsublicen[cs]e\w*\b", re.IGNORECASE),
    ),
    "access": (
        re.compile(r"\benter(?:ing|s|ed)?\s+(?:the\s+)?premises\b", re.IGNORECASE),
        re.compile(r"\b(?:inspect|inspection)\w*\s+(?:the\s+)?premises\b", re.IGNORECASE),
        re.compile(r"\b(?:repairs?|inspections?)\s+and\s+(?:repairs?|inspections?)\b", re.IGNORECASE),
        re.compile(r"\baccess\s+to\s+(?:the\s+)?premises\b", re.IGNORECASE),
        re.compile(r"\bright\s+of\s+entry\b", re.IGNORECASE),
        re.compile(r"\b(?:inspect|inspection)\w*\s+.{0,30}\b(?:repair|maintenance)\b", re.IGNORECASE | re.DOTALL),
    ),
    "late_fee": (
        re.compile(r"\blate\s+(?:fee|charge)\b", re.IGNORECASE),
        re.compile(r"\binterest\s+(?:on|upon)\s+overdue\b", re.IGNORECASE),
        re.compile(r"\b(?:penalt\w*|penal\s+sum)\b.{0,30}\b(?:rent|payment|overdue)\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bcharge\s+of\s+Rs\.?\b", re.IGNORECASE),
    ),
    "registration": (
        re.compile(r"\bregist(?:er|ration|ered)\b", re.IGNORECASE),
        re.compile(r"\bleasing\s+administrative\s+office\b", re.IGNORECASE),
        re.compile(r"\b(?:approval\s+and\s+)?regist(?:er|ration)\s+with\b", re.IGNORECASE),
    ),
    "dispute_resolution": (
        re.compile(r"\bdispute\b", re.IGNORECASE),
        re.compile(r"\barbitrat\w*\b", re.IGNORECASE),
        re.compile(r"\blitigat\w*\b", re.IGNORECASE),
        re.compile(r"\bmediation\b", re.IGNORECASE),
        re.compile(r"\bgoverning\s+law\b", re.IGNORECASE),
        re.compile(r"\bresolv\w*\s+.{0,20}\bdispute\b", re.IGNORECASE | re.DOTALL),
    ),
    "premises": (
        re.compile(r"\bleased\s+to\b", re.IGNORECASE),
        re.compile(r"\bflat\s+no\.?\b", re.IGNORECASE),
        re.compile(r"\bdescribed\s+as\b", re.IGNORECASE),
        re.compile(r"\bdemised?\s+premises\b", re.IGNORECASE),
    ),
    "pets": (
        re.compile(r"\bpet(?:s)?\b", re.IGNORECASE),
        re.compile(r"\bdomestic\s+animals?\b", re.IGNORECASE),
        re.compile(r"\bkeep\w*\s+.{0,20}\b(?:pet|animal)\b", re.IGNORECASE | re.DOTALL),
    ),
    "entire_agreement": (
        re.compile(r"\bentire\s+agreement\b", re.IGNORECASE),
        re.compile(r"\bwhole\s+agreement\b", re.IGNORECASE),
        re.compile(r"\bintegration\s+clause\b", re.IGNORECASE),
        re.compile(r"\bmerger\s+clause\b", re.IGNORECASE),
        re.compile(r"\bsupersed\w*\s+.{0,30}\b(?:prior|previous|representations)\b", re.IGNORECASE | re.DOTALL),
    ),
}

_KNOWN_TYPES = frozenset(_TRIGGERS)

# Per-type weights applied during tie-breaking in classifyClause.  `term` is
# the most generic type — its triggers ("expiration", "commencement date",
# "the Term") fire as background context in almost every clause, masking
# more specific types when evidence counts tie.  Downweighting it lets
# content-specific types (holdover, utilities, rent) win on equal evidence.
# §5.14, fix b.
_TYPE_WEIGHT: dict[str, float] = {"term": 0.5}


def evidenceCounts(text: str) -> dict[str, int]:
    """Total trigger matches per type (0 for types with no evidence)."""
    counts = {t: 0 for t in TAXONOMY}
    for t, patterns in _TRIGGERS.items():
        counts[t] = sum(len(p.findall(text)) for p in patterns)
    return counts


def classifyClause(text: str) -> tuple[str, int]:
    """Return (best_clause_type, evidence_count) for a section.

    Best type = max weighted evidence count.  Weights are applied from
    ``_TYPE_WEIGHT`` (default 1.0); ties after weighting are broken in
    taxonomy order (evidence count desc, taxonomy index asc — a spec,
    §5.14).  Returns (UNKNOWN, 0) when no trigger fires.
    """
    counts = evidenceCounts(text)
    best = UNKNOWN
    best_count = 0
    best_score = 0.0
    for t in TAXONOMY:
        count = counts[t]
        score = count * _TYPE_WEIGHT.get(t, 1.0)
        if score > best_score:
            best, best_count, best_score = t, count, score
    return best, best_count


def classifyText(text: str) -> str:
    """Fast-lane type only (no count); UNKNOWN when nothing fires."""
    return classifyClause(text)[0]


def confidenceFromCount(count: int) -> float:
    """Map a fast-lane evidence count to a confidence in [0, 1].

    A single trigger fires at 0.5 and each additional trigger halves the
    remaining gap, so confidence is monotone, bounded, and comparable with the
    classifier fallback's max softmax probability (before this mapping,
    fast-lane "confidence" was the raw evidence count, an incomparable scale).
    Returns 0.0 for count <= 0.
    """
    return 1.0 - 0.5 ** max(int(count), 0)


def _guard() -> None:
    """Assert every trigger set is a known taxonomy type (import-time check)."""
    assert _KNOWN_TYPES <= frozenset(TAXONOMY), f"unknown trigger keys: {_KNOWN_TYPES - frozenset(TAXONOMY)}"


_guard()
