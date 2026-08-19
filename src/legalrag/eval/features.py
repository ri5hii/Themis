# Deterministic feature extraction for the supervised tasks.
#
# (b) upgrade: binary deontic-trigger features (from the LEXDEMOD paper's
# top-trigger tables) combined with rebalancing, to attack the class-imbalance
# failure mode the TF-IDF baseline exposed (macro F1 0.35 vs acc 0.99).
#
# Pure logic (stdlib only) so it is unit-testable without dependencies.
from __future__ import annotations

import re
from collections.abc import Iterable

# Trigger lexicons per deontic type, condensed from the LEXDEMOD paper Table 3
# (top 10 triggers per type). Case-insensitive substring/word matching.
DEONTIC_TRIGGERS: dict[str, list[str]] = {
    "obl": [
        "shall be responsible for",
        "will be responsible for",
        "agrees to",
        "agree to",
        "acknowledges that",
        "shall pay",
        "shall provide",
        "undertakes to",
        "represents and warrants",
        "shall maintain",
    ],
    "ent": [
        "shall have the right to",
        "shall be entitled to",
        "will be entitled to",
        "has the right to",
        "retains all other rights",
        "shall have the option",
        "waives no rights",
        "right to terminate",
        "may at its option",
        "shall be permitted to",
    ],
    "pro": [
        "shall not",
        "will not",
        "may not",
        "nor shall",
        "not to be",
        "in no event shall",
        "neither lessor nor lessee may",
        "shall be prohibited from",
        "no obligation to",
        "without the prior written consent",
    ],
    "per": [
        "may",
        "is permitted to",
        "will allow",
        "shall be allowed to",
        "may at landlord's option",
        "shall have the right",
        "is authorized to",
        "has the right",
        "shall be permitted",
        "are permitted to",
    ],
    "nen": [
        "shall have no right to",
        "shall not be entitled",
        "waives any right",
        "shall not have the right",
        "no right to",
        "waive the right",
        "shall not be entitled to",
        "waives the right",
    ],
    "nobl": [
        "shall not be obligated",
        "shall not be required",
        "shall have no obligation",
        "shall have no liability",
        "shall not be liable",
        "no obligation to",
        "shall not bear",
        "shall not pay",
        "not responsible for",
        "shall not be obligated to",
    ],
}

_TRIGGER_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    key: [re.compile(re.escape(trig), re.IGNORECASE) for trig in trs]
    for key, trs in DEONTIC_TRIGGERS.items()
}


def triggerVector(text: str) -> list[int]:
    """Per deontic trigger group: 1 if any trigger in the group matches."""
    out: list[int] = []
    for group in DEONTIC_TRIGGERS:
        out.append(1 if any(p.search(text) for p in _TRIGGER_PATTERNS[group]) else 0)
    return out


def triggerCounts(text: str) -> list[int]:
    """Per group: total matches across all triggers in the group (count > binary)."""
    out: list[int] = []
    for group in DEONTIC_TRIGGERS:
        out.append(sum(len(p.findall(text)) for p in _TRIGGER_PATTERNS[group]))
    return out


def triggerFeatures(texts: Iterable[str]) -> list[list[int]]:
    return [triggerVector(t) for t in texts]


def partyVector(party: str) -> list[int]:
    """Binary party indicator: 1 for tenant, 0 for landlord (unknown -> 0)."""
    return [1 if str(party).strip().lower() == "tenant" else 0]


def deonticGroupCounts(texts: Iterable[str]) -> dict[str, int]:
    """How many texts hit each trigger group (diagnostics)."""
    counts = {g: 0 for g in DEONTIC_TRIGGERS}
    for text in texts:
        for g, vec in zip(DEONTIC_TRIGGERS, triggerVector(text)):
            counts[g] += vec
    return counts