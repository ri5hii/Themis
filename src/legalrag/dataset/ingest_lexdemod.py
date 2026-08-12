# LEXDEMOD ingestion (raw annotated CSVs -> canonical JSONL).
#
# Source: Sancheti et al., "Agent-Specific Deontic Modality Detection in Legal
# Language" (EMNLP 2022), https://github.com/adobe-research/LexDeMod (MIT).
#
# The two annotated CSVs (train_eval_annotated_data.csv, test_annotated_data.csv)
# contain agent-specific multi-label deontic modality annotations. Each row:
#   - cid   contract path (LEDGAR source), ending in <file>-<idx>
#   - text  "[party] sentence" with party in {tenant, landlord, lessee, lessor,
#     subtenant, sublandlord} (aliases collapsed to tenant/landlord)
#   - label 7-dim multi-hot vector
#   - span  {"obl": [[s,e],...], "ent": [...], ...} trigger spans per deontic type
#   - split train | eval | test
#
# Label dimension order (verified by single-active-span analysis):
#   0=obl (obligation), 1=ent (entitlement), 2=pro (prohibition),
#   3=per (permission), 4=oth (other), 5=nen (no-entitlement), 6=none.
#
# Pure logic (stdlib only) so it is unit-testable without dependencies.
from __future__ import annotations

import ast
import csv
from collections.abc import Iterable
from typing import Any

DEONTIC_LABELS = ["obl", "ent", "pro", "per", "oth", "nen", "none"]

PARTY_ALIASES = {
    "tenant": "tenant",
    "lessee": "tenant",
    "subtenant": "tenant",
    "landlord": "landlord",
    "lessor": "landlord",
    "sublandlord": "landlord",
}


def parseCsv(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def splitPartyText(text: str) -> tuple[str, str]:
    """Split a "[party] sentence" row into (normalized party, sentence text)."""
    if text.startswith("["):
        close = text.find("]")
        if close != -1:
            party = text[1:close].strip().lower()
            return PARTY_ALIASES.get(party, party), text[close + 1 :].strip()
    return "unknown", text.strip()


def contractSource(cid: str) -> str:
    """Contract id from a cid like ../LEDGAR/.../ex10_2.htm-7 -> ex10_2.htm."""
    base = cid.split("/")[-1]
    return base.rsplit("-", 1)[0] if "-" in base else base


def sentenceIndex(cid: str) -> int:
    base = cid.split("/")[-1]
    if "-" in base:
        try:
            return int(base.rsplit("-", 1)[1])
        except ValueError:
            return -1
    return -1


def parseLabel(value: str) -> list[int]:
    return [int(v) for v in ast.literal_eval(value)]


def activeDeonticTypes(label: list[int]) -> list[str]:
    return [DEONTIC_LABELS[i] for i, v in enumerate(label) if v]


def parseSpans(value: str) -> dict[str, list[list[int]]]:
    return {k: [list(span) for span in v] for k, v in ast.literal_eval(value).items()}


def ingestAnnotated(rows: Iterable[dict[str, str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("text", ""))
        party, sentence = splitPartyText(text)
        label = parseLabel(str(row.get("label", "[]")))
        out.append(
            {
                "source": contractSource(str(row.get("cid", ""))),
                "sentence_idx": sentenceIndex(str(row.get("cid", ""))),
                "party": party,
                "text": sentence,
                "label": label,
                "deontic_types": activeDeonticTypes(label),
                "spans": parseSpans(str(row.get("span", "{}"))),
                "split": str(row.get("split", "")),
            }
        )
    return out
