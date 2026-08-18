# Full Leivaditi redflag benchmark ingestion (raw CSV -> canonical JSONL).
#
# Source: Leivaditi et al., "A Benchmark for Lease Contract Review"
#   DOI 10.21942/uva.19732993; GitLab spyretta.leiv/lease_contract_review.
#
# The consolidated CSVs are richer than the extracted levi subset:
#   - docclass.csv        171 full-text lease documents (no 600-char cap)
#   - redflags.csv        52,853 paragraph-level rows, 19 redflag types + 'none'
#   - easy_redflag.csv    sentence-level redflag spans (1,242 positives)
#   - entities.csv        2,101 NER entity annotations, 12 classes
#   - clauses.csv         32,951 clause/part annotations
#
# Pure logic (stdlib only) so it is unit-testable without dependencies.
from __future__ import annotations

import csv
from typing import Any

csv.field_size_limit(10**9)

REDFLAG_TYPES = {
    "additional_remarks",
    "assignment_indeplaatsstelling_permitted",
    "break_option",
    "change_of_control",
    "compalsory_reconstraction",
    "damage",
    "expansion",
    "extension_period",
    "guarantee_transferable",
    "holdover",
    "landlord_repairs",
    "no_obligation_to_operate",
    "reinstatement_clause",
    "riders",
    "right_of_first_refusal_to_lease",
    "right_of_first_refusal_to_purchase",
    "services_charges",
    "special_stipulations",
    "sublease_permitted",
    "warrantees_of_the_owner",
}

ENTITY_CLASSES = {
    "designated_use",
    "end_date",
    "expiration_date_of_lease",
    "extension_period",
    "leased_space",
    "lessee",
    "lessor",
    "notice_period",
    "signing_date",
    "start_date",
    "term_of_payment",
    "vat",
}


def parseCsv(path: str) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def docId(uuid: str) -> str:
    """Normalize a uuid to the bare document id (everything before -lease_)."""
    return uuid.split("-lease_")[0]


def ingestRedflags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Paragraph-level classification rows; keeps 'none' as labeled negatives."""
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": docId(str(row.get("uuid", ""))),
                "text": str(row.get("text", "")),
                "raw_text": str(row.get("raw_text", "")),
                "type": str(row.get("type", "")),
                "start": int(row["start"]) if str(row.get("start", "")).lstrip("-").isdigit() else None,
                "end": int(row["end"]) if str(row.get("end", "")).lstrip("-").isdigit() else None,
            }
        )
    return out


def ingestEasyRedflags(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sentence-level redflag spans; keeps only positive (non-'none') labels."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("type", "")) == "none":
            continue
        out.append(
            {
                "source": docId(str(row.get("uuid", ""))),
                "part": str(row.get("part", "")),
                "text": str(row.get("text", "")),
                "raw_text": str(row.get("raw_text", "")),
                "type": str(row.get("type", "")),
                "start": int(row["start"]) if str(row.get("start", "")).lstrip("-").isdigit() else None,
                "end": int(row["end"]) if str(row.get("end", "")).lstrip("-").isdigit() else None,
            }
        )
    return out


def ingestDocs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        text = str(row.get("document_full_text", ""))
        out.append(
            {
                "source": docId(str(row.get("uuid", ""))),
                "document_class": str(row.get("document_class", "")),
                "text": text,
            }
        )
    return out


def ingestEntities(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": docId(str(row.get("uuid", ""))),
                "part": str(row.get("part_id", "")),
                "class_id": str(row.get("class_id", "")),
                "entity_text": str(row.get("entity_text", "")),
                "entity_start": int(row["entity_start"]) if str(row.get("entity_start", "")).isdigit() else None,
                "entity_end": int(row["entity_end"]) if str(row.get("entity_end", "")).isdigit() else None,
            }
        )
    return out


def ingestClauses(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        out.append(
            {
                "source": docId(str(row.get("uuid", ""))),
                "part": str(row.get("part", "")),
                "text": str(row.get("text", "")),
                "clause_begin": str(row.get("clause_begin", "")) == "True",
                "clause_type": str(row.get("clause_type", "")),
            }
        )
    return out
