# Dataset cleaning: schema validation, whitespace normalization, dedupe,
# truncation flagging and redflag_type label normalization.
#
# Pure logic (stdlib only) so it is unit-testable without dependencies.
from __future__ import annotations

from collections.abc import Iterable
from typing import Any

# Leivaditi lease sections are truncated by the scraper at 600 characters.
TRUNCATION_LIMIT = 600

# typos present in the raw leivaditi_redflags labels
REDFLAG_TYPE_FIXES = {
    "compalsory_reconstraction": "compulsory_reconstruction",
    "warrantees_of_the_owner": "warranties_of_the_owner",
}

LEASE_REQUIRED = {"source", "section_idx", "heading", "text"}
REDFLAG_REQUIRED = {"source", "type", "redflag_type", "text"}


def normalizeText(value: str) -> str:
    """Strip and collapse internal runs of whitespace to single spaces."""
    return " ".join(value.split())


def fixRedflagType(label: str) -> str:
    return REDFLAG_TYPE_FIXES.get(label, label)


def isTruncated(text: str) -> bool:
    """A section is a truncation candidate if it hits the scraper's cap."""
    return len(text) >= TRUNCATION_LIMIT


def validateLease(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = LEASE_REQUIRED - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    if "section_idx" in row and not isinstance(row["section_idx"], int):
        errors.append("section_idx not int")
    for key in ("source", "heading", "text"):
        if key in row and not isinstance(row[key], str):
            errors.append(f"{key} not str")
    return errors


def validateRedflag(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = REDFLAG_REQUIRED - set(row)
    if missing:
        errors.append(f"missing keys: {sorted(missing)}")
    for key in ("source", "type", "redflag_type", "text"):
        if key in row and not isinstance(row[key], str):
            errors.append(f"{key} not str")
    return errors


def cleanLease(row: dict[str, Any]) -> dict[str, Any]:
    """Return a cleaned lease section; original dict is not mutated."""
    out = dict(row)
    out["heading"] = normalizeText(str(out.get("heading", "")))
    out["text"] = normalizeText(str(out.get("text", "")))
    if isTruncated(out["text"]):
        out["truncated"] = True
    return out


def cleanRedflag(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["text"] = normalizeText(str(out.get("text", "")))
    out["redflag_type"] = fixRedflagType(str(out.get("redflag_type", "")))
    return out


def dedupe(rows: Iterable[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Drop exact duplicates on the given key tuple, preserving first-seen order.

    Returns (kept, dropped_count) via the last element of a tuple? No — returns
    the deduplicated list; callers can compute drops by length.
    """
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(row.get(k) for k in keys)
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out