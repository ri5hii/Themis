"""Chunk statute markdown sources into section-keyed chunks for grounding.

The shipped statute corpus is hand-built from PDF page exports, which
produces duplicate chunk ids (one per document), title/TOC/background-note
pages, and mojibake OCR pages. This module re-derives the corpus from the
markdown sources in data/statutes/ by splitting on section headings, so every
chunk gets a unique id (``mta_2021#s.11``) and junk pages are dropped.
"""
from __future__ import annotations

import re
from pathlib import Path

PAGE_RE = re.compile(r"^Page \d+ of \d+\s*$")
HEADER_RE = re.compile(r"^(\d{1,3}[A-Z]?\.)\s*(?!\d)(.+)$")
CHAPTER_RE = re.compile(r"^CHAPTER\s+[IVXLC]+\s*$")
SCHEDULE_RE = re.compile(r"^(THE\s+)?(FIRST|SECOND|THIRD|FOURTH) SCHEDULE\.?\s*$", re.IGNORECASE)
JUNK_MARKERS = ("ARRANGEMENT OF SECTIONS", "BACKGROUND NOTE", "BE it enacted", "An Act to establish", "ACT NO.")
MIN_LEN = 120
MAX_NON_ASCII_RATIO = 0.25


def _preambleEnd(lines: list[str]) -> int:
    """Index after the enacting clause; everything before it is title/TOC noise."""
    for i, ln in enumerate(lines):
        if "BE it enacted" in ln:
            return i + 1
    return 0


def chunkAct(source: Path, act: str) -> list[dict]:
    """Split a statute md file into section-keyed chunks ({id, text})."""
    lines = [ln.rstrip("\n") for ln in source.open(encoding="utf-8")]
    lines = [ln for ln in lines if not PAGE_RE.match(ln.strip())]
    lines = lines[_preambleEnd(lines):]

    chunks: list[dict] = []
    cur: dict | None = None
    pending: list[str] = []
    body_lines = 0

    def flush() -> None:
        nonlocal cur, body_lines
        if cur is not None and cur["id"] is not None and cur["lines"]:
            text = "\n".join(cur["lines"]).strip()
            if text:
                chunks.append({"id": cur["id"], "text": text, "n_body_lines": body_lines})
        cur = None
        body_lines = 0

    for raw in lines:
        ln = raw.strip()
        if not ln:
            continue
        if HEADER_RE.match(ln):
            flush()
            cur = {"id": f"{act}#s.{HEADER_RE.match(ln).group(1).rstrip('.')}", "lines": [ln] + pending}
            pending = []
        elif SCHEDULE_RE.match(ln):
            flush()
            cur = {"id": f"{act}#sch", "lines": [ln]}
        elif CHAPTER_RE.match(ln):
            flush()
            pending = [ln]
        elif pending:
            pending.append(ln)
        else:
            if cur is None:
                cur = {"id": None, "lines": []}
            cur["lines"].append(ln)
            body_lines += 1
    flush()
    return chunks


def isJunk(text: str) -> bool:
    """True for title/TOC/background-note chunks and mojibake pages."""
    flat = re.sub(r"\s+", " ", text[:300])
    if any(marker in flat for marker in JUNK_MARKERS):
        return True
    if len(text) < MIN_LEN:
        return True
    if not text:
        return True
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    return non_ascii / len(text) > MAX_NON_ASCII_RATIO


def cleanChunks(chunks: list[dict]) -> list[dict]:
    """Drop junk chunks and enforce unique ids (first occurrence wins)."""
    seen: set[str] = set()
    out: list[dict] = []
    for chunk in chunks:
        if chunk["id"] in seen or chunk["n_body_lines"] == 0 or isJunk(chunk["text"]):
            continue
        seen.add(chunk["id"])
        out.append(chunk)
    return out