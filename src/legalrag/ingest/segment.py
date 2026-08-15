# Document segmentation: extracted text -> engine row shapes.
#
# Pure logic (stdlib + regex only) so it is unit-testable without extraction
# or ML dependencies. Produces the two row shapes the supervised tasks consume:
#   redflag_paragraph  {source, text, raw_text, type, start, end}
#   deontic_multilabel {source, sentence_idx, party, text}
from __future__ import annotations

import re
from collections.abc import Iterable

# Sentence terminators . ! ? followed by optional closing quote/bracket/space.
_SENT = re.compile(r"[^.!?]+(?:[.!?]['\")\] ]*)?(?=\s|$|\Z)")
# Token that looks like an abbreviation (e.g. "Mr.", "No.", "Art.", "e.g.").
_ABBREV = re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof|No|Art|Sec|Sect|Cl|Clause|Inc|Ltd|Corp|Co|e\.g|i\.e|et al|vs|Fig|St|Ave|Rd|#)\.")
# placeholder for abbreviation dots so they don't split mid-token
_ABBREV_DOT = "\u0000"


def splitSentences(text: str) -> list[str]:
    """Split text into sentences, guarding against common abbreviations.

    Abbreviation periods (e.g. "Dr.", "No.", "e.g.") are shielded from the
    splitter and restored afterwards, so "Dr. Smith" stays one sentence.
    """
    text = " ".join(text.split())
    text = _ABBREV.sub(lambda m: m.group(0).replace(".", _ABBREV_DOT), text)
    parts = [m.strip().replace(_ABBREV_DOT, ".") for m in _SENT.findall(text) if m.strip()]
    return parts

# Numbered clause starts: "1.", "1.2", "6-1", "(a)", "43." at line start.
_CLAUSE_START = re.compile(r"^\s*(?:\(?\d+(?:\.\d+)*\)?\.?|[A-Z]\d+-\d+|\([a-z]\))\s*[^\n]{2,}")


def splitParagraphs(text: str) -> list[str]:
    """Split document text into clause/paragraph units.

    Blank lines hard-split. Runs of text with no blank lines are broken at
    numbered clause starts anywhere in the block (e.g. "4.2", "43. RIGHT OF",
    "6-1", "(a)"). Extractor output often uses CRLF line endings; those are
    normalized first so blank-line and clause splits still fire.
    """
    if not text.strip():
        return []
    paras: list[str] = []
    for block in text.replace("\r\n", "\n").replace("\r", "\n").split("\n\n"):
        block = block.strip()
        if not block:
            continue
        # split consecutive numbered clauses within a block
        lines = block.splitlines()
        if len(lines) == 1:
            paras.append(_collapse(lines))
            continue
        chunks: list[str] = []
        cur: list[str] = []
        for line in lines:
            if _CLAUSE_START.match(line) and cur:
                chunks.append(_collapse(cur))
                cur = [line]
            else:
                cur.append(line)
        if cur:
            chunks.append(_collapse(cur))
        paras.extend(chunks)
    return [p for p in paras if p.strip()]


def _collapse(lines: Iterable[str]) -> str:
    """Collapse a block's lines into a single normalized paragraph string."""
    return " ".join(" ".join(lines).split())


def buildRows(extraction) -> dict[str, list[dict]]:
    """Convert an Extraction into engine row shapes for both supervised tasks."""
    source = extraction.source
    sections: list[dict] = []
    sentences: list[dict] = []
    sent_idx = 0
    for page in extraction.pages:
        for para in splitParagraphs(page.text):
            sections.append(
                {
                    "source": source,
                    "text": "",
                    "raw_text": para,
                    "type": "",
                    "start": 0,
                    "end": 0,
                }
            )
        for sent in splitSentences(page.text):
            sentences.append(
                {
                    "source": source,
                    "sentence_idx": sent_idx,
                    "party": "",
                    "text": sent,
                }
            )
            sent_idx += 1
    return {"sections": sections, "sentences": sentences}