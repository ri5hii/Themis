"""Statute grounding: lexical anchor scan + dense FAISS retrieval + static fallback.

Two-stage process per finding:
  1. Lexical anchor scan (deterministic) - rule's anchors must ALL appear
  2. Dense FAISS tie-break (fallback) - top-1 from retrieve.queryEmbeddings,
     gated by the trained relevance head (models/grounding, if present)
  3. Static citation fallback - rule's statute_fallback if both fail
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .engine import Finding, RiskRule

GATE_DIR = Path(__file__).resolve().parents[3] / "models" / "grounding"


@lru_cache(maxsize=1)
def _loadGate(gate_dir: Path | None = None) -> object | None:
    """Cached relevance gate; None when absent/corrupt (ungated behavior)."""
    from .gate import _loadGate as _load

    return _load(Path(gate_dir) if gate_dir is not None else GATE_DIR)


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for matching."""
    import re
    return re.sub(r"\s+", " ", text.lower()).strip()


def lexicalGrounding(
    anchors: list[str],
    statute_chunks: list[dict],
) -> dict | None:
    """Return statute chunk containing ALL anchor keywords (case-insensitive).

    A section is accepted when its text contains EVERY anchor.
    Returns the first matching chunk, or None.
    """
    normalized_anchors = [_normalize(a) for a in anchors]

    for chunk in statute_chunks:
        text = _normalize(chunk.get("text", ""))
        if all(a in text for a in normalized_anchors):
            return chunk
    return None


def loadStatuteChunks(statute_index_dir: Path) -> list[dict]:
    """Load statute chunks from sections.jsonl."""
    ids_path = statute_index_dir / "sections.jsonl"
    if not ids_path.exists():
        return []
    chunks = []
    with ids_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def groundFinding(
    finding: Finding,
    rule: RiskRule,
    statute_chunks: list[dict],
    statute_index_dir: Path,
    dense_k: int = 3,
    gate: object | None = None,
) -> Finding:
    """Ground a finding against the statute corpus.

    Stage 1: Lexical anchor scan (deterministic)
    Stage 2: Dense FAISS retrieval (if lexical fails), gated by the trained
             relevance head when available
    Stage 3: Static fallback (if both fail)

    ``gate`` overrides the artifact loaded from models/grounding (tests).
    """
    # Stage 1: Lexical anchor scan
    if rule.statute_anchors:
        match = lexicalGrounding(rule.statute_anchors, statute_chunks)
        if match:
            finding.statute = match.get("id", rule.statute_fallback)
            finding.grounding = match.get("text", "")[:900]
            return finding

    # Stage 2: Dense FAISS retrieval
    if rule.statute_query:
        try:
            from legalrag.retrieve import queryEmbeddings

            hits = queryEmbeddings(
                rule.statute_query, statute_index_dir, k=dense_k
            )
            if hits:
                best = hits[0]
                gate = gate if gate is not None else _loadGate()
                if gate is None or gate.score(rule.statute_query, best["text"], best["score"]) >= gate.threshold:
                    finding.statute = best.get("id", rule.statute_fallback)
                    finding.grounding = best.get("text", "")[:900]
                    return finding
        except (OSError, ImportError):
            pass  # Fall through to static fallback

    # Stage 3: Static fallback
    finding.statute = rule.statute_fallback
    return finding


def groundAll(
    result,  # AnalysisResult
    rules_by_id: dict[str, RiskRule],
    statute_chunks: list[dict],
    statute_index_dir: Path,
) -> None:
    """Ground all findings in an AnalysisResult. Modifies in-place."""
    for finding in result.findings:
        rule = rules_by_id.get(finding.rule_id)
        if rule:
            groundFinding(finding, rule, statute_chunks, statute_index_dir)
