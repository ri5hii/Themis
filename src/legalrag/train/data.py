"""Dataset loaders for the trainable components.

- auto labels: fast-lane auto-labeled lease sections (candidate labels)
- gold labels: Leivaditi redflag sentences + ``themis annotate`` section rows
- finetune pairs: aligned finding->explanation pairs for SLM fine-tuning
- statute pairs: (rule query, anchor-matched chunk) positives vs random
  negatives for the grounding gate
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from legalrag.retrieve.statutes import chunkAct, cleanChunks
from legalrag.risk.grounding import lexicalGrounding

REDFLAGS_JSONL = "data/annotated/leivaditi_redflags.jsonl"
SECTIONS_GLOB = "data/annotated/*.sections.jsonl"


def load_auto_labels(path: Path) -> tuple[list[str], list[str]]:
    """Auto-labeled sections from ``type_fast_lane``; non-unknown, n>=2 classes."""
    texts: list[str] = []
    labels: list[str] = []
    if not path.exists():
        return texts, labels
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        label = row.get("type_fast_lane")
        text = row.get("text", "")
        if label and label != "unknown" and text.strip():
            texts.append(text)
            labels.append(label)
    # Drop classes with <2 members: they cannot be stratified and are not
    # trainable with a linear head (e.g. holdover n=1).
    keep = {c for c, n in Counter(labels).items() if n >= 2}
    kept_texts = [t for t, c in zip(texts, labels) if c in keep]
    kept_labels = [c for c in labels if c in keep]
    return kept_texts, kept_labels


def _rows_from_sections_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_gold_labels(annotated_dir: Path) -> tuple[list[str], list[str]]:
    """Gold labels: redflag sentences unioned with ``themis annotate`` rows.

    Redflag rows carry ``type`` (7-type vocabulary); annotate rows carry
    ``type`` from the 16-type taxonomy with ``unknown`` for skipped sections.
    Simple row union; unknown rows are dropped.
    """
    texts: list[str] = []
    labels: list[str] = []
    redflags = annotated_dir / "leivaditi_redflags.jsonl"
    if redflags.exists():
        for row in _rows_from_sections_jsonl(redflags):
            label = row.get("type")
            text = row.get("text", "")
            if label and label != "unknown" and text.strip():
                texts.append(text)
                labels.append(label)
    for path in sorted(annotated_dir.glob("*.sections.jsonl")):
        for row in _rows_from_sections_jsonl(path):
            label = row.get("type")
            text = row.get("text", "")
            if label and label != "unknown" and text.strip():
                texts.append(text)
                labels.append(label)
    return texts, labels


def load_finetune_pairs(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_statute_chunks(statutes_dir: Path) -> list[dict]:
    """Section-keyed chunks from the md sources (shared with index building)."""
    chunks: list[dict] = []
    for md in sorted(statutes_dir.glob("*.md")):
        chunks.extend(cleanChunks(chunkAct(md, md.stem)))
    return chunks


def build_statute_pairs(
    chunks: list[dict],
    rules: list,
    neg_per_pos: int = 3,
    seed: int = 42,
) -> tuple[list[str], list[str], list[int]]:
    """(query, chunk_text, label) pairs for gate training.

    Positives: each rule's statute_query paired with every lexically
    anchor-matched chunk. Negatives: the same query paired with random other
    chunks (any chunk not anchor-matched for that rule).
    """
    rng = random.Random(seed)
    queries: list[str] = []
    texts: list[str] = []
    labels: list[int] = []

    for rule in rules:
        if not rule.statute_query or not rule.statute_anchors:
            continue
        match = lexicalGrounding(rule.statute_anchors, chunks)
        if match is None:
            continue
        candidates = [c for c in chunks if c["id"] != match["id"]]
        queries.append(rule.statute_query)
        texts.append(match["text"])
        labels.append(1)
        for _ in range(neg_per_pos):
            if not candidates:
                break
            neg = rng.choice(candidates)
            queries.append(rule.statute_query)
            texts.append(neg["text"])
            labels.append(0)
    return queries, texts, labels