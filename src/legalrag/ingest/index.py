# Document index: content-hash dedup of ingested raw outputs.
#
# Pure logic (stdlib only) so it is unit-testable without extraction or ML
# dependencies. Scans data/raw/ for ingest output dirs (manifest.json +
# sections.jsonl + sentences.jsonl), dedups section/sentence units by SHA-256
# of normalized text, and writes hash-keyed indexes ready for retrieval.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Ingest output files that identify a document output dir.
_MANIFEST = "manifest.json"
_SECTIONS = "sections.jsonl"
_SENTENCES = "sentences.jsonl"

# Index output filenames.
INDEX_JSON = "index.json"
DOCS = "docs.jsonl"
SECTIONS_INDEX = "sections.jsonl"
SENTENCES_INDEX = "sentences.jsonl"


def contentHash(text: str) -> str:
    """SHA-256 of whitespace-normalized text; path-independent content identity."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    """Collapse runs of whitespace to single spaces (no case folding)."""
    return " ".join(text.split())


def discoverOutputs(raw_root: Path) -> list[Path]:
    """Return every dir under raw_root holding a complete ingest output set."""
    found: list[Path] = []
    for manifest in sorted(raw_root.rglob(_MANIFEST)):
        d = manifest.parent
        if (d / _SECTIONS).is_file() and (d / _SENTENCES).is_file():
            found.append(d)
    return found


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def buildIndex(raw_root: Path, indexes_dir: Path) -> dict:
    """Scan raw ingest outputs and write deduped indexes.

    Returns the index.json stats dict. docs.jsonl has one row per document
    (doc_id, source, path, chars, n_sections, n_sentences, sha256).
    sections.jsonl / sentences.jsonl have one row per unique unit
    (id=sha256, text, n_occurrences, sources=[doc_id...]).
    """
    indexes_dir.mkdir(parents=True, exist_ok=True)

    docs: list[dict] = []
    sections: dict[str, dict] = {}
    sentences: dict[str, dict] = {}

    for d in discoverOutputs(raw_root):
        manifest = _read_manifest(d / _MANIFEST)
        source = manifest.get("source", d.name)
        sections_rows = _read_jsonl(d / _SECTIONS)
        sentences_rows = _read_jsonl(d / _SENTENCES)

        section_texts = [r.get("raw_text") or r.get("text") or "" for r in sections_rows]
        doc_sha = hashlib.sha256(
            "\n\n".join(section_texts).encode("utf-8")
        ).hexdigest()

        docs.append(
            {
                "doc_id": source,
                "source": source,
                "path": str(d),
                "chars": manifest.get("chars", 0),
                "n_sections": manifest.get("n_sections", len(sections_rows)),
                "n_sentences": manifest.get("n_sentences", len(sentences_rows)),
                "sha256": doc_sha,
            }
        )

        for text in section_texts:
            if not text.strip():
                continue
            uid = contentHash(text)
            unit = sections.setdefault(
                uid, {"id": uid, "text": text, "n_occurrences": 0, "sources": []}
            )
            unit["n_occurrences"] += 1
            if source not in unit["sources"]:
                unit["sources"].append(source)

        for row in sentences_rows:
            text = row.get("text") or ""
            if not text.strip():
                continue
            uid = contentHash(text)
            unit = sentences.setdefault(
                uid, {"id": uid, "text": text, "n_occurrences": 0, "sources": []}
            )
            unit["n_occurrences"] += 1
            if source not in unit["sources"]:
                unit["sources"].append(source)

    n_sections_total = sum(d["n_sections"] for d in docs)
    n_sentences_total = sum(d["n_sentences"] for d in docs)
    stats = {
        "n_docs": len(docs),
        "n_sections_total": n_sections_total,
        "n_sections_unique": len(sections),
        "n_sentences_total": n_sentences_total,
        "n_sentences_unique": len(sentences),
        "section_duplicates": n_sections_total - len(sections),
        "sentence_duplicates": n_sentences_total - len(sentences),
    }

    _write_jsonl(indexes_dir / DOCS, docs)
    _write_jsonl(indexes_dir / SECTIONS_INDEX, list(sections.values()))
    _write_jsonl(indexes_dir / SENTENCES_INDEX, list(sentences.values()))
    (indexes_dir / INDEX_JSON).write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8"
    )
    return stats


def loadIndex(indexes_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Read back (docs, sections, sentences) from an existing index."""
    return (
        _read_jsonl(indexes_dir / DOCS),
        _read_jsonl(indexes_dir / SECTIONS_INDEX),
        _read_jsonl(indexes_dir / SENTENCES_INDEX),
    )


def lookup(indexes_dir: Path, text: str) -> list[dict]:
    """Exact-content lookup: units whose normalized text hashes to `text`'s."""
    uid = contentHash(text)
    matches: list[dict] = []
    _, sections, sentences = loadIndex(indexes_dir)
    for unit in (*sections, *sentences):
        if unit["id"] == uid:
            matches.append(unit)
    return matches


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")