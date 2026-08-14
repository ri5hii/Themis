# Unit tests for the document index step (content-hash dedup).
from __future__ import annotations

import json
from pathlib import Path

import pytest

from legalrag.ingest.index import (
    INDEX_JSON,
    buildIndex,
    contentHash,
    discoverOutputs,
    loadIndex,
    lookup,
)


@pytest.fixture
def ingest_output(tmp_path: Path) -> Path:
    """Two source dirs with overlapping sentence content, one unique each."""
    raw = tmp_path / "raw"
    for src, sections, sentences in [
        (
            "doc_a",
            ["Section A1 common clause", "Section A2 unique to A"],
            ["common sentence here", "unique sentence A"],
        ),
        (
            "doc_b",
            ["Section B1 common clause", "Section B2 unique to B"],
            ["common sentence here", "unique sentence B"],
        ),
    ]:
        d = raw / src
        d.mkdir(parents=True)
        (d / "manifest.json").write_text(
            json.dumps(
                {
                    "source": src,
                    "chars": 1,
                    "n_sections": len(sections),
                    "n_sentences": len(sentences),
                }
            )
        )
        with (d / "sections.jsonl").open("w") as fh:
            for text in sections:
                fh.write(json.dumps({"source": src, "raw_text": text, "text": ""}) + "\n")
        with (d / "sentences.jsonl").open("w") as fh:
            for i, text in enumerate(sentences):
                fh.write(json.dumps({"source": src, "sentence_idx": i, "text": text}) + "\n")
    return raw


class TestContentHash:
    def test_normalizes_whitespace(self) -> None:
        assert contentHash("a  b\nc") == contentHash("a b c")

    def test_differs_by_content(self) -> None:
        assert contentHash("hello world") != contentHash("hello there")


class TestDiscoverOutputs:
    def test_finds_complete_outputs(self, ingest_output: Path) -> None:
        found = [p.name for p in discoverOutputs(ingest_output)]
        assert sorted(found) == ["doc_a", "doc_b"]

    def test_skips_dir_without_jsonl(self, tmp_path: Path) -> None:
        d = tmp_path / "raw" / "partial"
        d.mkdir(parents=True)
        (d / "manifest.json").write_text("{}")
        assert discoverOutputs(tmp_path / "raw") == []


class TestBuildIndex:
    def test_dedups_common_content(self, ingest_output: Path, tmp_path: Path) -> None:
        out = tmp_path / "indexes"
        stats = buildIndex(ingest_output, out)
        assert stats["n_docs"] == 2
        assert stats["n_sections_total"] == 4
        assert stats["n_sections_unique"] == 4  # all four section texts differ
        assert stats["n_sentences_total"] == 4
        assert stats["n_sentences_unique"] == 3  # "common sentence here" dedups
        assert stats["sentence_duplicates"] == 1

    def test_writes_all_files(self, ingest_output: Path, tmp_path: Path) -> None:
        out = tmp_path / "indexes"
        buildIndex(ingest_output, out)
        assert (out / INDEX_JSON).is_file()
        docs, sections, sentences = loadIndex(out)
        assert len(docs) == 2
        assert len(sections) == 4
        assert len(sentences) == 3

    def test_doc_sha256_stable(self, ingest_output: Path, tmp_path: Path) -> None:
        out1, out2 = tmp_path / "i1", tmp_path / "i2"
        buildIndex(ingest_output, out1)
        buildIndex(ingest_output, out2)
        docs1, _, _ = loadIndex(out1)
        docs2, _, _ = loadIndex(out2)
        assert {d["sha256"] for d in docs1} == {d["sha256"] for d in docs2}

    def test_unit_records_sources(self, ingest_output: Path, tmp_path: Path) -> None:
        out = tmp_path / "indexes"
        buildIndex(ingest_output, out)
        _, _, sentences = loadIndex(out)
        common = [s for s in sentences if s["text"] == "common sentence here"]
        assert len(common) == 1
        assert sorted(common[0]["sources"]) == ["doc_a", "doc_b"]
        assert common[0]["n_occurrences"] == 2


class TestLookup:
    def test_exact_content_match(self, ingest_output: Path, tmp_path: Path) -> None:
        out = tmp_path / "indexes"
        buildIndex(ingest_output, out)
        hits = lookup(out, "common  sentence   here")
        assert len(hits) == 1
        assert hits[0]["text"] == "common sentence here"

    def test_no_match(self, ingest_output: Path, tmp_path: Path) -> None:
        out = tmp_path / "indexes"
        buildIndex(ingest_output, out)
        assert lookup(out, "no such content anywhere") == []