# Unit tests for dataset cleaning and EDA logic (stdlib only, no Qt).
from __future__ import annotations

import json

import pytest

from legalrag.dataset import clean, eda
from legalrag.dataset import ingest_leivaditi as ing


class TestClean:
    def test_normalizeText_collapses_whitespace(self):
        assert clean.normalizeText("  a\n\t b   c  ") == "a b c"

    def test_fixRedflagType_known_typo(self):
        assert clean.fixRedflagType("compalsory_reconstraction") == "compulsory_reconstruction"
        assert clean.fixRedflagType("warrantees_of_the_owner") == "warranties_of_the_owner"

    def test_fixRedflagType_unknown_passthrough(self):
        assert clean.fixRedflagType("break_option") == "break_option"

    def test_isTruncated(self):
        assert clean.isTruncated("x" * 600)
        assert not clean.isTruncated("x" * 599)

    def test_validateLease_missing_keys(self):
        errs = clean.validateLease({"source": "s"})
        assert any("missing" in e for e in errs)

    def test_validateLease_section_idx_not_int(self):
        errs = clean.validateLease(
            {"source": "s", "section_idx": "0", "heading": "h", "text": "t"}
        )
        assert any("section_idx not int" in e for e in errs)

    def test_validateRedflag_missing(self):
        errs = clean.validateRedflag({"source": "s"})
        assert len(errs) == 1

    def test_cleanLease_adds_truncation_flag(self):
        row = {"source": "s", "section_idx": 0, "heading": "  H  ", "text": "x" * 600}
        out = clean.cleanLease(row)
        assert out["heading"] == "H"
        assert out["truncated"] is True
        assert row["heading"] == "  H  "  # original not mutated

    def test_cleanRedflag_fixes_label(self):
        out = clean.cleanRedflag({"source": "s", "type": "maintenance",
                                  "redflag_type": "compalsory_reconstraction", "text": "  t  "})
        assert out["redflag_type"] == "compulsory_reconstruction"
        assert out["text"] == "t"

    def test_dedupe_preserves_first(self):
        rows = [{"a": 1}, {"a": 1}, {"a": 2}]
        assert clean.dedupe(rows, ("a",)) == [{"a": 1}, {"a": 2}]


class TestEda:
    def test_loadJsonl_skips_blanks(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text('{"a": 1}\n\n{"a": 2}\n')
        rows = eda.loadJsonl(path)
        assert len(rows) == 2

    def test_loadJsonl_rejects_non_object(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text("[1, 2]\n")
        with pytest.raises(TypeError):
            eda.loadJsonl(path)

    def test_textStats(self):
        stats = eda.textStats(["a", "bbbb", "ccc"])
        assert stats["min"] == 1 and stats["max"] == 4 and stats["rows"] == 3

    def test_textStats_empty(self):
        assert eda.textStats([]) == {"rows": 0}

    def test_summarizeRows_counts_truncation(self):
        rows = [
            {"source": "s1", "text": "x" * 600},
            {"source": "s1", "text": "y"},
            {"source": "s2", "text": "z" * 600},
        ]
        s = eda.summarizeRows(rows)
        assert s["unique_sources"] == 2
        assert s["truncated_at_limit"] == 2
        assert s["truncated_pct"] == 66.7

    def test_columnDistribution(self):
        dist = eda.columnDistribution(
            [{"k": "a"}, {"k": "b"}, {"k": "a"}], "k"
        )
        assert dist == {"a": 2, "b": 1}

    def test_buildReport_shape(self):
        report = eda.buildReport(
            [{"source": "s", "section_idx": 0, "heading": "h", "text": "t", "type_fast_lane": "x"}],
            [{"source": "s", "type": "t", "redflag_type": "r", "text": "tx"}],
        )
        assert report["leases"]["type_fast_lane"] == {"x": 1}
        assert report["redflags"]["type"] == {"t": 1}

    def test_truncationByColumn(self):
        rows = [
            {"k": "a", "text": "x" * 600},
            {"k": "a", "text": "y"},
            {"k": "b", "text": "z" * 600},
        ]
        info = eda.truncationByColumn(rows, "k")
        assert info["a"] == {"n": 2, "truncated": 1, "truncated_pct": 50.0}
        assert info["b"] == {"n": 1, "truncated": 1, "truncated_pct": 100.0}

    def test_sourceOverlap_disjoint(self):
        leases = [{"source": "s1"}, {"source": "s2"}]
        red = [{"source": "s3"}]
        ov = eda.sourceOverlap(leases, red)
        assert ov == {"leases_only": 2, "redflags_only": 1, "shared": 0}

    def test_sourceOverlap_shared(self):
        leases = [{"source": "s1"}, {"source": "s2"}]
        red = [{"source": "s1"}]
        ov = eda.sourceOverlap(leases, red)
        assert ov == {"leases_only": 1, "redflags_only": 0, "shared": 1}

    def test_crossTab(self):
        rows = [
            {"a": "x", "b": "p"},
            {"a": "x", "b": "q"},
            {"a": "y", "b": "p"},
        ]
        assert eda.crossTab(rows, "a", "b") == {
            "x": {"p": 1, "q": 1},
            "y": {"p": 1},
        }

    def test_headingFrequency(self):
        rows = [{"heading": "h1"}, {"heading": "h1"}, {"heading": "h2"}]
        assert eda.headingFrequency(rows) == {"h1": 2, "h2": 1}


def _writeSample(tmp_path):
    leases = tmp_path / "leases.jsonl"
    redflags = tmp_path / "redflags.jsonl"
    leases.write_text(
        json.dumps({"source": "a", "section_idx": 0, "heading": "H", "text": "t", "type_fast_lane": "x"}) + "\n"
    )
    redflags.write_text(
        json.dumps({"source": "a", "type": "m", "redflag_type": "compalsory_reconstraction", "text": " t "}) + "\n"
    )
    return leases, redflags


class TestIngestLeivaditi:
    def test_docId_strips_lease_suffix(self):
        assert ing.docId("abc123-lease_792") == "abc123"

    def test_ingestRedflags_keeps_none_negatives(self):
        rows = [
            {"uuid": "x-lease_1", "text": "t1", "raw_text": "r1", "type": "none", "start": "0", "end": "5"},
            {"uuid": "y-lease_2", "text": "t2", "raw_text": "r2", "type": "break_option", "start": "3", "end": "9"},
        ]
        out = ing.ingestRedflags(rows)
        assert len(out) == 2
        assert out[0]["type"] == "none"
        assert out[1] == {
            "source": "y",
            "text": "t2",
            "raw_text": "r2",
            "type": "break_option",
            "start": 3,
            "end": 9,
        }

    def test_ingestEasyRedflags_drops_none(self):
        rows = [
            {"uuid": "x-lease_1", "part": "s1p1", "text": "t1", "raw_text": "r1", "type": "none", "start": "0", "end": "5"},
            {"uuid": "y-lease_2", "part": "s1p2", "text": "t2", "raw_text": "r2", "type": "damage", "start": "1", "end": "2"},
        ]
        out = ing.ingestEasyRedflags(rows)
        assert len(out) == 1
        assert out[0]["source"] == "y"
        assert out[0]["type"] == "damage"

    def test_ingestDocs(self):
        rows = [{"uuid": "x-lease_1", "document_class": "lease agreement", "document_full_text": "full"}]
        assert ing.ingestDocs(rows) == [
            {"source": "x", "document_class": "lease agreement", "text": "full"}
        ]

    def test_ingestEntities(self):
        rows = [
            {
                "uuid": "x-lease_1",
                "part_id": "s1p2",
                "class_id": "lessor",
                "full_text": "ft",
                "entity_text": "ABC Co.",
                "entity_start": "8",
                "entity_end": "41",
            }
        ]
        assert ing.ingestEntities(rows) == [
            {
                "source": "x",
                "part": "s1p2",
                "class_id": "lessor",
                "entity_text": "ABC Co.",
                "entity_start": 8,
                "entity_end": 41,
            }
        ]

    def test_ingestClauses(self):
        rows = [
            {"uuid": "x-lease_1", "part": "s1p1", "text": "t", "clause_begin": "True", "clause_type": "clause_title"},
            {"uuid": "x-lease_1", "part": "s1p2", "text": "t2", "clause_begin": "False", "clause_type": "none"},
        ]
        out = ing.ingestClauses(rows)
        assert out[0]["clause_begin"] is True
        assert out[1]["clause_begin"] is False

    def test_parseCsv_handles_quoted_fields(self, tmp_path):
        p = tmp_path / "t.csv"
        p.write_text('a,b\n"x, y",z\n')
        assert ing.parseCsv(str(p)) == [{"a": "x, y", "b": "z"}]

    def test_redflag_types_known(self):
        assert "break_option" in ing.REDFLAG_TYPES
        assert "holdover" in ing.REDFLAG_TYPES

    def test_entity_classes_known(self):
        assert "lessor" in ing.ENTITY_CLASSES
        assert "vat" in ing.ENTITY_CLASSES


class TestEdaFullReport:
    def test_buildFullReport_counts(self):
        docs = [
            {"source": "d1", "document_class": "lease agreement", "text": "x" * 5000},
            {"source": "d2", "document_class": "amendment", "text": "y" * 200},
        ]
        redflags = [
            {"source": "d1", "text": "t", "type": "break_option"},
            {"source": "d1", "text": "t", "type": "none"},
            {"source": "d2", "text": "t", "type": "damage"},
        ]
        easy = [{"source": "d1", "text": "t", "type": "break_option"}]
        entities = [{"source": "d1", "class_id": "lessor"}]
        clauses = [{"source": "d1", "clause_begin": True, "clause_type": "clause_title"}]
        rep = eda.buildFullReport(docs, redflags, easy, entities, clauses)
        assert rep["docs"]["rows"] == 2
        assert rep["docs"]["len_chars"] == {"min": 200, "p50": 5000, "max": 5000}
        assert rep["redflags"]["positive"] == 2
        assert rep["redflags"]["negative_none"] == 1
        assert rep["redflags"]["docs_with_positive"] == 2
        assert rep["easy_redflags"]["rows"] == 1
        assert rep["entities"]["rows"] == 1
        assert rep["clauses"]["clause_begin_true"] == 1