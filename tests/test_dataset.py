# Unit tests for dataset cleaning and EDA logic (stdlib only, no Qt).
from __future__ import annotations

import json

import pytest

from legalrag.dataset import clean, eda


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