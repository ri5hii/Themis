# Unit tests for document segmentation (pure stdlib/regex logic).
from __future__ import annotations

from legalrag.ingest.extract import Extraction, PageText
from legalrag.ingest.segment import buildRows, splitParagraphs, splitSentences


class TestSplitSentences:
    def test_basic(self):
        assert splitSentences("Tenant pays rent. Landlord repairs.") == [
            "Tenant pays rent.",
            "Landlord repairs.",
        ]

    def test_semicolon_kept(self):
        assert splitSentences("One; two. Three!") == ["One; two.", "Three!"]

    def test_quotes(self):
        assert splitSentences('A says "done." Then b.') == ['A says "done."', "Then b."]

    def test_abbreviation_no_split(self):
        out = splitSentences("Dr. Smith agrees. No. 5 is void. e.g. rent and fees.")
        assert out == ["Dr. Smith agrees.", "No. 5 is void.", "e.g. rent and fees."]

    def test_trailing_terminator(self):
        assert splitSentences("A clause. Done.") == ["A clause.", "Done."]


class TestSplitParagraphs:
    def test_blank_line_hard_split(self):
        assert splitParagraphs("First clause.\n\nSecond clause.") == [
            "First clause.",
            "Second clause.",
        ]

    def test_numbered_clauses(self):
        out = splitParagraphs("12.2 Tenant pays.\n(a) subitem.\n\n43. ROFR granted.")
        assert out == ["12.2 Tenant pays.", "(a) subitem.", "43. ROFR granted."]

    def test_plain_text_no_split(self):
        assert splitParagraphs("Just one paragraph, no numbers.") == [
            "Just one paragraph, no numbers."
        ]

    def test_multi_dot_number_kept(self):
        out = splitParagraphs("4.2The Landlord enters:\n(a)to repair")
        assert out[0].startswith("4.2The Landlord")
        assert out[1].startswith("(a)to repair")

    def test_empty(self):
        assert splitParagraphs("") == []
        assert splitParagraphs("   ") == []

    def test_clause_within_block(self):
        out = splitParagraphs("1. First line.\n2. Second line.\n3. Third line.")
        assert out == ["1. First line.", "2. Second line.", "3. Third line."]

    def test_crlf_blank_line(self):
        out = splitParagraphs("First clause.\r\n\r\nSecond clause.")
        assert out == ["First clause.", "Second clause."]

    def test_crlf_numbered_clauses(self):
        out = splitParagraphs("1. TERM. One.\r\n2. RENT. Two.")
        assert out == ["1. TERM. One.", "2. RENT. Two."]

    def test_crlf_clause_after_running_text(self):
        out = splitParagraphs(
            "Page 2 of 9 Agreement, hereinafter referred to as rent.\r\n"
            "4. USE. Notwithstanding the foregoing.\r\n"
            "5. SIGNS. Following consent."
        )
        assert out == [
            "Page 2 of 9 Agreement, hereinafter referred to as rent.",
            "4. USE. Notwithstanding the foregoing.",
            "5. SIGNS. Following consent.",
        ]


class TestBuildRows:
    def test_shapes(self):
        ex = Extraction(
            source="lease",
            path="/tmp/lease.pdf",
            pages=[
                PageText(0, "Tenant pays rent. Landlord repairs.\n\n12.2 Clause two.", "text"),
            ],
        )
        rows = buildRows(ex)
        assert len(rows["sections"]) == 2
        # numbered heading "12.2" becomes a noisy extra sentence in the
        # sentence stream; paragraph rows keep it intact
        assert len(rows["sentences"]) == 3
        s0 = rows["sections"][0]
        assert set(s0) == {"source", "text", "raw_text", "type", "start", "end"}
        assert s0["source"] == "lease"
        assert s0["text"] == ""
        assert s0["type"] == ""
        sent0 = rows["sentences"][0]
        assert set(sent0) == {"source", "sentence_idx", "party", "text"}
        assert sent0["sentence_idx"] == 0
        assert sent0["party"] == ""
        assert rows["sentences"][1]["sentence_idx"] == 1