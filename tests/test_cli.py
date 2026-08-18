# CLI tests: formatters, review loop, annotate loop, subcommand dispatch.
from __future__ import annotations

import json

import pytest

from legalrag.cli import annotate, output
from legalrag.cli.review import ReviewAborted, review_findings

SAMPLE_OUTPUT = {
    "source": "test.pdf",
    "elapsed_s": 12.3,
    "sections": 4,
    "classified": 3,
    "findings": [
        {
            "rule_id": "deposit.cap_exceeded",
            "clause_type": "deposit",
            "risk_level": "medium",
            "confidence": 1.0,
            "rationale": "deposit of $62,000 exceeds cap",
            "statute": "mta_2021#0",
            "user_verdict": "accepted",
        }
    ],
    "slm": [
        {
            "clause_type": "deposit",
            "risk_level": "medium",
            "statute": "mta_2021#0",
            "plain_explanation": "You must pay a big deposit.",
            "tenant_impact": "Cash locked up.",
            "parse_ok": True,
        }
    ],
    "summary": {"n_findings": 1, "n_high": 0, "n_medium": 1, "n_low": 0, "n_info": 0},
}


class TestRender:
    def test_text_contains_finding_and_verdict(self):
        body = output.render(SAMPLE_OUTPUT, "text")
        assert "deposit.cap_exceeded" in body
        assert "your verdict: accepted" in body
        assert "You must pay a big deposit." in body

    def test_markdown_sections(self):
        body = output.render(SAMPLE_OUTPUT, "markdown")
        assert "## [MEDIUM] deposit.cap_exceeded" in body
        assert "## Plain language" in body

    def test_json_roundtrip(self):
        body = output.render(SAMPLE_OUTPUT, "json")
        assert json.loads(body)["summary"]["n_findings"] == 1

    def test_colorize_maps_levels(self):
        assert "\x1b[91m" in output.colorize("x", "high")
        assert output.colorize("x", "high", enabled=False) == "x"
        assert "\x1b[90m" in output.colorize("x", "info")


class TestReview:
    def _finding_dict(self, **kw) -> dict:
        base = {
            "rule_id": "r1",
            "clause_type": "rent",
            "risk_level": "medium",
            "confidence": 1.0,
            "rationale": "reason",
            "statute": "s",
        }
        base.update(kw)
        return base

    def test_accept_dismiss_edit_skip(self, capsys):
        answers = iter(["a", "d", "e", "high", "s"])
        out = review_findings(
            [self._finding_dict() for _ in range(4)],
            prompt=lambda _msg: next(answers),
        )
        assert [d["user_verdict"] for d in out] == ["accepted", "dismissed", "edited", "skipped"]
        assert out[2]["user_risk_level"] == "high"

    def test_note_added_without_verdict_change(self, capsys):
        answers = iter(["n", "worth checking", "a"])
        out = review_findings([self._finding_dict()], prompt=lambda _msg: next(answers))
        assert out[0]["user_note"] == "worth checking"
        assert out[0]["user_verdict"] == "accepted"

    def test_quit_raises(self, capsys):
        with pytest.raises(ReviewAborted):
            review_findings(
                [self._finding_dict()],
                prompt=lambda _msg: "q",
            )

    def test_eof_raises(self, capsys):
        def eof(_msg):
            raise EOFError

        with pytest.raises(ReviewAborted):
            review_findings([self._finding_dict()], prompt=eof)

    def test_empty_findings_no_prompt(self, capsys):
        out = review_findings([], prompt=lambda _msg: pytest.fail("should not prompt"))
        assert out == []


class TestAnnotate:
    def test_annotates_rows_and_writes(self, tmp_path, monkeypatch):

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(annotate, "extractText", lambda p: type("X", (), {
            "full_text": "Section A text.\n\nSection B text.",
            "n_pages": 1,
        })())

        answers = iter(["deposit", "u"])
        out_path = tmp_path / "out.jsonl"
        code = annotate.annotate_sections(pdf, out_path, prompt=lambda _msg: next(answers))
        assert code == 0
        rows = [json.loads(l) for l in out_path.read_text().splitlines()]
        assert [(r["section_idx"], r["type"]) for r in rows] == [(0, "deposit"), (1, "unknown")]
        assert rows[0]["source"] == "doc"

    def test_abort_persists_partial(self, tmp_path, monkeypatch):

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(annotate, "extractText", lambda p: type("X", (), {
            "full_text": "A.\n\nB.\n\nC.",
            "n_pages": 1,
        })())

        answers = iter(["term", "q"])
        out_path = tmp_path / "out.jsonl"
        with pytest.raises(annotate.AnnotateAborted) as exc:
            annotate.annotate_sections(pdf, out_path, prompt=lambda _msg: next(answers))
        annotate._write_rows(exc.value.rows, out_path)
        rows = [json.loads(l) for l in out_path.read_text().splitlines()]
        assert [(r["section_idx"], r["type"]) for r in rows] == [(0, "term")]

    def test_eof_aborts_with_partial_rows(self, tmp_path, monkeypatch):

        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(annotate, "extractText", lambda p: type("X", (), {
            "full_text": "A.\n\nB.\n\nC.",
            "n_pages": 1,
        })())

        answers = iter(["rent", "term"])
        out_path = tmp_path / "out.jsonl"

        def stub(_msg):
            try:
                return next(answers)
            except StopIteration:
                raise EOFError

        with pytest.raises(annotate.AnnotateAborted) as exc:
            annotate.annotate_sections(pdf, out_path, prompt=stub)
        assert [(r["section_idx"], r["type"]) for r in exc.value.rows] == [(0, "rent"), (1, "term")]


class TestDispatch:
    def test_parser_routes_subcommands(self):
        from legalrag.cli import build_parser

        parser = build_parser()
        ns = parser.parse_args(["analyze", "doc.pdf"])
        assert ns.func.__module__ == "legalrag.cli.analyze"
        ns2 = parser.parse_args(["annotate", "doc.pdf"])
        assert ns2.func.__module__ == "legalrag.cli.annotate"