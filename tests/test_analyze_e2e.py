"""End-to-end `themis analyze` contract test.

Stubs extraction and the classifier fallback (heavy/IO pieces) but runs the
real pipeline: segment -> fast-lane classify -> risk engine -> statute
grounding -> report rendering, pinning the output contract.
"""
from __future__ import annotations

import json

from legalrag.cli import analyze as analyze_mod

LEASE = (
    "Section A. Rent. Tenant shall pay base rent of one lakh per month. "
    "Tenant waives any right to offset or counterclaim against rent.\n\n"
    "Section B. Maintenance. Landlord shall have no duty to mitigate damages "
    "beyond what law requires.\n\n"
    "Section C. Term. This lease runs for ten years commencing on the "
    "commencement date."
)


class FakeExtraction:
    full_text = LEASE
    n_pages = 2
    methods = ("text",)


def test_analyze_e2e_pipeline(tmp_path, monkeypatch):
    pdf = tmp_path / "lease_test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(analyze_mod, "extractText", lambda p: FakeExtraction())
    monkeypatch.setattr(analyze_mod, "_load_fallback", lambda: None)

    args = analyze_mod.build_parser().parse_args(
        [str(pdf), "--format", "json", "--output", str(tmp_path / "out.json")]
    )
    code = analyze_mod.main(args)
    assert code == 0

    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out["source"] == str(pdf)
    assert out["sections"] == 3
    assert set(out["stage_times_s"]) >= {"parse", "segment", "classify", "risk"}
    assert out["summary"]["n_findings"] == len(out["findings"])

    rule_ids = {f["rule_id"] for f in out["findings"]}
    assert "rent.no_offset" in rule_ids
    assert "termination.no_mitigate" in rule_ids
    for finding in out["findings"]:
        assert finding["statute"], "every finding must carry a statute citation"
        assert finding["risk_level"] in ("high", "medium", "low", "info")
        assert finding["clause_type"]
        assert finding["section_id"]


def test_analyze_markdown_roundtrip(tmp_path, monkeypatch):
    pdf = tmp_path / "lease_test.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    monkeypatch.setattr(analyze_mod, "extractText", lambda p: FakeExtraction())
    monkeypatch.setattr(analyze_mod, "_load_fallback", lambda: None)

    args = analyze_mod.build_parser().parse_args(
        [str(pdf), "--format", "markdown", "--output", str(tmp_path / "out.md")]
    )
    code = analyze_mod.main(args)
    assert code == 0

    body = (tmp_path / "out.md").read_text(encoding="utf-8")
    assert body.startswith("# Lease analysis:")
    assert "## [MEDIUM] rent.no_offset" in body


def test_analyze_directory_arg_clean_error(tmp_path):
    code = analyze_mod.main(analyze_mod.build_parser().parse_args([str(tmp_path)]))
    assert code == 1
