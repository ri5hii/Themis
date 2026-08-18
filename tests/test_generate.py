"""SLM generation tests: parsing, field stamping, batch mapping (model stubbed)."""
from __future__ import annotations

import json

from legalrag.risk.engine import Finding
from legalrag.slm import generate
from legalrag.slm.generate import SLMOutput, simplifyAll, simplifyFinding


class StubChatModel:
    """Minimal llama_cpp-compatible stub with canned content."""

    def __init__(self, content: str) -> None:
        self.content = content

    def create_chat_completion(self, **kwargs) -> dict:
        return {"choices": [{"message": {"content": self.content}}]}


def _finding(**kw) -> Finding:
    base = {
        "rule_id": "deposit.cap_exceeded",
        "clause_type": "deposit",
        "risk_level": "medium",
        "confidence": 1.0,
        "rationale": "deposit of $62,000 exceeds cap",
        "clause_text": "Security Deposit required is $62,000",
        "statute": "MTA 2021 s.17",
        "grounding": "The deposit shall not exceed two months' rent",
    }
    base.update(kw)
    return Finding(**base)


def _stub_loader(holder):
    def load(model_path, n_ctx, n_threads):
        return holder["model"], holder["grammar"]

    return load


def _run(monkeypatch, content: str):
    holder = {"model": StubChatModel(content), "grammar": None}
    monkeypatch.setattr(generate, "_load_model", _stub_loader(holder))
    return holder


def test_parse_ok_valid_json(monkeypatch) -> None:
    _run(
        monkeypatch,
        json.dumps(
            {"plain_explanation": "You must pay a big deposit.", "tenant_impact": "Cash locked up."}
        ),
    )
    out = simplifyFinding(clause_text="x", rationale="r", risk_level="medium", statute="s")
    assert out.parse_ok
    assert out.plain_explanation == "You must pay a big deposit."
    assert out.tenant_impact == "Cash locked up."


def test_parse_fail_keeps_raw(monkeypatch) -> None:
    _run(monkeypatch, "not json at all")
    out = simplifyFinding(clause_text="x", rationale="r", risk_level="high", statute="s")
    assert not out.parse_ok
    assert out.plain_explanation == "not json at all"[:500]
    assert out.tenant_impact == ""


def test_engine_fields_stamped_post_inference(monkeypatch) -> None:
    _run(monkeypatch, json.dumps({"plain_explanation": "p", "tenant_impact": "t"}))
    out = simplifyFinding(
        clause_text="x",
        rationale="r",
        risk_level="low",
        statute="DRCA 1958 s.9",
        clause_type="rent",
    )
    assert out.clause_type == "rent"
    assert out.risk_level == "low"
    assert out.statute == "DRCA 1958 s.9"


def test_simplify_all_maps_findings(monkeypatch) -> None:
    _run(monkeypatch, json.dumps({"plain_explanation": "p", "tenant_impact": "t"}))
    outputs = simplifyAll([_finding(rule_id="a"), _finding(rule_id="b")])
    assert len(outputs) == 2
    assert all(isinstance(o, SLMOutput) for o in outputs)
    assert outputs[0].clause_type == "deposit"
    assert outputs[0].statute == "MTA 2021 s.17"