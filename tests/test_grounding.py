"""Statute grounding tests: lexical anchor scan, dense fallback, static fallback."""
from __future__ import annotations

from legalrag.risk.engine import Finding, RiskRule
from legalrag.risk.grounding import (
    groundAll,
    groundFinding,
    lexicalGrounding,
    loadStatuteChunks,
)


def _rule(**kw) -> RiskRule:
    base = {
        "rule_id": "rent.excessive_escalation",
        "clause_types": ("rent",),
        "triggers": [],
        "risk_level": "medium",
        "rationale_template": "escalation detected",
        "statute_query": "escalation cap",
        "statute_anchors": ["escalation", "rent"],
        "statute_fallback": "MTA 2021 general",
    }
    base.update(kw)
    return RiskRule(**base)


def _finding(**kw) -> Finding:
    base = {
        "rule_id": "rent.excessive_escalation",
        "clause_type": "rent",
        "risk_level": "medium",
        "confidence": 1.0,
        "rationale": "escalation detected",
        "clause_text": "rent shall escalate annually",
    }
    base.update(kw)
    return Finding(**base)


CHUNKS = [
    {"id": "mta_s17", "text": "The rent escalation shall not exceed 15 percent."},
    {"id": "mta_s18", "text": "The landlord shall maintain the premises."},
]


def test_lexical_grounding_all_anchors_required() -> None:
    hit = lexicalGrounding(["escalation", "rent"], CHUNKS)
    assert hit == CHUNKS[0]
    assert lexicalGrounding(["escalation", "refrigerator"], CHUNKS) is None
    assert lexicalGrounding(["landlord"], CHUNKS) == CHUNKS[1]


def test_ground_finding_stage1_lexical(tmp_path) -> None:
    f = _finding()
    groundFinding(f, _rule(), CHUNKS, tmp_path)
    assert f.statute == "mta_s17"
    assert f.grounding == CHUNKS[0]["text"]


def test_ground_finding_stage2_dense(tmp_path, monkeypatch) -> None:
    rule = _rule(statute_anchors=[])
    monkeypatch.setattr(
        "legalrag.retrieve.queryEmbeddings",
        lambda q, d, k=3: [{"id": "mta_s18", "text": CHUNKS[1]["text"], "score": 0.9}],
    )
    f = _finding()
    groundFinding(f, rule, CHUNKS, tmp_path, gate=_Gate(0.9))
    assert f.statute == "mta_s18"


def test_ground_finding_stage3_static(tmp_path, monkeypatch) -> None:
    rule = _rule(statute_anchors=[], statute_query="")
    f = _finding()
    groundFinding(f, rule, CHUNKS, tmp_path)
    assert f.statute == "MTA 2021 general"
    assert f.grounding == ""


def test_ground_finding_dense_error_falls_to_static(tmp_path, monkeypatch) -> None:
    rule = _rule(statute_anchors=[])

    def boom(q, d, k=3):
        raise OSError("index missing")

    monkeypatch.setattr("legalrag.retrieve.queryEmbeddings", boom)
    f = _finding()
    groundFinding(f, rule, CHUNKS, tmp_path)
    assert f.statute == "MTA 2021 general"


class _Gate:
    """Fake relevance gate for gating tests."""

    threshold = 0.5

    def __init__(self, prob: float) -> None:
        self.prob = prob

    def score(self, query: str, chunk_text: str, dense_score: float) -> float:
        return self.prob


def test_ground_finding_gate_suppresses_irrelevant_hit(tmp_path, monkeypatch) -> None:
    rule = _rule(statute_anchors=[])
    monkeypatch.setattr(
        "legalrag.retrieve.queryEmbeddings",
        lambda q, d, k=3: [{"id": "mta_s18", "text": CHUNKS[1]["text"], "score": 0.9}],
    )
    f = _finding()
    groundFinding(f, rule, CHUNKS, tmp_path, gate=_Gate(0.1))
    assert f.statute == "MTA 2021 general"
    assert f.grounding == ""


def test_ground_finding_gate_passes_relevant_hit(tmp_path, monkeypatch) -> None:
    rule = _rule(statute_anchors=[])
    monkeypatch.setattr(
        "legalrag.retrieve.queryEmbeddings",
        lambda q, d, k=3: [{"id": "mta_s18", "text": CHUNKS[1]["text"], "score": 0.9}],
    )
    f = _finding()
    groundFinding(f, rule, CHUNKS, tmp_path, gate=_Gate(0.9))
    assert f.statute == "mta_s18"
    assert f.grounding == CHUNKS[1]["text"]


def test_ground_all_mutates_in_place(tmp_path, monkeypatch) -> None:
    f1 = _finding(rule_id="rent.excessive_escalation")
    f2 = _finding(rule_id="unknown.rule", clause_type="unknown")
    result = type("AnalysisResult", (), {"findings": [f1, f2]})()
    rules = {"rent.excessive_escalation": _rule()}
    groundAll(result, rules, CHUNKS, tmp_path)
    assert f1.statute == "mta_s17"
    assert f2.statute == ""  # no rule entry: left untouched


def test_load_statute_chunks_missing_dir(tmp_path) -> None:
    assert loadStatuteChunks(tmp_path / "nope") == []