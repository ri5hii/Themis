"""Tests for the risk engine module."""
from __future__ import annotations

import re

from legalrag.risk.engine import (
    _check_triggers,
    _extract_values,
    analyzeRisk,
)
from legalrag.risk.rules import RULES


class TestCheckTriggers:
    def test_all_match(self):
        triggers = [re.compile(r"\bdeposit\b", re.IGNORECASE)]
        assert _check_triggers("security deposit", triggers) is True

    def test_partial_match_fails(self):
        triggers = [
            re.compile(r"\bdeposit\b", re.IGNORECASE),
            re.compile(r"\bcap\b", re.IGNORECASE),
        ]
        assert _check_triggers("security deposit", triggers) is False

    def test_empty_triggers(self):
        assert _check_triggers("any text", []) is True


class TestExtractValues:
    def test_amount_extraction(self):
        extractors = {"amount": re.compile(r"\$([\d,]+)")}
        values = _extract_values("The fee is $1,234", extractors)
        assert values["amount"] == "1,234"

    def test_no_match(self):
        extractors = {"amount": re.compile(r"\$([\d,]+)")}
        values = _extract_values("no amount here", extractors)
        assert "amount" not in values


class TestRiskRules:
    def test_rule_count(self):
        assert len(RULES) == 6

    def test_rule_ids_unique(self):
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids))

    def test_deposit_cap_rule(self):
        rule = next(r for r in RULES if r.rule_id == "deposit.cap_exceeded")
        assert "deposit" in rule.clause_types
        assert rule.risk_level == "high"
        assert len(rule.triggers) == 2

    def test_rent_escalation_rule(self):
        rule = next(r for r in RULES if r.rule_id == "rent.excessive_escalation")
        assert "rent" in rule.clause_types
        assert rule.risk_level == "medium"


class TestAnalyzeRisk:
    def _make_section(self, text: str, ctype: str, sid: str = "s1") -> dict:
        return {"id": sid, "text": text, "type": ctype, "confidence": 0.8}

    def test_no_findings_on_unknown(self):
        sections = [self._make_section("some text", "unknown")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 0

    def test_deposit_fires(self):
        sections = [self._make_section("security deposit of $10,000", "deposit")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "deposit.cap_exceeded"
        assert result.findings[0].risk_level == "high"

    def test_rent_escalation_fires(self):
        sections = [self._make_section(
            "rent shall increase every year by 5 percent", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "rent.excessive_escalation"

    def test_holdover_fires(self):
        sections = [self._make_section(
            "tenant at sufferance shall pay double rent", "holdover"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "holdover.punitive_rate"

    def test_late_fee_fires(self):
        sections = [self._make_section(
            "late fee of 5 percent within 10 days", "late_fee"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "late_fee.excessive"

    def test_termination_landlord_only_fires(self):
        sections = [self._make_section(
            "landlord may terminate this lease at any time", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "termination.landlord_only"

    def test_sole_discretion_fires(self):
        sections = [self._make_section(
            "reinstatement at landlord sole discretion", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "termination.sole_discretion"

    def test_sorted_by_severity(self):
        sections = [
            self._make_section("security deposit of $10,000", "deposit", "s1"),
            self._make_section("reinstatement at sole discretion", "termination", "s2"),
        ]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 2
        assert result.findings[0].risk_level == "high"
        assert result.findings[1].risk_level == "low"

    def test_counts(self):
        sections = [self._make_section("security deposit", "deposit")]
        result = analyzeRisk(sections, RULES)
        d = result.toDict()
        assert d["total_sections"] == 1
        assert d["classified_sections"] == 1
        assert d["n_findings"] == 1
        assert d["n_high"] == 1

    def test_wrong_clause_type_no_fire(self):
        sections = [self._make_section("security deposit", "term")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 0
