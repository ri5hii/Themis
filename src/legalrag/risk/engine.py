"""Risk engine: rule-based detection of lease-risk findings.

The engine takes classified sections (from extract.analyzeSections) and fires
matching RiskRules to produce Finding objects with severity, rationale, and
extracted values. Grounding (statute retrieval) is handled separately by
risk.grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Finding:
    """A single risk finding produced by the risk engine."""

    rule_id: str
    clause_type: str
    risk_level: str  # "high" | "medium" | "low" | "info"
    confidence: float
    rationale: str
    clause_text: str
    extracted_values: dict = field(default_factory=dict)
    statute: str = ""  # authoritative citation (filled by grounding)
    grounding: str = ""  # retrieved statute text (filled by grounding)
    section_id: str = ""

    def toDict(self) -> dict:
        return {
            "rule_id": self.rule_id,
            "clause_type": self.clause_type,
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "clause_text": self.clause_text[:600],
            "extracted_values": self.extracted_values,
            "statute": self.statute,
            "grounding": self.grounding[:900],
            "section_id": self.section_id,
        }


@dataclass
class AnalysisResult:
    """Full risk analysis result for a document."""

    findings: list[Finding] = field(default_factory=list)
    total_sections: int = 0
    classified_sections: int = 0

    def toDict(self) -> dict:
        return {
            "findings": [f.toDict() for f in self.findings],
            "total_sections": self.total_sections,
            "classified_sections": self.classified_sections,
            "n_findings": len(self.findings),
            "n_high": sum(1 for f in self.findings if f.risk_level == "high"),
            "n_medium": sum(1 for f in self.findings if f.risk_level == "medium"),
            "n_low": sum(1 for f in self.findings if f.risk_level == "low"),
            "n_info": sum(1 for f in self.findings if f.risk_level == "info"),
        }


@dataclass
class RiskRule:
    """A single risk-detection rule."""

    rule_id: str
    clause_types: tuple[str, ...]  # only fire on these clause types
    triggers: list[re.Pattern]  # ALL must match (AND logic)
    extractors: dict[str, re.Pattern] = field(default_factory=dict)
    risk_level: str = "medium"
    rationale_template: str = ""
    statute_query: str = ""
    statute_anchors: list[str] = field(default_factory=list)
    statute_fallback: str = ""


def _check_triggers(text: str, triggers: list[re.Pattern]) -> bool:
    """Return True if ALL trigger patterns match the text."""
    return all(t.search(text) for t in triggers)


def _extract_values(text: str, extractors: dict[str, re.Pattern]) -> dict:
    """Run each extractor pattern against the text, return named captures."""
    values: dict = {}
    for name, pattern in extractors.items():
        m = pattern.search(text)
        if m:
            # Try named groups first, then group(1)
            if m.groupdict():
                values[name] = m.groupdict()
            else:
                values[name] = m.group(1)
    return values


def _format_rationale(template: str, values: dict, clause_text: str) -> str:
    """Fill the rationale template with extracted values."""
    try:
        return template.format(values=values, clause=clause_text[:200])
    except (KeyError, IndexError):
        return template


def analyzeRisk(
    sections: list[dict],
    rules: list[RiskRule],
) -> AnalysisResult:
    """Run all risk rules against classified sections.

    Each section dict must have:
      - "type": clause type string (from classifyClause)
      - "text": section text
      - "id": section identifier (optional)
      - "confidence": classification confidence (optional)

    Returns AnalysisResult with findings sorted by severity (high first).
    """
    findings: list[Finding] = []
    classified = 0

    for section in sections:
        ctype = section.get("type", "unknown")
        if ctype == "unknown":
            continue
        classified += 1
        text = section.get("text", "")
        section_id = section.get("id", "")
        confidence = section.get("confidence", 0.0)

        for rule in rules:
            if ctype not in rule.clause_types:
                continue
            if not _check_triggers(text, rule.triggers):
                continue

            extracted = _extract_values(text, rule.extractors)
            rationale = _format_rationale(rule.rationale_template, extracted, text)

            findings.append(
                Finding(
                    rule_id=rule.rule_id,
                    clause_type=ctype,
                    risk_level=rule.risk_level,
                    confidence=confidence,
                    rationale=rationale,
                    clause_text=text[:600],
                    extracted_values=extracted,
                    section_id=section_id,
                    statute=rule.statute_fallback,
                )
            )

    # Sort by severity
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    findings.sort(key=lambda f: severity_order.get(f.risk_level, 9))

    total = len(sections)
    return AnalysisResult(
        findings=findings,
        total_sections=total,
        classified_sections=classified,
    )
