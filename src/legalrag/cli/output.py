# Terminal output helpers: ANSI colors, severity styling, formatters.
from __future__ import annotations

import json
import sys
from typing import Any

_LEVEL_COLOR = {
    "high": "91",  # bright red
    "medium": "93",  # bright yellow
    "low": "94",  # bright blue
    "info": "90",  # bright black
}


def _color(text: str, code: str) -> str:
    return f"\x1b[{code}m{text}\x1b[0m"


def colorize(text: str, level: str, enabled: bool = True) -> str:
    code = _LEVEL_COLOR.get(level, "")
    return _color(text, code) if enabled and code else text


def tty() -> bool:
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def format_finding(f: Any, colors: bool = False) -> list[str]:
    """Human-readable lines for a single finding (toDict-shaped or object)."""
    d = f.toDict() if hasattr(f, "toDict") else f
    lines = [
        colorize(f"[{d.get('risk_level', '').upper()}] {d.get('rule_id', '')}", d.get("risk_level", ""), colors),
    ]
    if d.get("confidence") is not None:
        lines.append(f"  confidence: {d['confidence']:.2f}")
    if d.get("clause_type"):
        lines.append(f"  clause type: {d['clause_type']}")
    if d.get("rationale"):
        lines.append(f"  rationale: {d['rationale']}")
    if d.get("statute"):
        lines.append(f"  statute: {d['statute']}")
    if d.get("section_id"):
        lines.append(f"  section: {d['section_id']}")
    if d.get("user_verdict"):
        lines.append(f"  your verdict: {d['user_verdict']}")
    if d.get("user_risk_level"):
        lines.append(f"  your risk level: {d['user_risk_level']}")
    if d.get("user_note"):
        lines.append(f"  your note: {d['user_note']}")
    return lines


def format_summary(summary: dict[str, int]) -> str:
    n = summary.get("n_findings", 0)
    parts = [f"findings: {n}"]
    for level in ("high", "medium", "low", "info"):
        parts.append(f"{level}={summary.get(f'n_{level}', 0)}")
    return ", ".join(parts)


def render_text(output: dict[str, Any], colors: bool = False) -> str:
    """Plain-text report of a full analysis output dict."""
    lines: list[str] = []
    lines.append(f"source: {output.get('source', '')}")
    lines.append(
        f"sections: {output.get('sections', 0)} total, {output.get('classified', 0)} classified "
        f"({format_summary(output.get('summary', {}))})"
    )
    if output.get("elapsed_s") is not None:
        lines.append(f"elapsed: {output['elapsed_s']:.1f}s")
    lines.append("")
    for f in output.get("findings", []):
        lines.extend(format_finding(f, colors))
        lines.append("")
    if output.get("slm"):
        slm_lines = []
        for o in output["slm"]:
            if o.get("parse_ok"):
                slm_lines.append(
                    f"- {o.get('plain_explanation', '')} "
                    f"[{o.get('tenant_impact', '')}]"
                )
        if slm_lines:
            lines.append("plain language:")
            lines.extend(slm_lines)
    return "\n".join(lines).rstrip() + "\n"


def render_markdown(output: dict[str, Any]) -> str:
    """Markdown report of a full analysis output dict."""
    lines: list[str] = []
    lines.append(f"# Lease analysis: `{output.get('source', '')}`")
    lines.append("")
    lines.append(
        f"- Sections: {output.get('sections', 0)} total, "
        f"{output.get('classified', 0)} classified"
    )
    if output.get("elapsed_s") is not None:
        lines.append(f"- Elapsed: {output['elapsed_s']:.1f}s")
    lines.append(f"- Findings: {format_summary(output.get('summary', {}))}")
    lines.append("")
    for f in output.get("findings", []):
        lines.append(f"## [{f.get('risk_level', '').upper()}] {f.get('rule_id', '')}")
        if f.get("confidence") is not None:
            lines.append(f"- Confidence: {f['confidence']:.2f}")
        if f.get("clause_type"):
            lines.append(f"- Clause type: {f['clause_type']}")
        if f.get("rationale"):
            lines.append(f"- Rationale: {f['rationale']}")
        if f.get("statute"):
            lines.append(f"- Statute: {f['statute']}")
        if f.get("section_id"):
            lines.append(f"- Section: {f['section_id']}")
        if f.get("user_verdict"):
            lines.append(f"- Your verdict: {f['user_verdict']}")
        if f.get("user_risk_level"):
            lines.append(f"- Your risk level: {f['user_risk_level']}")
        if f.get("user_note"):
            lines.append(f"- Your note: {f['user_note']}")
        lines.append("")
    if output.get("slm"):
        lines.append("## Plain language")
        for o in output["slm"]:
            if o.get("parse_ok"):
                lines.append(f"- {o.get('plain_explanation', '')}")
                lines.append(f"  - Impact: {o.get('tenant_impact', '')}")
        lines.append("")
    return "\n".join(lines)


def render_json(output: dict[str, Any]) -> str:
    return json.dumps(output, indent=2, ensure_ascii=False) + "\n"


def render(output: dict[str, Any], fmt: str = "text", colors: bool = False) -> str:
    if fmt == "json":
        return render_json(output)
    if fmt == "markdown":
        return render_markdown(output)
    return render_text(output, colors)