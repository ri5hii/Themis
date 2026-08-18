"""SLM generation: lazy-loaded GGUF model for plain-language prose.

The engine provides all authoritative fields (clause_type, risk_level,
statute). The SLM only generates plain_explanation and tenant_impact.
Engine stamps authoritative fields post-inference.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .grammar import GRAMMAR, SYSTEM_PROMPT, make_finding_prompt


@dataclass
class SLMOutput:
    """SLM output for a single finding."""

    plain_explanation: str = ""
    tenant_impact: str = ""
    clause_type: str = ""  # engine-authoritative, stamped post-inference
    risk_level: str = ""  # engine-authoritative, stamped post-inference
    statute: str = ""  # engine-authoritative, stamped post-inference
    parse_ok: bool = False
    raw: str = ""

    def toDict(self) -> dict:
        return {
            "clause_type": self.clause_type,
            "risk_level": self.risk_level,
            "statute": self.statute,
            "plain_explanation": self.plain_explanation,
            "tenant_impact": self.tenant_impact,
            "parse_ok": self.parse_ok,
        }


# Lazy-loaded model singleton
_model = None
_grammar = None


def _load_model(model_path: str, n_ctx: int = 4096, n_threads: int = 8):
    """Lazy-load the GGUF model (single instance, reused across findings)."""
    global _model, _grammar
    if _model is not None:
        return _model, _grammar

    from llama_cpp import Llama, LlamaGrammar

    _grammar = LlamaGrammar.from_string(GRAMMAR)
    _model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=0,
        verbose=False,
    )
    return _model, _grammar


def simplifyFinding(
    clause_text: str,
    rationale: str,
    risk_level: str,
    statute: str,
    grounding: str = "",
    clause_type: str = "",
    model_path: str | None = None,
    n_ctx: int = 4096,
    n_threads: int = 8,
) -> SLMOutput:
    """Generate plain-language explanation for a single finding.

    The engine stamps clause_type, risk_level, and statute post-inference.
    The SLM only generates plain_explanation and tenant_impact.
    """
    if model_path is None:
        model_path = str(
            Path(__file__).resolve().parents[3]
            / "models/llama-3.2-3b/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        )

    model, grammar = _load_model(model_path, n_ctx, n_threads)
    user_prompt = make_finding_prompt(clause_text, rationale, risk_level, statute, grounding)

    out = model.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        grammar=grammar,
        max_tokens=400,
        temperature=0.0,
    )

    raw = out["choices"][0]["message"]["content"]
    output = SLMOutput(raw=raw, clause_type=clause_type, risk_level=risk_level, statute=statute)

    try:
        parsed = json.loads(raw)
        output.plain_explanation = parsed.get("plain_explanation", "")
        output.tenant_impact = parsed.get("tenant_impact", "")
        output.parse_ok = True
    except json.JSONDecodeError:
        output.plain_explanation = raw[:500]
        output.tenant_impact = ""

    return output


def simplifyAll(findings: list, model_path: str | None = None) -> list[SLMOutput]:
    """Generate plain-language explanations for all findings."""
    outputs = []
    for finding in findings:
        out = simplifyFinding(
            clause_text=finding.clause_text,
            rationale=finding.rationale,
            risk_level=finding.risk_level,
            statute=finding.statute,
            grounding=finding.grounding,
            clause_type=finding.clause_type,
            model_path=model_path,
        )
        outputs.append(out)
    return outputs
