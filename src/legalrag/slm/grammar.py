"""GBNF grammar for constrained SLM output.

The grammar guarantees valid JSON with engine-authoritative fields
(clause_type, risk_level, statute) and SLM-generated prose fields
(plain_explanation, tenant_impact). The engine stamps authoritative
fields post-inference.
"""
from __future__ import annotations

# Valid clause types (must match taxonomy.py)
CLAUSE_TYPES = (
    '"term"', '"rent"', '"deposit"', '"maintenance"', '"utilities"',
    '"termination"', '"holdover"', '"subletting"', '"access"', '"late_fee"',
    '"registration"', '"dispute_resolution"', '"premises"', '"pets"',
    '"entire_agreement"', '"no_obligation"',
)

# Valid risk levels
RISK_LEVELS = ('"high"', '"medium"', '"low"', '"info"')

# Build the grammar dynamically from the valid values
_clause_type_alt = " | ".join(CLAUSE_TYPES)
_risk_level_alt = " | ".join(RISK_LEVELS)

GRAMMAR = f"""
root   ::= object
object ::= "{{" ws
           "\"clause_type\"" ws ":" ws ctype ws ","
           ws "\"risk_level\"" ws ":" ws rlevel ws ","
           ws "\"statute\"" ws ":" ws string ws ","
           ws "\"plain_explanation\"" ws ":" ws string ws ","
           ws "\"tenant_impact\"" ws ":" ws string ws
           "}}"
ctype  ::= {_clause_type_alt}
rlevel ::= {_risk_level_alt}
string ::= "\\"" chars+ "\\""
chars  ::= [^"\\\\] |\\
           "\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
ws     ::= [ \\t\\n]*
"""

SYSTEM_PROMPT = (
    "You review a clause from a lease contract and rewrite it in plain "
    "language for a non-lawyer tenant. You produce a JSON object with: "
    "clause_type (the type of clause), risk_level (high/medium/low/info), "
    "statute (the applicable statute reference), plain_explanation (plain "
    "language explanation), and tenant_impact (how this affects the tenant "
    "in terms of costs, restrictions, or rights)."
)


def make_finding_prompt(
    clause_text: str,
    rationale: str,
    risk_level: str,
    statute: str,
    grounding: str = "",
) -> str:
    """Build the user prompt for a single finding."""
    parts = [f"Lease clause:\n{clause_text[:600]}"]
    if statute:
        parts.append(f"\nStatute reference:\n{statute}")
    if grounding:
        parts.append(f"\n{grounding[:900]}")
    parts.append(f"\nRisk: {risk_level}. {rationale}")
    parts.append("\nRewrite in plain language for the tenant.")
    return "\n".join(parts)
