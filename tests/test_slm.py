"""SLM grammar tests: GBNF must compile and enforce quoted-JSON output."""
from __future__ import annotations

import re

from llama_cpp import LlamaGrammar

from legalrag.slm.grammar import GRAMMAR


def test_grammar_compiles() -> None:
    LlamaGrammar.from_string(GRAMMAR)


def test_grammar_keys_are_quoted_literals() -> None:
    for key in ("clause_type", "risk_level", "statute", "plain_explanation", "tenant_impact"):
        assert f'"\\"{key}\\""' in GRAMMAR, f"key {key!r} not quoted in GBNF"


def test_grammar_values_are_quoted_literals() -> None:
    for value in ("term", "rent", "high", "medium", "info"):
        assert f'"\\"{value}\\""' in GRAMMAR, f"value {value!r} not quoted in GBNF"


def test_grammar_is_single_line_rules() -> None:
    for line in GRAMMAR.splitlines():
        assert "::=" not in line or line.count("::=") == 1
        assert re.match(r"^\S+.*::=.*$", line), f"malformed rule line: {line!r}"


def test_grammar_object_rule_orders_keys_exactly() -> None:
    """The object rule must emit the five keys once, in schema order."""
    obj_line = next(line for line in GRAMMAR.splitlines() if line.startswith("object"))
    keys = ["clause_type", "risk_level", "statute", "plain_explanation", "tenant_impact"]
    pos = -1
    for key in keys:
        lit = '\\"' + key + '\\"'
        found = obj_line.find(lit, pos + 1)
        assert found > pos, f"key {key!r} missing or out of order in object rule"
        pos = found
