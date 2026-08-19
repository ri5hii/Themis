"""SLM module: grammar-constrained plain-language generation."""
from .generate import SLMOutput, simplifyAll, simplifyFinding
from .grammar import GRAMMAR, SYSTEM_PROMPT, make_finding_prompt

__all__ = [
    "GRAMMAR",
    "SYSTEM_PROMPT",
    "SLMOutput",
    "make_finding_prompt",
    "simplifyAll",
    "simplifyFinding",
]
