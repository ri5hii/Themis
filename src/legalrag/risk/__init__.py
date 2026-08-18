"""Risk engine: rule-based lease-risk detection with statute grounding."""
from .engine import AnalysisResult, Finding, RiskRule, analyzeRisk
from .grounding import groundAll, groundFinding, loadStatuteChunks
from .rules import RULES

__all__ = [
    "RULES",
    "AnalysisResult",
    "Finding",
    "RiskRule",
    "analyzeRisk",
    "groundAll",
    "groundFinding",
    "loadStatuteChunks",
]
