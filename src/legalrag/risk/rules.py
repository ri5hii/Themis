"""Core risk-detection rules for lease clause analysis.

Each RiskRule defines:
  - clause_types: which clause types to fire on
  - triggers: regex patterns that ALL must match (AND logic)
  - extractors: named patterns to pull numeric values
  - risk_level: severity rating
  - rationale_template: plain-language explanation template
  - statute_query / statute_anchors / statute_fallback: for grounding
"""
from __future__ import annotations

import re

from .engine import RiskRule

# --- Extractors (reusable across rules) ---

# Match monetary amounts: $1,234.56 or Rs. 50,000 or INR 100000
_AMOUNT = r"(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)"

# Match percentage values: 5% or 5 percent or 5.5%
_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"

# Match time periods: 2 months, 30 days, 1 year
_PERIOD = r"(\d+)\s+(month|months|day|days|year|years)"

# Multiplier patterns: double, 150%, 1.5x
_MULTIPLIER_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"
_MULTIPLIER_WORD = r"(?:double|twice|triple|150%|200%)"

# --- Rules ---

RULES: list[RiskRule] = [
    # 1. Deposit cap exceeded (HIGH)
    RiskRule(
        rule_id="deposit.cap_exceeded",
        clause_types=("deposit",),
        triggers=[
            re.compile(r"\bsecurity\s+deposit\b", re.IGNORECASE),
            re.compile(r"\bdeposit\b", re.IGNORECASE),
        ],
        extractors={
            "deposit_amount": re.compile(_AMOUNT, re.IGNORECASE),
        },
        risk_level="high",
        rationale_template=(
            "Security deposit clause detected. Verify deposit does not exceed "
            "statutory cap (MTA s.11: max 2 months rent). "
            "Extracted deposit: {deposit_amount}"
        ),
        statute_query="security deposit cap maximum two months rent",
        statute_anchors=["deposit", "two months", "rent", "not exceed"],
        statute_fallback="MTA 2021 s.11(1)(a)",
    ),

    # 2. Rent excessive escalation (MEDIUM)
    RiskRule(
        rule_id="rent.excessive_escalation",
        clause_types=("rent",),
        triggers=[
            re.compile(r"\b(?:increas\w*|escalat\w*|adjust\w*|revision)\b", re.IGNORECASE),
            re.compile(r"\b(?:every|annual|yearly|per\s+year)\b", re.IGNORECASE),
        ],
        extractors={
            "pct": re.compile(_PCT, re.IGNORECASE),
            "cadence": re.compile(r"every\s+(\d+)\s+(month|months|year|years)", re.IGNORECASE),
        },
        risk_level="medium",
        rationale_template=(
            "Rent escalation clause detected. Verify increase cadence and "
            "percentage are within norms. "
            "Extracted: {pct} every {cadence}"
        ),
        statute_query="rent increase revision percentage limit annual cap",
        statute_anchors=["revision", "rent", "percent"],
        statute_fallback="MTA 2021 s.9",
    ),

    # 3. Holdover punitive rate (MEDIUM)
    RiskRule(
        rule_id="holdover.punitive_rate",
        clause_types=("holdover",),
        triggers=[
            re.compile(r"\bhold\w*\s+over\b|\btenant\s+at\s+sufferance\b", re.IGNORECASE),
        ],
        extractors={
            "multiplier": re.compile(
                r"(\d+(?:\.\d+)?)\s*(?:%|percent)|(double|twice|triple)",
                re.IGNORECASE,
            ),
        },
        risk_level="medium",
        rationale_template=(
            "Holdover penalty clause detected. Verify penalty rate is "
            "reasonable. Extracted multiplier: {multiplier}"
        ),
        statute_query="holdover tenant sufferance rent multiplier penalty",
        statute_anchors=["holdover", "sufferance", "rent"],
        statute_fallback="MTA 2021 s.23",
    ),

    # 4. Late fee excessive (MEDIUM)
    RiskRule(
        rule_id="late_fee.excessive",
        clause_types=("late_fee",),
        triggers=[
            re.compile(r"\blate\s+(?:fee|charge|payment)\b", re.IGNORECASE),
        ],
        extractors={
            "pct": re.compile(_PCT, re.IGNORECASE),
            "days": re.compile(r"within\s+(\d+)\s+days", re.IGNORECASE),
        },
        risk_level="medium",
        rationale_template=(
            "Late fee clause detected. Verify fee rate is within "
            "jurisdictional norms. Extracted: {pct} after {days} days"
        ),
        statute_query="late fee penalty rent maximum percentage",
        statute_anchors=["late fee", "interest", "rent"],
        statute_fallback="general",
    ),

    # 5. Termination landlord-only (LOW)
    RiskRule(
        rule_id="termination.landlord_only",
        clause_types=("termination",),
        triggers=[
            re.compile(r"\blandlord\s+(?:may|shall|can)\s+terminat", re.IGNORECASE),
        ],
        risk_level="low",
        rationale_template=(
            "Termination right appears to be granted to Landlord only. "
            "Check for corresponding Tenant termination right."
        ),
        statute_query="termination right tenant landlord mutual",
        statute_anchors=["termination", "tenant", "landlord"],
        statute_fallback="MTA 2021 s.7",
    ),

    # 6. Termination sole discretion (LOW)
    RiskRule(
        rule_id="termination.sole_discretion",
        clause_types=("termination",),
        triggers=[
            re.compile(r"\bsole\s+discretion\b", re.IGNORECASE),
        ],
        risk_level="low",
        rationale_template=(
            "Clause uses 'sole discretion' language, which gives the landlord "
            "unilateral control. This may limit tenant remedies."
        ),
        statute_query="sole discretion landlord obligation good faith",
        statute_anchors=["sole discretion", "landlord"],
        statute_fallback="general",
    ),
]
