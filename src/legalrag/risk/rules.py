"""Risk-detection rules for lease clause analysis.

Each RiskRule defines:
  - clause_types: which clause types to fire on
  - triggers: regex patterns that ALL must match (AND logic)
  - extractors: named patterns to pull numeric values
  - risk_level: severity rating
  - rationale_template: plain-language explanation template
  - statute_query / statute_anchors / statute_fallback: for grounding

Trigger methodology (v0.4.2 redesign)
-------------------------------------
Trigger patterns are derived from published lease-risk taxonomies and
practitioner guidance rather than from inspection of the test corpus:

  [LEIV2020] Leivaditi, Rossi & Kanoulas, "A Benchmark for Lease Contract
             Review", arXiv:2010.10386. Defines 19 lease red-flag types with
             retrieval keyword queries (Table 1).
             https://arxiv.org/abs/2010.10386
  [CUAD2021] Hendrycks et al., "CUAD: An Expert-Annotated NLP Dataset for
             Legal Contract Review", NeurIPS 2021. 41 clause categories.
             https://arxiv.org/abs/2103.06228
  [HARV2018] Harvard Law School Transactional Law Clinics, "Commercial
             Leases 101 Legal Toolkit" (28 lease provisions, tenant focus).
             http://clinics.law.harvard.edu/tlc/files/2018/12/Commercial-Leases-101-Legal-Toolkit.pdf
  [LEASELENS] LeaseLens, "12 Commercial Lease Red Flags Tenants Should Never
             Ignore" and companion negotiation guides.
             https://leaselens.org/blog/commercial-lease-red-flags
  [MASSGOV]  Mass.gov RE80C13, "Commercial Lease Clauses of Tenant Concerns:
             Part III". https://www.mass.gov/info-details/re80c13-commercial-lease-clauses-of-tenant-concerns-part-iii
  [NYCGUIDE] NYC SBS, "Commercial Leasing" guide for tenants.
             https://www.nyc.gov/assets/sbs/downloads/pdf/about/reports/commercial-lease-guide-accessible.pdf

Severity is a screening signal only: rules flag clauses for human review,
they do not compute statutory compliance.
"""
from __future__ import annotations

import re

from .engine import RiskRule

# --- Extractors (reusable across rules) ---

_AMOUNT = r"(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)"
_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"

# --- Rules ---

RULES: list[RiskRule] = [

    # ── DEPOSIT ──────────────────────────────────────────────────────

    # 1. Deposit cap exceeded (MEDIUM)
    # Source: [HARV2018] §5 Security Deposit; [MASSGOV] §B; [NYCGUIDE].
    # Deposit amounts are only extracted when a currency value appears
    # within 40 chars of the word "deposit" so thresholds (e.g. net-worth
    # floors) are not mistaken for deposit amounts.
    RiskRule(
        rule_id="deposit.cap_exceeded",
        clause_types=("deposit",),
        triggers=[
            re.compile(r"\bsecurity\s+deposit\b|\bdeposit\s+(?:of|shall|to)\b", re.IGNORECASE),
        ],
        extractors={
            "deposit_amount": re.compile(
                r"(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?).{0,40}\b(?:security\s+)?deposit\b"
                r"|(?:security\s+)?deposit\b.{0,40}?(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)",
                re.IGNORECASE | re.DOTALL,
            ),
        },
        risk_level="medium",
        rationale_template=(
            "Security deposit clause detected. Verify deposit does not "
            "exceed statutory cap (MTA s.11: max 2 months rent). "
            "Extracted: {deposit_amount}"
        ),
        statute_query="security deposit cap maximum two months rent",
        statute_anchors=["deposit", "two months", "rent", "not exceed"],
        statute_fallback="MTA 2021 s.11(1)(a)",
    ),

    # 2. Guaranty auto-transfer (LOW)
    # Source: [LEIV2020] red flag "guarantee transferable" (Table 1
    # keywords: "non transferable security", "guarantee").
    RiskRule(
        rule_id="deposit.guaranty_transfer",
        clause_types=("deposit",),
        triggers=[
            re.compile(r"\bguarant\w*\b", re.IGNORECASE),
            re.compile(r"\btransfer\w*|\bassign\w*|\bsuccess\w*\b", re.IGNORECASE),
        ],
        risk_level="low",
        rationale_template=(
            "Guaranty appears transferable to assignees or successors "
            "without requiring a new guaranty instrument."
        ),
        statute_query="guaranty assignment transfer successor",
        statute_anchors=["guaranty", "assign", "transfer"],
        statute_fallback="general",
    ),

    # ── RENT ─────────────────────────────────────────────────────────

    # 3. Rent excessive escalation (MEDIUM)
    # Source: [LEIV2020] red flag "indexation" (keywords: "indexation,
    # index, price increase"); [HARV2018] §6 Rent Adjustment.
    RiskRule(
        rule_id="rent.excessive_escalation",
        clause_types=("rent",),
        triggers=[
            re.compile(
                r"\b(?:increas\w*|escalat\w*|adjust\w*|revision|revis\w*|index\w*)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:every|annual|yearly|per\s+year|each\s+year|compounding|year-over-year|per\s+annum)\b",
                re.IGNORECASE,
            ),
        ],
        extractors={
            "pct": re.compile(_PCT, re.IGNORECASE),
            "cadence": re.compile(
                r"every\s+(\d+)\s+(month|months|year|years)", re.IGNORECASE
            ),
        },
        risk_level="medium",
        rationale_template=(
            "Rent escalation clause detected. Verify increase cadence and "
            "percentage are within norms. Extracted: {pct}"
        ),
        statute_query="rent increase revision percentage limit annual cap",
        statute_anchors=["revision", "rent", "percent"],
        statute_fallback="MTA 2021 s.9",
    ),

    # 4. Rent no-offset waiver (MEDIUM)
    # Source: [HARV2018] §19 Default (tenant set-off right) and §7.1
    # offset/counterclaim language; [LEASELENS] mitigation of damages.
    RiskRule(
        rule_id="rent.no_offset",
        clause_types=("rent",),
        triggers=[
            re.compile(
                r"\b(?:without|no|not)\s+(?:deduction|offset|counterclaim)\b"
                r"|\bwaive\w*\s+.{0,30}\b(?:offset|counterclaim|deduction)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="medium",
        rationale_template=(
            "Tenant waives right to offset or counterclaim against rent, "
            "removing a key self-help remedy for landlord non-performance."
        ),
        statute_query="rent offset counterclaim waiver tenant remedy",
        statute_anchors=["rent", "offset", "counterclaim"],
        statute_fallback="general",
    ),

    # 5. Uncapped operating expense passthrough (MEDIUM)
    # Source: [LEIV2020] red flag "service charges"; [LEASELENS] uncapped
    # CAM / operating expenses guidance.
    RiskRule(
        rule_id="rent.uncapped_passthrough",
        clause_types=("rent", "utilities"),
        triggers=[
            re.compile(
                r"\b(?:operat\w*|cam|common\s+area|maintenance|tax|insurance)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:pass[- ]?through|tenant.*share|proportionate|reimburs\w*)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="medium",
        rationale_template=(
            "Operating expense passthrough has no stated annual cap, "
            "exposing tenant to uncapped cost growth."
        ),
        statute_query="operating expense cap annual limit pass-through",
        statute_anchors=["operating", "expense", "cap", "annual"],
        statute_fallback="general",
    ),

    # ── HOLDOVER ─────────────────────────────────────────────────────

    # 6. Holdover punitive rate (MEDIUM)
    # Source: [LEASELENS] red flag #2 (holdover at 200% with no cure);
    # [NYCGUIDE] holdover definitions. Market range 110-200%.
    RiskRule(
        rule_id="holdover.punitive_rate",
        clause_types=("holdover",),
        triggers=[
            re.compile(
                r"\bhold\w*\s+over\b|\btenant\s+at\s+sufferance\b"
                r"|\bfail\w*\s+to\s+vacate\b|\bremain\w*\s+in\s+possession\b",
                re.IGNORECASE,
            ),
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

    # ── LATE FEE ─────────────────────────────────────────────────────

    # 7. Late fee excessive (MEDIUM)
    # Source: [MASSGOV] §G Late Rent Payments; [HARV2018] §4.
    RiskRule(
        rule_id="late_fee.excessive",
        clause_types=("late_fee",),
        triggers=[
            re.compile(
                r"\blate\s+(?:fee|charge|payment)\b|\binterest.*overdue\b",
                re.IGNORECASE,
            ),
        ],
        extractors={
            "pct": re.compile(_PCT, re.IGNORECASE),
            "days": re.compile(r"within\s+(\d+)\s+days", re.IGNORECASE),
        },
        risk_level="medium",
        rationale_template=(
            "Late fee clause detected. Verify fee rate is within "
            "jurisdictional norms. Extracted: {pct}"
        ),
        statute_query="late fee penalty rent maximum percentage",
        statute_anchors=["late fee", "interest", "rent"],
        statute_fallback="general",
    ),

    # ── TERMINATION ──────────────────────────────────────────────────

    # 8. Termination landlord-only (LOW)
    # Source: [LEIV2020] red flag "termination" (keywords: "termination,
    # limit of, finality"); [MASSGOV] §G / [NYCGUIDE] termination rights.
    RiskRule(
        rule_id="termination.landlord_only",
        clause_types=("termination",),
        triggers=[
            re.compile(
                r"\blandlord\s+(?:may|shall|can|is\s+entitled\s+to)\s+terminat"
                r"|\blandlord.*terminat\w*\s+(?:this|the)\s+(?:lease|agreement)\b",
                re.IGNORECASE,
            ),
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

    # 9. Termination sole discretion (LOW)
    # Source: [LEASELENS] sole-discretion subletting/assignment guidance;
    # Seattle Wayfind lease toolkit ("sole and absolute discretion").
    # Fires on rent-classified sections too: discretion language often
    # appears in rent/payment contexts.
    RiskRule(
        rule_id="termination.sole_discretion",
        clause_types=("termination", "rent"),
        triggers=[
            re.compile(r"\bsole\s+discretion\b", re.IGNORECASE),
        ],
        risk_level="low",
        rationale_template=(
            "Clause uses 'sole discretion' language, which gives the "
            "landlord unilateral control. This may limit tenant remedies."
        ),
        statute_query="sole discretion landlord obligation good faith",
        statute_anchors=["sole discretion", "landlord"],
        statute_fallback="general",
    ),

    # ── SUBLETTING / CHANGE OF CONTROL ───────────────────────────────

    # 10. Change of control restrictive (LOW)
    # Source: [LEIV2020] red flag "change of control" (keywords: "ownership,
    # change in management, change of lessor"); [CUAD2021] "Change of
    # Control" category (merger / restructuring / control transfer).
    RiskRule(
        rule_id="subletting.change_of_control",
        clause_types=("subletting",),
        triggers=[
            re.compile(
                r"\bchange\s+of\s+control\b|\bequity\s+transfer\b"
                r"|\bvoting\s+(?:interest|threshold|percent)\b"
                r"|\bmerger\b|\bconsolidat\w*\b"
                r"|\bsale\s+of\s+(?:substantially\s+)?all\b",
                re.IGNORECASE,
            ),
        ],
        extractors={
            "threshold": re.compile(r"(\d+)\s*%", re.IGNORECASE),
        },
        risk_level="low",
        rationale_template=(
            "Change-of-control consent provision detected. Verify threshold "
            "is reasonable. Extracted threshold: {threshold}"
        ),
        statute_query="change of control consent equity threshold assignment",
        statute_anchors=["change", "control", "consent", "assignment"],
        statute_fallback="general",
    ),

    # ── DISPUTE RESOLUTION ───────────────────────────────────────────

    # 11. Jury trial waiver (INFO)
    # Source: [CUAD2021] "Jury Trial Waiver" category; [NYCGUIDE].
    RiskRule(
        rule_id="dispute_resolution.jury_waiver",
        clause_types=("dispute_resolution",),
        triggers=[
            re.compile(
                r"\b(?:waive|waiver|right\s+to)\s+.{0,20}\bjury\b"
                r"|\bjury\s+(?:trial|verdict|right)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Tenant waives jury trial right, removing access to jury "
            "in lease-related litigation."
        ),
        statute_query="jury trial waiver tenant right lease",
        statute_anchors=["jury", "trial", "waive"],
        statute_fallback="general",
    ),

    # 12. One-way indemnity (INFO)
    # Source: [HARV2018] §23 Indemnification (mutuality guidance).
    RiskRule(
        rule_id="dispute_resolution.one_way_indemnity",
        clause_types=("dispute_resolution",),
        triggers=[
            re.compile(r"\bindemn\w*\b", re.IGNORECASE),
            re.compile(
                r"\b(?:lessee|tenant)\s+(?:shall|will|agrees?\s+to)\s+indemnif"
                r"|\b(?:runs?\s+from|one[- ]?way)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Indemnification obligation appears to run one-way from tenant "
            "to landlord with no reciprocal landlord indemnity."
        ),
        statute_query="indemnity mutual obligation landlord tenant",
        statute_anchors=["indemnity", "tenant", "landlord"],
        statute_fallback="general",
    ),

    # 13. Contra proferentem waiver (INFO)
    # Source: [CUAD2021] "Contra Proferentem Clause" category; general
    # contract-interpretation doctrine (ambiguity construed against
    # drafting party).
    RiskRule(
        rule_id="dispute_resolution.contra_proferentem",
        clause_types=("dispute_resolution",),
        triggers=[
            re.compile(
                r"\bcontra\s+proferentem\b"
                r"|\bambiguit\w*.{0,40}\bconstrued\s+against"
                r"|\bnot\s+construed\s+against\s+(?:the\s+)?drafting\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Clause waives the contra proferentem interpretation rule, "
            "removing a tenant-protective ambiguity resolution principle."
        ),
        statute_query="contra proferentem ambiguity drafting party interpretation",
        statute_anchors=["ambiguity", "construed", "drafting"],
        statute_fallback="general",
    ),

    # ── MAINTENANCE / PREMISES ───────────────────────────────────────

    # 14. Liability disclaimer (LOW)
    # Source: [HARV2018] §26 Liability of Owner; [MASSGOV].
    RiskRule(
        rule_id="maintenance.liability_disclaim",
        clause_types=("maintenance", "premises"),
        triggers=[
            re.compile(r"\bliab\w*\b", re.IGNORECASE),
            re.compile(
                r"\b(?:not|no|shall\s+not)\b.{0,40}\b(?:theft|vandal\w*|loss|damage)\b"
                r"|\b(?:theft|vandal\w*|loss)\b.{0,40}\b(?:not|no)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Landlord disclaims liability for theft or vandalism of tenant "
            "property, shifting security-related loss risk entirely to tenant."
        ),
        statute_query="landlord liability tenant property theft vandalism",
        statute_anchors=["liability", "tenant", "property"],
        statute_fallback="general",
    ),

    # 15. Weak warranty (INFO)
    # Source: [LEIV2020] red flag "warranties of the owner" (keywords:
    # "warranties"). Trigger 1 requires the word "warrant*" itself —
    # "represented by counsel" is not a warranty (false-positive guard).
    # Clause types include "term" because warranty sections frequently
    # mention commencement dates and are classified as term clauses.
    RiskRule(
        rule_id="maintenance.weak_warranty",
        clause_types=("maintenance", "premises", "term"),
        triggers=[
            re.compile(r"\bwarrant\w*\b", re.IGNORECASE),
            re.compile(r"\bto\s+its\s+knowledge\b|\bas\s+of\s+the\s+date\b", re.IGNORECASE),
        ],
        risk_level="info",
        rationale_template=(
            "Warranty or representation is qualified 'to its knowledge' or "
            "'as of the date', materially weakening the protection."
        ),
        statute_query="warranty knowledge qualifier landlord representation",
        statute_anchors=["warranty", "knowledge", "landlord"],
        statute_fallback="general",
    ),

    # 16. Incorporation by reference (INFO)
    # Source: [CUAD2021] "Incorporation by Reference" category.
    # Clause types include "termination" because rider clauses often
    # grant termination rights and are classified as termination.
    RiskRule(
        rule_id="maintenance.incorporation_by_ref",
        clause_types=("maintenance", "utilities", "termination"),
        triggers=[
            re.compile(
                r"\b(?:incorporat\w*|made\s+a\s+part)\s+.{0,20}\bby\s+reference\b"
                r"|\b(?:rider|schedule|annex|exhibit)\b.{0,30}\b(?:attached|enclosed|hereto)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Terms are incorporated by reference without reproduction in "
            "the document, creating a blind spot for review."
        ),
        statute_query="incorporation by reference terms schedule rider",
        statute_anchors=["incorporated", "reference", "schedule"],
        statute_fallback="general",
    ),

    # ── ACCESS ───────────────────────────────────────────────────────

    # 17. No duty to mitigate (LOW)
    # Source: [HARV2018] §34 Mitigation of Damages; [LEASELENS]
    # mitigation-of-damages guidance.
    RiskRule(
        rule_id="termination.no_mitigate",
        clause_types=("termination", "rent"),
        triggers=[
            re.compile(
                r"\bno\s+(?:duty|obligation)\s+to\s+mitigat"
                r"|\bnot\s+(?:required|obligated)\s+to\s+mitigat"
                r"|\bmitigat\w*\s+.{0,30}\b(?:beyond|greater\s+than)\s+.{0,20}\blaw\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Landlord has no duty to mitigate damages beyond what law "
            "requires, increasing tenant exposure on default."
        ),
        statute_query="duty to mitigate damages landlord tenant default",
        statute_anchors=["mitigate", "damages", "landlord"],
        statute_fallback="general",
    ),

    # ── NO OBLIGATION ────────────────────────────────────────────────

    # 18. Go-dark right (INFO)
    # Source: [LEIV2020] red flag "no obligation to operate" (keywords:
    # "no obligation to operate, no commitment obligations"). The verb
    # list excludes "relet": landlord reletting duties are not a tenant
    # go-dark right (false-positive guard).
    RiskRule(
        rule_id="no_obligation.go_dark",
        clause_types=("no_obligation",),
        triggers=[
            re.compile(
                r"\bno\s+(?:obligation|duty|requirement)\s+to\s+(?:operate|conduct|open|maintain)\b"
                r"|\bmay\s+cease\s+(?:operations?|business|conducting)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Tenant may cease operations without breach. Verify this does "
            "not conflict with co-tenancy or percentage-rent provisions."
        ),
        statute_query="no obligation operate tenant cease operations",
        statute_anchors=["obligation", "operate", "tenant"],
        statute_fallback="general",
    ),
]