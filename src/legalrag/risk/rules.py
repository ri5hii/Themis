"""Risk-detection rules for lease clause analysis.

Each RiskRule defines:
  - clause_types: which clause types to fire on
  - triggers: regex patterns that ALL must match (AND logic)
  - extractors: named patterns to pull numeric values
  - risk_level: severity rating
  - rationale_template: plain-language explanation template
  - statute_query / statute_anchors / statute_fallback: for grounding

Trigger methodology (v0.4.2 redesign, v0.4.3 coverage expansion)
-------------------------------------------------------------
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
  [INDIA]    Indian Registration Act 1908 s.17(1)(d) (leases >1 year must be
             registered); Indian Stamp Act 1899 (stamp duty on leases).

v0.4.3 added rules 19-27 for red-flag concepts identified in the held-out
gap analysis (access notice, stamp-duty registration, as-is reinstatement,
mandatory arbitration, fee shifting, tenant lock-in, automatic termination,
upfront full-term payment, landlord-insurance cost shift) — each grounded in
the sources above. `no_renewal` was NOT added: it is an absence-detection
concern (a lease lacking a renewal clause), which the engine cannot detect
from clause text. Rule exclusions (negative guards) suppress false positives
on explicit waiver language, e.g. "No Security Deposit is required".

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
    # Source: [HARV2018] §5 Security Deposit; [MASSGOV] §B; [NYCGUIDE];
    # [LEIV2020] red flag "guarantee transferable" defines the deposit red
    # flag to include cash deposits AND letters of credit. Deposit clauses
    # frequently co-occur in rent- and late-fee-classified sections (e.g.
    # "Rent and Deposit"), so the rule fires on those types too.
    # Exclusion guard: explicit "No Security Deposit is required" waivers
    # are not deposit risk (false-positive guard, v0.4.3).
    # v0.4.4 trigger mining from the Leivaditi benchmark sentences
    # (data/annotated/leivaditi_redflags.jsonl) added the non-"security
    # deposit" phrasings: "deposit for", "deposit equal to", "as deposit",
    # "lease security", "guaranty money", and amount-vs-rent comparisons.
    RiskRule(
        rule_id="deposit.cap_exceeded",
        clause_types=("deposit", "rent", "late_fee"),
        triggers=[
            re.compile(
                r"\bsecurity\s+deposit\b|\bletter\s+of\s+credit\b"
                r"|\bdeposit\s+(?:of|shall|to|required|payable|held|for|is|equal\s+to|equaling|equal\b)\b"
                r"|\b(?:as\s+a?\s+deposit|lease\s+security|guaranty\s+money)\b"
                r"|\bsecurity\b.{0,60}\b(?:twice|double|equal\w*|equivalent\w*)\b.{0,40}\brent\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        exclusions=[
            re.compile(r"\bno\s+security\s+deposit\b|\bsecurity\s+deposit\s+(?:is\s+)?not\s+required\b",
                       re.IGNORECASE),
        ],
        extractors={
            "deposit_amount": re.compile(
                r"(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?).{0,40}\b(?:security\s+)?deposit\b"
                r"|(?:security\s+)?deposit\b.{0,40}?(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?)"
                r"|(?:\$|Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?).{0,40}\bletter\s+of\s+credit\b",
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
                r"\b(?:without|no|not)\s+(?:deduction|offset|counterclaim|abatement)\b"
                r"|\bwaive\w*\s+.{0,30}\b(?:offset|counterclaim|deduction|abatement)\b",
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
    # CAM / operating expenses guidance. v0.4.4 mining added the utility-
    # and-expense vocabulary from the Leivaditi benchmark sentences
    # ("expenses ... undertaken by the lessee" for power-supply increases).
    RiskRule(
        rule_id="rent.uncapped_passthrough",
        clause_types=("rent", "utilities"),
        triggers=[
            re.compile(
                r"\b(?:operat\w*|cam|common\s+area|maintenance|tax|insurance|expense\w*|utility\w*|service)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:pass[- ]?through|tenant.*share|proportionate|reimburs\w*|undertak\w*|borne\s+by)\b",
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
    # Clause type "term" included: the Leivaditi-benchmark regression
    # (progress.md §7 Fix b) showed holdover sections are frequently
    # classified as "term" by the tie-break order.
    # v0.4.4 trigger mining from the Leivaditi benchmark added
    # penalty-linked surrender drafting ("fails to return ... double
    # rent", "move out ... twice"), the dominant international pattern.
    RiskRule(
        rule_id="holdover.punitive_rate",
        clause_types=("holdover", "term"),
        triggers=[
            re.compile(
                r"\bhold\w*\s+over\b|\btenant\s+at\s+sufferance\b"
                r"|\bfail\w*\s+to\s+vacate\b|\bremain\w*\s+in\s+possession\b"
                r"|\b(?:fails?\s+to\s+return|shall\s+return|move\s+out|does\s+not\s+move\s+out)\b"
                r".{0,60}\b(?:double|twice|\d+(?:\.\d+)?\s*(?:%|percent))\b"
                r".{0,30}\b(?:rent|rental|penalty)\b"
                r"|\bdouble\s+rent\b|\btwice\s+the\s+rent\b",
                re.IGNORECASE | re.DOTALL,
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
    # Interest-on-unpaid-rent drafting without the word "late" is a
    # common late-fee variant ([MASSGOV] §G).
    RiskRule(
        rule_id="late_fee.excessive",
        clause_types=("late_fee",),
        triggers=[
            re.compile(
                r"\blate\s+(?:fee|charge|payment)\b|\binterest.*overdue\b"
                r"|\binterest\s+at\s+\d+\s*(?:%|percent)\b",
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
    # v0.4.4: "lessor" added alongside "landlord" (both appear in
    # published benchmark drafting).
    RiskRule(
        rule_id="termination.landlord_only",
        clause_types=("termination",),
        triggers=[
            re.compile(
                r"\b(?:landlord|lessor)\s+(?:may|shall|can|is\s+entitled\s+to)\s+terminat"
                r"|\b(?:landlord|lessor).*terminat\w*\s+(?:this|the)\s+(?:lease|agreement)\b",
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
                r"\bchange\s+of\s+control\b|\bchange\s+in\s+(?:management|ownership|control)\b"
                r"|\bequity\s+transfer\b"
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
    # Source: [CUAD2021] "Incorporation by Reference" category;
    # [LEIV2020] red flag "special stipulations" (rider supplements and
    # modifies lease provisions). "Rider/addendum/appendix" words alone
    # indicate external terms even without attachment verbs.
    RiskRule(
        rule_id="maintenance.incorporation_by_ref",
        clause_types=("maintenance", "utilities", "termination"),
        triggers=[
            re.compile(
                r"\b(?:incorporat\w*|made\s+a\s+part)\s+.{0,20}\bby\s+reference\b"
                r"|\b(?:rider|schedule|annex|exhibit|addendum|appendix)\b.{0,30}\b(?:attached|enclosed|hereto)\b"
                r"|\b(?:rider|addendum|appendix)\b",
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

    # ── ACCESS ───────────────────────────────────────────────────────

    # 19. Unrestricted landlord entry (LOW)
    # Source: [MASSGOV] §D Access / covenant of quiet enjoyment;
    # [NYCGUIDE] access provisions (tenant should ensure reasonable notice).
    # Tenant risk: landlord may enter at any time without notice.
    RiskRule(
        rule_id="access.unrestricted_entry",
        clause_types=("access", "premises", "maintenance", "utilities"),
        triggers=[
            re.compile(
                r"\b(?:landlord|lessor|owner|landlord'?s?\s+(?:agents?|representatives?|employees?))\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:enter|access|inspect|examine|tour)\w*\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:at\s+any\s+time|at\s+all\s+times|without\s+(?:notice|prior\s+notice|permission|advance\s+notice)|upon\s+demand|at\s+will)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Landlord retains unrestricted access to the premises (any time "
            "or without notice), which can disrupt tenant operations."
        ),
        statute_query="landlord access premises notice inspection quiet enjoyment",
        statute_anchors=["access", "notice", "premises", "landlord"],
        statute_fallback="general",
    ),

    # ── TRANSACTION / COSTS ──────────────────────────────────────────

    # 20. Tenant bears stamp duty + registration (LOW)
    # Source: Indian Registration Act 1908 s.17(1)(d) (leases exceeding one
    # year must be registered); Indian Stamp Act 1899 (stamp duty on lease
    # instruments); standard Indian commercial-leasing drafting practice.
    RiskRule(
        rule_id="transaction.registration_costs",
        clause_types=("registration", "rent", "term", "termination"),
        triggers=[
            re.compile(r"\b(?:stamp\s+duty|registration|registry|notariz\w*)\b", re.IGNORECASE),
            re.compile(r"\b(?:tenant|lessee)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:bear|pay|cost|expense|charge|liab\w*|borne|incurred)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Stamp duty and registration costs are placed on the Tenant. "
            "Verify the applicable duty (stamp duty can be several percent "
            "of rent) is not an outsized, landlord-shifting burden."
        ),
        statute_query="lease registration stamp duty tenant cost",
        statute_anchors=["registration", "stamp duty", "tenant", "cost"],
        statute_fallback="general",
    ),

    # ── REINSTATEMENT ────────────────────────────────────────────────

    # 21. As-is reinstatement obligation (LOW)
    # Source: [LEIV2020] Table 1 red flag "as is reinstatement" (keywords:
    # "as is reinstatement, as it is, restore").
    RiskRule(
        rule_id="reinstatement.as_is_restoration",
        clause_types=("maintenance", "premises", "term", "termination"),
        triggers=[
            re.compile(r"\b(?:restore|restoration|reinstat\w*)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:original\s+condition|as\s+it\s+was|as\s+it\s+existed|prior\s+to|same\s+condition|as\s+is)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Tenant must restore the premises to its original condition, "
            "including ordinary wear-and-tear items. Verify scope of the "
            "restoration obligation."
        ),
        statute_query="reinstatement restoration original condition tenant surrender",
        statute_anchors=["restore", "condition", "tenant"],
        statute_fallback="general",
    ),

    # ── DISPUTE RESOLUTION ───────────────────────────────────────────

    # 22. Mandatory binding arbitration (INFO)
    # Source: [MASSGOV] RE80C13 dispute-resolution guidance; [NYCGUIDE].
    # Binding arbitration waives court access and can shift cost burdens
    # to the tenant.
    RiskRule(
        rule_id="dispute_resolution.mandatory_arbitration",
        clause_types=("dispute_resolution", "termination", "rent"),
        triggers=[
            re.compile(r"\barbitrat\w*\b", re.IGNORECASE),
            re.compile(
                r"\b(?:binding|final|exclusive\s+means|in\s+lieu\s+of|sole\s+remedy)\b"
                r"|\bshall\s+(?:submit|be\s+settled|be\s+resolved)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Disputes must be resolved by binding arbitration, waiving "
            "court access for tenant claims."
        ),
        statute_query="arbitration binding dispute resolution tenant",
        statute_anchors=["arbitration", "binding", "dispute"],
        statute_fallback="general",
    ),

    # 23. One-way fee shifting (INFO)
    # Source: [MASSGOV] RE80C13 attorney-fee provisions; [LEASELENS]
    # fee-shifting guidance. One-way fee shifting exposes the tenant to
    # landlord attorney costs on any dispute.
    RiskRule(
        rule_id="dispute_resolution.fee_shifting",
        clause_types=("dispute_resolution", "rent", "termination"),
        triggers=[
            re.compile(
                r"\b(?:attorney'?s?\s+fees?|attorney\s+costs?|legal\s+fees?|lawyers?\s+fees?|solicitor'?s?\s+fees?)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:tenant|lessee)\b", re.IGNORECASE),
            re.compile(r"\b(?:pay|bear|reimburs\w*|cost|expense)\b", re.IGNORECASE),
        ],
        risk_level="info",
        rationale_template=(
            "Attorney-fee obligation is placed on the Tenant. Verify "
            "fee-shifting is not one-way in the landlord's favor."
        ),
        statute_query="attorney fees prevailing party fee shifting tenant",
        statute_anchors=["fees", "attorney", "tenant"],
        statute_fallback="general",
    ),

    # ── TERMINATION ──────────────────────────────────────────────────

    # 24. Tenant lock-in / no early exit (LOW)
    # Source: [MASSGOV] RE80C13 early-termination guidance; [NYCGUIDE]
    # termination-rights guidance. A lease with no tenant exit right for
    # the full term is a lock-in risk.
    RiskRule(
        rule_id="termination.no_early_exit",
        clause_types=("termination", "rent", "term"),
        triggers=[
            re.compile(r"\b(?:tenant|lessee)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:no\s+right|may\s+not|shall\s+not|is\s+not\s+entitled|has\s+no\s+right|without\s+the\s+right)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:terminat\w*|cancel\w*|break)\b", re.IGNORECASE),
        ],
        risk_level="low",
        rationale_template=(
            "Tenant has no early-termination right for the lease term, "
            "creating a lock-in risk."
        ),
        statute_query="tenant early termination right lock in",
        statute_anchors=["termination", "tenant", "right"],
        statute_fallback="general",
    ),

    # 25. Automatic termination (INFO)
    # Source: [LEIV2020] red flag "termination" / "break option";
    # [NYCGUIDE] termination drafting. Automatic termination on an event
    # (e.g. cessation of business) removes tenant tenure security.
    RiskRule(
        rule_id="termination.automatic",
        clause_types=("termination", "term"),
        triggers=[
            re.compile(
                r"\b(?:shall\s+)?(?:be\s+deemed\s+)?(?:automatically\s+terminat\w*"
                r"|automatically\s+expire|terminat\w*\s+automatically|expire\s+automatically)\b",
                re.IGNORECASE,
            ),
        ],
        risk_level="info",
        rationale_template=(
            "Lease terminates automatically upon a stated event, removing "
            "tenant tenure security. Verify the triggering event."
        ),
        statute_query="automatic termination event lease tenant",
        statute_anchors=["termination", "automatic"],
        statute_fallback="general",
    ),

    # ── RENT ─────────────────────────────────────────────────────────

    # 26. Upfront full-term payment (LOW)
    # Source: [NYCGUIDE] prepaid-rent guidance; [HARV2018] §4 Rent.
    # Full-term rent due in advance at signing is an outsized cash
    # commitment risk. Two drafting patterns: "full rent payable in
    # advance" and "rent/license fee due in full prior to commencement".
    RiskRule(
        rule_id="rent.upfront_payment",
        clause_types=("rent", "deposit"),
        triggers=[
            re.compile(
                r"\b(?:full|entire|whole)\s+(?:rent|rental|annual\s+rent)\b.{0,60}\b"
                r"(?:in\s+advance|prior\s+to|at\s+signing|upon\s+execution|before\s+possession)\b"
                r"|\b(?:rent|rental|license\s+fee|payment|installment)\b.{0,60}\b"
                r"(?:due\s+in\s+full|payable\s+in\s+full)\b.{0,40}\b"
                r"(?:prior\s+to|before|in\s+advance|at\s+signing|upon\s+execution)\b",
                re.IGNORECASE | re.DOTALL,
            ),
        ],
        risk_level="low",
        rationale_template=(
            "Full rent is payable in advance at signing, an outsized "
            "upfront cash commitment for the tenant."
        ),
        statute_query="rent advance payment full term upfront tenant",
        statute_anchors=["rent", "advance", "payment"],
        statute_fallback="general",
    ),

    # ── INSURANCE ────────────────────────────────────────────────────

    # 27. Tenant pays landlord's insurance (INFO)
    # Source: [MASSGOV] RE80C13 insurance provisions; [NYCGUIDE].
    # Tenant bearing landlord's insurance premiums is a cost-shift risk.
    RiskRule(
        rule_id="insurance.tenant_pays_all",
        clause_types=("maintenance", "premises", "rent", "utilities"),
        triggers=[
            re.compile(r"\binsurance\b", re.IGNORECASE),
            re.compile(r"\b(?:landlord'?s?|owner'?s?|lessor'?s?)\b", re.IGNORECASE),
            re.compile(r"\b(?:cost|expense|premium|reimburs\w*|bear|pay\w*)\b", re.IGNORECASE),
        ],
        risk_level="info",
        rationale_template=(
            "Tenant bears costs for the Landlord's insurance. Verify the "
            "insurance-cost allocation is not one-sided."
        ),
        statute_query="insurance premiums tenant landlord cost allocation",
        statute_anchors=["insurance", "tenant", "cost"],
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