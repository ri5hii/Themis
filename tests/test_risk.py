"""Tests for the risk engine module."""
from __future__ import annotations

import re

from legalrag.risk.engine import (
    _check_triggers,
    _extract_values,
    analyzeRisk,
)
from legalrag.risk.rules import RULES


class TestCheckTriggers:
    def test_all_match(self):
        triggers = [re.compile(r"\bdeposit\b", re.IGNORECASE)]
        assert _check_triggers("security deposit", triggers) is True

    def test_partial_match_fails(self):
        triggers = [
            re.compile(r"\bdeposit\b", re.IGNORECASE),
            re.compile(r"\bcap\b", re.IGNORECASE),
        ]
        assert _check_triggers("security deposit", triggers) is False

    def test_empty_triggers(self):
        assert _check_triggers("any text", []) is True


class TestExtractValues:
    def test_amount_extraction(self):
        extractors = {"amount": re.compile(r"\$([\d,]+)")}
        values = _extract_values("The fee is $1,234", extractors)
        assert values["amount"] == "1,234"

    def test_no_match(self):
        extractors = {"amount": re.compile(r"\$([\d,]+)")}
        values = _extract_values("no amount here", extractors)
        assert "amount" not in values


class TestRiskRules:
    def test_rule_count(self):
        assert len(RULES) == 27

    def test_rule_ids_unique(self):
        ids = [r.rule_id for r in RULES]
        assert len(ids) == len(set(ids))

    def test_deposit_cap_rule(self):
        rule = next(r for r in RULES if r.rule_id == "deposit.cap_exceeded")
        assert "deposit" in rule.clause_types
        assert "rent" in rule.clause_types
        assert rule.risk_level == "medium"
        assert len(rule.exclusions) == 1

    def test_new_gap_rules_exist(self):
        expected = {
            "access.unrestricted_entry",
            "transaction.registration_costs",
            "reinstatement.as_is_restoration",
            "dispute_resolution.mandatory_arbitration",
            "dispute_resolution.fee_shifting",
            "termination.no_early_exit",
            "termination.automatic",
            "rent.upfront_payment",
            "insurance.tenant_pays_all",
        }
        ids = {r.rule_id for r in RULES}
        assert expected <= ids

    def test_rent_escalation_rule(self):
        rule = next(r for r in RULES if r.rule_id == "rent.excessive_escalation")
        assert "rent" in rule.clause_types
        assert rule.risk_level == "medium"

    def test_guaranty_transfer_rule(self):
        rule = next(r for r in RULES if r.rule_id == "deposit.guaranty_transfer")
        assert rule.risk_level == "low"

    def test_no_offset_rule(self):
        rule = next(r for r in RULES if r.rule_id == "rent.no_offset")
        assert rule.risk_level == "medium"

    def test_uncapped_passthrough_rule(self):
        rule = next(r for r in RULES if r.rule_id == "rent.uncapped_passthrough")
        assert rule.risk_level == "medium"

    def test_jury_waiver_rule(self):
        rule = next(r for r in RULES if r.rule_id == "dispute_resolution.jury_waiver")
        assert rule.risk_level == "info"

    def test_one_way_indemnity_rule(self):
        rule = next(r for r in RULES if r.rule_id == "dispute_resolution.one_way_indemnity")
        assert rule.risk_level == "info"

    def test_contra_proferentem_rule(self):
        rule = next(r for r in RULES if r.rule_id == "dispute_resolution.contra_proferentem")
        assert rule.risk_level == "info"

    def test_liability_disclaim_rule(self):
        rule = next(r for r in RULES if r.rule_id == "maintenance.liability_disclaim")
        assert rule.risk_level == "low"

    def test_weak_warranty_rule(self):
        rule = next(r for r in RULES if r.rule_id == "maintenance.weak_warranty")
        assert rule.risk_level == "info"

    def test_incorporation_by_ref_rule(self):
        rule = next(r for r in RULES if r.rule_id == "maintenance.incorporation_by_ref")
        assert rule.risk_level == "info"

    def test_no_mitigate_rule(self):
        rule = next(r for r in RULES if r.rule_id == "termination.no_mitigate")
        assert rule.risk_level == "low"

    def test_go_dark_rule(self):
        rule = next(r for r in RULES if r.rule_id == "no_obligation.go_dark")
        assert rule.risk_level == "info"


class TestAnalyzeRisk:
    def _make_section(self, text: str, ctype: str, sid: str = "s1") -> dict:
        return {"id": sid, "text": text, "type": ctype, "confidence": 0.8}

    def test_no_findings_on_unknown(self):
        sections = [self._make_section("some text", "unknown")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 0

    def test_deposit_fires(self):
        sections = [self._make_section("security deposit of $10,000", "deposit")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "deposit.cap_exceeded"
        assert result.findings[0].risk_level == "medium"

    def test_deposit_extracts_amount_near_deposit(self):
        sections = [self._make_section(
            "The Security Deposit required under this Lease is $62,000", "deposit"
        )]
        result = analyzeRisk(sections, RULES)
        f = result.findings[0]
        assert f.rule_id == "deposit.cap_exceeded"
        assert f.extracted_values.get("deposit_amount") == "62,000"

    def test_deposit_does_not_extract_net_worth_threshold(self):
        sections = [self._make_section(
            "If Tenant's Guarantor's net worth falls below $5,000,000, "
            "Landlord may require an increase in the Security Deposit",
            "deposit",
        )]
        result = analyzeRisk(sections, RULES)
        f = result.findings[0]
        assert f.rule_id == "deposit.cap_exceeded"
        assert "deposit_amount" not in f.extracted_values

    def test_rent_escalation_fires(self):
        sections = [self._make_section(
            "rent shall increase every year by 5 percent", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "rent.excessive_escalation"

    def test_holdover_fires(self):
        sections = [self._make_section(
            "tenant at sufferance shall pay double rent", "holdover"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "holdover.punitive_rate"

    def test_late_fee_fires(self):
        sections = [self._make_section(
            "late fee of 5 percent within 10 days", "late_fee"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "late_fee.excessive"

    def test_termination_landlord_only_fires(self):
        sections = [self._make_section(
            "landlord may terminate this lease at any time", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "termination.landlord_only"

    def test_sole_discretion_fires(self):
        sections = [self._make_section(
            "reinstatement at landlord sole discretion", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 1
        assert result.findings[0].rule_id == "termination.sole_discretion"

    def test_sole_discretion_fires_on_rent_classified(self):
        sections = [self._make_section(
            "landlord may waive the late fee at its sole discretion", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "termination.sole_discretion" for f in result.findings)

    def test_incorporation_by_ref_fires_on_termination_classified(self):
        sections = [self._make_section(
            "the Special Stipulations Rider attached to this Lease grants "
            "Tenant a one-time termination right",
            "termination",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "maintenance.incorporation_by_ref" for f in result.findings)

    def test_no_offset_fires(self):
        sections = [self._make_section(
            "rent is payable without deduction, offset, or counterclaim", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "rent.no_offset" for f in result.findings)

    def test_no_offset_fires_on_waive_variant(self):
        sections = [self._make_section(
            "tenant hereby waives any right to offset rent", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "rent.no_offset" for f in result.findings)

    def test_uncapped_passthrough_fires(self):
        sections = [self._make_section(
            "tenant shall reimburse landlord for all operating expenses "
            "including CAM taxes and insurance", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "rent.uncapped_passthrough" for f in result.findings)

    def test_change_of_control_fires(self):
        sections = [self._make_section(
            "change of control at 50 percent equity transfer requires consent",
            "subletting",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "subletting.change_of_control" for f in result.findings)

    def test_change_of_control_fires_on_merger(self):
        sections = [self._make_section(
            "tenant may assign this lease in connection with a merger or "
            "sale of substantially all of its assets without consent",
            "subletting",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "subletting.change_of_control" for f in result.findings)

    def test_jury_waiver_fires(self):
        sections = [self._make_section(
            "tenant waives right to jury trial", "dispute_resolution"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "dispute_resolution.jury_waiver" for f in result.findings)

    def test_one_way_indemnity_fires(self):
        sections = [self._make_section(
            "lessee shall indemnify landlord for all claims", "dispute_resolution"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "dispute_resolution.one_way_indemnity" for f in result.findings)

    def test_contra_proferentem_fires(self):
        sections = [self._make_section(
            "any ambiguity shall not be construed against the drafting party",
            "dispute_resolution",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "dispute_resolution.contra_proferentem" for f in result.findings)

    def test_liability_disclaim_fires(self):
        sections = [self._make_section(
            "landlord shall not be liable for theft or vandalism of tenant property",
            "maintenance",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "maintenance.liability_disclaim" for f in result.findings)

    def test_weak_warranty_fires(self):
        sections = [self._make_section(
            "landlord warrants the premises to its knowledge as of the date of this lease",
            "maintenance",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "maintenance.weak_warranty" for f in result.findings)

    def test_weak_warranty_no_fire_on_represented_by_counsel(self):
        sections = [self._make_section(
            "the parties acknowledge this lease was negotiated with each "
            "party represented by legal counsel as of the date first written",
            "maintenance",
        )]
        result = analyzeRisk(sections, RULES)
        assert not any(f.rule_id == "maintenance.weak_warranty" for f in result.findings)

    def test_weak_warranty_fires_on_term_classified_section(self):
        sections = [self._make_section(
            "landlord warrants that, to its knowledge, the premises comply "
            "with building codes as of the Commencement Date",
            "term",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "maintenance.weak_warranty" for f in result.findings)

    def test_incorporation_by_ref_fires(self):
        sections = [self._make_section(
            "HVAC rider is incorporated by reference and made a part hereof",
            "maintenance",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "maintenance.incorporation_by_ref" for f in result.findings)

    def test_no_mitigate_fires(self):
        sections = [self._make_section(
            "landlord has no duty to mitigate damages beyond what applicable law requires",
            "termination",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "termination.no_mitigate" for f in result.findings)

    def test_go_dark_fires(self):
        sections = [self._make_section(
            "tenant has no obligation to operate the premises at any time",
            "no_obligation",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "no_obligation.go_dark" for f in result.findings)

    def test_go_dark_no_fire_on_landlord_relet(self):
        sections = [self._make_section(
            "landlord shall have no obligation to relet the premises "
            "in preference to other available space",
            "no_obligation",
        )]
        result = analyzeRisk(sections, RULES)
        assert not any(f.rule_id == "no_obligation.go_dark" for f in result.findings)

    def test_sorted_by_severity(self):
        sections = [
            self._make_section("security deposit of $10,000", "deposit", "s1"),
            self._make_section("reinstatement at sole discretion", "termination", "s2"),
        ]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 2
        assert result.findings[0].risk_level == "medium"
        assert result.findings[1].risk_level == "low"

    def test_counts(self):
        sections = [self._make_section("security deposit", "deposit")]
        result = analyzeRisk(sections, RULES)
        d = result.toDict()
        assert d["total_sections"] == 1
        assert d["classified_sections"] == 1
        assert d["n_findings"] == 1
        assert d["n_medium"] == 1

    def test_wrong_clause_type_no_fire(self):
        sections = [self._make_section("security deposit", "term")]
        result = analyzeRisk(sections, RULES)
        assert len(result.findings) == 0

    def test_deposit_fires_on_rent_classified_with_letter_of_credit(self):
        sections = [self._make_section(
            "Tenant shall provide a Letter of Credit in the amount of $100,000 "
            "as security for its obligations", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "deposit.cap_exceeded" for f in result.findings)

    def test_deposit_no_fire_on_explicit_waiver(self):
        sections = [self._make_section(
            "No Security Deposit is required of Tenant given its investment "
            "grade credit rating", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert not any(f.rule_id == "deposit.cap_exceeded" for f in result.findings)

    def test_access_unrestricted_entry_fires(self):
        sections = [self._make_section(
            "Landlord shall have the right to enter the premises at any time "
            "without notice for inspection", "access"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "access.unrestricted_entry" for f in result.findings)

    def test_access_no_fire_on_notice_provision(self):
        sections = [self._make_section(
            "Landlord may enter the premises upon 24 hours prior written notice", "access"
        )]
        result = analyzeRisk(sections, RULES)
        assert not any(f.rule_id == "access.unrestricted_entry" for f in result.findings)

    def test_registration_costs_fires(self):
        sections = [self._make_section(
            "Tenant shall bear the stamp duty and registration costs of this Lease",
            "registration",
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "transaction.registration_costs" for f in result.findings)

    def test_reinstatement_as_is_fires(self):
        sections = [self._make_section(
            "Tenant shall restore the premises to its original condition as it "
            "existed prior to the commencement", "maintenance"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "reinstatement.as_is_restoration" for f in result.findings)

    def test_mandatory_arbitration_fires(self):
        sections = [self._make_section(
            "all disputes shall be resolved by binding arbitration in favor of "
            "expedited resolution", "dispute_resolution"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "dispute_resolution.mandatory_arbitration" for f in result.findings)

    def test_fee_shifting_fires(self):
        sections = [self._make_section(
            "Tenant shall pay all of Landlord's attorney's fees and costs in "
            "any dispute", "dispute_resolution"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "dispute_resolution.fee_shifting" for f in result.findings)

    def test_no_early_exit_fires(self):
        sections = [self._make_section(
            "Tenant shall have no right to terminate this Lease prior to the "
            "expiration of the Term", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "termination.no_early_exit" for f in result.findings)

    def test_automatic_termination_fires(self):
        sections = [self._make_section(
            "This Lease shall automatically terminate if Tenant ceases "
            "business operations", "termination"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "termination.automatic" for f in result.findings)

    def test_upfront_payment_fires(self):
        sections = [self._make_section(
            "The full annual rent shall be payable in advance at signing", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "rent.upfront_payment" for f in result.findings)

    def test_upfront_payment_due_in_full_fires(self):
        sections = [self._make_section(
            "Tenant shall pay a flat License Fee of $6,750 for the full Term, "
            "due in full prior to commencement", "rent"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "rent.upfront_payment" for f in result.findings)

    def test_insurance_tenant_pays_fires(self):
        sections = [self._make_section(
            "Tenant shall bear the cost of Landlord's property insurance "
            "premiums", "maintenance"
        )]
        result = analyzeRisk(sections, RULES)
        assert any(f.rule_id == "insurance.tenant_pays_all" for f in result.findings)
