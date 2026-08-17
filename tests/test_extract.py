# Unit tests for the fast-lane regex clause classifier.
from __future__ import annotations

from legalrag.extract import fast_lane
from legalrag.extract.taxonomy import TAXONOMY, UNKNOWN, taxonomyIndex


# §5.8: every trigger set maps to a taxonomy type (import-time guard).
def test_trigger_keys_are_taxonomy_types() -> None:
    for t in fast_lane._TRIGGERS:
        assert t in TAXONOMY


def test_taxonomy_order_is_stable_spec() -> None:
    assert TAXONOMY == tuple(dict.fromkeys(TAXONOMY))  # no dupes
    assert taxonomyIndex("deposit") < taxonomyIndex("termination")  # §5.14 tie order
    assert taxonomyIndex(UNKNOWN) == len(TAXONOMY)  # unknown sorts last


def test_confidence_from_count_bounds() -> None:
    # Evidence-count -> [0, 1] mapping: monotone, 0 for no evidence, one
    # trigger fires at 0.5 (comparable with classifier softmax probability).
    assert fast_lane.confidenceFromCount(0) == 0.0
    assert fast_lane.confidenceFromCount(1) == 0.5
    assert fast_lane.confidenceFromCount(2) == 0.75
    assert fast_lane.confidenceFromCount(-3) == 0.0
    assert fast_lane.confidenceFromCount(10) > fast_lane.confidenceFromCount(3)
    assert all(0.0 <= fast_lane.confidenceFromCount(n) < 1.0 for n in range(20))


# §5.14: a termination clause that says "refund the security deposit" must
# classify as termination, not deposit (evidence count beats the refund link).
def test_refund_overmatch_resolved_by_evidence_count() -> None:
    text = (
        "Clause 9. Termination. The Lessor shall refund the security deposit, "
        "but shall have the right to terminate this lease upon failure to pay rent."
    )
    t, count = fast_lane.classifyClause(text)
    assert t == "termination"
    assert count >= 2


# §5.15: trigger inflection must not be a hard miss (OOD phrasings).
def test_access_inflection_variants() -> None:
    for text in (
        "Landlord may enter the premises upon notice",
        "Landlord shall be permitted access, entering the premises at reasonable hours",
        "repairs and inspections require notice",
        "inspection and repairs require notice",
        "inspect the premises upon reasonable notice",
    ):
        t, _ = fast_lane.classifyClause(text)
        assert t == "access", f"{text!r} -> {t}"


def test_term_license_synonym() -> None:
    t, _ = fast_lane.classifyClause("The term of this license shall be 11 months")
    assert t == "term"


def test_holdover_double_and_twice_rent() -> None:
    for text in (
        "tenant holding over shall pay double the monthly rent",
        "tenant holding over shall pay twice the monthly rent",
        "in the event of holdover the tenant shall pay double rent",
    ):
        t, count = fast_lane.classifyClause(text)
        assert t == "holdover", f"{text!r} -> {t}"
        assert count >= 1


def test_holdover_wins_tie_over_term_on_sufferance_clause() -> None:
    # claudeTestDocs lease_03: a holdover article with "tenant at sufferance"
    # and 150% rent also matches term (expiration/the term).  With _TYPE_WEIGHT
    # (term=0.5, §5.14 fix b), holdover's higher specificity wins the tie
    # instead of term's taxonomy-index advantage.
    text = (
        "Should Lessee remain in possession of the Premises after the expiration "
        "of the Term without Lessor's written consent, Lessee shall be deemed a "
        "tenant at sufferance and shall pay Rent equal to one hundred fifty "
        "percent (150%) of the Base Rent in effect immediately prior to expiration."
    )
    counts = fast_lane.evidenceCounts(text)
    assert counts["holdover"] >= 1
    assert counts["holdover"] == counts["term"] >= 1
    t, _ = fast_lane.classifyClause(text)
    assert t == "holdover"


# §5.8: premises must use distinctive phrases, not the generic word "premises".
def test_premises_distinctive_phrases() -> None:
    for text in (
        "The flat no. 42, described as the demised premises, is leased to tenant",
        "The premises described as Suite 100 are leased to the Tenant",
    ):
        t, _ = fast_lane.classifyClause(text)
        assert t == "premises", f"{text!r} -> {t}"


def test_generic_premises_not_enough() -> None:
    # Mentions "premises" everywhere but no distinctive phrase -> not premises.
    t, _ = fast_lane.classifyClause("the premises and the building and the premises again")
    assert t != "premises"


# No evidence -> UNKNOWN (the classifier-fallback handoff signal).
def test_unknown_when_no_evidence() -> None:
    t, count = fast_lane.classifyClause("THIS LEASE made this day between Landlord and Tenant.")
    assert t == UNKNOWN
    assert count == 0


def test_evidence_counts_all_taxonomy_keys() -> None:
    counts = fast_lane.evidenceCounts("the tenant shall sublet the premises")
    assert set(counts) == set(TAXONOMY)
    assert counts["subletting"] >= 1


def test_pets() -> None:
    t, _ = fast_lane.classifyClause("no pets allowed in the premises without consent")
    assert t == "pets"


def test_utilities() -> None:
    t, _ = fast_lane.classifyClause("tenant shall pay for water, gas and electricity charges")
    assert t == "utilities"
