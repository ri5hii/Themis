"""Compare Themis risk findings against Claude/human risk-flag annotations.

Usage:
    python scripts/eval_risk_flags.py [--findings-dir eval/artifacts/themis_run]

Reports per-document recall (Themis flags / annotated flags) and precision
(annotated flags / Themis flags) using keyword-concept matching, split into
in-sample (lease_01-07, used during trigger design) and held-out
(lease_08-22, never seen during trigger design) sets.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

REPO = Path(__file__).resolve().parent.parent
ANNOT_DIR = REPO / "eval" / "claude_test_docs"

FLAG_CONCEPTS = {
    "holdover": re.compile(r"holdover|sufferance|hold\s+over"),
    "deposit": re.compile(r"deposit|letter\s+of\s+credit|security"),
    "sole_discretion": re.compile(r"sole\s+discretion|sole\s+and\s+absolute"),
    "late_fee": re.compile(r"late\s+fe|late\s+charg|late\s+pay|interest"),
    "no_offset": re.compile(r"offset|counterclaim|set-?off|without\s+deduction"),
    "no_mitigate": re.compile(r"mitigat"),
    "jury_waiver": re.compile(r"jury"),
    "termination_landlord": re.compile(
        r"terminat.{0,60}landlord|landlord.{0,60}terminat|unilateral"
    ),
    "go_dark": re.compile(r"go-?dark|no obligation to operate|cease operations|cease business"),
    "one_way_indemnity": re.compile(r"one-?way|no reciprocal.*indemn|indemnif"),
    "change_of_control": re.compile(r"change of control|merger|consolidat|sale of"),
    "guaranty_transfer": re.compile(r"guarant.*(?:transfer|assign)|transfer.*guarant"),
    "rent_escalation": re.compile(r"escalat|increas|index|compounding|step-?up"),
    "uncapped_passthrough": re.compile(r"uncapped|no stated cap|pass-?through|proportionate share"),
    "liability_disclaim": re.compile(r"no liability|disclaim|shall not be liable"),
    "weak_warranty": re.compile(r"warrant.*knowledge|as of the date|to its knowledge"),
    "incorporation_by_ref": re.compile(r"incorporat.*reference|rider|handbook|master lease.*reference"),
    "contra_proferentem": re.compile(r"contra proferentem|construed against|ambiguity"),
    "access_control": re.compile(r"access|advance notice"),
    "arbitration": re.compile(r"arbitrat|mediation"),
    "termination_restriction": re.compile(r"lock-?in|no early exit|no early termination"),
    "upfront_payment": re.compile(r"upfront|in full prior|installment"),
    "registration": re.compile(r"stamp duty|registration|record"),
    "insurance": re.compile(r"insurance|indemnif"),
    "automatic_termination": re.compile(r"automatically terminated|auto-?termination"),
    "reinstatement_window": re.compile(r"reinstatement|45-day|180 days"),
    "fee_shifting": re.compile(r"prevailing party|fee-?shift"),
    "no_renewal": re.compile(r"no renewal"),
}

RULE_TO_CONCEPT = {
    "deposit.cap_exceeded": "deposit",
    "deposit.guaranty_transfer": "guaranty_transfer",
    "rent.excessive_escalation": "rent_escalation",
    "rent.no_offset": "no_offset",
    "rent.uncapped_passthrough": "uncapped_passthrough",
    "holdover.punitive_rate": "holdover",
    "late_fee.excessive": "late_fee",
    "termination.landlord_only": "termination_landlord",
    "termination.sole_discretion": "sole_discretion",
    "subletting.change_of_control": "change_of_control",
    "dispute_resolution.jury_waiver": "jury_waiver",
    "dispute_resolution.one_way_indemnity": "one_way_indemnity",
    "dispute_resolution.contra_proferentem": "contra_proferentem",
    "maintenance.liability_disclaim": "liability_disclaim",
    "maintenance.weak_warranty": "weak_warranty",
    "maintenance.incorporation_by_ref": "incorporation_by_ref",
    "termination.no_mitigate": "no_mitigate",
    "no_obligation.go_dark": "go_dark",
    "access.unrestricted_entry": "access_control",
    "transaction.registration_costs": "registration",
    "reinstatement.as_is_restoration": "reinstatement_window",
    "dispute_resolution.mandatory_arbitration": "arbitration",
    "dispute_resolution.fee_shifting": "fee_shifting",
    "termination.no_early_exit": "termination_restriction",
    "termination.automatic": "automatic_termination",
    "rent.upfront_payment": "upfront_payment",
    "insurance.tenant_pays_all": "insurance",
}


def annot_concepts(flags: list) -> set[str]:
    """Map annotated risk flags to concept keywords."""
    concepts = set()
    for f in flags:
        text = f if isinstance(f, str) else f.get("flag", f.get("description", f.get("text", str(f))))
        for concept, pattern in FLAG_CONCEPTS.items():
            if pattern.search(text.lower()):
                concepts.add(concept)
    return concepts


def themis_concepts(findings: list) -> set[str]:
    """Map Themis findings to concept keywords."""
    concepts = set()
    for f in findings:
        concept = RULE_TO_CONCEPT.get(f["rule_id"])
        if concept:
            concepts.add(concept)
    return concepts


def load_annotations() -> dict[str, dict]:
    """Load all annotation documents (Claude batch 1+2, human batch 1)."""
    docs: dict[str, dict] = {}
    for name in ("claude_inference.json", "claude_inference_08_22.json"):
        path = ANNOT_DIR / name
        if path.exists():
            with open(path) as fh:
                docs_in = json.load(fh)["documents"]
            for doc_name, doc in docs_in.items():
                docs.setdefault(doc_name, {"claude": [], "human": []})
                docs[doc_name]["claude"] = doc.get("risk_flags", [])
    path = ANNOT_DIR / "my_inference.json"
    if path.exists():
        with open(path) as fh:
            docs_in = json.load(fh)["documents"]
        for doc_name, doc in docs_in.items():
            docs.setdefault(doc_name, {"claude": [], "human": []})
            docs[doc_name]["human"] = doc.get("risk_flags", [])
    return docs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--findings-dir", default=str(REPO / "eval" / "artifacts" / "themis_run"))
    args = parser.parse_args()

    findings_dir = Path(args.findings_dir)
    if not findings_dir.exists():
        print(f"[error] findings dir not found: {findings_dir}")
        return 1

    themis: dict[str, list] = {}
    for f in sorted(findings_dir.glob("*_analysis.json")):
        with open(f) as fh:
            themis[f.stem.replace("_analysis", "")] = json.load(fh)["findings"]

    annotations = load_annotations()

    in_sample = {f"lease_{i:02d}" for i in range(1, 8)}
    held_out = {f"lease_{i:02d}" for i in range(8, 23)}

    def prefix(name: str) -> str:
        return name.split("_", 2)[0] + "_" + name.split("_", 2)[1]

    def clean_key(name: str) -> str:
        return name.replace("_SCANNED", "")

    # Aggregate by clean doc name (SCANNED variants merge into base doc)
    all_names = set()
    for name in themis:
        all_names.add(clean_key(name))
    for name in annotations:
        all_names.add(clean_key(name))

    print(f"{'='*78}")
    print("RISK-FLAG EVALUATION: Themis vs Claude+Human annotations")
    print(f"{'='*78}")

    agg = {"in_sample": {"n_annot": 0, "n_caught": 0, "n_themis": 0},
           "held_out": {"n_annot": 0, "n_caught": 0, "n_themis": 0}}
    matched: dict[str, set] = {}

    for name in sorted(all_names):
        base = clean_key(name)
        if base not in annotations:
            continue
        flags = annotations[base]["claude"] + annotations[base]["human"]
        findings = themis.get(base, []) + themis.get(base + "_SCANNED", [])

        annot = annot_concepts(flags)
        pred = themis_concepts(findings)
        caught = annot & pred

        group = "in_sample" if prefix(base) in in_sample else "held_out" if prefix(base) in held_out else None
        if group is None:
            continue

        agg[group]["n_annot"] += len(annot)
        agg[group]["n_caught"] += len(caught)
        agg[group]["n_themis"] += len(pred)
        matched[base] = caught

        recall = f"{len(caught)}/{len(annot)}" if annot else "n/a"
        print(f"{base:42s} R={recall:<8s} P={len(caught)}/{len(pred) if pred else 0}")

    print(f"\n{'='*78}")
    for group, label in (("in_sample", "IN-SAMPLE (lease_01-07, used in design)"),
                         ("held_out", "HELD-OUT (lease_08-22, never seen in design)")):
        a = agg[group]
        recall = a["n_caught"] / a["n_annot"] if a["n_annot"] else float("nan")
        precision = a["n_caught"] / a["n_themis"] if a["n_themis"] else float("nan")
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0
        print(f"{label}")
        print(f"  recall    {a['n_caught']}/{a['n_annot']} = {recall:.3f}")
        print(f"  precision {a['n_caught']}/{a['n_themis']} = {precision:.3f}")
        print(f"  F1        {f1:.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())