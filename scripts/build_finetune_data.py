"""Build aligned finding→explanation pairs for SLM LoRA fine-tuning.

Takes risk engine findings from labeled lease documents and produces
training data in the format: user message → corrected JSON with engine-
authoritative fields (clause_type, risk_level, statute) stamped over the
SLM's self-reported values.

Usage:
    python scripts/build_finetune_data.py [--augment] [--output data/finetune/]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.extract.analyze import analyzeSections
from legalrag.ingest.extract import extractText
from legalrag.ingest.segment import splitParagraphs
from legalrag.risk import RULES, analyzeRisk
from legalrag.slm.grammar import SYSTEM_PROMPT, make_finding_prompt


def build_pairs_from_pdf(pdf_path: Path) -> list[dict]:
    """Extract findings from a single PDF and build training pairs."""
    extraction = extractText(pdf_path)
    raw_sections = splitParagraphs(extraction.full_text)

    section_dicts = []
    for i, sec_text in enumerate(raw_sections):
        sec = {"id": f"section_{i}", "text": sec_text[:2000]}
        result = analyzeSections([sec])
        best = result[0] if result else {"type": "unknown", "confidence": 0.0}
        section_dicts.append({
            "id": f"section_{i}",
            "text": sec_text[:2000],
            "type": best.get("clause_type", "unknown"),
            "confidence": best.get("confidence", 0.0),
        })

    analysis = analyzeRisk(section_dicts, RULES)
    pairs = []

    for finding in analysis.findings:
        user_msg = make_finding_prompt(
            finding.clause_text,
            finding.rationale,
            finding.risk_level,
            finding.statute,
            finding.grounding,
        )
        # Engine-authoritative fields stamped over SLM's self-reported values
        assistant_msg = json.dumps({
            "clause_type": finding.clause_type,
            "risk_level": finding.risk_level,
            "statute": finding.statute,
            "plain_explanation": finding.rationale[:300],
            "tenant_impact": f"This is a {finding.risk_level}-risk clause of type {finding.clause_type}.",
        }, ensure_ascii=False)

        pairs.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ],
            "source": pdf_path.stem,
            "rule_id": finding.rule_id,
        })

    return pairs


def augment_pair(pair: dict, rng: random.Random) -> dict:
    """Create an augmented variant by rephrasing the user prompt."""
    # Simple augmentation: shuffle the prompt sections
    msgs = pair["messages"]
    user_content = msgs[1]["content"]
    # Split on double newlines and shuffle
    parts = user_content.split("\n\n")
    if len(parts) > 2:
        rng.shuffle(parts)
    augmented = "\n\n".join(parts)

    return {
        "messages": [
            msgs[0],  # system
            {"role": "user", "content": augmented},
            msgs[2],  # assistant (same)
        ],
        "source": pair["source"] + "_aug",
        "rule_id": pair["rule_id"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--augment", action="store_true", help="Include augmented pairs")
    parser.add_argument("--output", "-o", default="data/finetune", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-split", type=float, default=0.15, help="Fraction for eval")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find all test PDFs
    test_dir = Path("claudeTestDocs")
    pdfs = sorted(test_dir.glob("*.pdf"))

    if not pdfs:
        print(f"[error] no PDFs found in {test_dir}")
        return 1

    all_pairs = []
    for pdf in pdfs:
        pairs = build_pairs_from_pdf(pdf)
        all_pairs.extend(pairs)
        print(f"[pair] {pdf.name}: {len(pairs)} findings")

    print(f"\n[total] {len(all_pairs)} base pairs")

    # Augment
    if args.augment and all_pairs:
        rng = random.Random(args.seed)
        augmented = [augment_pair(p, rng) for p in all_pairs]
        # Some get a second augmentation at higher temperature
        augmented2 = [augment_pair(p, rng) for p in all_pairs[:len(all_pairs)//2]]
        all_pairs.extend(augmented)
        all_pairs.extend(augmented2)
        print(f"[augment] {len(augmented) + len(augmented2)} augmented pairs")

    # Shuffle and split
    rng = random.Random(args.seed)
    rng.shuffle(all_pairs)
    n_eval = max(1, int(len(all_pairs) * args.eval_split))
    eval_pairs = all_pairs[:n_eval]
    train_pairs = all_pairs[n_eval:]

    # Write
    train_path = out_dir / "train.jsonl"
    eval_path = out_dir / "eval.jsonl"

    for path, pairs in [(train_path, train_pairs), (eval_path, eval_pairs)]:
        with path.open("w", encoding="utf-8") as fh:
            for p in pairs:
                fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n[train] {len(train_pairs)} pairs -> {train_path}")
    print(f"[eval] {len(eval_pairs)} pairs -> {eval_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
