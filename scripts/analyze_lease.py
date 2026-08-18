"""End-to-end lease analysis: parse -> classify -> risk -> ground -> SLM.

Usage:
    python scripts/analyze_lease.py <pdf_path> [--slm] [--output results.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.extract.analyze import analyzeSections
from legalrag.ingest.extract import extractText
from legalrag.ingest.segment import splitParagraphs
from legalrag.risk import analyzeRisk, loadStatuteChunks
from legalrag.risk.grounding import groundAll
from legalrag.risk.rules import RULES as RISK_RULES

STATUTE_INDEX = Path("data/indexes/statutes")
ROOT = Path(__file__).resolve().parent.parent
CLASSIFIER_PATH = ROOT / "models" / "classifier.npz"


def _load_fallback() -> object | None:
    """Trained classifier for fast-lane `unknown` sections, if present."""
    if not CLASSIFIER_PATH.exists():
        return None
    from legalrag.extract.classifier import TrainedClassifier

    return TrainedClassifier.load(CLASSIFIER_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", help="Path to lease PDF")
    parser.add_argument("--slm", action="store_true", help="Run SLM simplification")
    parser.add_argument("--output", "-o", help="Output JSON path")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[error] {pdf_path} not found")
        return 1

    t0 = time.time()

    # 1. Extract text
    print(f"[1/5] Extracting text from {pdf_path.name}...")
    extraction = extractText(pdf_path)
    print(f"  {extraction.n_pages} pages, methods: {extraction.methods}")

    # 2. Segment into sections
    print("[2/5] Segmenting into sections...")
    raw_sections = splitParagraphs(extraction.full_text)
    print(f"  {len(raw_sections)} sections")

    # 3. Classify sections (fast-lane + trained classifier fallback, batched)
    print("[3/5] Classifying sections...")
    fallback = _load_fallback()
    if fallback is not None:
        print(f"  classifier fallback: {fallback.model_name}")
    section_dicts = [{"id": f"section_{i}", "text": t[:2000]} for i, t in enumerate(raw_sections)]
    classified_rows = analyzeSections(section_dicts, fallback)
    section_dicts = [
        {
            "id": row["id"],
            "text": row["text"],
            "type": row["clause_type"],
            "confidence": row["confidence"],
        }
        for row in classified_rows
    ]

    classified = sum(1 for s in section_dicts if s["type"] != "unknown")
    print(f"  {classified}/{len(section_dicts)} sections classified")

    # 4. Risk engine + grounding
    print("[4/5] Running risk engine...")
    analysis = analyzeRisk(section_dicts, RISK_RULES)
    print(f"  {len(analysis.findings)} findings")

    # Ground findings
    statute_chunks = loadStatuteChunks(STATUTE_INDEX)
    if statute_chunks:
        rules_by_id = {r.rule_id: r for r in RISK_RULES}
        groundAll(analysis, rules_by_id, statute_chunks, STATUTE_INDEX)
        grounded = sum(1 for f in analysis.findings if f.statute)
        print(f"  {grounded}/{len(analysis.findings)} findings grounded")

    # 5. SLM (optional)
    slm_outputs = []
    if args.slm:
        print("[5/5] Running SLM simplification...")
        from legalrag.slm import simplifyAll
        slm_outputs = simplifyAll(analysis.findings, n_threads=args.threads)
        parsed = sum(1 for o in slm_outputs if o.parse_ok)
        print(f"  {parsed}/{len(slm_outputs)} parsed OK")
    else:
        print("[5/5] SLM skipped (use --slm to enable)")

    elapsed = time.time() - t0

    # Build output
    output = {
        "source": str(pdf_path),
        "elapsed_s": round(elapsed, 1),
        "sections": len(section_dicts),
        "classified": classified,
        "findings": [f.toDict() for f in analysis.findings],
        "slm": [o.toDict() for o in slm_outputs],
        "summary": {
            "n_findings": len(analysis.findings),
            "n_high": sum(1 for f in analysis.findings if f.risk_level == "high"),
            "n_medium": sum(1 for f in analysis.findings if f.risk_level == "medium"),
            "n_low": sum(1 for f in analysis.findings if f.risk_level == "low"),
            "n_info": sum(1 for f in analysis.findings if f.risk_level == "info"),
        },
    }

    # Print findings
    print(f"\n{'='*60}")
    print(f"Analysis complete in {elapsed:.1f}s")
    print(f"Sections: {len(section_dicts)} total, {classified} classified")
    print(f"Findings: {len(analysis.findings)} "
          f"(high={output['summary']['n_high']}, "
          f"medium={output['summary']['n_medium']}, "
          f"low={output['summary']['n_low']}, "
          f"info={output['summary']['n_info']})")
    print(f"{'='*60}")

    for i, f in enumerate(analysis.findings):
        print(f"\n[{f.risk_level.upper()}] {f.rule_id}")
        print(f"  Rationale: {f.rationale[:120]}")
        print(f"  Statute: {f.statute}")
        if args.slm and i < len(slm_outputs) and slm_outputs[i].parse_ok:
            print(f"  Plain: {slm_outputs[i].plain_explanation[:120]}")
            print(f"  Impact: {slm_outputs[i].tenant_impact[:120]}")

    # Write output
    out_path = args.output or str(pdf_path.stem) + "_analysis.json"
    Path(out_path).write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"\n[output] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
