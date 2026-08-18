# `themis analyze` — end-to-end lease analysis CLI.
#
# Pipeline: parse -> segment -> classify (fast-lane + classifier fallback)
# -> risk engine + statute grounding -> optional SLM simplification.
# Output as text (TTY-colored), markdown, or JSON; `--interactive` reviews
# each finding before writing.
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from legalrag.extract.analyze import analyzeSections
from legalrag.ingest.extract import extractText
from legalrag.ingest.segment import splitParagraphs
from legalrag.risk import analyzeRisk, loadStatuteChunks
from legalrag.risk.grounding import groundAll
from legalrag.risk.rules import RULES as RISK_RULES

from .output import render, tty
from .review import ReviewAborted, review_findings

ROOT = Path(__file__).resolve().parents[3]
CLASSIFIER_PATH = ROOT / "models" / "classifier.npz"
STATUTE_INDEX = ROOT / "data" / "indexes" / "statutes"


def _load_fallback() -> object | None:
    """Trained classifier for fast-lane `unknown` sections, if present."""
    if not CLASSIFIER_PATH.exists():
        return None
    from legalrag.extract.classifier import TrainedClassifier

    return TrainedClassifier.load(CLASSIFIER_PATH)


def analyze_lease(
    pdf_path: Path,
    *,
    slm: bool = False,
    interactive: bool = False,
    threads: int = 8,
    stage_log: bool = True,
) -> dict[str, Any]:
    """Run the full pipeline; returns the analysis output dict."""
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    def log(msg: str) -> None:
        if stage_log:
            print(msg)

    t0 = time.time()
    stage_times: dict[str, float] = {}

    def stage(name: str, fn):
        s = time.time()
        result = fn()
        stage_times[name] = round(time.time() - s, 1)
        return result

    # 1. Extract text
    log(f"[1/5] Extracting text from {pdf_path.name}...")
    extraction = stage("parse", lambda: extractText(pdf_path))
    log(f"  {extraction.n_pages} pages, methods: {extraction.methods}")

    # 2. Segment into sections
    log("[2/5] Segmenting into sections...")
    raw_sections = stage("segment", lambda: splitParagraphs(extraction.full_text))
    log(f"  {len(raw_sections)} sections")

    # 3. Classify (fast-lane + trained classifier fallback, batched)
    log("[3/5] Classifying sections...")
    fallback = _load_fallback()
    if fallback is not None:
        log(f"  classifier fallback: {fallback.model_name}")
    section_dicts = [
        {"id": f"section_{i}", "text": t[:2000]} for i, t in enumerate(raw_sections)
    ]
    classified_rows = stage("classify", lambda: analyzeSections(section_dicts, fallback))
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
    log(f"  {classified}/{len(section_dicts)} sections classified")

    # 4. Risk engine + grounding
    log("[4/5] Running risk engine...")
    analysis = stage("risk", lambda: analyzeRisk(section_dicts, RISK_RULES))
    log(f"  {len(analysis.findings)} findings")

    statute_chunks = loadStatuteChunks(STATUTE_INDEX)
    if not STATUTE_INDEX.is_dir():
        log(f"  [warn] statute index not found: {STATUTE_INDEX} (skipping grounding)")
    elif statute_chunks:
        rules_by_id = {r.rule_id: r for r in RISK_RULES}
        s = time.time()
        groundAll(analysis, rules_by_id, statute_chunks, STATUTE_INDEX)
        stage_times["grounding"] = round(time.time() - s, 1)
        grounded = sum(1 for f in analysis.findings if f.statute)
        log(f"  {grounded}/{len(analysis.findings)} findings grounded")

    # 5. SLM (optional)
    slm_outputs: list[dict[str, Any]] = []
    if slm:
        log("[5/5] Running SLM simplification...")
        from legalrag.slm import simplifyAll

        s = time.time()
        outputs = simplifyAll(analysis.findings, n_threads=threads)
        stage_times["slm"] = round(time.time() - s, 1)
        slm_outputs = [o.toDict() for o in outputs]
        parsed = sum(1 for o in outputs if o.parse_ok)
        log(f"  {parsed}/{len(slm_outputs)} parsed OK")
    else:
        log("[5/5] SLM skipped (use --slm to enable)")

    # Interactive review of findings
    if interactive and analysis.findings:
        log("")
        try:
            findings_out = review_findings(analysis.findings)
        except ReviewAborted:
            log("Review aborted — verdicts not persisted.")
            findings_out = [f.toDict() for f in analysis.findings]
    else:
        findings_out = [f.toDict() for f in analysis.findings]

    elapsed = time.time() - t0

    return {
        "source": str(pdf_path),
        "elapsed_s": round(elapsed, 1),
        "stage_times_s": stage_times,
        "sections": len(section_dicts),
        "classified": classified,
        "findings": findings_out,
        "slm": slm_outputs,
        "summary": {
            "n_findings": len(analysis.findings),
            "n_high": sum(1 for f in analysis.findings if f.risk_level == "high"),
            "n_medium": sum(1 for f in analysis.findings if f.risk_level == "medium"),
            "n_low": sum(1 for f in analysis.findings if f.risk_level == "low"),
            "n_info": sum(1 for f in analysis.findings if f.risk_level == "info"),
        },
    }


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis analyze",
        description="End-to-end lease analysis: parse -> classify -> risk -> ground -> SLM.",
    )
    parser.add_argument("pdf", help="Path to lease PDF")
    parser.add_argument("--slm", action="store_true", help="Run SLM simplification")
    parser.add_argument("--interactive", "-i", action="store_true", help="Review findings interactively")
    parser.add_argument("--format", "-f", choices=("text", "markdown", "json"), default="text")
    parser.add_argument("--output", "-o", help="Output file (default: <stem>_analysis.<ext>)")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = args if args is not None else build_parser().parse_args()
    colors = tty() and not args.no_color

    try:
        output = analyze_lease(
            Path(args.pdf),
            slm=args.slm,
            interactive=args.interactive,
            threads=args.threads,
        )
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
        name = getattr(e, "filename", None) or args.pdf
        print(f"[error] {name} not a readable file")
        return 1

    body = render(output, args.format, colors)

    ext = {"text": "txt", "markdown": "md", "json": "json"}[args.format]
    if args.output:
        out_path = Path(args.output)
    elif args.format == "json":
        out_path = Path(args.pdf).with_name(Path(args.pdf).stem + "_analysis.json")
    else:
        out_path = Path(args.pdf).with_suffix(f".analysis.{ext}")
    out_path.write_text(body, encoding="utf-8")
    print(body, end="")
    print(f"[output] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())