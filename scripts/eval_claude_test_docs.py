#!/usr/bin/env python3
"""Run Themis Extract engine on the claudeTestDocs PDFs and dump per-section results.

Uses the shipped product path: legalrag.ingest.extractText (docling extraction)
-> legalrag.ingest.buildRows (segmentation) -> legalrag.extract.analyze.analyzeSections
(fast-lane + trained classifier fallback). Writes eval/claude_test_docs/themis_out.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from legalrag.extract.analyze import analyzeSections
from legalrag.extract.classifier import TrainedClassifier
from legalrag.ingest import buildRows, extractText

DOCS_DIR = ROOT / "claudeTestDocs"
CLASSIFIER = ROOT / "models" / "classifier.npz"


def main() -> int:
    fallback = TrainedClassifier.load(CLASSIFIER) if CLASSIFIER.is_file() else None
    print(f"[themis_eval] classifier fallback: {CLASSIFIER.is_file()} ({len(fallback.classes)} classes)" if fallback else "[themis_eval] classifier fallback: none")

    out: dict = {}
    for pdf in sorted(DOCS_DIR.glob("*.pdf")):
        name = pdf.stem
        try:
            extraction = extractText(str(pdf))
        except Exception as e:  # noqa: BLE001
            out[name] = {"error": str(e)}
            print(f"[themis_eval] {name}: EXTRACTION FAILED: {e}")
            continue
        rows = buildRows(extraction)
        sections = rows.get("sections", [])
        analyzed = analyzeSections(sections, fallback=fallback)
        out[name] = {
            "n_pages": extraction.n_pages,
            "n_sections": len(sections),
            "n_sentences": len(rows.get("sentences", [])),
            "sections": [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "clause_type": r["clause_type"],
                    "method": r["method"],
                    "confidence": r["confidence"],
                }
                for r in analyzed
            ],
        }
        print(f"[themis_eval] {name}: {len(sections)} sections")

    dst = ROOT / "eval" / "claude_test_docs" / "themis_out.json"
    dst.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[themis_eval] wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
