# `themis annotate` — interactive section-level re-annotation.
#
# Loads a lease document, segments it, and walks the sections for a human
# annotator to label with a clause type from the taxonomy (or `unknown`).
# Output is JSONL ready for the risk/eval tooling:
#   {source, section_idx, text, type, confidence=1.0}
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

from legalrag.ingest.extract import extractText
from legalrag.ingest.segment import splitParagraphs

TAXONOMY = (
    "term", "rent", "deposit", "maintenance", "utilities", "termination",
    "holdover", "subletting", "access", "late_fee", "registration",
    "dispute_resolution", "premises", "pets", "entire_agreement",
    "no_obligation",
)


class AnnotateAborted(Exception):
    """User quit; carries the rows annotated so far."""

    def __init__(self, rows: list[dict]) -> None:
        super().__init__("annotation aborted")
        self.rows = rows


def _write_rows(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read(prompt: Callable[[str], str | None], msg: str) -> str | None:
    """Prompt for input; None on EOF (non-interactive stdin)."""
    try:
        return prompt(msg)
    except EOFError:
        return None


def annotate_sections(
    pdf_path: Path,
    out_path: Path,
    prompt: Callable[[str], str | None] = input,
    max_sections: int | None = None,
) -> int:
    """Walk sections, collect {source, section_idx, text, type} rows."""
    extraction = extractText(pdf_path)
    sections = splitParagraphs(extraction.full_text)
    n = len(sections)
    if max_sections is not None:
        n = min(n, max_sections)

    print(f"{pdf_path.name}: {n} sections to annotate ({extraction.n_pages} pages)")
    print(f"types: {', '.join(TAXONOMY)}, or 'u' for unknown, '?' to repeat list, 'q' to quit")
    print("=" * 60)

    rows: list[dict] = []
    for i, text in enumerate(sections[:n]):
        body = text[:600] + ("..." if len(text) > 600 else "")
        print(f"\n--- Section {i + 1}/{n} ---")
        print(body)
        while True:
            choice = _read(prompt, "  type (u=unknown, ?=types, q=quit): ")
            if choice is None:
                raise AnnotateAborted(rows)
            choice = choice.strip().lower()
            if choice == "q":
                raise AnnotateAborted(rows)
            if choice == "?":
                print(f"  {', '.join(TAXONOMY)}")
                continue
            if choice == "u":
                ctype = "unknown"
                break
            if choice in TAXONOMY:
                ctype = choice
                break
            print("  not a valid type")
        rows.append(
            {
                "source": pdf_path.stem,
                "section_idx": i,
                "text": text,
                "type": ctype,
                "confidence": 1.0,
            }
        )

    _write_rows(rows, out_path)
    print(f"\nannotated {len(rows)} sections -> {out_path}")
    return 0


def build_parser(parser: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    parser = parser or argparse.ArgumentParser(
        prog="themis annotate",
        description="Interactively re-annotate lease sections with clause types.",
    )
    parser.add_argument("pdf", help="Path to lease PDF")
    parser.add_argument("--output", "-o", help="Output JSONL path")
    parser.add_argument("--max-sections", type=int, default=None, help="Only annotate the first N sections")
    return parser


def main(args: argparse.Namespace | None = None) -> int:
    args = args if args is not None else build_parser().parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[error] {pdf_path} not found")
        return 1
    out_path = (
        Path(args.output)
        if args.output
        else Path("data/annotated") / f"{pdf_path.stem}.sections.jsonl"
    )
    try:
        return annotate_sections(pdf_path, out_path, max_sections=args.max_sections)
    except AnnotateAborted as e:
        _write_rows(e.rows, out_path)
        print(f"\nannotation aborted; {len(e.rows)} rows written -> {out_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())