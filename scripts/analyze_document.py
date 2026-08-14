"""Ingest a user document into engine row shapes.

Extracts text (PDF text layer + OCR, images, DOCX, TXT), segments it into
paragraph/section and sentence rows, and writes the result into
data/raw/<source>/ as manifest.json + sections.jsonl + sentences.jsonl.

Usage:
    python scripts/analyze_document.py <path> [--out data/raw]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from legalrag.ingest import buildRows, extractText


def write_rows(path: Path, rows: list[dict]) -> int:
    """Write JSONL rows; returns count."""
    count = 0
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to a PDF, image, DOCX, or TXT document")
    ap.add_argument("--out", default="data/raw", help="output directory (default data/raw)")
    args = ap.parse_args()

    src = Path(args.path)
    if not src.exists():
        print(f"[analyze] missing: {src}", file=sys.stderr)
        return 1

    t0 = time.time()
    try:
        extraction = extractText(str(src))
    except IsADirectoryError:
        print(f"[analyze] is a directory, not a document: {src}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[analyze] {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001 -- CLI boundary: report, don't traceback
        print(f"[analyze] extraction failed for {src}: {e}", file=sys.stderr)
        return 1

    rows = buildRows(extraction)

    out_dir = Path(args.out) / extraction.source
    out_dir.mkdir(parents=True, exist_ok=True)

    error_pages = [p.page_idx for p in extraction.pages if p.method == "error"]
    by_method: dict[str, int] = {}
    for p in extraction.pages:
        by_method[p.method] = by_method.get(p.method, 0) + 1

    manifest = {
        "source": extraction.source,
        "path": str(src),
        "n_pages": extraction.n_pages,
        "methods": sorted(extraction.methods),
        "pages_by_method": by_method,
        "n_error_pages": len(error_pages),
        "error_pages": error_pages,
        "chars": len(extraction.full_text),
        "n_sections": len(rows["sections"]),
        "n_sentences": len(rows["sentences"]),
        "elapsed_s": round(time.time() - t0, 2),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    n_sections = write_rows(out_dir / "sections.jsonl", rows["sections"])
    n_sentences = write_rows(out_dir / "sentences.jsonl", rows["sentences"])

    print(f"[analyze] {extraction.source}: {n_sections} sections, {n_sentences} sentences")
    if error_pages:
        print(f"[analyze] warning: {len(error_pages)} page(s) failed: {error_pages}", file=sys.stderr)
    if not extraction.full_text.strip():
        print(f"[analyze] warning: no text extracted from {src}", file=sys.stderr)
        return 2
    print(f"[analyze] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())