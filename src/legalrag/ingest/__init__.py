# User document ingestion: extraction + segmentation into engine row shapes.
#
# extract.py handles file -> text records (PDF text layer with lazy OCR fallback,
# images via rapidocr, docx/txt read directly). segment.py is pure logic that
# turns extracted text into the section/sentence rows the supervised tasks consume.
from __future__ import annotations

from .extract import Extraction, extractText, extractPageText

__all__ = [
    "Extraction",
    "extractText",
    "extractPageText",
]