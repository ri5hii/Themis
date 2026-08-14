# Unit tests for user document extraction (pypdfium2 + rapidocr + docx/txt).
from __future__ import annotations

from pathlib import Path

import pytest

from legalrag.ingest.extract import (
    Extraction,
    PageText,
    extractText,
)


class TestExtractionModel:
    def test_toDict_shape(self):
        ex = Extraction(source="doc", path="/x.pdf", pages=[PageText(0, "hi", "text")])
        d = ex.toDict()
        assert d["n_pages"] == 1
        assert d["methods"] == ["text"]
        assert d["pages"][0]["chars"] == 2

    def test_full_text_joins_pages(self):
        ex = Extraction(source="doc", path="/x.pdf", pages=[PageText(0, "a", "text"), PageText(1, "b", "text")])
        assert ex.full_text == "a\n\nb"


class TestExtractText:
    def test_txt_file(self, tmp_path: Path):
        p = tmp_path / "lease.txt"
        p.write_text("First line.\nSecond line.")
        ex = extractText(str(p))
        assert ex.source == "lease"
        assert ex.n_pages == 1
        assert ex.methods == {"raw"}
        assert ex.pages[0].text == "First line.\nSecond line."

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            extractText(str(tmp_path / "nope.pdf"))

    def test_unsupported_extension(self, tmp_path: Path):
        p = tmp_path / "doc.xyz"
        p.write_text("hi")
        with pytest.raises(ValueError, match="unsupported file type"):
            extractText(str(p))

    def test_docx(self, tmp_path: Path):
        import docx

        doc = docx.Document()
        doc.add_paragraph("Clause one.")
        doc.add_paragraph("Clause two.")
        p = tmp_path / "lease.docx"
        doc.save(str(p))
        ex = extractText(str(p))
        assert ex.methods == {"paragraphs"}
        assert "Clause one." in ex.pages[0].text
        assert "Clause two." in ex.pages[0].text

    def test_extension_case_insensitive(self, tmp_path: Path):
        p = tmp_path / "lease.TXT"
        p.write_text("plain")
        ex = extractText(str(p))
        assert ex.pages[0].text == "plain"


class TestPdfTextLayer:
    def test_text_layer_pdf(self, tmp_path: Path):
        from reportlab.pdfgen import canvas

        out = tmp_path / "text.pdf"
        c = canvas.Canvas(str(out))
        c.drawString(72, 720, "Tenant shall pay rent in accordance with this agreement")
        c.save()

        ex = extractText(str(out))
        assert ex.n_pages == 1
        assert ex.methods == {"text"}
        assert "rent" in ex.pages[0].text