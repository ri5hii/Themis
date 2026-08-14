# User document extraction: file -> page/chunk text records.
#
# PDFs use pypdfium2 for the text layer; pages that yield almost no text are
# treated as scanned and OCR'd lazily with rapidocr (ONNX). Images are OCR'd
# directly, docx via python-docx, and plain text/markdown read as-is. No new
# heavyweight dependencies: everything ships in the existing venv.
#
# The rapidocr engine is only instantiated on first OCR need, so text-layer
# PDFs never pay the model-load cost.
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

# A page with fewer than this many text characters is treated as scanned.
MIN_TEXT_CHARS = 40

# render scale for OCR: 2.0 upscales scanned pages for better detection
OCR_SCALE = 2.0

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}
DOCX_EXTS = {".docx"}


@dataclass
class PageText:
    """Text content of one page/chunk plus how it was obtained."""

    page_idx: int
    text: str
    method: str  # "text" | "ocr" | "paragraphs" | "raw"


@dataclass
class Extraction:
    """Extracted text for a whole document."""

    source: str
    path: str
    pages: list[PageText] = field(default_factory=list)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def methods(self) -> set[str]:
        return {p.method for p in self.pages}

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)

    def toDict(self) -> dict:
        return {
            "source": self.source,
            "path": self.path,
            "n_pages": self.n_pages,
            "methods": sorted(self.methods),
            "pages": [
                {"page_idx": p.page_idx, "method": p.method, "chars": len(p.text)}
                for p in self.pages
            ],
        }


def _ocr_instance():
    from rapidocr import RapidOCR

    return RapidOCR()


def _ocr_text(image_bytes: bytes, engine) -> str:
    """OCR a rendered page image, joining detected text boxes by line."""
    result = engine(image_bytes)
    if not result:
        return ""
    lines: list[str] = []
    # result is (boxes, texts, scores)
    try:
        _, texts, _ = result
    except (TypeError, ValueError):
        return ""
    for line in texts or []:
        lines.append(str(line))
    return "\n".join(lines).strip()


def extractPageText(page, ocr_engine) -> tuple[str, str]:
    """Text + method for one pypdfium2 page. OCRs if the text layer is empty."""
    textpage = page.get_textpage()
    text = textpage.get_text_bounded().strip()
    if len(text) >= MIN_TEXT_CHARS:
        return text, "text"
    bitmap = page.render(scale=OCR_SCALE).to_pil()
    buf = io.BytesIO()
    bitmap.save(buf, format="PNG")
    ocr = _ocr_text(buf.getvalue(), ocr_engine)
    return ocr, "ocr"


def extractPdf(path: Path, source: str) -> Extraction:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(path))
    pages: list[PageText] = []
    engine = None
    try:
        for i in range(len(doc)):
            page = doc.get_page(i)
            textpage = page.get_textpage()
            text = textpage.get_text_bounded().strip()
            if len(text) >= MIN_TEXT_CHARS:
                pages.append(PageText(page_idx=i, text=text, method="text"))
                continue
            if engine is None:
                engine = _ocr_instance()
            bitmap = page.render(scale=OCR_SCALE).to_pil()
            buf = io.BytesIO()
            bitmap.save(buf, format="PNG")
            ocr = _ocr_text(buf.getvalue(), engine)
            pages.append(PageText(page_idx=i, text=ocr, method="ocr"))
    finally:
        doc.close()
    return Extraction(source=source, path=str(path), pages=pages)


def extractImage(path: Path, source: str) -> Extraction:
    engine = _ocr_instance()
    text = _ocr_text(path.read_bytes(), engine)
    return Extraction(source=source, path=str(path), pages=[PageText(page_idx=0, text=text, method="ocr")])


def extractDocx(path: Path, source: str) -> Extraction:
    import docx

    document = docx.Document(str(path))
    paras = [p.text for p in document.paragraphs]
    pages = [PageText(page_idx=0, text="\n".join(paras).strip(), method="paragraphs")]
    return Extraction(source=source, path=str(path), pages=pages)


def extractText(path: str | Path) -> Extraction:
    """Extract text from a user document, dispatching on extension."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    source = path.stem
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extractPdf(path, source)
    if suffix in IMAGE_EXTS:
        return extractImage(path, source)
    if suffix in DOCX_EXTS:
        return extractDocx(path, source)
    if suffix in TEXT_EXTS or suffix == "":
        text = path.read_text(encoding="utf-8", errors="replace")
        return Extraction(source=source, path=str(path), pages=[PageText(page_idx=0, text=text, method="raw")])
    raise ValueError(f"unsupported file type: {suffix}")