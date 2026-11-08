from __future__ import annotations

import pymupdf as fitz
import pytest

from services.chunking import SectionAwareChunker, detect_section, lead_text
from services.pdf_service import PDFService
from tests.conftest import make_paper_pdf
from utils.exceptions import InvalidPDFError, PDFTooLargeError, ScannedPDFError


def test_extracts_title_columns_and_strips_headers(settings, paper_pdf):
    doc = PDFService(settings).extract(paper_pdf, "paper.pdf")
    assert doc.page_count == 3
    assert "Lightweight CNNs" in doc.title
    assert doc.source == "IEEE"
    text = doc.full_text
    assert "IEEE TRANSACTIONS ON TEST" not in text  # running header removed (repeats on every page)
    # Reading order: left column (Abstract) comes before the right column (Introduction).
    assert text.index("Abstract") < text.index("1 Introduction") < text.index("2 Proposed Method")


@pytest.mark.parametrize(
    ("data", "name", "error"),
    [(b"", "a.pdf", InvalidPDFError), (b"hello", "a.txt", InvalidPDFError), (b"not a pdf at all", "a.pdf", InvalidPDFError)],
)
def test_rejects_invalid_files(settings, data, name, error):
    with pytest.raises(error):
        PDFService(settings).extract(data, name)


def test_rejects_corrupt_pdf(settings):
    with pytest.raises(InvalidPDFError):
        PDFService(settings).extract(b"%PDF-1.7\n garbage garbage", "broken.pdf")


def test_rejects_too_many_pages(settings):
    with pytest.raises(PDFTooLargeError):
        PDFService(settings).extract(make_paper_pdf(pages=settings.max_pages + 1), "long.pdf")


def test_rejects_scanned_pdf(settings):
    doc = fitz.open()
    for _ in range(3):
        doc.new_page()
    with pytest.raises(ScannedPDFError):
        PDFService(settings).extract(doc.tobytes(), "scan.pdf")


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ("1 Introduction", "introduction"),
        ("III. PROPOSED METHOD FOR ROUTING", "methodology"),
        ("Abstract—We propose a new method.", "abstract"),
        ("5.2 Experimental Setup", "experiments"),
        ("References", "references"),
        ("Conclusion and Future Work", "conclusion"),
        ("The method works well in practice.", None),
        ("In this introduction we describe a long sentence that is clearly body text and not a heading at all", None),
    ],
)
def test_detect_section(block, expected):
    assert detect_section(block) == expected


def test_chunker_is_section_aware_and_skips_references(settings, paper_pdf):
    doc = PDFService(settings).extract(paper_pdf, "paper.pdf")
    chunks = SectionAwareChunker(600, 100).chunk(doc)
    sections = {c.section for c in chunks}
    assert {"abstract", "introduction", "methodology", "experiments", "conclusion"} <= sections
    assert "references" not in sections
    assert all("Some cited work" not in c.text for c in chunks)
    assert all(len(c.text) <= 600 + 100 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))
    assert "Lightweight" in lead_text(doc)


def test_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        SectionAwareChunker(100, 100)
