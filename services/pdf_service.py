"""PDF validation and layout-aware text extraction with PyMuPDF.

Handles the realities of academic PDFs:
* two-column layouts (IEEE/ACM) - blocks are re-ordered column by column;
* running headers/footers and page numbers - removed when they repeat;
* hyphenation and ligatures - normalised;
* encrypted, scanned, corrupt or oversized files - rejected with clear errors.

Pages are processed one at a time, so memory stays flat even for 100-page papers.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (modern import name)

from utils.config import Settings
from utils.exceptions import (
    EncryptedPDFError,
    InvalidPDFError,
    PDFProcessingError,
    PDFTooLargeError,
    ScannedPDFError,
)
from utils.logger import get_logger
from utils.progress import ProgressCallback, noop_progress
from utils.text import clean_block_text, normalize_whitespace

logger = get_logger(__name__)

# PyMuPDF block tuple: (x0, y0, x1, y1, text, block_no, block_type)
_Block = tuple[float, float, float, float, str, int, int]

_PAGE_NUMBER = re.compile(r"^(?:page\s*)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?$", re.IGNORECASE)
_DIGITS = re.compile(r"\d+")
_SOURCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("arXiv", re.compile(r"arXiv:\s*\d{4}\.\d{4,5}|arxiv\.org", re.IGNORECASE)),
    ("IEEE", re.compile(r"\bIEEE\b")),
    ("ACM", re.compile(r"\bACM\b|dl\.acm\.org|\bACM Reference Format\b")),
    ("Springer", re.compile(r"\bSpringer\b|link\.springer\.com", re.IGNORECASE)),
    ("Elsevier", re.compile(r"\bElsevier\b|sciencedirect", re.IGNORECASE)),
)
_BAD_TITLES = re.compile(r"^(untitled|microsoft word|title|paper|document|\d+)\b", re.IGNORECASE)


@dataclass(frozen=True)
class PageContent:
    number: int  # 1-based page number
    blocks: tuple[str, ...]

    @property
    def text(self) -> str:
        return "\n\n".join(self.blocks)


@dataclass(frozen=True)
class ExtractedDocument:
    filename: str
    file_hash: str
    title: str
    page_count: int
    pages: tuple[PageContent, ...]
    source: str = "Unknown"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def word_count(self) -> int:
        return sum(len(page.text.split()) for page in self.pages)


class PDFService:
    """Validates uploads and converts PDFs into clean, ordered text blocks."""

    def __init__(self, settings: Settings) -> None:
        self._max_pages = settings.max_pages
        self._max_bytes = settings.max_file_size_bytes
        self._max_mb = settings.max_file_size_mb
        self._min_chars_per_page = settings.min_chars_per_page

    # ---------------------------------------------------------------- public API
    @staticmethod
    def compute_hash(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def validate(self, data: bytes, filename: str) -> None:
        """Cheap checks that run before the PDF is parsed."""
        if not data:
            raise InvalidPDFError(user_message="The uploaded file is empty.")
        if not filename.lower().endswith(".pdf"):
            raise InvalidPDFError(user_message="Only .pdf files are supported.")
        if len(data) > self._max_bytes:
            size_mb = len(data) / (1024 * 1024)
            raise PDFTooLargeError(
                user_message=f"The file is {size_mb:.1f} MB; the limit is {self._max_mb} MB."
            )
        if b"%PDF" not in data[:1024]:
            raise InvalidPDFError(user_message="The file does not look like a PDF (missing %PDF header).")

    def extract(
        self, data: bytes, filename: str, progress: ProgressCallback | None = None
    ) -> ExtractedDocument:
        """Validate and extract a PDF into an ``ExtractedDocument``."""
        progress = progress or noop_progress
        self.validate(data, filename)

        try:
            document = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # PyMuPDF raises several error types for corrupt files
            raise InvalidPDFError(
                f"PyMuPDF could not open {filename}: {exc}",
                user_message="The PDF appears to be corrupted and could not be opened.",
            ) from exc

        with document:
            if document.needs_pass and not document.authenticate(""):
                raise EncryptedPDFError()
            page_count = document.page_count
            if page_count == 0:
                raise InvalidPDFError(user_message="The PDF has no pages.")
            if page_count > self._max_pages:
                raise PDFTooLargeError(
                    user_message=(
                        f"The PDF has {page_count} pages; the maximum supported is {self._max_pages}. "
                        "Try uploading only the main paper (without long appendices)."
                    )
                )

            pages: list[PageContent] = []
            for index in range(page_count):
                try:
                    blocks = self._ordered_blocks(document.load_page(index))
                except Exception as exc:  # one broken page must not kill the whole paper
                    logger.warning("Skipping unreadable page %d of %s: %s", index + 1, filename, exc)
                    blocks = []
                pages.append(PageContent(number=index + 1, blocks=tuple(blocks)))
                progress((index + 1) / page_count, f"Extracting text: page {index + 1}/{page_count}")

            metadata = {k: str(v) for k, v in (document.metadata or {}).items() if v}
            title = self._detect_title(document, metadata, pages, filename)

        source = self._detect_source(pages, metadata)  # before stripping: headers often name the venue
        pages = self._strip_repeating_blocks(pages)
        total_chars = sum(len(page.text) for page in pages)
        if total_chars / page_count < self._min_chars_per_page:
            raise ScannedPDFError()

        logger.info("Extracted %s: %d pages, %d chars, source=%s", filename, page_count, total_chars, source)
        return ExtractedDocument(
            filename=filename,
            file_hash=self.compute_hash(data),
            title=title,
            page_count=page_count,
            pages=tuple(pages),
            source=source,
            metadata=metadata,
        )

    def extract_file(self, path: Path, progress: ProgressCallback | None = None) -> ExtractedDocument:
        """Convenience wrapper for CLI/tests."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise PDFProcessingError(f"Cannot read {path}: {exc}") from exc
        return self.extract(data, path.name, progress)

    # ------------------------------------------------------------ layout helpers
    @staticmethod
    def _ordered_blocks(page: fitz.Page) -> list[str]:
        """Return the page's text blocks in natural reading order.

        Two-column detection: blocks fully in the left or right half are treated as
        column blocks; blocks spanning the middle (title, abstract, wide figures)
        are emitted before the columns if above them, otherwise after.
        """
        raw: list[_Block] = page.get_text("blocks")
        blocks = [b for b in raw if b[6] == 0 and b[4].strip()]
        if not blocks:
            return []

        width = page.rect.width
        mid, tolerance = width / 2, width * 0.04
        left = [b for b in blocks if b[2] <= mid + tolerance]
        right = [b for b in blocks if b[0] >= mid - tolerance and b[2] > mid + tolerance]
        spanning = [b for b in blocks if b not in left and b not in right]

        def by_y(items: list[_Block]) -> list[_Block]:
            return sorted(items, key=lambda b: (round(b[1], 1), b[0]))

        if len(left) >= 2 and len(right) >= 2:
            column_top = min(b[1] for b in left + right)
            header = [b for b in spanning if b[1] < column_top]
            footer = [b for b in spanning if b[1] >= column_top]
            ordered = by_y(header) + by_y(left) + by_y(right) + by_y(footer)
        else:
            ordered = by_y(blocks)

        cleaned = (clean_block_text(b[4]) for b in ordered)
        return [text for text in cleaned if text and not _PAGE_NUMBER.match(text)]

    @staticmethod
    def _strip_repeating_blocks(pages: list[PageContent]) -> list[PageContent]:
        """Remove running headers/footers that repeat on many pages."""
        if len(pages) < 3:
            return pages

        def signature(text: str) -> str:
            return _DIGITS.sub("#", text.lower())

        counts: Counter[str] = Counter()
        for page in pages:
            edge_blocks = set(page.blocks[:2] + page.blocks[-2:])
            counts.update({signature(b) for b in edge_blocks if len(b) < 160})

        threshold = max(3, int(len(pages) * 0.4))
        repeating = {sig for sig, count in counts.items() if count >= threshold}
        if not repeating:
            return pages
        return [
            PageContent(p.number, tuple(b for b in p.blocks if signature(b) not in repeating))
            for p in pages
        ]

    @staticmethod
    def _detect_title(
        document: fitz.Document, metadata: dict[str, str], pages: list[PageContent], filename: str
    ) -> str:
        meta_title = normalize_whitespace(metadata.get("title", ""))
        if 10 <= len(meta_title) <= 250 and not _BAD_TITLES.match(meta_title):
            return meta_title

        # Fallback: the largest-font text in the top half of the first page.
        try:
            first = document.load_page(0)
            spans = [
                (round(span["size"], 1), span["text"].strip(), line["bbox"][1])
                for block in first.get_text("dict")["blocks"]
                if block.get("type") == 0
                for line in block["lines"]
                for span in line["spans"]
                if len(span["text"].strip()) > 2 and line["bbox"][1] < first.rect.height / 2
            ]
            if spans:
                max_size = max(size for size, _, _ in spans)
                title_parts = [text for size, text, _ in spans if abs(size - max_size) <= 0.5]
                candidate = normalize_whitespace(" ".join(title_parts))
                if 10 <= len(candidate) <= 250 and not _BAD_TITLES.match(candidate):
                    return candidate
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Font-based title detection failed: %s", exc)

        if pages and pages[0].blocks:
            first_block = pages[0].blocks[0]
            if 10 <= len(first_block) <= 200:
                return first_block
        return Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Untitled paper"

    @staticmethod
    def _detect_source(pages: list[PageContent], metadata: dict[str, str]) -> str:
        sample = " ".join(metadata.values()) + " " + " ".join(p.text for p in pages[:2])
        for name, pattern in _SOURCE_PATTERNS:
            if pattern.search(sample):
                return name
        return "Unknown"
