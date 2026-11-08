"""Section-aware chunking of extracted papers for retrieval.

Chunks never cross a section boundary, carry their section label and page
range as metadata, and the References/Bibliography section is excluded because
citation lists add noise to semantic search.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from services.pdf_service import ExtractedDocument
from utils.text import split_sentences, tail_at_boundary

# Canonical section name -> heading regex (matched after optional numbering).
SECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("abstract", r"abstract"),
    ("introduction", r"introduction"),
    ("related_work", r"related works?|background|literature (?:review|survey)|prior work|preliminaries"),
    ("methodology", r"methodology|methods?|proposed (?:method|approach|system|framework|model|scheme|architecture|algorithm)"
                    r"|our approach|approach|system (?:design|architecture|model|overview)|problem (?:formulation|definition|statement)"),
    ("implementation", r"implementation(?: details)?"),
    ("experiments", r"experiments?|experimental (?:setup|settings?|results|evaluation)|evaluation|performance evaluation"
                    r"|simulations?(?: results)?|datasets?"),
    ("results", r"results(?: and (?:discussion|analysis))?|discussion|analysis"),
    ("limitations", r"limitations?|threats to validity"),
    ("future_work", r"future (?:work|directions|research)"),
    ("conclusion", r"conclusions?(?: and future (?:work|directions|scope))?|concluding remarks|summary"),
    ("references", r"references|bibliography|works cited"),
    ("acknowledgments", r"acknowledge?ments?"),
    ("appendix", r"appendi(?:x|ces)"),
)

_NUMBER_PREFIX = r"(?:(?:\d+(?:\.\d+)*|[IVXLC]+|[A-H])[.)]?\s+)?"
_HEADING_REGEXES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(rf"^{_NUMBER_PREFIX}(?:{pattern})\b[\w\s:,&\-]{{0,50}}$", re.IGNORECASE))
    for name, pattern in SECTION_PATTERNS
)
_INLINE_ABSTRACT = re.compile(r"^abstract\s*[^\w\s]\s*\S", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str
    section: str
    page_start: int
    page_end: int

    def chunk_id(self, paper_id: str) -> str:
        return f"{paper_id}-{self.index:05d}"


def detect_section(block: str) -> str | None:
    """Return the canonical section name if ``block`` looks like a heading."""
    text = block.strip()
    if _INLINE_ABSTRACT.match(text):
        return "abstract"
    if len(text) > 80 or len(text.split()) > 10:
        return None
    # Unnumbered sentences ending with a period are body text, not headings.
    has_number = bool(re.match(r"^(?:\d+(?:\.\d+)*|[IVXLC]+)[.)]?\s", text))
    if text.endswith(".") and not has_number and text.lower().rstrip(".") not in {"abstract", "references"}:
        return None
    for name, regex in _HEADING_REGEXES:
        if regex.match(text.rstrip(".")):
            return name
    return None


class SectionAwareChunker:
    """Split an ``ExtractedDocument`` into overlapping, section-pure chunks."""

    def __init__(self, chunk_size: int = 1400, chunk_overlap: int = 200, min_chunk_chars: int = 120) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    def chunk(self, document: ExtractedDocument) -> list[Chunk]:
        chunks: list[Chunk] = []
        section = "front_matter"
        buffer: list[str] = []
        buffer_pages: list[int] = []
        fresh = 0  # pieces added since the last flush (excludes carried-over overlap)
        skipping = False

        def flush(carry_overlap: bool) -> None:
            nonlocal buffer, buffer_pages, fresh
            if fresh == 0:  # only overlap text left: nothing new to store
                buffer, buffer_pages = [], []
                return
            text = " ".join(buffer).strip()
            if len(text) < self.min_chunk_chars and chunks and chunks[-1].section == section:
                # Too small to stand alone: append the new text to the previous chunk.
                prev = chunks[-1]
                merged = f"{prev.text} {' '.join(buffer[-fresh:])}"
                chunks[-1] = replace(prev, text=merged, page_end=max(prev.page_end, *buffer_pages))
            else:
                chunks.append(Chunk(len(chunks), text, section, min(buffer_pages), max(buffer_pages)))
            if carry_overlap:
                buffer = [tail_at_boundary(text, self.chunk_overlap)]
                buffer_pages = [buffer_pages[-1]]
            else:
                buffer, buffer_pages = [], []
            fresh = 0

        for page in document.pages:
            for block in page.blocks:
                heading = detect_section(block)
                if heading is not None:
                    if buffer:
                        flush(carry_overlap=False)
                    section = heading
                    skipping = heading in {"references", "acknowledgments"}
                    if _INLINE_ABSTRACT.match(block):  # "Abstract—text..." keeps its text
                        block = re.sub(r"^abstract\s*[^\w\s]\s*", "", block, flags=re.IGNORECASE)
                    else:
                        continue
                if skipping:
                    continue

                for piece in self._split_long(block):
                    if buffer and len(" ".join(buffer)) + len(piece) + 1 > self.chunk_size:
                        flush(carry_overlap=True)
                    buffer.append(piece)
                    buffer_pages.append(page.number)
                    fresh += 1

        if buffer:
            flush(carry_overlap=False)
        return chunks

    def _split_long(self, block: str) -> list[str]:
        """Split a block longer than ``chunk_size`` along sentence boundaries."""
        if len(block) <= self.chunk_size:
            return [block]
        pieces: list[str] = []
        current = ""
        for sentence in split_sentences(block):
            while len(sentence) > self.chunk_size:  # pathological sentence: hard split
                pieces.append(sentence[: self.chunk_size])
                sentence = sentence[self.chunk_size :]
            if current and len(current) + len(sentence) + 1 > self.chunk_size:
                pieces.append(current)
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            pieces.append(current)
        return pieces


def lead_text(document: ExtractedDocument, max_chars: int = 2500) -> str:
    """Title + abstract + start of introduction: the densest summary of any paper."""
    parts: list[str] = []
    total = 0
    for page in document.pages[:3]:
        for block in page.blocks:
            if detect_section(block) in {"related_work", "methodology", "references"}:
                return "\n".join(parts)
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                return "\n".join(parts)[:max_chars]
    return "\n".join(parts)[:max_chars]
