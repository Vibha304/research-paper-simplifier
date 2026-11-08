"""Pure text-processing helpers (no third-party dependencies)."""

from __future__ import annotations

import re
import unicodedata

_HYPHEN_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE = re.compile(r"\s+")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")

# Ligatures and typographic characters that PDF extraction frequently produces.
_TYPOGRAPHIC_MAP = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "−": "-", "•": "-", " ": " ",
    "…": "...", "­": "",
}


def clean_block_text(text: str) -> str:
    """Normalise a raw PDF text block into a single clean paragraph."""
    if not text:
        return ""
    text = _CONTROL_CHARS.sub("", text)
    for src, dst in (("ﬀ", "ff"), ("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬃ", "ffi"), ("ﬄ", "ffl"), ("­", "")):
        text = text.replace(src, dst)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    return _WHITESPACE.sub(" ", text).strip()


def normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text or "").strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences with a lightweight regex heuristic."""
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate at a word boundary so that the result is at most ``max_chars`` long."""
    if len(text) <= max_chars:
        return text
    cut = text[: max(0, max_chars - len(suffix))]
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:") + suffix


def tail_at_boundary(text: str, max_chars: int) -> str:
    """Return roughly the last ``max_chars`` characters, starting on a word boundary."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    space = tail.find(" ")
    return tail[space + 1 :] if 0 <= space < len(tail) // 2 else tail


def slugify(text: str, max_length: int = 60, separator: str = "-") -> str:
    """ASCII slug for filenames and identifiers."""
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", separator, ascii_text).strip(separator).lower()
    return slug[:max_length].rstrip(separator) or "paper"


def to_latin1(text: str) -> str:
    """Make text safe for PDF core fonts (Latin-1), replacing common symbols."""
    for src, dst in _TYPOGRAPHIC_MAP.items():
        text = text.replace(src, dst)
    normalized = unicodedata.normalize("NFKC", text)
    return normalized.encode("latin-1", "replace").decode("latin-1")


def dedupe_preserve_order(items: list[str]) -> list[str]:
    """Case-insensitive de-duplication that keeps the first spelling seen."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        key = normalize_whitespace(item).lower().rstrip(".")
        if key and key not in seen:
            seen.add(key)
            result.append(normalize_whitespace(item))
    return result
