"""Robust JSON extraction from LLM output.

Small local models occasionally wrap JSON in prose or Markdown fences, or emit
trailing commas. ``extract_json`` recovers the first well-formed JSON object.
"""

from __future__ import annotations

import json
import re
from typing import Any

from utils.exceptions import LLMResponseFormatError

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


def _balanced_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, respecting string literals."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : index + 1]
        start = text.find("{", start + 1)
    return None


def _try_load(candidate: str) -> dict[str, Any] | None:
    for attempt in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            value = json.loads(attempt)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def extract_json(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in ``text``.

    Raises:
        LLMResponseFormatError: if no JSON object can be recovered.
    """
    if not text or not text.strip():
        raise LLMResponseFormatError("Empty response from model")

    candidates: list[str] = [text.strip()]
    candidates.extend(match.strip() for match in _FENCE.findall(text))
    balanced = _balanced_object(text)
    if balanced:
        candidates.append(balanced)

    for candidate in candidates:
        parsed = _try_load(candidate)
        if parsed is not None:
            return parsed
    raise LLMResponseFormatError(f"No JSON object found in model output: {text[:200]!r}")
