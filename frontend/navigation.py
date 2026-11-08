"""Page registry so views can navigate without importing each other."""

from __future__ import annotations

from typing import Any

import streamlit as st

from frontend import state

ANALYZE, LIBRARY, CHAT, SIMILAR, SYSTEM = "analyze", "library", "chat", "similar", "system"
_PAGES: dict[str, Any] = {}


def register(pages: dict[str, Any]) -> None:
    _PAGES.clear()
    _PAGES.update(pages)


def go_to(page: str, *, paper_id: str | None = None) -> None:
    """Navigate to a page, optionally selecting a paper first."""
    if paper_id is not None:
        state.set_value(state.ACTIVE_PAPER, paper_id)
        state.set_value(state.CHAT_PAPER, paper_id)
    st.switch_page(_PAGES[page])
