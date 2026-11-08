"""Typed accessors for Streamlit session state and cached resources."""

from __future__ import annotations

from typing import Any

import streamlit as st

from backend.container import AppContainer, build_container
from services.llm_service import HealthStatus

ACTIVE_PAPER = "active_paper_id"
CHAT_PAPER = "chat_paper_id"
QUEUED_QUESTION = "queued_question"
FLASH = "flash_message"


@st.cache_resource(show_spinner="Starting local services...")
def get_container() -> AppContainer:
    """One container per server process (shared across sessions)."""
    return build_container()


@st.cache_data(ttl=20, show_spinner=False)
def get_health() -> HealthStatus:
    return get_container().llm.health_check()  # type: ignore[attr-defined]


def get(key: str, default: Any = None) -> Any:
    return st.session_state.get(key, default)


def set_value(key: str, value: Any) -> None:
    st.session_state[key] = value


def pop(key: str, default: Any = None) -> Any:
    return st.session_state.pop(key, default)


def flash(message: str, icon: str = "✅") -> None:
    """Show a toast on the next rerun (survives st.rerun / switch_page)."""
    st.session_state[FLASH] = (message, icon)


def show_flash() -> None:
    pending = st.session_state.pop(FLASH, None)
    if pending:
        st.toast(pending[0], icon=pending[1])
