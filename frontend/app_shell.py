"""Application shell: global styles, sidebar and multipage navigation."""

from __future__ import annotations

import streamlit as st

from frontend import navigation, state
from frontend.styles import inject_css
from frontend.views import analyze, chat, library, similarity, system
from utils.exceptions import AppError


def _sidebar() -> None:
    with st.sidebar:
        st.markdown("### 📄 Paper Simplifier")
        st.caption("Offline research assistant for engineering students")
        settings = state.get_container().settings
        health = state.get_health()
        if settings.engine != "ollama":
            st.success("Built-in engine ready", icon="🟢")
            st.caption("No installs, no internet, no API keys.")
            return
        if health.ready:
            st.success("Local AI ready", icon="🟢")
        elif health.reachable:
            st.warning("Models missing", icon="🟠")
        else:
            st.error("Ollama offline", icon="🔴")
        st.caption(f"LLM: `{settings.llm_model}` · Embeddings: `{settings.embedding_model}`")


def run() -> None:
    inject_css()
    try:
        state.get_container()
    except AppError as exc:
        st.error(f"Startup failed: {exc.user_message}")
        st.stop()

    pages = {
        navigation.ANALYZE: st.Page(analyze.render, title="Analyze Paper", icon="📤", default=True),
        navigation.LIBRARY: st.Page(library.render, title="Library", icon="📚", url_path="library"),
        navigation.CHAT: st.Page(chat.render, title="Chat Assistant", icon="💬", url_path="chat"),
        navigation.SIMILAR: st.Page(similarity.render, title="Similar Papers", icon="🔍", url_path="similar"),
        navigation.SYSTEM: st.Page(system.render, title="System Status", icon="⚙️", url_path="system"),
    }
    navigation.register(pages)
    current = st.navigation(list(pages.values()))
    _sidebar()
    state.show_flash()
    current.run()
