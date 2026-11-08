"""Research paper chat assistant (RAG over the selected paper)."""

from __future__ import annotations

import streamlit as st

from frontend import components as ui
from frontend import navigation, state
from services.chat_service import SUGGESTED_QUESTIONS
from utils.exceptions import AppError
from utils.logger import get_logger

logger = get_logger("frontend.chat")


def _paper_selector() -> tuple[str, str] | None:
    papers = state.get_container().library.analyzed_papers()
    if not papers:
        ui.empty_state("💬", "No papers to chat with", "Analyze a paper first - then ask it anything.")
        if st.button("Go to Analyze", type="primary"):
            navigation.go_to(navigation.ANALYZE)
        return None
    ids = [p.id for p in papers]
    titles = {p.id: p.title for p in papers}
    current = state.get(state.CHAT_PAPER)
    index = ids.index(current) if current in ids else 0
    selected = st.selectbox("Paper", ids, index=index, format_func=lambda pid: titles[pid])
    state.set_value(state.CHAT_PAPER, selected)
    return selected, titles[selected]


def render() -> None:
    ui.hero("💬 Chat Assistant", "Ask questions about a paper. Answers are grounded in retrieved excerpts with page citations.")
    selection = _paper_selector()
    if selection is None:
        return
    paper_id, title = selection
    chat = state.get_container().chat

    history = chat.history(paper_id)
    top = st.columns([4, 1])
    top[0].caption(f"{len(history)} messages · stored locally")
    if top[1].button("🧹 Clear chat", use_container_width=True, disabled=not history):
        chat.clear(paper_id)
        st.rerun()

    for message in history:
        with st.chat_message(message.role, avatar="🧑‍🎓" if message.role == "user" else "📄"):
            st.markdown(message.content)
            if message.role == "assistant":
                ui.source_citations(message.sources)

    if not history:
        st.markdown("**Try asking:**")
        cols = st.columns(2)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            if cols[i % 2].button(question, key=f"suggest_{i}", use_container_width=True):
                state.set_value(state.QUEUED_QUESTION, question)
                st.rerun()

    question = st.chat_input("Ask about the methodology, equations, results...") or state.pop(state.QUEUED_QUESTION)
    if not question:
        return

    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(question)
    with st.chat_message("assistant", avatar="📄"):
        try:
            with st.spinner("Searching the paper..."):
                answer = chat.ask(paper_id, title, question)
            st.write_stream(answer.tokens)
            ui.source_citations(answer.sources)
        except AppError as exc:
            st.error(exc.user_message)
        except Exception:
            logger.exception("Chat failed")
            st.error("Unexpected error while answering. See data/logs/app.log.")
