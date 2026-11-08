"""History library of previously analyzed papers."""

from __future__ import annotations

import streamlit as st

from frontend import components as ui
from frontend import navigation, state
from utils.exceptions import AppError


def render() -> None:
    container = state.get_container()
    ui.hero("📚 Library", "Every analysis is stored locally in SQLite - reopen, chat or delete at any time.")

    stats = container.library.stats()
    ui.metric_cards(
        [
            ui.MetricCard("Papers", str(stats["papers"])),
            ui.MetricCard("Analyzed", str(stats["analyzed"])),
            ui.MetricCard("Pages processed", f"{stats['pages']:,}"),
            ui.MetricCard("Chunks indexed", f"{stats['chunks']:,}"),
        ]
    )

    search = st.text_input("Search by title or filename", placeholder="e.g. transformer, intrusion detection...")
    entries = container.library.list(search)
    if not entries:
        ui.empty_state("🗂️", "Nothing here yet", "Analyze a paper and it will appear in your library.")
        return

    for entry in entries:
        paper = entry.paper
        with st.container(border=True):
            info, actions = st.columns([3, 2])
            with info:
                st.markdown(f"**{paper.title}**")
                ui.badges(
                    [
                        ui.Badge(paper.source if paper.source != "Unknown" else "", "accent"),
                        ui.Badge(f"{paper.page_count} pages"),
                        ui.Badge(entry.difficulty_level or "Not analyzed", entry.difficulty_level or ""),
                        ui.Badge(f"Resume {entry.resume_impact_score}/10" if entry.resume_impact_score else ""),
                        ui.Badge(f"~{entry.estimated_hours} h" if entry.estimated_hours else ""),
                        ui.Badge(f"🗓 {(entry.analyzed_at or paper.created_at)[:10]}"),
                    ]
                )
            with actions:
                b1, b2, b3 = st.columns(3)
                if b1.button("Open", key=f"open_{paper.id}", use_container_width=True,
                             disabled=entry.analysis_id is None):
                    navigation.go_to(navigation.ANALYZE, paper_id=paper.id)
                if b2.button("Chat", key=f"chat_{paper.id}", use_container_width=True):
                    navigation.go_to(navigation.CHAT, paper_id=paper.id)
                with b3.popover("Delete", use_container_width=True):
                    st.write("Delete this paper, its analyses, chat history and vectors?")
                    if st.button("Yes, delete", key=f"del_{paper.id}", type="primary"):
                        try:
                            container.library.delete(paper.id)
                        except AppError as exc:
                            st.error(exc.user_message)
                        else:
                            for key in (state.ACTIVE_PAPER, state.CHAT_PAPER):
                                if state.get(key) == paper.id:
                                    state.pop(key)
                            state.flash("Paper deleted", "🗑️")
                            st.rerun()
