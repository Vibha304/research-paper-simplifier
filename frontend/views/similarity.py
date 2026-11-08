"""Research paper similarity search across the local library."""

from __future__ import annotations

import streamlit as st

from frontend import components as ui
from frontend import navigation, state
from services.similarity_service import SimilarPaper
from utils.exceptions import AppError


def _results(results: list[SimilarPaper], key: str) -> None:
    if not results:
        st.info("No similar papers found yet. Analyze more papers to grow your library.")
        return
    for rank, item in enumerate(results, 1):
        with st.container(border=True):
            info, action = st.columns([4, 1])
            with info:
                st.markdown(f"**{rank}. {item.title}**")
                st.progress(item.percent / 100, text=f"Similarity {item.percent}%")
                if item.keywords:
                    ui.chips(item.keywords[:6])
                if item.snippet:
                    st.caption(item.snippet + "...")
            if action.button("Open", key=f"{key}_{item.paper_id}", use_container_width=True):
                navigation.go_to(navigation.ANALYZE, paper_id=item.paper_id)


def render() -> None:
    container = state.get_container()
    ui.hero("🔍 Similar Papers", "Semantic search over your analyzed papers using local embeddings.")
    papers = container.library.analyzed_papers()
    if len(papers) == 0:
        ui.empty_state("🔍", "Library is empty", "Analyze at least two papers to compare them.")
        return

    top_k = st.slider("Number of results", 1, 10, 5)
    by_paper, by_topic = st.tabs(["📄 Similar to a paper", "🔤 Search by topic"])
    with by_paper:
        ids = [p.id for p in papers]
        titles = {p.id: p.title for p in papers}
        current = state.get(state.ACTIVE_PAPER)
        selected = st.selectbox(
            "Reference paper", ids, index=ids.index(current) if current in ids else 0,
            format_func=lambda pid: titles[pid],
        )
        try:
            _results(container.similarity.similar_to_paper(selected, top_k), "paper")
        except AppError as exc:
            st.error(exc.user_message)
    with by_topic:
        query = st.text_input("Describe a topic, idea or abstract", placeholder="e.g. lightweight intrusion detection for IoT")
        if query:
            try:
                with st.spinner("Searching..."):
                    _results(container.similarity.search(query, top_k), "topic")
            except AppError as exc:
                st.error(exc.user_message)
