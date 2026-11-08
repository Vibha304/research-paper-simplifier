"""Renders a stored analysis as a tabbed, student-friendly report."""

from __future__ import annotations

import json

import streamlit as st

from backend.library_service import StoredAnalysis
from backend.models import PaperAnalysis
from frontend import components as ui
from frontend import navigation, state
from utils.exceptions import AppError
from utils.text import slugify

TABS = (
    "📘 Overview", "⚙️ Methodology", "🧰 Resources", "🔭 Limitations & Future",
    "🚀 Build This Paper", "🎓 Viva & Interview", "📊 Scores", "📥 Export",
)


@st.cache_data(show_spinner=False, max_entries=32)
def _pdf_bytes(analysis_id: int, analysis_json: str, filename: str, pages: int) -> bytes:
    analysis = PaperAnalysis.model_validate_json(analysis_json)
    return state.get_container().exporter.to_pdf(analysis, filename=filename, page_count=pages)


@st.cache_data(show_spinner=False, max_entries=32)
def _kit_bytes(analysis_id: int, analysis_json: str) -> tuple[str, bytes]:
    return state.get_container().project_kit.build_zip(PaperAnalysis.model_validate_json(analysis_json))


def render(stored: StoredAnalysis) -> None:
    paper, record, a = stored.paper, stored.record, stored.analysis
    s = a.assessment

    st.markdown(f"## {a.title or paper.title}")
    ui.badges(
        [
            ui.Badge(paper.source if paper.source != "Unknown" else "", "accent"),
            ui.Badge(f"📄 {paper.page_count} pages"),
            ui.Badge(f"📝 {paper.word_count:,} words"),
            ui.Badge(f"🧩 {paper.chunk_count} chunks indexed"),
            ui.Badge(s.difficulty_level, s.difficulty_level),
            ui.Badge(f"🤖 {record.llm_model}"),
            ui.Badge(f"⏱ analyzed in {record.duration_seconds:.0f}s" if record.duration_seconds else ""),
        ]
    )
    ui.metric_cards(
        [
            ui.MetricCard("Difficulty", f"{s.difficulty_level}", f"{s.difficulty_score}/10",
                          s.difficulty_score / 10, ui.score_color(s.difficulty_score, invert=True)),
            ui.MetricCard("Resume impact", f"{s.resume_impact_score}/10", "How much it boosts your CV",
                          s.resume_impact_score / 10, ui.score_color(s.resume_impact_score)),
            ui.MetricCard("Estimated time", f"{s.estimated_hours} h", s.estimated_time_label.split("(")[-1].rstrip(")")),
            ui.MetricCard("Viva / Interview Qs", f"{len(a.viva_questions)} / {len(a.interview_questions)}",
                          "Ready for practice"),
        ]
    )

    actions = st.columns([1, 1, 2])
    if actions[0].button("💬 Chat with this paper", use_container_width=True, key=f"chat_{paper.id}"):
        navigation.go_to(navigation.CHAT, paper_id=paper.id)
    if actions[1].button("🔍 Find similar papers", use_container_width=True, key=f"sim_{paper.id}"):
        navigation.go_to(navigation.SIMILAR, paper_id=paper.id)

    if a.warnings:
        with st.expander(f"⚠️ {len(a.warnings)} note(s) about this analysis"):
            for warning in a.warnings:
                st.markdown(f"- {warning}")

    tabs = st.tabs(list(TABS))
    with tabs[0]:
        ui.text_block("Simple Summary", a.simple_summary)
        ui.text_block("Problem Statement", a.problem_statement)
        ui.bullet_list("Key Contributions", a.key_contributions)
        if a.keywords:
            ui.section_title("Keywords")
            ui.chips(a.keywords)
    with tabs[1]:
        ui.text_block("Methodology Explained Simply", a.methodology)
        ui.bullet_list("Implementation Steps", a.implementation_steps, numbered=True)
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            ui.bullet_list("Datasets Used", a.datasets)
        with col2:
            ui.bullet_list("Tools & Frameworks", a.tools_frameworks)
        st.caption("Items marked *(detected in text)* were found by exact keyword match in the PDF.")
    with tabs[3]:
        col1, col2 = st.columns(2)
        with col1:
            ui.bullet_list("Limitations", a.limitations)
        with col2:
            ui.bullet_list("Future Work", a.future_work)
    with tabs[4]:
        _render_build_tab(record.id, a)
    with tabs[5]:
        viva_tab, interview_tab = st.tabs(["🎓 Viva questions", "💼 Interview questions"])
        with viva_tab:
            ui.qa_list(a.viva_questions, key_prefix="viva")
        with interview_tab:
            ui.qa_list(a.interview_questions, key_prefix="interview")
    with tabs[6]:
        _render_scores_tab(a)
    with tabs[7]:
        _render_export_tab(stored)


def _render_build_tab(analysis_id: int, a: PaperAnalysis) -> None:
    p, b = a.project_idea, a.build_plan
    st.markdown("#### ⚡ One-click: Build this Paper as a Project")
    st.caption(
        "Download a ready-to-code starter kit: README, roadmap checklist, requirements.txt, "
        "module skeletons, smoke test and viva prep notes."
    )
    try:
        kit_name, kit_bytes = _kit_bytes(analysis_id, a.model_dump_json())
        st.download_button(
            "📦 Download project starter kit (.zip)", kit_bytes, file_name=kit_name,
            mime="application/zip", type="primary", use_container_width=True,
        )
    except AppError as exc:
        st.error(exc.user_message)

    ui.section_title(f"💡 Project Idea: {p.title or 'Student project'}")
    st.markdown(p.description or "_Not generated._")
    col1, col2 = st.columns(2)
    with col1:
        ui.bullet_list("Features", p.features)
        ui.bullet_list("Deliverables", p.deliverables)
    with col2:
        ui.section_title("Tech Stack")
        ui.chips(p.tech_stack) if p.tech_stack else st.markdown("_Not specified._")
        ui.text_block("Objective", b.objective)
        ui.text_block("Architecture", b.architecture)
    ui.bullet_list("Modules", b.modules)
    col3, col4 = st.columns(2)
    with col3:
        ui.bullet_list("Setup Steps", b.setup_steps, numbered=True)
    with col4:
        ui.bullet_list("Evaluation Metrics", b.evaluation_metrics)
        ui.bullet_list("Stretch Goals", b.stretch_goals)
    ui.section_title("🗺️ Implementation Roadmap")
    ui.roadmap_timeline(a.roadmap)


def _render_scores_tab(a: PaperAnalysis) -> None:
    s = a.assessment
    ui.section_title(f"Difficulty: {s.difficulty_level} ({s.difficulty_score}/10)")
    st.progress(s.difficulty_score / 10)
    st.markdown(s.difficulty_rationale or "_No rationale._")
    ui.section_title(f"Resume Impact: {s.resume_impact_score}/10")
    st.progress(s.resume_impact_score / 10)
    st.markdown(s.resume_impact_rationale or "_No rationale._")
    ui.section_title("Estimated Implementation Time")
    st.markdown(f"**{s.estimated_time_label or f'{s.estimated_hours} hours'}**")
    ui.bullet_list("Skills You Will Gain", s.skills_gained)
    st.caption(
        "Scores blend the local LLM's judgement (60%) with deterministic text heuristics (40%): "
        "math density, advanced terminology, paper length, topic trends and resource availability."
    )


def _render_export_tab(stored: StoredAnalysis) -> None:
    paper, record, a = stored.paper, stored.record, stored.analysis
    exporter = state.get_container().exporter
    base_name = slugify(a.title or paper.title, 60)
    markdown = exporter.to_markdown(a, filename=paper.filename, page_count=paper.page_count)

    col1, col2, col3 = st.columns(3)
    col1.download_button("⬇️ Markdown (.md)", markdown, file_name=f"{base_name}.md",
                         mime="text/markdown", use_container_width=True)
    try:
        pdf = _pdf_bytes(record.id, a.model_dump_json(), paper.filename, paper.page_count)
        col2.download_button("⬇️ PDF report", pdf, file_name=f"{base_name}.pdf",
                             mime="application/pdf", use_container_width=True)
    except AppError as exc:
        col2.error(exc.user_message)
    col3.download_button("⬇️ Raw JSON", json.dumps(a.model_dump(), indent=2, ensure_ascii=False),
                         file_name=f"{base_name}.json", mime="application/json", use_container_width=True)
    with st.expander("Preview Markdown", expanded=False):
        st.markdown(markdown)
