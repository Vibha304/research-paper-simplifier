"""Upload page: validates the PDF, runs the pipeline with live progress, shows results."""

from __future__ import annotations

import time

import streamlit as st

from frontend import components as ui
from frontend import state
from frontend.views import analysis_view
from utils.exceptions import AppError
from utils.logger import get_logger

logger = get_logger("frontend.analyze")


def _health_banner() -> bool:
    """Show Ollama status; return True when analysis can run."""
    health = state.get_health()
    if state.get_container().settings.engine != "ollama":
        return True  # built-in engine: always ready
    if not health.reachable:
        st.error(
            f"**Ollama is not reachable.** {health.error or ''} Then refresh this page.",
            icon="🔌",
        )
        return False
    if health.missing_models:
        commands = "\n".join(f"ollama pull {m}" for m in health.missing_models)
        st.warning(f"**Missing local models.** Run in a terminal:\n```bash\n{commands}\n```", icon="📦")
        return False
    return True


def _run_analysis(data: bytes, filename: str, force: bool) -> None:
    container = state.get_container()
    with st.status(f"Analyzing **{filename}**...", expanded=True) as status:
        bar = st.progress(0, text="Starting...")
        log = st.empty()
        steps: list[str] = []
        started = time.perf_counter()

        def on_progress(fraction: float, message: str) -> None:
            elapsed = time.perf_counter() - started
            bar.progress(min(100, int(fraction * 100)), text=f"{message}  ·  {elapsed:.0f}s")
            stage = message.split(":")[0].split(" - ")[0]
            if not steps or steps[-1] != stage:
                steps.append(stage)
                log.markdown("\n".join(f"- {'✅' if i < len(steps) - 1 else '⏳'} {s}" for i, s in enumerate(steps)))

        try:
            outcome = container.pipeline.run(data, filename, on_progress, force=force)
        except AppError as exc:
            logger.warning("Analysis failed for %s: %s", filename, exc)
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error(exc.user_message, icon="❌")
            return
        except Exception:  # unexpected bug: log full traceback, show a safe message
            logger.exception("Unexpected error analysing %s", filename)
            status.update(label="Analysis failed", state="error", expanded=True)
            st.error("An unexpected error occurred. Details were written to data/logs/app.log.", icon="❌")
            return

        label = "Loaded saved analysis" if outcome.reused else f"Analysis complete in {outcome.duration_seconds:.0f}s"
        status.update(label=label, state="complete", expanded=False)
    state.set_value(state.ACTIVE_PAPER, outcome.paper.id)
    state.set_value(state.CHAT_PAPER, outcome.paper.id)
    st.rerun()


def render() -> None:
    container = state.get_container()
    settings = container.settings
    ui.hero(
        "AI Research Paper Simplifier",
        "Upload an IEEE, ACM, Springer or arXiv paper and get a student-friendly breakdown - 100% offline.",
    )
    ready = _health_banner()

    uploaded = st.file_uploader(
        "Upload a research paper (PDF)",
        type=["pdf"],
        help=f"Text-based PDFs up to {settings.max_pages} pages and {settings.max_file_size_mb} MB.",
    )
    if uploaded is not None:
        data = uploaded.getvalue()
        size_mb = len(data) / (1024 * 1024)
        ui.badges([ui.Badge(f"📄 {uploaded.name}", "accent"), ui.Badge(f"{size_mb:.1f} MB")])
        existing = container.pipeline.find_existing(data)
        force = False
        if existing is not None:
            st.info(f"You analyzed this paper before (**{existing.title}**).", icon="🗂️")
            force = st.checkbox("Re-analyze from scratch instead of loading the saved result")
        label = "🔁 Re-analyze paper" if force else ("📂 Open saved analysis" if existing else "🚀 Analyze paper")
        if st.button(label, type="primary", disabled=not ready and (existing is None or force),
                     use_container_width=True):
            _run_analysis(data, uploaded.name, force)

    active_id = state.get(state.ACTIVE_PAPER)
    if active_id:
        st.divider()
        try:
            analysis_view.render(container.library.load(active_id))
        except AppError as exc:
            state.pop(state.ACTIVE_PAPER)
            st.warning(exc.user_message)
    elif uploaded is None:
        ui.empty_state(
            "📄", "No paper selected",
            "Upload a PDF above, or open a previous analysis from the Library.",
        )
        with st.expander("What will I get?"):
            st.markdown(
                "- **Simple summary, problem statement & key contributions**\n"
                "- **Methodology explained simply** + implementation steps\n"
                "- **Datasets, tools/frameworks, limitations & future work**\n"
                "- **Student project idea** and a one-click **starter kit** (Build this Paper)\n"
                "- **Implementation roadmap**, difficulty, resume impact & time estimate\n"
                "- **Viva & interview questions**, chat assistant and similarity search\n"
                "- Export as **Markdown** or **PDF**"
            )
