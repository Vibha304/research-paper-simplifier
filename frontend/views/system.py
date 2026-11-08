"""System status: active engine, configuration and storage."""

from __future__ import annotations

import streamlit as st

from frontend import components as ui
from frontend import state


def _builtin_status() -> None:
    ui.metric_cards(
        [
            ui.MetricCard("Engine", "Built-in", "Ready - nothing to install", 1.0, "#16a34a"),
            ui.MetricCard("Internet", "Not needed", "Runs 100% on this computer"),
            ui.MetricCard("Cost", "Free", "No API keys or subscriptions"),
        ]
    )
    st.markdown(
        "The built-in engine analyses papers with classic NLP techniques: it ranks the most important sentences "
        "of each section, extracts datasets, tools, metrics and jargon, and fills paper-specific templates for "
        "the project kit, roadmap, scores and viva/interview questions. Everything it states about the paper is "
        "taken from the paper itself, with page citations."
    )
    with st.expander("Optional: use a local AI model for more natural, rewritten explanations"):
        st.markdown(
            "You don't need this. If you ever want generated (not extracted) text, install the free "
            "[Ollama](https://ollama.com) app, run `ollama pull llama3.2` and `ollama pull nomic-embed-text`, "
            "then add `RPS_ENGINE=ollama` to your `.env` file and restart the app."
        )


def _ollama_status() -> None:
    settings = state.get_container().settings
    if st.button("🔄 Re-check Ollama"):
        state.get_health.clear()
    health = state.get_health()
    ui.metric_cards(
        [
            ui.MetricCard("Ollama", "Online" if health.reachable else "Offline", settings.ollama_base_url,
                          1.0 if health.reachable else 0.0, "#16a34a" if health.reachable else "#dc2626"),
            ui.MetricCard("LLM", settings.llm_model,
                          "installed" if health.reachable and settings.llm_model not in health.missing_models else "missing"),
            ui.MetricCard("Embeddings", settings.embedding_model,
                          "installed" if health.reachable and settings.embedding_model not in health.missing_models
                          else "missing"),
        ]
    )
    if not health.ready:
        st.code("ollama serve\n" + "\n".join(f"ollama pull {m}" for m in
                (health.missing_models or [settings.llm_model, settings.embedding_model])), language="bash")
        st.info("Or remove `RPS_ENGINE=ollama` from `.env` to switch back to the built-in engine.")


def render() -> None:
    container = state.get_container()
    settings = container.settings
    ui.hero("⚙️ System Status", "Everything runs locally: ChromaDB for search vectors, SQLite for your history.")
    if settings.engine == "ollama":
        _ollama_status()
    else:
        _builtin_status()

    ui.section_title("Configuration")
    st.caption("Override any value with environment variables or a .env file (see .env.example).")
    st.table(
        {
            "Setting": ["Engine", "Max pages", "Max file size", "Chunk size / overlap", "Retrieval top-k",
                        "SQLite", "ChromaDB", "Logs"],
            "Value": [settings.engine, str(settings.max_pages), f"{settings.max_file_size_mb} MB",
                      f"{settings.chunk_size} / {settings.chunk_overlap} chars", str(settings.retrieval_top_k),
                      str(settings.sqlite_path), str(settings.chroma_dir), str(settings.log_dir)],
        }
    )
