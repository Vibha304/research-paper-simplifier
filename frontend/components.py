"""Reusable, presentation-only Streamlit components.

All model-generated text is HTML-escaped before it is placed inside custom
HTML, so LLM output can never inject markup into the page.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st

from backend.models import NOT_FOUND, QAItem, RoadmapPhase


@dataclass(frozen=True)
class Badge:
    text: str
    variant: str = ""  # "", "accent", "Beginner", "Intermediate", "Advanced"


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="rps-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True
    )


def badges(items: list[Badge]) -> None:
    html = "".join(f'<span class="rps-badge {escape(b.variant)}">{escape(b.text)}</span>' for b in items if b.text)
    st.markdown(f'<div class="rps-badges">{html}</div>', unsafe_allow_html=True)


def score_color(score: int, *, invert: bool = False) -> str:
    """Green/amber/red by score (invert for difficulty: high = red)."""
    value = 11 - score if invert else score
    return "#16a34a" if value >= 7 else "#d97706" if value >= 4 else "#dc2626"


@dataclass(frozen=True)
class MetricCard:
    label: str
    value: str
    hint: str = ""
    fraction: float | None = None  # 0..1 fills a progress bar
    color: str = "#2563eb"


def metric_cards(cards: list[MetricCard]) -> None:
    parts = []
    for card in cards:
        bar = ""
        if card.fraction is not None:
            pct = max(0, min(100, round(card.fraction * 100)))
            bar = f'<div class="rps-bar"><span style="width:{pct}%;background:{escape(card.color)}"></span></div>'
        parts.append(
            f'<div class="rps-card"><div class="label">{escape(card.label)}</div>'
            f'<div class="value">{escape(card.value)}</div>{bar}'
            f'<div class="hint">{escape(card.hint)}</div></div>'
        )
    st.markdown(f'<div class="rps-grid">{"".join(parts)}</div>', unsafe_allow_html=True)


def section_title(title: str) -> None:
    st.markdown(f'<div class="rps-section"><h3>{escape(title)}</h3></div>', unsafe_allow_html=True)


def text_block(title: str, text: str) -> None:
    section_title(title)
    st.markdown(text or f"_{NOT_FOUND}_")


def bullet_list(title: str, items: list[str], *, numbered: bool = False) -> None:
    section_title(title)
    if not items:
        st.markdown(f"_{NOT_FOUND}_")
        return
    st.markdown("\n".join(f"{i}. {item}" if numbered else f"- {item}" for i, item in enumerate(items, 1)))


def chips(items: list[str]) -> None:
    badges([Badge(item, "accent") for item in items])


def qa_list(items: list[QAItem], *, key_prefix: str) -> None:
    if not items:
        st.info("No questions were generated for this paper.")
        return
    for index, qa in enumerate(items, 1):
        with st.expander(f"Q{index}. {qa.question}"):
            st.markdown(qa.answer or "_Answer not generated._")


def roadmap_timeline(phases: list[RoadmapPhase]) -> None:
    if not phases:
        st.markdown(f"_{NOT_FOUND}_")
        return
    steps = []
    for phase in phases:
        tasks = "".join(f"<li>{escape(task)}</li>" for task in phase.tasks)
        duration = f'<span class="duration">{escape(phase.duration)}</span>' if phase.duration else ""
        steps.append(
            f'<div class="rps-step"><span class="title">{escape(phase.phase)}</span>{duration}'
            f"{f'<ul>{tasks}</ul>' if tasks else ''}</div>"
        )
    st.markdown(f'<div class="rps-timeline">{"".join(steps)}</div>', unsafe_allow_html=True)


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="rps-empty"><div class="icon">{escape(icon)}</div><h4>{escape(title)}</h4>'
        f"<p>{escape(body)}</p></div>",
        unsafe_allow_html=True,
    )


def source_citations(sources: list[dict[str, object]]) -> None:
    if not sources:
        return
    with st.expander(f"📎 Sources from the paper ({len(sources)})"):
        for source in sources:
            st.markdown(
                f'<div class="rps-source"><b>{escape(str(source.get("pages", "")))}</b> · '
                f'{escape(str(source.get("section", "")).replace("_", " "))}<br>'
                f'{escape(str(source.get("excerpt", "")))}…</div>',
                unsafe_allow_html=True,
            )
