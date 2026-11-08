"""Global CSS. Colors use translucent tints so they work in light and dark themes."""

from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --rps-accent: #2563eb;
  --rps-accent-soft: rgba(37, 99, 235, 0.10);
  --rps-green: #16a34a;
  --rps-amber: #d97706;
  --rps-red: #dc2626;
  --rps-border: rgba(128, 128, 128, 0.25);
  --rps-card: rgba(128, 128, 128, 0.06);
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

.rps-hero { padding: 1.25rem 1.5rem; border-radius: 16px; margin-bottom: 1rem;
  background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(124,58,237,0.12));
  border: 1px solid var(--rps-border); }
.rps-hero h1 { font-size: clamp(1.4rem, 3.2vw, 2.1rem); margin: 0 0 .25rem 0; line-height: 1.2; }
.rps-hero p { margin: 0; opacity: .8; }

.rps-badges { display: flex; flex-wrap: wrap; gap: .4rem; margin: .5rem 0 1rem 0; }
.rps-badge { display: inline-flex; align-items: center; gap: .3rem; padding: .18rem .65rem;
  border-radius: 999px; font-size: .78rem; font-weight: 600; border: 1px solid var(--rps-border);
  background: var(--rps-card); white-space: nowrap; }
.rps-badge.accent { background: var(--rps-accent-soft); color: var(--rps-accent); border-color: transparent; }
.rps-badge.Beginner { background: rgba(22,163,74,.12); color: var(--rps-green); border-color: transparent; }
.rps-badge.Intermediate { background: rgba(217,119,6,.12); color: var(--rps-amber); border-color: transparent; }
.rps-badge.Advanced { background: rgba(220,38,38,.12); color: var(--rps-red); border-color: transparent; }

.rps-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: .8rem; margin: .5rem 0 1.2rem 0; }
.rps-card { border: 1px solid var(--rps-border); background: var(--rps-card); border-radius: 14px; padding: 1rem 1.1rem; }
.rps-card .label { font-size: .75rem; text-transform: uppercase; letter-spacing: .06em; opacity: .7; font-weight: 600; }
.rps-card .value { font-size: 1.55rem; font-weight: 700; margin: .15rem 0 .35rem 0; line-height: 1.2; }
.rps-card .hint { font-size: .8rem; opacity: .75; }
.rps-bar { height: 8px; border-radius: 999px; background: rgba(128,128,128,.2); overflow: hidden; }
.rps-bar > span { display: block; height: 100%; border-radius: 999px; }

.rps-section { border-left: 4px solid var(--rps-accent); padding: .1rem 0 .1rem .9rem; margin: 1.2rem 0 .6rem 0; }
.rps-section h3 { margin: 0; font-size: 1.1rem; }

.rps-timeline { position: relative; margin: .5rem 0 1rem .4rem; padding-left: 1.4rem; border-left: 2px solid var(--rps-border); }
.rps-step { position: relative; margin-bottom: 1rem; }
.rps-step::before { content: ""; position: absolute; left: -1.85rem; top: .3rem; width: .8rem; height: .8rem;
  border-radius: 50%; background: var(--rps-accent); box-shadow: 0 0 0 4px var(--rps-accent-soft); }
.rps-step .title { font-weight: 700; }
.rps-step .duration { font-size: .78rem; opacity: .7; margin-left: .4rem; }
.rps-step ul { margin: .3rem 0 0 0; padding-left: 1.1rem; }

.rps-empty { text-align: center; padding: 2.5rem 1rem; border: 2px dashed var(--rps-border); border-radius: 16px; opacity: .85; }
.rps-empty .icon { font-size: 2.4rem; }

.rps-source { font-size: .8rem; border-left: 3px solid var(--rps-border); padding-left: .6rem; margin: .4rem 0; opacity: .85; }

div[data-testid="stTabs"] button[role="tab"] p { font-size: .92rem; font-weight: 600; }
div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button { border-radius: 10px; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1rem; }
  .rps-hero { padding: 1rem; }
  .rps-card .value { font-size: 1.25rem; }
  .rps-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: .5rem; }
  div[data-testid="stTabs"] button[role="tab"] p { font-size: .8rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
