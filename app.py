"""Streamlit entry point.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Cloud hosts (e.g. Streamlit Community Cloud) can ship an SQLite older than ChromaDB
# needs (>= 3.35). If the optional pysqlite3-binary package is installed (Linux only),
# use its bundled modern SQLite instead. On Windows/macOS this is a no-op.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    pass

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="AI Research Paper Simplifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

from frontend.app_shell import run  # noqa: E402  (must follow set_page_config)

run()
