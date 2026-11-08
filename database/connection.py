"""SQLite connection management.

A new short-lived connection is opened per unit of work. This is the safest
pattern under Streamlit, which executes reruns on different threads.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from utils.exceptions import DatabaseError
from utils.logger import get_logger

logger = get_logger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    """Thin wrapper that owns the database file path and schema bootstrap."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(self) -> None:
        """Apply the (idempotent) schema and enable WAL for concurrent reads."""
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        except sqlite3.Error as exc:
            raise DatabaseError(f"Schema initialisation failed: {exc}") from exc
        logger.info("SQLite database ready at %s", self.path)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection inside a transaction (commit on success, rollback on error)."""
        try:
            conn = sqlite3.connect(self.path, timeout=30)
        except sqlite3.Error as exc:
            raise DatabaseError(f"Cannot open database: {exc}") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise DatabaseError(f"Database operation failed: {exc}") from exc
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
