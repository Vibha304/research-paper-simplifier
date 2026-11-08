"""Repository classes: the only place that contains SQL statements."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from database.connection import Database


# --------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class PaperRecord:
    id: str
    title: str
    filename: str
    file_hash: str
    page_count: int
    word_count: int
    source: str
    chunk_count: int
    created_at: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> PaperRecord:
        return cls(**{k: row[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class AnalysisRecord:
    id: int
    paper_id: str
    llm_model: str
    embedding_model: str
    analysis: dict[str, Any]
    difficulty_level: str | None
    difficulty_score: int | None
    resume_impact_score: int | None
    estimated_hours: int | None
    duration_seconds: float
    created_at: str

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> AnalysisRecord:
        return cls(
            id=row["id"],
            paper_id=row["paper_id"],
            llm_model=row["llm_model"],
            embedding_model=row["embedding_model"],
            analysis=json.loads(row["analysis_json"]),
            difficulty_level=row["difficulty_level"],
            difficulty_score=row["difficulty_score"],
            resume_impact_score=row["resume_impact_score"],
            estimated_hours=row["estimated_hours"],
            duration_seconds=row["duration_seconds"],
            created_at=row["created_at"],
        )


@dataclass(frozen=True)
class LibraryEntry:
    """A paper with summary fields of its latest analysis (from ``v_paper_library``)."""

    paper: PaperRecord
    analysis_id: int | None
    difficulty_level: str | None
    difficulty_score: int | None
    resume_impact_score: int | None
    estimated_hours: int | None
    analyzed_at: str | None


@dataclass(frozen=True)
class ChatMessageRecord:
    id: int
    paper_id: str
    role: str
    content: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = ""


# --------------------------------------------------------------------------------------
# Repositories
# --------------------------------------------------------------------------------------
class PaperRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def upsert(self, paper: PaperRecord) -> None:
        with self._db.connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (id, title, filename, file_hash, page_count, word_count, source, chunk_count)
                VALUES (:id, :title, :filename, :file_hash, :page_count, :word_count, :source, :chunk_count)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title, filename = excluded.filename,
                    page_count = excluded.page_count, word_count = excluded.word_count,
                    source = excluded.source, chunk_count = excluded.chunk_count
                """,
                {k: getattr(paper, k) for k in PaperRecord.__dataclass_fields__ if k != "created_at"},
            )

    def get(self, paper_id: str) -> PaperRecord | None:
        with self._db.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE id = ?", (paper_id,)).fetchone()
        return PaperRecord.from_row(row) if row else None

    def get_by_hash(self, file_hash: str) -> PaperRecord | None:
        with self._db.connect() as conn:
            row = conn.execute("SELECT * FROM papers WHERE file_hash = ?", (file_hash,)).fetchone()
        return PaperRecord.from_row(row) if row else None

    def list_library(self, search: str = "", limit: int = 200) -> list[LibraryEntry]:
        query = "SELECT * FROM v_paper_library"
        params: list[Any] = []
        if search.strip():
            query += " WHERE title LIKE ? OR filename LIKE ?"
            like = f"%{search.strip()}%"
            params.extend([like, like])
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._db.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            LibraryEntry(
                paper=PaperRecord.from_row(row),
                analysis_id=row["analysis_id"],
                difficulty_level=row["difficulty_level"],
                difficulty_score=row["difficulty_score"],
                resume_impact_score=row["resume_impact_score"],
                estimated_hours=row["estimated_hours"],
                analyzed_at=row["analyzed_at"],
            )
            for row in rows
        ]

    def count(self) -> int:
        with self._db.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0])

    def delete(self, paper_id: str) -> bool:
        with self._db.connect() as conn:
            cursor = conn.execute("DELETE FROM papers WHERE id = ?", (paper_id,))
        return cursor.rowcount > 0


class AnalysisRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(
        self,
        *,
        paper_id: str,
        llm_model: str,
        embedding_model: str,
        analysis: dict[str, Any],
        duration_seconds: float,
    ) -> int:
        assessment = analysis.get("assessment", {})
        with self._db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analyses (paper_id, llm_model, embedding_model, analysis_json,
                    difficulty_level, difficulty_score, resume_impact_score, estimated_hours, duration_seconds)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    paper_id,
                    llm_model,
                    embedding_model,
                    json.dumps(analysis, ensure_ascii=False),
                    assessment.get("difficulty_level"),
                    assessment.get("difficulty_score"),
                    assessment.get("resume_impact_score"),
                    assessment.get("estimated_hours"),
                    duration_seconds,
                ),
            )
        return int(cursor.lastrowid or 0)

    def latest_for_paper(self, paper_id: str) -> AnalysisRecord | None:
        with self._db.connect() as conn:
            row = conn.execute(
                "SELECT * FROM analyses WHERE paper_id = ? ORDER BY id DESC LIMIT 1", (paper_id,)
            ).fetchone()
        return AnalysisRecord.from_row(row) if row else None

    def list_for_paper(self, paper_id: str) -> list[AnalysisRecord]:
        with self._db.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM analyses WHERE paper_id = ? ORDER BY id DESC", (paper_id,)
            ).fetchall()
        return [AnalysisRecord.from_row(row) for row in rows]


class ChatRepository:
    def __init__(self, db: Database) -> None:
        self._db = db

    def add(self, paper_id: str, role: str, content: str, sources: list[dict[str, Any]] | None = None) -> int:
        with self._db.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO chat_messages (paper_id, role, content, sources_json) VALUES (?, ?, ?, ?)",
                (paper_id, role, content, json.dumps(sources) if sources else None),
            )
        return int(cursor.lastrowid or 0)

    def list_for_paper(self, paper_id: str, limit: int | None = None) -> list[ChatMessageRecord]:
        query = "SELECT * FROM chat_messages WHERE paper_id = ? ORDER BY id"
        with self._db.connect() as conn:
            rows = conn.execute(query, (paper_id,)).fetchall()
        records = [
            ChatMessageRecord(
                id=row["id"],
                paper_id=row["paper_id"],
                role=row["role"],
                content=row["content"],
                sources=json.loads(row["sources_json"]) if row["sources_json"] else [],
                created_at=row["created_at"],
            )
            for row in rows
        ]
        return records[-limit:] if limit else records

    def clear(self, paper_id: str) -> None:
        with self._db.connect() as conn:
            conn.execute("DELETE FROM chat_messages WHERE paper_id = ?", (paper_id,))
