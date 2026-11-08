-- =====================================================================
-- Research Paper Simplifier - SQLite schema (version 1)
-- Idempotent: safe to execute on every application start.
-- =====================================================================

PRAGMA foreign_keys = ON;

-- One row per unique uploaded PDF (deduplicated by SHA-256 content hash).
CREATE TABLE IF NOT EXISTS papers (
    id            TEXT    PRIMARY KEY,                 -- uuid4 hex
    title         TEXT    NOT NULL,
    filename      TEXT    NOT NULL,
    file_hash     TEXT    NOT NULL UNIQUE,             -- sha256 of file bytes
    page_count    INTEGER NOT NULL CHECK (page_count > 0),
    word_count    INTEGER NOT NULL DEFAULT 0,
    source        TEXT    NOT NULL DEFAULT 'Unknown',  -- IEEE / ACM / Springer / arXiv / ...
    chunk_count   INTEGER NOT NULL DEFAULT 0,          -- chunks indexed in ChromaDB
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_created ON papers (created_at DESC);

-- Every analysis run of a paper (re-analysis keeps history).
CREATE TABLE IF NOT EXISTS analyses (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id             TEXT    NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    llm_model            TEXT    NOT NULL,
    embedding_model      TEXT    NOT NULL,
    analysis_json        TEXT    NOT NULL,             -- serialized PaperAnalysis
    difficulty_level     TEXT    CHECK (difficulty_level IN ('Beginner', 'Intermediate', 'Advanced')),
    difficulty_score     INTEGER CHECK (difficulty_score BETWEEN 1 AND 10),
    resume_impact_score  INTEGER CHECK (resume_impact_score BETWEEN 1 AND 10),
    estimated_hours      INTEGER CHECK (estimated_hours >= 0),
    duration_seconds     REAL    NOT NULL DEFAULT 0,
    created_at           TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_analyses_paper ON analyses (paper_id, id DESC);

-- Research-paper chat assistant conversation history.
CREATE TABLE IF NOT EXISTS chat_messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    paper_id      TEXT    NOT NULL REFERENCES papers (id) ON DELETE CASCADE,
    role          TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content       TEXT    NOT NULL,
    sources_json  TEXT,                                -- JSON list of cited chunks
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_paper ON chat_messages (paper_id, id);

-- Library view: each paper joined with its most recent analysis.
CREATE VIEW IF NOT EXISTS v_paper_library AS
SELECT
    p.id, p.title, p.filename, p.file_hash, p.page_count, p.word_count, p.source,
    p.chunk_count, p.created_at,
    a.id                  AS analysis_id,
    a.difficulty_level,
    a.difficulty_score,
    a.resume_impact_score,
    a.estimated_hours,
    a.created_at          AS analyzed_at
FROM papers p
LEFT JOIN analyses a
       ON a.id = (SELECT MAX(id) FROM analyses WHERE paper_id = p.id);

CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER PRIMARY KEY,
    applied_at  TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

INSERT OR IGNORE INTO schema_version (version) VALUES (1);
