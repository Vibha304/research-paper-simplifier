"""Centralised, environment-driven application settings.

All tunables are read once from environment variables (optionally loaded from a
``.env`` file in the project root) and exposed as an immutable ``Settings``
object. Every variable is prefixed with ``RPS_`` (Research Paper Simplifier).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is listed in requirements
    load_dotenv = None  # type: ignore[assignment]

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable application configuration."""

    # --- Analysis engine: "builtin" (default, zero setup) or "ollama" (optional) ---
    engine: str

    # --- Optional local LLM (Ollama) ---
    ollama_base_url: str
    llm_model: str
    embedding_model: str
    llm_temperature: float
    llm_num_ctx: int
    llm_timeout_seconds: int
    llm_max_retries: int

    # --- Storage ---
    data_dir: Path
    sqlite_path: Path
    chroma_dir: Path
    log_dir: Path
    log_level: str

    # --- PDF limits ---
    max_pages: int
    max_file_size_mb: int
    min_chars_per_page: int

    # --- Chunking & retrieval ---
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    max_context_chars: int
    embed_batch_size: int
    chat_history_turns: int

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create every directory the app writes to."""
        for directory in (self.data_dir, self.chroma_dir, self.log_dir, self.sqlite_path.parent):
            directory.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build a fresh ``Settings`` object from the current environment."""
    data_dir = Path(_env_str("RPS_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
    return Settings(
        engine=_env_str("RPS_ENGINE", "builtin").lower(),
        ollama_base_url=_env_str("RPS_OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/"),
        llm_model=_env_str("RPS_LLM_MODEL", "llama3.2"),
        embedding_model=_env_str("RPS_EMBEDDING_MODEL", "nomic-embed-text"),
        llm_temperature=_env_float("RPS_LLM_TEMPERATURE", 0.2),
        llm_num_ctx=_env_int("RPS_LLM_NUM_CTX", 8192),
        llm_timeout_seconds=_env_int("RPS_LLM_TIMEOUT_SECONDS", 300),
        llm_max_retries=_env_int("RPS_LLM_MAX_RETRIES", 2),
        data_dir=data_dir,
        sqlite_path=Path(_env_str("RPS_SQLITE_PATH", str(data_dir / "history.db"))).expanduser(),
        chroma_dir=Path(_env_str("RPS_CHROMA_DIR", str(data_dir / "chroma"))).expanduser(),
        log_dir=Path(_env_str("RPS_LOG_DIR", str(data_dir / "logs"))).expanduser(),
        log_level=_env_str("RPS_LOG_LEVEL", "INFO").upper(),
        max_pages=_env_int("RPS_MAX_PAGES", 100),
        max_file_size_mb=_env_int("RPS_MAX_FILE_SIZE_MB", 50),
        min_chars_per_page=_env_int("RPS_MIN_CHARS_PER_PAGE", 80),
        chunk_size=_env_int("RPS_CHUNK_SIZE", 1400),
        chunk_overlap=_env_int("RPS_CHUNK_OVERLAP", 200),
        retrieval_top_k=_env_int("RPS_RETRIEVAL_TOP_K", 6),
        max_context_chars=_env_int("RPS_MAX_CONTEXT_CHARS", 9000),
        embed_batch_size=_env_int("RPS_EMBED_BATCH_SIZE", 16),
        chat_history_turns=_env_int("RPS_CHAT_HISTORY_TURNS", 4),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton (directories are created)."""
    settings = load_settings()
    settings.ensure_directories()
    return settings
