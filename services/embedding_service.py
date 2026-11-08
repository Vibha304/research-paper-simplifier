"""Batched text embedding on top of any ``LLMClient``."""

from __future__ import annotations

from collections.abc import Sequence

from services.llm_service import LLMClient
from utils.exceptions import LLMResponseFormatError
from utils.progress import ProgressCallback, noop_progress

# nomic-embed-text was trained with task prefixes; using them improves retrieval.
_PREFIXED_MODELS = ("nomic-embed-text",)


class EmbeddingService:
    def __init__(self, llm: LLMClient, batch_size: int = 16) -> None:
        self._llm = llm
        self._batch_size = max(1, batch_size)
        self._use_prefixes = any(m in llm.embedding_model for m in _PREFIXED_MODELS)

    @property
    def model_name(self) -> str:
        return self._llm.embedding_model

    def embed_documents(
        self, texts: Sequence[str], progress: ProgressCallback | None = None
    ) -> list[list[float]]:
        progress = progress or noop_progress
        prepared = [self._prefix(t, "search_document") for t in texts]
        vectors: list[list[float]] = []
        total = len(prepared)
        for start in range(0, total, self._batch_size):
            batch = prepared[start : start + self._batch_size]
            vectors.extend(self._embed_batch(batch))
            done = min(start + self._batch_size, total)
            progress(done / total, f"Embedding chunks: {done}/{total}")
        return vectors

    def embed_queries(self, queries: Sequence[str]) -> list[list[float]]:
        return self._embed_batch([self._prefix(q, "search_query") for q in queries])

    def embed_query(self, query: str) -> list[float]:
        return self.embed_queries([query])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        vectors = self._llm.embed(batch)
        if len(vectors) != len(batch) or any(not v for v in vectors):
            raise LLMResponseFormatError(
                f"Embedding model returned {len(vectors)} vectors for {len(batch)} inputs"
            )
        return vectors

    def _prefix(self, text: str, task: str) -> str:
        return f"{task}: {text}" if self._use_prefixes else text
