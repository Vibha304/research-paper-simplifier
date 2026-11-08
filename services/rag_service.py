"""Retrieval-Augmented Generation: indexing, section-aware retrieval and context packing."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from services.chunking import Chunk
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore
from utils.progress import ProgressCallback

# Bonus added to cosine similarity when a chunk comes from a section that is
# known to answer the question (e.g. "limitations" for the limitations task).
SECTION_BOOST = 0.08


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    section: str
    page_start: int
    page_end: int
    score: float

    @property
    def page_label(self) -> str:
        return f"p. {self.page_start}" if self.page_start == self.page_end else f"pp. {self.page_start}-{self.page_end}"

    def to_source(self) -> dict[str, object]:
        return {
            "section": self.section,
            "pages": self.page_label,
            "score": round(self.score, 3),
            "excerpt": self.text[:300],
        }


class RAGService:
    def __init__(self, embeddings: EmbeddingService, store: VectorStore, default_top_k: int = 6) -> None:
        self._embeddings = embeddings
        self._store = store
        self._default_top_k = default_top_k

    def index(self, paper_id: str, chunks: Sequence[Chunk], progress: ProgressCallback | None = None) -> int:
        """Embed and persist chunks (replacing any previous vectors for the paper)."""
        self._store.delete_paper(paper_id)
        if not chunks:
            return 0
        vectors = self._embeddings.embed_documents([c.text for c in chunks], progress)
        self._store.add_chunks(
            paper_id,
            ids=[c.chunk_id(paper_id) for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[
                {"section": c.section, "page_start": c.page_start, "page_end": c.page_end, "index": c.index}
                for c in chunks
            ],
        )
        return len(chunks)

    def is_indexed(self, paper_id: str) -> bool:
        return self._store.chunk_count(paper_id) > 0

    def retrieve(
        self,
        paper_id: str,
        queries: str | Sequence[str],
        *,
        top_k: int | None = None,
        preferred_sections: Sequence[str] = (),
    ) -> list[RetrievedChunk]:
        """Multi-query retrieval with section-aware re-ranking and de-duplication."""
        query_list = [queries] if isinstance(queries, str) else list(queries)
        if not query_list:
            return []
        k = top_k or self._default_top_k
        best: dict[str, RetrievedChunk] = {}
        for vector in self._embeddings.embed_queries(query_list):
            for hit in self._store.query_chunks(paper_id, vector, top_k=k * 2):
                section = str(hit.metadata.get("section", "body"))
                score = hit.similarity + (SECTION_BOOST if section in preferred_sections else 0.0)
                current = best.get(hit.id)
                if current is None or score > current.score:
                    best[hit.id] = RetrievedChunk(
                        text=hit.document,
                        section=section,
                        page_start=int(hit.metadata.get("page_start", 0)),
                        page_end=int(hit.metadata.get("page_end", 0)),
                        score=score,
                    )
        return sorted(best.values(), key=lambda c: c.score, reverse=True)[:k]

    @staticmethod
    def format_context(chunks: Sequence[RetrievedChunk], max_chars: int) -> str:
        """Pack chunks (highest score first) into a citation-tagged context block."""
        parts: list[str] = []
        used = 0
        for number, chunk in enumerate(chunks, start=1):
            entry = f"[Excerpt {number} | {chunk.section} | {chunk.page_label}]\n{chunk.text}"
            if used + len(entry) > max_chars:
                remaining = max_chars - used
                if remaining > 300:
                    parts.append(entry[:remaining])
                break
            parts.append(entry)
            used += len(entry) + 2
        return "\n\n".join(parts)
