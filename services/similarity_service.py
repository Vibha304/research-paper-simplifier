"""Paper-to-paper and topic-to-paper similarity search over analysis profiles."""

from __future__ import annotations

from dataclasses import dataclass

from backend.models import PaperAnalysis
from services.embedding_service import EmbeddingService
from services.vector_store import VectorStore


@dataclass(frozen=True)
class SimilarPaper:
    paper_id: str
    title: str
    similarity: float  # 0..1
    keywords: list[str]
    snippet: str

    @property
    def percent(self) -> int:
        return round(max(0.0, min(1.0, self.similarity)) * 100)


class SimilarityService:
    def __init__(self, embeddings: EmbeddingService, store: VectorStore) -> None:
        self._embeddings = embeddings
        self._store = store

    @staticmethod
    def build_profile_text(analysis: PaperAnalysis) -> str:
        """Condensed representation of a paper used for similarity search."""
        parts = [
            analysis.title,
            analysis.simple_summary,
            analysis.problem_statement,
            "Contributions: " + "; ".join(analysis.key_contributions),
            "Keywords: " + ", ".join(analysis.keywords),
            "Methods: " + analysis.methodology[:600],
        ]
        return "\n".join(p for p in parts if p and p.strip(": "))

    def index_paper(self, paper_id: str, analysis: PaperAnalysis) -> None:
        text = self.build_profile_text(analysis)
        vector = self._embeddings.embed_documents([text])[0]
        self._store.upsert_profile(
            paper_id,
            vector,
            document=analysis.simple_summary[:1000],
            metadata={"title": analysis.title[:500], "keywords": ", ".join(analysis.keywords)[:500]},
        )

    def similar_to_paper(self, paper_id: str, top_k: int = 5) -> list[SimilarPaper]:
        vector = self._store.get_profile_embedding(paper_id)
        if vector is None:
            return []
        return self._to_results(self._store.query_profiles(vector, top_k, exclude_id=paper_id))

    def search(self, query: str, top_k: int = 5) -> list[SimilarPaper]:
        if not query.strip():
            return []
        vector = self._embeddings.embed_query(query)
        return self._to_results(self._store.query_profiles(vector, top_k))

    @staticmethod
    def _to_results(hits: list) -> list[SimilarPaper]:
        return [
            SimilarPaper(
                paper_id=hit.id,
                title=str(hit.metadata.get("title", "Untitled")),
                similarity=hit.similarity,
                keywords=[k.strip() for k in str(hit.metadata.get("keywords", "")).split(",") if k.strip()],
                snippet=hit.document[:300],
            )
            for hit in hits
        ]
