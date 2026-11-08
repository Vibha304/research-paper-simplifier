"""Read/delete operations over stored papers (history library)."""

from __future__ import annotations

from dataclasses import dataclass

from backend.models import PaperAnalysis
from database.repository import AnalysisRecord, AnalysisRepository, LibraryEntry, PaperRecord, PaperRepository
from services.vector_store import VectorStore
from utils.exceptions import AppError, NotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class StoredAnalysis:
    paper: PaperRecord
    record: AnalysisRecord
    analysis: PaperAnalysis


class LibraryService:
    def __init__(self, papers: PaperRepository, analyses: AnalysisRepository, store: VectorStore) -> None:
        self._papers = papers
        self._analyses = analyses
        self._store = store

    def list(self, search: str = "") -> list[LibraryEntry]:
        return self._papers.list_library(search)

    def analyzed_papers(self) -> list[PaperRecord]:
        return [entry.paper for entry in self._papers.list_library() if entry.analysis_id is not None]

    def get_paper(self, paper_id: str) -> PaperRecord:
        paper = self._papers.get(paper_id)
        if paper is None:
            raise NotFoundError(user_message="This paper no longer exists in your library.")
        return paper

    def load(self, paper_id: str) -> StoredAnalysis:
        paper = self.get_paper(paper_id)
        record = self._analyses.latest_for_paper(paper_id)
        if record is None:
            raise NotFoundError(user_message="This paper has no completed analysis yet.")
        return StoredAnalysis(paper, record, PaperAnalysis.model_validate(record.analysis))

    def delete(self, paper_id: str) -> None:
        """Delete a paper, its analyses, chat history (cascade) and its vectors."""
        try:
            self._store.delete_paper(paper_id)
        except AppError as exc:
            logger.warning("Vector cleanup failed for %s: %s", paper_id, exc)
        self._papers.delete(paper_id)

    def stats(self) -> dict[str, int]:
        entries = self._papers.list_library(limit=10_000)
        return {
            "papers": len(entries),
            "analyzed": sum(1 for e in entries if e.analysis_id is not None),
            "pages": sum(e.paper.page_count for e in entries),
            "chunks": sum(e.paper.chunk_count for e in entries),
        }
