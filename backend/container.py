"""Composition root: wires every service together (dependency injection by hand).

The UI asks for one ``AppContainer``; tests can pass a fake ``LLMClient`` and a
temporary ``Settings`` to get a fully isolated application.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.analysis_pipeline import AnalysisPipeline
from backend.library_service import LibraryService
from database.connection import Database
from database.repository import AnalysisRepository, ChatRepository, PaperRepository
from services.chat_service import ChatService
from services.chunking import SectionAwareChunker
from services.embedding_service import EmbeddingService
from services.export_service import ExportService
from services.builtin_engine import BuiltinEngine
from services.llm_service import LLMClient, OllamaClient
from services.pdf_service import PDFService
from services.project_kit_service import ProjectKitService
from services.rag_service import RAGService
from services.scoring_service import ScoringService
from services.similarity_service import SimilarityService
from services.vector_store import VectorStore
from utils.config import Settings, get_settings
from utils.logger import setup_logging


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    llm: LLMClient
    pipeline: AnalysisPipeline
    library: LibraryService
    chat: ChatService
    similarity: SimilarityService
    exporter: ExportService
    project_kit: ProjectKitService


def build_container(settings: Settings | None = None, llm: LLMClient | None = None) -> AppContainer:
    settings = settings or get_settings()
    settings.ensure_directories()
    setup_logging(settings.log_dir, settings.log_level)

    database = Database(settings.sqlite_path)
    database.initialize()
    papers = PaperRepository(database)
    analyses = AnalysisRepository(database)
    chats = ChatRepository(database)

    llm = llm or _build_engine(settings)
    embeddings = EmbeddingService(llm, settings.embed_batch_size)
    store = VectorStore(settings.chroma_dir, llm.embedding_model)
    rag = RAGService(embeddings, store, settings.retrieval_top_k)
    similarity = SimilarityService(embeddings, store)

    pipeline = AnalysisPipeline(
        settings=settings,
        llm=llm,
        pdf_service=PDFService(settings),
        chunker=SectionAwareChunker(settings.chunk_size, settings.chunk_overlap),
        rag=rag,
        scoring=ScoringService(),
        similarity=similarity,
        papers=papers,
        analyses=analyses,
    )
    return AppContainer(
        settings=settings,
        llm=llm,
        pipeline=pipeline,
        library=LibraryService(papers, analyses, store),
        chat=ChatService(
            llm, rag, chats, history_turns=settings.chat_history_turns,
            max_context_chars=min(settings.max_context_chars, 7000),
        ),
        similarity=similarity,
        exporter=ExportService(),
        project_kit=ProjectKitService(),
    )


def _build_engine(settings: Settings) -> LLMClient:
    """Default: the built-in engine (no installs). Set RPS_ENGINE=ollama to use a local LLM instead."""
    if settings.engine == "ollama":
        return OllamaClient(
            settings.ollama_base_url,
            settings.llm_model,
            settings.embedding_model,
            temperature=settings.llm_temperature,
            num_ctx=settings.llm_num_ctx,
            timeout_seconds=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
    return BuiltinEngine()
