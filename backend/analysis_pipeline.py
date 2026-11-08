"""End-to-end analysis orchestration.

    PDF bytes -> extract -> chunk -> embed & index (ChromaDB)
              -> for each AnalysisTask: retrieve -> prompt -> JSON -> validate
              -> resource cross-check -> scoring -> similarity profile -> SQLite

The pipeline is fault tolerant: if one generation task fails (timeout, bad
JSON), the remaining sections are still produced and a warning is attached.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from backend.models import PaperAnalysis
from backend.prompts import (
    ANALYSIS_TASKS,
    JSON_RETRY_SUFFIX,
    KNOWN_FACTS_TEMPLATE,
    SYSTEM_PROMPT,
    TASK_TEMPLATE,
    AnalysisTask,
)
from database.repository import AnalysisRepository, PaperRecord, PaperRepository
from services.builtin_engine import StructuredGenerator, TaskInput
from services.chunking import Chunk, SectionAwareChunker, lead_text
from services.llm_service import LLMClient
from services.pdf_service import ExtractedDocument, PDFService
from services.rag_service import RAGService
from services.scoring_service import ScoringService, merge_detected_resources
from services.similarity_service import SimilarityService
from utils.config import Settings
from utils.exceptions import AppError, LLMConnectionError, LLMResponseFormatError, ModelNotFoundError, PDFProcessingError
from utils.json_utils import extract_json
from utils.logger import get_logger
from utils.progress import ProgressCallback, ProgressTracker
from utils.text import truncate

logger = get_logger(__name__)

# Progress-bar allocation of each stage (fractions of 1.0).
_EXTRACT, _CHUNK, _INDEX, _GENERATE, _FINALIZE = (0.0, 0.12), (0.12, 0.15), (0.15, 0.35), (0.35, 0.95), (0.95, 1.0)


@dataclass(frozen=True)
class AnalysisOutcome:
    paper: PaperRecord
    analysis: PaperAnalysis
    analysis_id: int
    duration_seconds: float
    reused: bool = False


class AnalysisPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        pdf_service: PDFService,
        chunker: SectionAwareChunker,
        rag: RAGService,
        scoring: ScoringService,
        similarity: SimilarityService,
        papers: PaperRepository,
        analyses: AnalysisRepository,
        tasks: tuple[AnalysisTask, ...] = ANALYSIS_TASKS,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._pdf = pdf_service
        self._chunker = chunker
        self._rag = rag
        self._scoring = scoring
        self._similarity = similarity
        self._papers = papers
        self._analyses = analyses
        self._tasks = tasks

    # ----------------------------------------------------------------- public
    def find_existing(self, data: bytes) -> PaperRecord | None:
        """Return the stored paper if these exact bytes were analysed before."""
        paper = self._papers.get_by_hash(self._pdf.compute_hash(data))
        if paper and self._analyses.latest_for_paper(paper.id):
            return paper
        return None

    def run(
        self, data: bytes, filename: str, progress: ProgressCallback | None = None, *, force: bool = False
    ) -> AnalysisOutcome:
        started = time.perf_counter()
        tracker = ProgressTracker(progress)

        if not force:
            existing = self.find_existing(data)
            if existing is not None:
                record = self._analyses.latest_for_paper(existing.id)
                assert record is not None
                tracker.report(1.0, "Loaded previous analysis from history")
                return AnalysisOutcome(
                    existing, PaperAnalysis.model_validate(record.analysis), record.id, 0.0, reused=True
                )

        # 1. Extract -----------------------------------------------------------
        tracker.report(0.0, "Reading PDF...")
        document = self._pdf.extract(data, filename, tracker.stage(*_EXTRACT))

        # 2. Chunk -------------------------------------------------------------
        tracker.report(_CHUNK[0], "Splitting paper into sections and chunks...")
        chunks = self._chunker.chunk(document)
        if not chunks:
            raise PDFProcessingError(user_message="No usable text could be extracted from this PDF.")
        logger.info("%s -> %d chunks", filename, len(chunks))

        existing_paper = self._papers.get_by_hash(document.file_hash)
        paper_id = existing_paper.id if existing_paper else uuid.uuid4().hex

        # 3. Index -------------------------------------------------------------
        tracker.report(_INDEX[0], f"Embedding {len(chunks)} chunks with {self._llm.embedding_model}...")
        self._rag.index(paper_id, chunks, tracker.stage(*_INDEX))

        paper = PaperRecord(
            id=paper_id,
            title=document.title,
            filename=filename,
            file_hash=document.file_hash,
            page_count=document.page_count,
            word_count=document.word_count,
            source=document.source,
            chunk_count=len(chunks),
        )
        self._papers.upsert(paper)  # persisted early so chat works even if generation is interrupted

        # 4. Generate ----------------------------------------------------------
        analysis = self._generate(paper_id, document, chunks, tracker.stage(*_GENERATE))

        # 5. Finalize ----------------------------------------------------------
        tracker.report(_FINALIZE[0], "Scoring difficulty, resume impact and time...")
        full_text = document.full_text
        analysis = merge_detected_resources(analysis, full_text)
        analysis = analysis.model_copy(
            update={
                "assessment": self._scoring.finalize(
                    full_text, document.page_count, analysis,
                    llm_weight=1.0 if isinstance(self._llm, StructuredGenerator) else 0.6,
                )
            }
        )

        try:
            self._similarity.index_paper(paper_id, analysis)
        except AppError as exc:  # similarity is a nice-to-have; never fail the analysis for it
            logger.warning("Similarity indexing failed for %s: %s", paper_id, exc)
            analysis.warnings.append("Similarity profile could not be stored.")

        duration = time.perf_counter() - started
        analysis_id = self._analyses.add(
            paper_id=paper_id,
            llm_model=self._llm.model,
            embedding_model=self._llm.embedding_model,
            analysis=analysis.model_dump(),
            duration_seconds=duration,
        )
        tracker.report(1.0, f"Done in {duration:.0f}s")
        logger.info("Analysis of %s finished in %.1fs (%d warnings)", filename, duration, len(analysis.warnings))
        return AnalysisOutcome(paper, analysis, analysis_id, duration)

    # ------------------------------------------------------------- generation
    def _generate(
        self, paper_id: str, document: ExtractedDocument, chunks: list[Chunk], progress: ProgressCallback
    ) -> PaperAnalysis:
        analysis = PaperAnalysis(title=document.title)
        lead = lead_text(document, max_chars=2500)
        total = len(self._tasks)
        failures = 0

        for index, task in enumerate(self._tasks):
            progress(index / total, f"[{index + 1}/{total}] {task.label}...")
            try:
                data = self._run_task(paper_id, document, chunks, lead, task, analysis)
                analysis = self._merge(analysis, data, task)
            except (LLMConnectionError, ModelNotFoundError):
                raise  # infrastructure problems: abort with a clear message
            except AppError as exc:
                failures += 1
                logger.warning("Task %s failed: %s", task.key, exc)
                analysis.warnings.append(f"Section '{task.key}' could not be generated: {exc.user_message}")
            progress((index + 1) / total, f"[{index + 1}/{total}] {task.label} - done")

        if failures == total:
            raise LLMResponseFormatError(
                user_message="The language model failed to produce any section. Check the model and try again."
            )
        return analysis

    def _run_task(
        self,
        paper_id: str,
        document: ExtractedDocument,
        all_chunks: list[Chunk],
        lead: str,
        task: AnalysisTask,
        analysis: PaperAnalysis,
    ) -> dict[str, Any]:
        title = document.title
        chunks = self._rag.retrieve(
            paper_id, task.queries, top_k=task.top_k, preferred_sections=task.preferred_sections
        )
        if isinstance(self._llm, StructuredGenerator):  # built-in engine: no prompting needed
            try:
                return self._llm.generate_task(
                    TaskInput(paper_id, task.key, title, lead, chunks, all_chunks, document.page_count, analysis)
                )
            except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
                raise LLMResponseFormatError(f"Built-in engine failed on {task.key}: {exc}",
                                             user_message="This section could not be extracted.") from exc
        context = self._rag.format_context(chunks, self._settings.max_context_chars)
        prompt = TASK_TEMPLATE.format(
            title=title,
            lead=lead,
            context=context or "(no additional excerpts retrieved)",
            known_facts=KNOWN_FACTS_TEMPLATE.format(facts=self._known_facts(analysis)) if task.uses_known_facts else "",
            instructions=task.instructions,
            schema=task.schema,
        )
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

        try:
            data = extract_json(self._llm.chat(messages, json_mode=True))
        except LLMResponseFormatError:
            logger.info("Task %s returned invalid JSON; retrying once", task.key)
            messages[-1]["content"] += JSON_RETRY_SUFFIX
            data = extract_json(self._llm.chat(messages, json_mode=True, temperature=0.0))
        return task.transform(data) if task.transform else data

    @staticmethod
    def _merge(analysis: PaperAnalysis, data: dict[str, Any], task: AnalysisTask) -> PaperAnalysis:
        """Validate the task's keys into the analysis, dropping only invalid keys."""
        updates = {k: data[k] for k in task.output_keys if k in data}
        missing = [k for k in task.output_keys if k not in data]
        if missing:
            analysis.warnings.append(f"Model omitted: {', '.join(missing)}")
        base = analysis.model_dump()
        try:
            return PaperAnalysis.model_validate({**base, **updates})
        except ValidationError:
            for key, value in updates.items():
                try:
                    candidate = PaperAnalysis.model_validate({**base, key: value})
                    base = candidate.model_dump()
                except ValidationError as exc:
                    logger.warning("Dropping invalid field %s: %s", key, exc.errors()[:1])
                    base["warnings"] = [*base.get("warnings", []), f"Field '{key}' had an invalid format."]
            return PaperAnalysis.model_validate(base)

    @staticmethod
    def _known_facts(analysis: PaperAnalysis) -> str:
        facts = [
            f"Summary: {truncate(analysis.simple_summary, 600)}" if analysis.simple_summary else "",
            f"Contributions: {'; '.join(analysis.key_contributions[:5])}" if analysis.key_contributions else "",
            f"Method: {truncate(analysis.methodology, 700)}" if analysis.methodology else "",
            f"Datasets: {', '.join(analysis.datasets[:6])}" if analysis.datasets else "",
            f"Tools: {', '.join(analysis.tools_frameworks[:10])}" if analysis.tools_frameworks else "",
            f"Project: {analysis.project_idea.title} - {truncate(analysis.project_idea.description, 300)}"
            if analysis.project_idea.title else "",
        ]
        return "\n".join(f for f in facts if f) or "(nothing yet)"
