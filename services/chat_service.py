"""RAG-powered "chat with the paper" assistant with persisted history."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from backend.prompts import CHAT_SYSTEM_PROMPT, CHAT_USER_TEMPLATE
from database.repository import ChatMessageRecord, ChatRepository
from services.builtin_engine import StructuredGenerator
from services.llm_service import LLMClient, Message
from services.rag_service import RAGService
from utils.logger import get_logger

logger = get_logger(__name__)

SUGGESTED_QUESTIONS: tuple[str, ...] = (
    "Explain the core idea of this paper like I'm a beginner.",
    "What are the key equations or algorithms, and what do they mean?",
    "How could I implement a minimal version of this in Python?",
    "How does this approach compare to previous work?",
    "What would be a good mini-project based on this paper?",
)


@dataclass
class ChatAnswer:
    """A streaming answer. Iterate ``tokens``; ``sources`` are known up front."""

    sources: list[dict[str, Any]]
    tokens: Iterator[str]
    full_text: list[str] = field(default_factory=list)


class ChatService:
    def __init__(
        self, llm: LLMClient, rag: RAGService, repository: ChatRepository,
        *, history_turns: int = 4, max_context_chars: int = 7000,
    ) -> None:
        self._llm = llm
        self._rag = rag
        self._repo = repository
        self._history_turns = history_turns
        self._max_context_chars = max_context_chars

    def history(self, paper_id: str) -> list[ChatMessageRecord]:
        return self._repo.list_for_paper(paper_id)

    def clear(self, paper_id: str) -> None:
        self._repo.clear(paper_id)

    def ask(self, paper_id: str, paper_title: str, question: str) -> ChatAnswer:
        """Retrieve context, persist the question and return a streaming answer."""
        question = question.strip()
        # Retrieval query includes the previous user turn so follow-ups ("and why?") work.
        previous = [m for m in self._repo.list_for_paper(paper_id, limit=2) if m.role == "user"]
        retrieval_query = f"{previous[-1].content} {question}" if previous and len(question) < 40 else question
        chunks = self._rag.retrieve(paper_id, [question, retrieval_query] if retrieval_query != question else question)
        context = self._rag.format_context(chunks, self._max_context_chars)

        messages: list[Message] = [{"role": "system", "content": CHAT_SYSTEM_PROMPT.format(title=paper_title)}]
        for record in self._repo.list_for_paper(paper_id, limit=self._history_turns * 2):
            messages.append({"role": record.role, "content": record.content})
        messages.append({"role": "user", "content": CHAT_USER_TEMPLATE.format(context=context, question=question)})

        self._repo.add(paper_id, "user", question)
        sources = [c.to_source() for c in chunks]
        answer = ChatAnswer(sources=sources, tokens=iter(()))
        if isinstance(self._llm, StructuredGenerator):
            stream = self._llm.answer_question(question, chunks, paper_title)
        else:
            stream = self._llm.stream_chat(messages)
        answer.tokens = self._stream_and_store(paper_id, stream, sources, answer.full_text)
        return answer

    def _stream_and_store(
        self, paper_id: str, stream: Iterator[str], sources: list[dict[str, Any]], sink: list[str]
    ) -> Iterator[str]:
        try:
            for token in stream:
                sink.append(token)
                yield token
        finally:
            text = "".join(sink).strip()
            if text:
                self._repo.add(paper_id, "assistant", text, sources)
            else:
                logger.warning("Empty chat answer for paper %s", paper_id)
