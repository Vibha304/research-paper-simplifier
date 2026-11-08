"""ChromaDB persistence for chunk embeddings and paper-level profiles.

Embeddings are always computed by us (Ollama) and passed in explicitly, so
Chroma never tries to download its default ONNX embedding model - keeping the
app fully offline. Collection names include the embedding model, so switching
models never mixes vectors of different dimensionality.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from utils.exceptions import VectorStoreError
from utils.logger import get_logger
from utils.text import slugify

logger = get_logger(__name__)

_ADD_BATCH = 500


@dataclass(frozen=True)
class VectorHit:
    id: str
    document: str
    metadata: dict[str, Any]
    similarity: float  # cosine similarity in [-1, 1]


class VectorStore:
    def __init__(self, persist_dir: Path, embedding_model: str) -> None:
        try:
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
            )
            suffix = slugify(embedding_model, max_length=40, separator="_")
            self._chunks = self._collection(f"chunks_{suffix}")
            self._profiles = self._collection(f"profiles_{suffix}")
        except Exception as exc:
            raise VectorStoreError(f"Cannot initialise ChromaDB at {persist_dir}: {exc}") from exc

    def _collection(self, name: str) -> Any:
        return self._client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    # ------------------------------------------------------------------ chunks
    def add_chunks(
        self,
        paper_id: str,
        ids: Sequence[str],
        documents: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]],
    ) -> None:
        if not (len(ids) == len(documents) == len(embeddings) == len(metadatas)):
            raise VectorStoreError("ids, documents, embeddings and metadatas must have equal length")
        try:
            for start in range(0, len(ids), _ADD_BATCH):
                end = start + _ADD_BATCH
                self._chunks.add(
                    ids=list(ids[start:end]),
                    documents=list(documents[start:end]),
                    embeddings=[list(e) for e in embeddings[start:end]],
                    metadatas=[{**m, "paper_id": paper_id} for m in metadatas[start:end]],
                )
        except Exception as exc:
            raise VectorStoreError(f"Failed to add chunks for {paper_id}: {exc}") from exc

    def query_chunks(self, paper_id: str, embedding: Sequence[float], top_k: int) -> list[VectorHit]:
        try:
            available = self.chunk_count(paper_id)
            if available == 0:
                return []
            result = self._chunks.query(
                query_embeddings=[list(embedding)],
                n_results=min(top_k, available),
                where={"paper_id": paper_id},
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Chunk query failed for {paper_id}: {exc}") from exc
        return self._to_hits(result)

    def chunk_count(self, paper_id: str) -> int:
        try:
            return len(self._chunks.get(where={"paper_id": paper_id}, include=[])["ids"])
        except Exception as exc:
            raise VectorStoreError(f"Chunk count failed: {exc}") from exc

    def delete_paper(self, paper_id: str) -> None:
        try:
            self._chunks.delete(where={"paper_id": paper_id})
            self._profiles.delete(ids=[paper_id])
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete vectors for {paper_id}: {exc}") from exc

    # ---------------------------------------------------------------- profiles
    def upsert_profile(
        self, paper_id: str, embedding: Sequence[float], document: str, metadata: dict[str, Any]
    ) -> None:
        try:
            self._profiles.upsert(
                ids=[paper_id], embeddings=[list(embedding)], documents=[document], metadatas=[metadata]
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to store profile for {paper_id}: {exc}") from exc

    def get_profile_embedding(self, paper_id: str) -> list[float] | None:
        try:
            result = self._profiles.get(ids=[paper_id], include=["embeddings"])
        except Exception as exc:
            raise VectorStoreError(f"Profile lookup failed: {exc}") from exc
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return [float(x) for x in embeddings[0]]

    def query_profiles(self, embedding: Sequence[float], top_k: int, exclude_id: str | None = None) -> list[VectorHit]:
        try:
            total = self._profiles.count()
            if total == 0:
                return []
            result = self._profiles.query(
                query_embeddings=[list(embedding)],
                n_results=min(top_k + (1 if exclude_id else 0), total),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Profile query failed: {exc}") from exc
        return [hit for hit in self._to_hits(result) if hit.id != exclude_id][:top_k]

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _to_hits(result: dict[str, Any]) -> list[VectorHit]:
        ids = result.get("ids", [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        return [
            VectorHit(id=i, document=d or "", metadata=dict(m or {}), similarity=1.0 - float(dist))
            for i, d, m, dist in zip(ids, documents, metadatas, distances)
        ]
