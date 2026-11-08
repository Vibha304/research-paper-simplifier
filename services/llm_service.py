"""Local LLM access through the Ollama HTTP API (fully offline).

``LLMClient`` is a Protocol so the rest of the app depends on an interface, not
on Ollama; tests inject a fake implementation.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

from utils.exceptions import LLMConnectionError, LLMServiceError, ModelNotFoundError
from utils.logger import get_logger

logger = get_logger(__name__)

Message = dict[str, str]


class LLMClient(Protocol):
    """Interface required by the pipeline, chat and embedding services."""

    model: str
    embedding_model: str

    def chat(self, messages: list[Message], *, json_mode: bool = False, temperature: float | None = None) -> str: ...

    def stream_chat(self, messages: list[Message], *, temperature: float | None = None) -> Iterator[str]: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


@dataclass(frozen=True)
class HealthStatus:
    reachable: bool
    available_models: list[str] = field(default_factory=list)
    missing_models: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.reachable and not self.missing_models


def _normalize_model(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


class OllamaClient:
    """Minimal, dependency-light Ollama client with retries and clear errors."""

    def __init__(
        self,
        base_url: str,
        model: str,
        embedding_model: str,
        *,
        temperature: float = 0.2,
        num_ctx: int = 8192,
        timeout_seconds: int = 300,
        max_retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout_seconds
        self.max_retries = max_retries
        self._session = requests.Session()
        self._use_legacy_embeddings = False

    # ------------------------------------------------------------------ health
    def list_models(self) -> list[str]:
        response = self._request("GET", "/api/tags", timeout=10, retries=0)
        return [m.get("name", "") for m in response.json().get("models", [])]

    def health_check(self) -> HealthStatus:
        try:
            available = self.list_models()
        except LLMServiceError as exc:
            return HealthStatus(reachable=False, error=exc.user_message)
        normalized = {_normalize_model(m) for m in available}
        missing = [m for m in (self.model, self.embedding_model) if _normalize_model(m) not in normalized]
        return HealthStatus(reachable=True, available_models=sorted(available), missing_models=missing)

    # -------------------------------------------------------------- generation
    def chat(self, messages: list[Message], *, json_mode: bool = False, temperature: float | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": self._options(temperature),
        }
        if json_mode:
            payload["format"] = "json"
        started = time.perf_counter()
        data = self._request("POST", "/api/chat", json=payload).json()
        logger.debug("chat completed in %.1fs", time.perf_counter() - started)
        return data.get("message", {}).get("content", "")

    def stream_chat(self, messages: list[Message], *, temperature: float | None = None) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": self._options(temperature),
        }
        response = self._request("POST", "/api/chat", json=payload, stream=True)
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMServiceError(chunk["error"])
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
                if chunk.get("done"):
                    break
        except requests.RequestException as exc:
            raise LLMConnectionError(f"Stream interrupted: {exc}") from exc
        finally:
            response.close()

    # --------------------------------------------------------------- embedding
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self._use_legacy_embeddings:
            try:
                data = self._request(
                    "POST", "/api/embed", json={"model": self.embedding_model, "input": list(texts)}
                ).json()
                return data["embeddings"]
            except _EndpointMissing:
                logger.info("Ollama /api/embed unavailable; falling back to legacy /api/embeddings")
                self._use_legacy_embeddings = True
        return [
            self._request("POST", "/api/embeddings", json={"model": self.embedding_model, "prompt": text}).json()[
                "embedding"
            ]
            for text in texts
        ]

    # ---------------------------------------------------------------- internals
    def _options(self, temperature: float | None) -> dict[str, Any]:
        return {
            "temperature": self.temperature if temperature is None else temperature,
            "num_ctx": self.num_ctx,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        stream: bool = False,
        timeout: int | None = None,
        retries: int | None = None,
    ) -> requests.Response:
        url = f"{self.base_url}{path}"
        attempts = (self.max_retries if retries is None else retries) + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self._session.request(
                    method, url, json=json, stream=stream, timeout=timeout or self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_error = exc
                if attempt < attempts:
                    wait = 2 ** (attempt - 1)
                    logger.warning("Ollama request to %s failed (%s); retry %d in %ss", path, exc, attempt, wait)
                    time.sleep(wait)
                    continue
                if isinstance(exc, requests.Timeout):
                    raise LLMConnectionError(
                        f"Timeout calling {url}",
                        user_message=(
                            f"Ollama did not respond within {timeout or self.timeout}s. The model may still be "
                            "loading or your machine may be under heavy load - please retry."
                        ),
                    ) from exc
                raise LLMConnectionError(
                    f"Connection to {url} failed: {exc}",
                    user_message=f"Cannot reach Ollama at {self.base_url}. Start it with `ollama serve`.",
                ) from exc

            if response.status_code < 400:
                return response
            self._raise_for_status(response, path)
        raise LLMConnectionError(str(last_error))  # pragma: no cover - loop always returns/raises

    def _raise_for_status(self, response: requests.Response, path: str) -> None:
        body = response.text[:500]
        try:
            message = response.json().get("error", body)
        except ValueError:
            message = body
        if response.status_code == 404:
            if "model" in message.lower():
                model = self.embedding_model if "embed" in path else self.model
                raise ModelNotFoundError(
                    message, user_message=f"Model '{model}' is not installed. Run: `ollama pull {model}`"
                )
            raise _EndpointMissing(message)
        raise LLMServiceError(
            f"Ollama {path} returned HTTP {response.status_code}: {message}",
            user_message=f"Ollama returned an error: {message}",
        )


class _EndpointMissing(LLMServiceError):
    """Internal: the Ollama server is too old to provide this endpoint."""
