"""OllamaClient error mapping and fallbacks, with HTTP mocked out."""

from __future__ import annotations

import json

import pytest
import requests

from services.llm_service import OllamaClient
from utils.exceptions import LLMConnectionError, LLMServiceError, ModelNotFoundError


class FakeResponse:
    def __init__(self, status: int, payload: dict | None = None, text: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.text = text or json.dumps(payload or {})

    def json(self) -> dict:
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


def make_client(monkeypatch, handler) -> OllamaClient:
    client = OllamaClient("http://ollama:11434", "llama3.2", "nomic-embed-text", max_retries=1)
    monkeypatch.setattr(client._session, "request", lambda method, url, **kw: handler(method, url, kw))
    monkeypatch.setattr("services.llm_service.time.sleep", lambda _: None)
    return client


def test_health_check_reports_missing_models(monkeypatch):
    client = make_client(monkeypatch, lambda m, u, kw: FakeResponse(200, {"models": [{"name": "llama3.2:latest"}]}))
    health = client.health_check()
    assert health.reachable and health.missing_models == ["nomic-embed-text"] and not health.ready


def test_connection_error_is_retried_then_mapped(monkeypatch):
    calls = []

    def handler(m, u, kw):
        calls.append(u)
        raise requests.ConnectionError("refused")

    client = make_client(monkeypatch, handler)
    with pytest.raises(LLMConnectionError, match="refused"):
        client.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2  # 1 retry
    assert not client.health_check().reachable


def test_missing_model_maps_to_pull_instruction(monkeypatch):
    client = make_client(monkeypatch, lambda m, u, kw: FakeResponse(404, {"error": 'model "llama3.2" not found'}))
    with pytest.raises(ModelNotFoundError) as info:
        client.chat([{"role": "user", "content": "hi"}])
    assert "ollama pull llama3.2" in info.value.user_message


def test_server_error_maps_to_service_error(monkeypatch):
    client = make_client(monkeypatch, lambda m, u, kw: FakeResponse(500, {"error": "out of memory"}))
    with pytest.raises(LLMServiceError, match="out of memory"):
        client.chat([{"role": "user", "content": "hi"}], json_mode=True)


def test_embed_falls_back_to_legacy_endpoint(monkeypatch):
    def handler(m, url, kw):
        if url.endswith("/api/embed"):
            return FakeResponse(404, None, text="404 page not found")
        return FakeResponse(200, {"embedding": [0.1, 0.2]})

    client = make_client(monkeypatch, handler)
    assert client.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
