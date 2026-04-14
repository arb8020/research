from __future__ import annotations

from typing import Any

import pytest

from rollouts.models import (
    MODELS,
    ModelCost,
    ModelMetadata,
    _is_syncable_openai_model_id,
    fetch_openai_models,
    sync_openai_models,
)


def test_is_syncable_openai_model_id_filters_non_generation_products() -> None:
    assert _is_syncable_openai_model_id("gpt-5.4")
    assert _is_syncable_openai_model_id("o4-mini")
    assert not _is_syncable_openai_model_id("text-embedding-3-large")
    assert not _is_syncable_openai_model_id("omni-moderation-latest")
    assert not _is_syncable_openai_model_id("gpt-4o-realtime-preview")
    assert not _is_syncable_openai_model_id("whisper-1")


def test_fetch_openai_models_uses_bearer_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return

        def json(self) -> dict[str, object]:
            return {"data": [{"id": "gpt-5.4"}]}

    class _FakeClient:
        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        async def get(self, url: str, headers: dict[str, str]) -> _FakeResponse:
            captured["url"] = url
            captured["headers"] = headers
            return _FakeResponse()

    monkeypatch.setattr("httpx.AsyncClient", _FakeClient)

    result = pytest.importorskip("trio").run(fetch_openai_models, "sk-test")

    assert result == [{"id": "gpt-5.4"}]
    assert captured == {
        "url": "https://api.openai.com/v1/models",
        "headers": {"Authorization": "Bearer sk-test"},
    }


def test_sync_openai_models_compares_only_syncable_generation_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_openai_models = MODELS["openai"]
    MODELS["openai"] = {
        "gpt-4.1": ModelMetadata(
            id="gpt-4.1",
            name="GPT-4.1",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=False,
            input_types=["text", "image"],
            cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=0.5),
            context_window=128000,
            max_tokens=16384,
        ),
        "gpt-5.2": ModelMetadata(
            id="gpt-5.2",
            name="GPT-5.2",
            provider="openai",
            api="openai-responses",
            base_url="https://api.openai.com/v1",
            reasoning=True,
            input_types=["text", "image"],
            cost=ModelCost(input=1.0, output=2.0, cache_read=0.5, cache_write=0.5),
            context_window=400000,
            max_tokens=32768,
        ),
    }

    async def _fake_fetch_openai_models(_api_key: str) -> list[dict]:
        return [
            {"id": "gpt-5.2"},
            {"id": "gpt-5.4"},
            {"id": "gpt-5.4-2026-03-05"},
            {"id": "text-embedding-3-large"},
            {"id": "omni-moderation-latest"},
            {"id": "gpt-4o-realtime-preview"},
        ]

    monkeypatch.setattr("rollouts.models.fetch_openai_models", _fake_fetch_openai_models)

    try:
        diff = pytest.importorskip("trio").run(sync_openai_models, "sk-test")
    finally:
        MODELS["openai"] = original_openai_models

    assert diff.missing == ["gpt-5.4", "gpt-5.4-2026-03-05"]
    assert diff.extra == ["gpt-4.1"]
    assert diff.updated == {}
