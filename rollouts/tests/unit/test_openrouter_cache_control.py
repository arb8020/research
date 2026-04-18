from __future__ import annotations

import copy

import pytest

from rollouts.providers.openai_completions import _maybe_add_openrouter_cache_control


def _msgs(content: str | list) -> list[dict]:
    return [{"role": "user", "content": content}]


# --- models that should get markers ---


def test_anthropic_string_content_converted_to_block() -> None:
    msgs = _msgs("hello")
    _maybe_add_openrouter_cache_control("anthropic/claude-sonnet-4-5", msgs)
    content = msgs[0]["content"]
    assert isinstance(content, list)
    assert content[0]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_list_content_last_text_block_marked() -> None:
    msgs = _msgs([
        {"type": "text", "text": "stable part"},
        {"type": "text", "text": "variable part"},
    ])
    _maybe_add_openrouter_cache_control("anthropic/claude-opus-4-7", msgs)
    content = msgs[0]["content"]
    assert "cache_control" not in content[0]
    assert content[1]["cache_control"] == {"type": "ephemeral"}


def test_google_model_gets_marker() -> None:
    msgs = _msgs("hello")
    _maybe_add_openrouter_cache_control("google/gemini-2.5-flash", msgs)
    assert msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_does_not_double_mark() -> None:
    msgs = _msgs([{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}])
    original = copy.deepcopy(msgs)
    _maybe_add_openrouter_cache_control("anthropic/claude-sonnet-4-5", msgs)
    assert msgs == original


# --- models that should NOT get markers ---


@pytest.mark.parametrize(
    "model_id",
    [
        "deepseek/deepseek-r1",
        "deepseek-ai/DeepSeek-V3.2",
        "openai/gpt-4o",
        "meta-llama/llama-3.3-70b-instruct",
        "mistralai/mistral-small",
    ],
)
def test_automatic_caching_models_untouched(model_id: str) -> None:
    msgs = _msgs("hello")
    original = copy.deepcopy(msgs)
    _maybe_add_openrouter_cache_control(model_id, msgs)
    assert msgs == original


def test_walks_back_past_non_user_messages() -> None:
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
        {"role": "user", "content": "follow up"},
    ]
    _maybe_add_openrouter_cache_control("anthropic/claude-sonnet-4-5", msgs)
    # Should mark the last user message, not the first
    assert msgs[2]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert msgs[0]["content"] == "first"
