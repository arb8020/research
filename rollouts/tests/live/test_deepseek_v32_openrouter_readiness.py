#!/usr/bin/env python3
"""Live DeepSeek V3.2 readiness checks for OpenRouter-facing behavior."""

import json
import os

import pytest
from openai import OpenAI

from rollouts.tools.deepseek_vendor_verifier import (
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    make_client,
    weather_tools,
)

pytestmark = pytest.mark.live


def _client() -> OpenAI:
    return make_client(
        base_url=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL),
        api_key=os.environ.get("KIMI_API_KEY", DEFAULT_API_KEY),
        timeout=120.0,
    )


@pytest.mark.live
def test_deepseek_v32_tool_loop_regression() -> None:
    """Verify the exact working tool loop stays valid on the served stack."""
    model = os.environ.get("KIMI_MODEL", DEFAULT_MODEL)
    client = _client()

    first = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Use the tool then answer: what is the weather in San Francisco?",
            }
        ],
        tools=weather_tools(),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=256,
        temperature=0.0,
    )
    first_message = first.choices[0].message
    assert first_message.tool_calls, f"missing tool_calls; content={first_message.content!r}"

    tool_call = first_message.tool_calls[0]
    second = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Use the tool then answer: what is the weather in San Francisco?",
            },
            {
                "role": "assistant",
                "content": first_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({
                    "city": "San Francisco",
                    "unit": "F",
                    "temperature": 61,
                    "condition": "foggy",
                }),
            },
        ],
        tools=weather_tools(),
        tool_choice="none",
        max_tokens=256,
        temperature=0.0,
    )

    final_text = (second.choices[0].message.content or "").lower()
    assert "61" in final_text, (
        f"missing temperature in final answer: {second.choices[0].message.content!r}"
    )
    assert "fog" in final_text, (
        f"missing weather condition in final answer: {second.choices[0].message.content!r}"
    )


@pytest.mark.live
def test_deepseek_v32_models_schema_contains_target_model() -> None:
    """Verify /v1/models exposes the served DeepSeek model in a usable shape."""
    model = os.environ.get("KIMI_MODEL", DEFAULT_MODEL)
    client = _client()

    models = list(client.models.list().data)
    assert models, "/v1/models returned no models"
    matched = next((item for item in models if item.id == model), None)
    assert matched is not None, f"{model!r} missing from /v1/models"
    assert matched.object == "model"
    assert matched.owned_by
