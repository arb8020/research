"""vLLM/SGLang provider implementation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import httpx
from dacite import from_dict

from rollouts._retry import async_retry
from rollouts.dtypes import Actor, ChatCompletion, StreamEvent

from .base import (
    NonRetryableError,
    VLLMErrorType,
    _classify_vllm_error,
    _format_context_length_error,
    _format_invalid_param_error,
    _prepare_messages_for_llm,
)
from .openai_completions import _message_to_openai, _tool_to_openai


def _normalize_vllm_api_base(api_base: str) -> str:
    """Normalize API base URL to include /chat/completions. Pure function."""
    assert isinstance(api_base, str), f"api_base must be str, got {type(api_base)}"
    assert len(api_base) > 0, "api_base cannot be empty"

    result = None
    if api_base.endswith("/chat/completions"):
        result = api_base
    elif api_base.endswith("/v1"):
        result = api_base.rstrip("/") + "/chat/completions"
    else:
        result = api_base.rstrip("/") + "/v1/chat/completions"

    assert result.endswith("/chat/completions"), (
        f"result must end with /chat/completions, got {result}"
    )
    return result


def _build_vllm_params(actor: Actor) -> dict:
    """Build vLLM API parameters. Pure function."""
    assert actor is not None
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert actor.trajectory.messages is not None

    # Strip details before sending to LLM
    llm_messages = _prepare_messages_for_llm(actor.trajectory.messages)
    messages = [_message_to_openai(m) for m in llm_messages]
    assert isinstance(messages, list)
    assert len(messages) > 0, "messages list cannot be empty"

    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "max_tokens": actor.endpoint.max_tokens,
        "temperature": actor.endpoint.temperature,
        "stream": False,
        "logprobs": True,
        "echo": True,
    }

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    assert "model" in params, "params must contain model"
    assert "messages" in params, "params must contain messages"
    return params


def _conform_tool_calls(raw_tool_calls: list) -> list:
    """Convert raw OpenAI tool calls to conformed format. Pure function."""
    assert isinstance(raw_tool_calls, list), (
        f"raw_tool_calls must be list, got {type(raw_tool_calls)}"
    )
    assert len(raw_tool_calls) > 0, "raw_tool_calls cannot be empty"

    conformed = []
    for tc in raw_tool_calls:
        conformed.append({
            "id": tc["id"],
            "name": tc["function"]["name"],
            "args": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {},
        })

    assert len(conformed) == len(raw_tool_calls), "conformed length must match input length"
    return conformed


async def _execute_vllm_request(
    api_base: str, params: dict, headers: dict, max_retries: int, backoff_base: int, timeout: float
) -> dict:
    """Execute vLLM API request with retry logic. Returns completion dict."""
    assert isinstance(api_base, str)
    assert isinstance(params, dict)
    assert isinstance(headers, dict)
    assert max_retries > 0
    assert backoff_base > 0
    assert timeout > 0
    assert "messages" in params, "params must contain messages"
    assert len(api_base) > 0, "api_base cannot be empty"

    @async_retry(
        max_attempts=max_retries,
        delay=backoff_base,
        backoff=2,
        jitter=True,
        exceptions=(httpx.HTTPError, httpx.TimeoutException),
    )
    async def _call_with_retry():
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(api_base, json=params, headers=headers)
            error_type = _classify_vllm_error(response.status_code, response.text)

            if error_type == VLLMErrorType.SUCCESS:
                return response.json()

            print(f"❌ Server returned {response.status_code}: {response.text}")

            if error_type == VLLMErrorType.CONTEXT_LENGTH:
                print(_format_context_length_error(params.get("max_tokens", 8192)))
                raise NonRetryableError(f"Context length exceeded: {response.text}")

            if error_type == VLLMErrorType.INVALID_PARAM:
                print(_format_invalid_param_error(list(params.keys())))
                raise NonRetryableError(f"Invalid parameter: {response.text}")

            response.raise_for_status()

    result = await _call_with_retry()
    assert result is not None, "result cannot be None"
    assert isinstance(result, dict), f"result must be dict, got {type(result)}"
    return result


async def rollout_sglang(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Invoke a vLLM server and return the updated actor.

    Note: **kwargs accepts but ignores provider-specific params (e.g., anthropic thinking params)
    """
    # Tiger Style: Assert all inputs
    assert actor is not None
    assert isinstance(actor, Actor)
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # Use endpoint's retry configuration (consistent with OpenAI/Anthropic providers)
    max_api_retries = actor.endpoint.max_retries
    timeout = actor.endpoint.timeout
    backoff_base = 4  # Keep this constant for now

    assert max_api_retries > 0
    assert max_api_retries <= 100
    assert backoff_base > 0
    assert timeout > 0

    # Build request using pure helpers
    params = _build_vllm_params(actor)
    api_base = _normalize_vllm_api_base(actor.endpoint.api_base)
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    # Execute API call
    completion = await _execute_vllm_request(
        api_base, params, headers, max_api_retries, backoff_base, timeout
    )
    assert completion

    # Process tool calls
    message = completion["choices"][0]["message"]
    raw_tool_calls = message.get("tool_calls")
    message["tool_calls"] = _conform_tool_calls(raw_tool_calls) if raw_tool_calls else []

    # Parse and validate
    completion = from_dict(ChatCompletion, completion)
    assert completion is not None
    completion = replace(completion, model=actor.endpoint.model)
    assert completion.choices is not None
    assert len(completion.choices) > 0
    final_message = completion.choices[0].message
    assert final_message is not None

    # Update trajectory
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )
    assert new_trajectory is not None

    result_actor = replace(actor, trajectory=new_trajectory)
    assert result_actor is not None
    assert result_actor.trajectory is not None
    return result_actor
