"""Shared utilities for provider implementations."""

from __future__ import annotations

import copy
from dataclasses import replace
from enum import Enum

from rollouts.dtypes import Cost, Message, Usage
from rollouts.models import ModelCost


class NonRetryableError(Exception):
    """Exception for errors that should not be retried."""

    pass


class VLLMErrorType(Enum):
    """Classification of vLLM server errors."""

    SUCCESS = "success"
    CONTEXT_LENGTH = "context_length"
    INVALID_PARAM = "invalid_param"
    HTTP_ERROR = "http_error"


def _classify_vllm_error(status_code: int, error_body: str) -> VLLMErrorType:
    """Classify vLLM error type. Pure function - no I/O."""
    assert isinstance(status_code, int), f"status_code must be int, got {type(status_code)}"
    assert isinstance(error_body, str), f"error_body must be str, got {type(error_body)}"

    if status_code == 200:
        return VLLMErrorType.SUCCESS
    if "maximum context length" in error_body.lower():
        return VLLMErrorType.CONTEXT_LENGTH
    if "not a valid parameter" in error_body.lower():
        return VLLMErrorType.INVALID_PARAM
    return VLLMErrorType.HTTP_ERROR


def _format_context_length_error(max_tokens: int) -> str:
    """Format context length error message. Pure function."""
    assert isinstance(max_tokens, int), f"max_tokens must be int, got {type(max_tokens)}"
    assert max_tokens > 0, f"max_tokens must be > 0, got {max_tokens}"

    suggested_value = max_tokens // 2
    return (
        "💡 CONTEXT LENGTH ERROR DETECTED:\n"
        "   • This is NOT a server startup failure - server is working correctly\n"
        f"   • Your max_tokens ({max_tokens}) exceeds server's limit\n"
        f"   • FIX: Reduce max_tokens to a smaller value (try {suggested_value})\n"
        "   • OR: Redeploy server with larger --max-model-len\n"
        "🛑 Stopping retries - context length errors cannot be fixed by retrying"
    )


def _format_invalid_param_error(param_keys: list) -> str:
    """Format invalid parameter error message. Pure function."""
    assert isinstance(param_keys, list), f"param_keys must be list, got {type(param_keys)}"

    return (
        "💡 PARAMETER ERROR DETECTED:\n"
        "   • Server doesn't support one of your parameters\n"
        f"   • Your parameters: {param_keys}\n"
        "   • Try removing 'logprobs' or 'echo' parameters"
    )


def sanitize_request_for_logging(params: dict) -> dict:
    """Tiger Style: Sanitize request parameters to remove large base64 image data.

    Vision messages can contain 100KB+ base64 images. Replace with bounded
    placeholders to prevent terminal spam while preserving useful debug info.
    """
    sanitized = copy.deepcopy(params)

    if "messages" in sanitized:
        for msg in sanitized["messages"]:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                # Handle vision messages (list of content parts)
                if isinstance(content, list):
                    sanitized_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            if part.get("type") == "image_url":
                                # Replace base64 data with placeholder
                                # image_url field is a dict with url key containing base64
                                url_str = str(part.get("image_url", {}).get("url", ""))
                                if url_str.startswith("data:image") and len(url_str) > 100:
                                    url_preview = f"{url_str[:50]}... ({len(url_str)} chars)"
                                else:
                                    url_preview = url_str
                                sanitized_parts.append({
                                    "type": "image_url",
                                    "image_url": {"url": url_preview},
                                })
                            else:
                                # Keep text parts
                                sanitized_parts.append(part)
                        else:
                            sanitized_parts.append(part)
                    msg["content"] = sanitized_parts
                # Handle long text content
                elif isinstance(content, str) and len(content) > 500:
                    msg["content"] = content[:500] + f"... ({len(content)} chars total)"

    return sanitized


def add_cache_control_to_last_content(
    messages, cache_control={"type": "ephemeral"}, max_cache_controls: int = 4
):
    """Adds cache control metadata to the final content block if possible."""
    assert cache_control is not None
    assert isinstance(cache_control, dict)
    assert max_cache_controls > 0
    assert max_cache_controls <= 10  # Reasonable upper bound

    if not messages:
        return messages

    assert isinstance(messages, list)
    new_messages = copy.deepcopy(messages)
    assert new_messages is not None

    cache_control_count = sum(
        1
        for msg in new_messages
        for content in (
            msg["content"] if isinstance(msg.get("content"), list) else [msg.get("content")]
        )
        if isinstance(content, dict) and "cache_control" in content
    )

    if cache_control_count >= max_cache_controls:
        return new_messages

    last_message = new_messages[-1]
    if isinstance(last_message.get("content"), list) and last_message["content"]:
        last_content = last_message["content"][-1]
        if (
            isinstance(last_content, dict)
            and "type" in last_content
            and "cache_control" not in last_content
        ):
            last_content["cache_control"] = cache_control
    elif isinstance(last_message.get("content"), dict):
        if "cache_control" not in last_message["content"]:
            last_message["content"]["cache_control"] = cache_control

    assert isinstance(new_messages, list)
    return new_messages


def _prepare_messages_for_llm(messages: list[Message]) -> list[Message]:
    """Strip tool result details before sending to LLM.

    Tiger Style: Explicit filtering, no magic.
    Tools return both `content` (for LLM) and `details` (for UI).
    This function removes `details` to reduce token usage.

    Args:
        messages: List of messages, some may have details field

    Returns:
        New list with details stripped from tool messages
    """
    assert messages is not None
    assert isinstance(messages, list)

    filtered = []
    for msg in messages:
        if msg.role == "tool":
            # Strip details field - UI-only data
            # Keep only content for LLM context
            filtered_msg = replace(msg, details=None)  # Remove details
            filtered.append(filtered_msg)
        else:
            filtered.append(msg)

    assert len(filtered) == len(messages)  # No messages dropped
    return filtered


def verbose(level: int) -> bool:
    """Simple verbose checker - just return True for now."""
    return True


def calculate_cost_from_usage(usage: Usage, model_cost: ModelCost | None) -> Cost:
    """Pure function: Usage + ModelCost -> Cost.

    Following CLASSES_VS_FUNCTIONAL: pure math, no class needed.
    Following FAVORITES: explicit inputs/outputs, no hidden state.

    Args:
        usage: Token usage from API response
        model_cost: Pricing info from models.py (per million tokens)

    Returns:
        Cost breakdown in USD
    """
    if model_cost is None:
        return Cost()

    return Cost(
        input=(usage.input_tokens / 1_000_000) * model_cost.input,
        output=(usage.output_tokens / 1_000_000) * model_cost.output,
        cache_read=(usage.cache_read_tokens / 1_000_000) * model_cost.cache_read,
        cache_write=(usage.cache_write_tokens / 1_000_000) * model_cost.cache_write,
    )
