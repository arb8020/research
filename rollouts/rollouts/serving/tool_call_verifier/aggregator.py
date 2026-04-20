"""Raw-SSE aggregator specialized for K2VV-shape tool-call conformance.

Produces a ChatCompletion-shape dict (matching OpenAI's response_format)
from a chat-completions request, honoring streaming when the row sets
stream=true. Tool-call delta accumulation is bit-compatible with KVV's
_accumulate_tool_calls so schema validation applies to the same
assembled arguments KVV would have seen.

# TODO(tool-call-aggregator-convergence): this is a third OpenAI-compatible
# aggregator alongside aggregate_stream (SDK-object based) and
# aggregate_openai_compatible_sse (raw SSE, text-only) in
# rollouts/providers/openai_completions.py. It exists because we want
# K2VV-bit-exact tool_call delta accumulation against raw SSE, and neither
# existing aggregator does that. These three should converge: one raw-SSE
# aggregator that handles text + tool_calls + usage, with the SDK-object
# path either deleted or kept as a thin wrapper. Don't add a fourth variant;
# fold into this one or extend aggregate_openai_compatible_sse.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _accumulate_tool_calls(
    delta_tool_calls: list[dict[str, Any]],
    acc: dict[int, dict[str, Any]],
) -> None:
    """Fold tool-call deltas into an indexed accumulator.

    Mirrors KVV ToolCallsValidator._accumulate_tool_calls but operates on
    raw dicts rather than SDK objects.
    """
    for tc in delta_tool_calls:
        idx = tc.get("index")
        if idx is None:
            idx = 0

        if idx not in acc:
            acc[idx] = {
                "id": tc.get("id"),
                "type": tc.get("type", "function"),
                "function": {"name": "", "arguments": ""},
            }

        # Late-arriving id/type after the first delta for an index.
        if tc.get("id") and not acc[idx].get("id"):
            acc[idx]["id"] = tc["id"]
        if tc.get("type") and acc[idx].get("type") in (None, "function"):
            acc[idx]["type"] = tc["type"]

        func = tc.get("function") or {}
        name = func.get("name")
        arguments = func.get("arguments")
        if name:
            acc[idx]["function"]["name"] = name
        if arguments:
            acc[idx]["function"]["arguments"] += arguments


def _parse_sse_line(line: str) -> dict[str, Any] | None:
    """Return the JSON object carried by one SSE data line, or None."""
    if not line.startswith("data:"):
        return None
    payload = line[len("data:") :].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


async def _aggregate_streaming_response(
    response: httpx.Response,
    model_hint: str,
) -> dict[str, Any]:
    """Read an SSE stream and return a ChatCompletion-shape dict."""
    response_id: str | None = None
    created: int | None = None
    model = model_hint
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls_acc: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None

    async for raw_line in response.aiter_lines():
        chunk = _parse_sse_line(raw_line)
        if chunk is None:
            continue

        if response_id is None:
            response_id = chunk.get("id")
        if created is None:
            created = chunk.get("created")
        if chunk.get("model"):
            model = chunk["model"]

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str) and content:
                content_parts.append(content)
            reasoning = delta.get("reasoning_content")
            if isinstance(reasoning, str) and reasoning:
                reasoning_parts.append(reasoning)
            deltas_tc = delta.get("tool_calls")
            if isinstance(deltas_tc, list):
                _accumulate_tool_calls(deltas_tc, tool_calls_acc)

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

        choice_usage = choice.get("usage")
        if isinstance(choice_usage, dict):
            usage = choice_usage

    tool_calls_list = (
        [tool_calls_acc[idx] for idx in sorted(tool_calls_acc.keys())] if tool_calls_acc else None
    )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(content_parts),
        "tool_calls": tool_calls_list,
    }
    if reasoning_parts:
        message["reasoning_content"] = "".join(reasoning_parts)

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


async def send_row(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    api_key: str | None,
    row: dict[str, Any],
    model: str,
    extra_body: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Send one K2VV row to /chat/completions and return (status, response).

    status is "success" or "failed"; on failure `response` carries an error
    descriptor rather than a ChatCompletion-shape dict. Preserves the row's
    stream flag — when stream=true, accumulates the SSE into the same
    ChatCompletion shape as the non-streaming path.
    """

    # Row-provided values win; extra_body fills gaps. Model is always
    # overridden to the configured endpoint's model so a corpus authored
    # against kimi-k2-* still hits our deployed model.
    request_body: dict[str, Any] = dict(row)
    if extra_body:
        for key, value in extra_body.items():
            request_body.setdefault(key, value)
    request_body["model"] = model

    if request_body.get("stream"):
        stream_options = request_body.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options.setdefault("include_usage", True)
        request_body["stream_options"] = stream_options

    # K2's _input role is a KVV convention — map to system so OpenAI-compatible
    # servers accept it. This mirrors KVV.prepare_request.
    messages = request_body.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "_input":
                msg["role"] = "system"

    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        if request_body.get("stream"):
            async with client.stream("POST", url, json=request_body, headers=headers) as resp:
                if resp.status_code >= 400:
                    error_body = (await resp.aread()).decode("utf-8", errors="replace")
                    return "failed", {
                        "error_type": "HTTPStatusError",
                        "error_message": f"status={resp.status_code} body={error_body[:500]}",
                        "error": f"status={resp.status_code}",
                    }
                payload = await _aggregate_streaming_response(resp, model_hint=model)
                return "success", payload
        else:
            resp = await client.post(url, json=request_body, headers=headers)
            if resp.status_code >= 400:
                return "failed", {
                    "error_type": "HTTPStatusError",
                    "error_message": f"status={resp.status_code} body={resp.text[:500]}",
                    "error": f"status={resp.status_code}",
                }
            return "success", resp.json()
    except httpx.RequestError as exc:
        return "failed", {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "error": str(exc),
        }
