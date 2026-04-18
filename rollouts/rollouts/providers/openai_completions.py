"""OpenAI Chat Completions API provider implementation."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import httpx
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam

from ..agents import Actor
from ..core import Message, Tool, ToolCall, Usage
from ..dtypes import (
    ChatCompletion,
    Choice,
    FirstToken,
    ImageContent,
    Logprob,
    Logprobs,
    StreamDone,
    StreamEvent,
    StreamStart,
    TextContent,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingContent,
    ThinkingDelta,
    ThinkingEnd,
    ThinkingStart,
    ToolCallContent,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallError,
    ToolCallStart,
    parse_streaming_json,
)
from .base import _prepare_messages_for_llm, calculate_cost_from_usage, sanitize_request_for_logging

logger = logging.getLogger(__name__)


def _serialize_top_logprob_entry(entry: Any) -> dict[str, Any]:
    """Normalize one SDK top-logprob entry into plain data."""
    token = getattr(entry, "token", "")
    logprob = getattr(entry, "logprob", 0.0)
    token_id = getattr(entry, "token_id", None)
    raw_bytes = getattr(entry, "bytes", None)
    return {
        "token": token,
        "logprob": float(logprob),
        "bytes": list(raw_bytes) if raw_bytes else [],
        "token_id": token_id,
    }


def _build_logprob_from_sdk_entry(entry: Any) -> Logprob:
    top_candidates = []
    raw_top = getattr(entry, "top_logprobs", None)
    if raw_top:
        top_candidates = [_serialize_top_logprob_entry(candidate) for candidate in raw_top]
    return Logprob(
        token=getattr(entry, "token", ""),
        logprob=getattr(entry, "logprob", 0.0),
        bytes=list(entry.bytes) if hasattr(entry, "bytes") and entry.bytes else [],
        top_logprobs=[candidate["logprob"] for candidate in top_candidates],
        top_candidates=top_candidates,
        token_id=getattr(entry, "token_id", None),
    )


def _normalize_openai_stream_chunk_payload(payload: str) -> dict[str, Any] | None:
    """Parse one SSE payload line from an OpenAI-compatible stream."""
    stripped = payload.strip()
    if not stripped or stripped == "[DONE]":
        return None
    return json.loads(stripped)


def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    """Convert framework `Message` objects to the OpenAI SDK schema.

    Handles new ContentBlock-based message structure:
    - Extracts text from TextContent blocks
    - Converts ThinkingContent to text (OpenAI chat completions don't support thinking)
    - Converts ToolCallContent to tool_calls format
    - Converts ImageContent to image_url format
    """

    assert m is not None
    assert isinstance(m, Message)
    assert m.role is not None
    assert isinstance(m.role, str)
    assert len(m.role) > 0

    # Validate message content - catch empty messages early
    # Tiger Style: Use assertions for programmer errors (bugs in our code)
    # Note: assistant messages can have empty content if tool calls failed to parse
    tool_calls = m.get_tool_calls()
    if not m.content and not tool_calls and m.role == "user":
        # Only crash on empty user messages - that's definitely a bug
        logger.error(f"❌ Empty message content detected! Role: {m.role}")
        logger.error("   This usually means prepare_messages() is using the wrong dataset field.")
        logger.error(f"   Message object: {m}")
        raise AssertionError(
            f"Message has empty content (role={m.role}). "
            f"Check that prepare_messages() is using the correct dataset field name. "
            f"Common issue: using 'prompt' when dataset has 'problem_description'."
        )
    if not m.content and not tool_calls and m.role == "assistant":
        # Empty assistant message - likely from failed tool call parsing
        # Add placeholder to prevent downstream errors
        logger.warning("Empty assistant message detected - adding placeholder")
        m = Message(role="assistant", content="[No response generated]")

    msg: dict[str, Any] = {"role": m.role}

    # Tiger Style: Explicit control flow - handle each content type
    # Handle string content (simple text messages)
    if isinstance(m.content, str):
        msg["content"] = m.content
    elif isinstance(m.content, list):
        # Handle ContentBlock list (structured messages with text/thinking/tools/images)
        if m.role == "user":
            # User messages: convert ContentBlocks to OpenAI format
            content_parts = []
            for block in m.content:
                if isinstance(block, TextContent):
                    content_parts.append({"type": "text", "text": block.text})
                elif isinstance(block, ImageContent):
                    # Convert ImageContent to OpenAI image_url format
                    if block.data.startswith("http"):
                        url = block.data
                    else:
                        # Base64-encoded image
                        url = f"data:{block.mime_type};base64,{block.data}"
                    content_parts.append({"type": "image_url", "image_url": {"url": url}})
                elif isinstance(block, ThinkingContent):
                    # Convert thinking to text for OpenAI (they don't support thinking)
                    content_parts.append({
                        "type": "text",
                        "text": f"<thinking>\n{block.thinking}\n</thinking>",
                    })

            # If only one text block, use string format; otherwise use array
            if len(content_parts) == 1 and content_parts[0].get("type") == "text":
                msg["content"] = content_parts[0]["text"]
            else:
                msg["content"] = content_parts
        elif m.role == "assistant":
            # Assistant messages: extract text blocks, handle thinking, tool calls handled separately
            text_blocks = [b for b in m.content if isinstance(b, TextContent)]
            thinking_blocks = [b for b in m.content if isinstance(b, ThinkingContent)]

            # Combine text and thinking into content
            content_parts = []
            for block in text_blocks:
                content_parts.append({"type": "text", "text": block.text})
            for block in thinking_blocks:
                # Convert thinking to text (OpenAI chat completions don't support thinking)
                content_parts.append({
                    "type": "text",
                    "text": f"<thinking>\n{block.thinking}\n</thinking>",
                })

            if len(content_parts) == 1:
                msg["content"] = content_parts[0]["text"]
            elif len(content_parts) > 1:
                msg["content"] = content_parts
            else:
                msg["content"] = None
        elif m.role == "tool":
            # Tool messages: extract text from ContentBlocks
            if isinstance(m.content, list):
                text_blocks = [b for b in m.content if isinstance(b, TextContent)]
                msg["content"] = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            else:
                msg["content"] = m.content
        else:
            # Unknown role - pass through
            msg["content"] = m.content

    if tool_calls and m.role == "assistant":
        assert isinstance(tool_calls, list)
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args),
                },
            }
            for tc in tool_calls
        ]

    if m.role == "tool":
        assert m.tool_call_id is not None
        msg["tool_call_id"] = m.tool_call_id

    assert "role" in msg
    return msg


def _tool_to_openai(tool: Tool) -> dict[str, Any]:
    """Convert a framework `Tool` definition into OpenAI's schema."""
    assert tool is not None
    assert isinstance(tool, Tool)
    assert tool.function is not None
    assert tool.function.name is not None
    assert len(tool.function.name) > 0
    assert tool.function.parameters is not None

    result = {
        "type": tool.type,
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": {
                "type": tool.function.parameters.type,
                "properties": tool.function.parameters.properties,
                "required": tool.function.required,
            },
        },
    }
    assert "type" in result
    assert "function" in result
    return result


def _maybe_add_openrouter_cache_control(
    model_id: str,
    messages: list[dict[str, Any]],
) -> None:
    """Inject cache_control on the last user/assistant text block for OpenRouter models
    that require explicit cache breakpoints (anthropic/* and google/* via Gemini).

    DeepSeek, OpenAI, Grok, Moonshot, and Groq via OpenRouter use automatic caching —
    no markers needed. Anthropic and Gemini require explicit cache_control on content
    blocks to opt into caching, even when routed through OpenRouter.

    Mutates messages in-place (same pattern as pi-mono's maybeAddOpenRouterAnthropicCacheControl).
    """
    provider_prefix = model_id.split("/")[0] if "/" in model_id else ""
    if provider_prefix not in ("anthropic", "google"):
        return

    # Walk backwards to find the last user or assistant message with text content
    # and stamp cache_control on its last text block.
    for msg in reversed(messages):
        if msg.get("role") not in ("user", "assistant"):
            continue
        content = msg.get("content")
        if isinstance(content, str):
            msg["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
            return
        if isinstance(content, list):
            for part in reversed(content):
                if (
                    isinstance(part, dict)
                    and part.get("type") == "text"
                    and "cache_control" not in part
                ):
                    part["cache_control"] = {"type": "ephemeral"}
                    return


def _parse_usage(u: CompletionUsage) -> Usage:
    """Parse OpenAI CompletionUsage into our Usage dataclass with cache/reasoning tokens."""
    assert u is not None
    assert hasattr(u, "prompt_tokens")
    assert hasattr(u, "completion_tokens")

    # Extract cache tokens from prompt_tokens_details.
    # cache_write_tokens lives at the same nesting level as cached_tokens.
    # OpenRouter (and some other providers) report cached_tokens as
    # (cache_hits + cache_writes). Subtract cache_write to get true reads.
    # TODO: parse CompletionUsage into a trusted frozen dataclass at the boundary
    # so downstream code operates on real types instead of getattr gymnastics.
    reported_cached_tokens = 0
    cache_write_tokens = 0
    if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        reported_cached_tokens = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0
        cache_write_tokens = getattr(u.prompt_tokens_details, "cache_write_tokens", 0) or 0

    assert reported_cached_tokens >= 0
    assert cache_write_tokens >= 0
    # When writes are present, reported_cached_tokens >= cache_write_tokens must hold
    # because reported_cached_tokens = reads + writes (the overlapping representation).
    if cache_write_tokens > 0:
        assert reported_cached_tokens >= cache_write_tokens

    cache_read_tokens = (
        reported_cached_tokens - cache_write_tokens
        if cache_write_tokens > 0
        else reported_cached_tokens
    )
    assert cache_read_tokens >= 0
    assert cache_read_tokens <= reported_cached_tokens

    reasoning_tokens = 0
    if hasattr(u, "completion_tokens_details") and u.completion_tokens_details:
        reasoning_tokens = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
    assert reasoning_tokens >= 0

    input_tokens = max(0, (u.prompt_tokens or 0) - cache_read_tokens - cache_write_tokens)
    output_tokens = max(0, (u.completion_tokens or 0) - reasoning_tokens)

    result = Usage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
    )

    assert result.input_tokens >= 0
    assert result.output_tokens >= 0
    assert result.cache_read_tokens >= 0
    assert result.cache_write_tokens >= 0
    return result


def _parse_completion(resp: Any) -> ChatCompletion:
    """Convert an OpenAI SDK response into the framework `ChatCompletion`."""
    assert resp is not None
    assert hasattr(resp, "choices")
    assert hasattr(resp, "id")
    assert hasattr(resp, "object")
    assert hasattr(resp, "created")
    assert hasattr(resp, "model")
    assert hasattr(resp, "usage")

    choices = []
    for c in resp.choices:
        assert c is not None
        assert hasattr(c, "message")
        tool_calls = []
        if hasattr(c.message, "tool_calls") and c.message.tool_calls:
            for tc in c.message.tool_calls:
                assert tc is not None
                assert hasattr(tc, "id")
                assert hasattr(tc, "function")
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        args=json.loads(tc.function.arguments) if tc.function.arguments else {},
                    )
                )

        # Build ContentBlocks
        content_blocks: list = []
        if c.message.content:
            content_blocks.append(TextContent(text=c.message.content))
        for tc in tool_calls:
            content_blocks.append(
                ToolCallContent(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.args,
                )
            )

        msg = Message(
            role=c.message.role,
            content=content_blocks if content_blocks else c.message.content,
        )
        assert msg is not None
        choice_logprobs = None
        if hasattr(c, "logprobs") and c.logprobs is not None:
            lp_content = getattr(c.logprobs, "content", None)
            if lp_content:
                choice_logprobs = Logprobs(
                    content=[_build_logprob_from_sdk_entry(entry) for entry in lp_content]
                )
        token_ids = None
        if choice_logprobs is not None:
            candidate_token_ids = tuple(
                entry.token_id for entry in choice_logprobs.content if entry.token_id is not None
            )
            if candidate_token_ids:
                token_ids = candidate_token_ids
        choices.append(
            Choice(
                c.index,
                msg,
                c.finish_reason,
                logprobs=choice_logprobs,
                token_ids=token_ids,
            )
        )

    assert len(choices) > 0
    result = ChatCompletion(
        id=resp.id,
        object=resp.object,
        created=resp.created,
        model=resp.model,
        usage=_parse_usage(resp.usage),
        choices=choices,
    )
    assert result is not None
    assert result.choices is not None
    assert len(result.choices) > 0
    return result


async def aggregate_stream(
    stream: AsyncIterator,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    request_start: float | None = None,
) -> tuple[ChatCompletion, float | None]:
    """Aggregate streaming chunks into a complete `ChatCompletion` with granular events.

    Emits granular streaming events following pi-ai pattern:
    - start: Stream begins
    - text_start/delta/end: Text content lifecycle
    - toolcall_start/delta/end: Tool call lifecycle with partial JSON parsing
    - done: Stream completes successfully
    - error: Stream encounters error

    Args:
        stream: Async iterator of streaming chunks
        on_chunk: Callback for streaming events
        request_start: perf_counter timestamp when request was initiated (for TTFT calc)

    Returns:
        Tuple of (ChatCompletion, ttft_ms) where ttft_ms is time to first token in ms,
        or None if request_start wasn't provided.

    TODO: Add doom loop detection (same as Anthropic provider)
    Article quote: "Rarely, model responses run into 'doom loops', i.e., the model
    re-generates part of its response endlessly, until it reaches the max_tokens limit."

    TODO: Detect truncated responses
    Article quote: "Some providers return empty or cut-off responses, although the
    max_tokens are not reached."
    Check finish_reason and add warning if not "stop" or "tool_calls".
    """
    import time

    assert stream is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # Emit start event
    await on_chunk(StreamStart())

    accumulated_content = ""
    ttft_ms: float | None = None  # Time to first token
    finish_reason = None
    response_id = None
    created = None
    stream_usage: Usage | None = (
        None  # Captured from final chunk if stream_options.include_usage=True
    )

    # Track content blocks by index (text is content_index 0, tool calls start at 1)
    content_index = 0
    text_started = False

    # Track thinking/reasoning content (kimi, qwen, deepseek, etc.)
    thinking_started = False
    thinking_content_index = 0
    accumulated_thinking = ""

    # Track tool calls: idx -> {id, name, arguments, content_index, started}
    call_buf: dict[int, dict[str, Any]] = {}
    next_auto_index = 0

    # Accumulate logprobs for GRPO importance sampling
    accumulated_logprobs: list[Logprob] = []

    async for chunk in stream:
        # Capture usage from final chunk (when stream_options.include_usage=True)
        if hasattr(chunk, "usage") and chunk.usage is not None:
            stream_usage = _parse_usage(chunk.usage)

        # Skip chunks with no choices (e.g., usage-only final chunk)
        if not chunk.choices:
            continue

        choice = chunk.choices[0]
        delta = choice.delta

        # Accumulate logprobs from streaming chunks (for GRPO importance sampling)
        # OpenAI streaming includes choice.logprobs with per-token logprobs
        if hasattr(choice, "logprobs") and choice.logprobs is not None:
            lp_content = getattr(choice.logprobs, "content", None)
            if lp_content:
                for lp in lp_content:
                    # Convert OpenAI logprob to our Logprob format
                    accumulated_logprobs.append(_build_logprob_from_sdk_entry(lp))

        if response_id is None:
            response_id = chunk.id
            created = chunk.created

        # Check for reasoning content (kimi, qwen, deepseek, llama.cpp, etc.)
        # Different providers use different field names for reasoning/thinking content
        reasoning_fields = ["reasoning_content", "reasoning", "reasoning_text"]
        reasoning_delta = None
        delta_dict = delta.model_dump() if hasattr(delta, "model_dump") else vars(delta)
        for field in reasoning_fields:
            value = delta_dict.get(field)
            if value is not None and len(value) > 0:
                reasoning_delta = value
                break

        # Capture TTFT on first content chunk (text, reasoning, or tool call)
        if ttft_ms is None and request_start is not None:
            if delta.content or delta.tool_calls or reasoning_delta:
                ttft_ms = (time.perf_counter() - request_start) * 1000
                await on_chunk(FirstToken(ttft_ms=ttft_ms))

        # Handle reasoning/thinking content (emitted before text content)
        if reasoning_delta:
            if not thinking_started:
                await on_chunk(ThinkingStart(content_index=thinking_content_index))
                thinking_started = True

            accumulated_thinking += reasoning_delta
            await on_chunk(
                ThinkingDelta(content_index=thinking_content_index, delta=reasoning_delta)
            )

        # Handle text content
        if delta.content:
            if not text_started:
                await on_chunk(TextStart(content_index=content_index))
                text_started = True

            accumulated_content += delta.content
            await on_chunk(TextDelta(content_index=content_index, delta=delta.content))

        # Handle tool calls
        if delta.tool_calls:
            for tool_call in delta.tool_calls:
                idx = tool_call.index
                if idx is None:
                    idx = next_auto_index
                    next_auto_index += 1

                # Initialize tool call buffer if new
                if idx not in call_buf:
                    # Assign next content_index (text is 0, first tool is 1, etc.)
                    tool_content_index = content_index + idx + 1

                    call_buf[idx] = {
                        "id": "",
                        "name": "",
                        "arguments": "",
                        "content_index": tool_content_index,
                        "started": False,
                    }

                # Update tool call metadata
                if tool_call.id:
                    call_buf[idx]["id"] = tool_call.id
                if tool_call.function:
                    if tool_call.function.name:
                        call_buf[idx]["name"] = tool_call.function.name
                    if tool_call.function.arguments:
                        call_buf[idx]["arguments"] += tool_call.function.arguments

                # Emit start event if this is the first chunk for this tool call
                tc_buf = call_buf[idx]
                if not tc_buf["started"] and tc_buf["name"]:
                    await on_chunk(
                        ToolCallStart(
                            content_index=tc_buf["content_index"],
                            tool_call_id=tc_buf["id"],
                            tool_name=tc_buf["name"],
                        )
                    )
                    tc_buf["started"] = True

                # Emit delta event with partial JSON parsing
                if tool_call.function and tool_call.function.arguments:
                    partial_args = parse_streaming_json(tc_buf["arguments"])
                    await on_chunk(
                        ToolCallDelta(
                            content_index=tc_buf["content_index"],
                            tool_call_id=tc_buf["id"],
                            delta=tool_call.function.arguments,
                            partial_args=partial_args,
                        )
                    )

        if choice.finish_reason:
            finish_reason = choice.finish_reason

    # Emit thinking_end if we started thinking
    if thinking_started:
        await on_chunk(
            ThinkingEnd(content_index=thinking_content_index, content=accumulated_thinking)
        )

    # Emit text_end if we started text
    if text_started:
        await on_chunk(TextEnd(content_index=content_index, content=accumulated_content))

    # Emit tool_end events and build final tool_calls list
    # Track both successful tool calls and parse errors
    tool_calls: list[ToolCall] = []
    tool_call_errors: dict[str, tuple[str, str]] = {}  # id -> (error_msg, raw_arguments)

    for idx, tc_buf in sorted(call_buf.items()):
        if tc_buf["name"]:
            try:
                args = json.loads(tc_buf["arguments"]) if tc_buf["arguments"] else {}
                # Verify args is a dict - model might return "8" instead of {"result": 8}
                if not isinstance(args, dict):
                    raise ValueError(f"Tool args must be object, got {type(args).__name__}")
                tool_call = ToolCall(id=tc_buf["id"], name=tc_buf["name"], args=args)

                await on_chunk(
                    ToolCallEnd(
                        content_index=tc_buf["content_index"],
                        tool_call=tool_call,
                    )
                )

                tool_calls.append(tool_call)

            except (json.JSONDecodeError, ValueError) as e:
                error_msg = f"Invalid tool arguments: {str(e)}"
                await on_chunk(
                    ToolCallError(
                        content_index=tc_buf["content_index"],
                        tool_call_id=tc_buf["id"],
                        tool_name=tc_buf["name"],
                        error=error_msg,
                        raw_arguments=tc_buf["arguments"],
                    )
                )
                # Still create a ToolCall so it appears in the message and agent loop
                # can return an error to the model (like verifiers pattern)
                tool_call = ToolCall(id=tc_buf["id"], name=tc_buf["name"], args={})
                tool_calls.append(tool_call)
                tool_call_errors[tc_buf["id"]] = (error_msg, tc_buf["arguments"])

    # Emit done event
    await on_chunk(StreamDone(finish_reason=finish_reason or "stop"))

    # Build final message with ContentBlocks
    content_blocks: list = []
    # Add thinking content first (if any) - mirrors Anthropic's thinking block ordering
    if accumulated_thinking:
        content_blocks.append(ThinkingContent(thinking=accumulated_thinking))
    if accumulated_content:
        content_blocks.append(TextContent(text=accumulated_content))
    for tc in tool_calls:
        # Check if this tool call had a parse error
        if tc.id in tool_call_errors:
            error_msg, raw_args = tool_call_errors[tc.id]
            content_blocks.append(
                ToolCallContent(
                    id=tc.id,
                    name=tc.name,
                    arguments={},
                    parse_error=error_msg,
                    raw_arguments=raw_args,
                )
            )
        else:
            content_blocks.append(
                ToolCallContent(
                    id=tc.id,
                    name=tc.name,
                    arguments=tc.args,
                )
            )

    final_message = Message(role="assistant", content=content_blocks if content_blocks else "")

    assert final_message is not None
    assert isinstance(final_message, Message)
    assert isinstance(tool_calls, list)

    # Build logprobs for Choice (for GRPO importance sampling)
    choice_logprobs = Logprobs(content=accumulated_logprobs) if accumulated_logprobs else None

    token_ids = tuple(
        entry.token_id for entry in accumulated_logprobs if entry.token_id is not None
    )
    completion = ChatCompletion(
        id=response_id or "unknown",
        object="chat.completion",
        created=created or 0,
        model="",
        usage=stream_usage or Usage(),  # From final chunk if stream_options.include_usage=True
        choices=[
            Choice(
                0,
                final_message,
                finish_reason or "stop",
                logprobs=choice_logprobs,
                token_ids=token_ids or None,
            )
        ],
    )

    assert completion is not None
    assert completion.choices is not None
    assert len(completion.choices) > 0
    return completion, ttft_ms


async def aggregate_openai_compatible_sse(
    stream: AsyncIterator[str],
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    request_start: float | None = None,
) -> tuple[ChatCompletion, float | None]:
    """Aggregate a raw SSE stream from a loosely OpenAI-compatible backend.

    This is the compatibility path for servers that speak the same transport
    shape as OpenAI chat completions but do not satisfy the Python SDK's stricter
    response model assumptions.
    """
    import time

    assert stream is not None
    assert on_chunk is not None

    await on_chunk(StreamStart())

    accumulated_content = ""
    ttft_ms: float | None = None
    finish_reason = None
    response_id = None
    created = None
    model = ""
    text_started = False

    async for payload in stream:
        chunk = _normalize_openai_stream_chunk_payload(payload)
        if chunk is None:
            continue

        if response_id is None:
            response_id = chunk.get("id")
        if created is None:
            created = chunk.get("created")
        if not model:
            model = chunk.get("model", "")

        choices = chunk.get("choices") or []
        if not choices:
            continue

        choice = choices[0]
        delta = choice.get("delta") or {}

        if (
            ttft_ms is None
            and request_start is not None
            and (delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls"))
        ):
            ttft_ms = (time.perf_counter() - request_start) * 1000
            await on_chunk(FirstToken(ttft_ms=ttft_ms))

        if content_delta := delta.get("content"):
            if not text_started:
                await on_chunk(TextStart(content_index=0))
                text_started = True
            accumulated_content += content_delta
            await on_chunk(TextDelta(content_index=0, delta=content_delta))

        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    if text_started:
        await on_chunk(TextEnd(content_index=0, content=accumulated_content))
    await on_chunk(StreamDone(finish_reason=finish_reason or "stop"))

    return (
        ChatCompletion(
            id=response_id or "unknown",
            object="chat.completion",
            created=created or 0,
            model=model,
            usage=Usage(),
            choices=[
                Choice(
                    0,
                    Message(role="assistant", content=accumulated_content),
                    finish_reason or "stop",
                )
            ],
        ),
        ttft_ms,
    )


async def _iter_openai_sse_payloads(response: httpx.Response) -> AsyncIterator[str]:
    """Yield raw `data:` payloads from an SSE response body."""
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        yield line[len("data:") :].strip()


async def _rollout_openai_raw_sse(
    *,
    actor: Actor,
    params: dict[str, Any],
    request_options: dict[str, Any],
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    request_start: float,
) -> tuple[ChatCompletion, float | None]:
    """Fallback for self-hosted OpenAI-compatible servers with looser streams."""
    headers: dict[str, str] = {}
    if actor.endpoint.api_key:
        headers["Authorization"] = f"Bearer {actor.endpoint.api_key}"

    payload = dict(params)
    extra_body = request_options.get("extra_body")
    if isinstance(extra_body, dict):
        payload.update(extra_body)

    api_base = actor.endpoint.api_base.rstrip("/")
    url = (
        f"{api_base}/chat/completions"
        if api_base.endswith("/v1")
        else f"{api_base}/v1/chat/completions"
    )

    timeout = httpx.Timeout(actor.endpoint.timeout)
    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        async with client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            return await aggregate_openai_compatible_sse(
                _iter_openai_sse_payloads(response),
                on_chunk,
                request_start,
            )


async def rollout_openai(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Make an OpenAI API call with streaming and update the actor.

    Note: **kwargs accepts but ignores provider-specific params (e.g., anthropic thinking params)
    """
    assert actor is not None
    assert isinstance(actor, Actor)
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    client_kwargs = {
        "api_key": actor.endpoint.api_key,
        "max_retries": actor.endpoint.max_retries,
        "timeout": actor.endpoint.timeout,
    }
    # Only set base_url if it's provided (non-empty)
    if actor.endpoint.api_base:
        client_kwargs["base_url"] = actor.endpoint.api_base
    client = AsyncOpenAI(**client_kwargs)

    # Strip details before sending to LLM
    llm_messages = _prepare_messages_for_llm(actor.trajectory.messages)
    messages = [_message_to_openai(m) for m in llm_messages]

    if actor.endpoint.provider == "openrouter":
        _maybe_add_openrouter_cache_control(actor.endpoint.model_id, messages)

    params = {
        "model": actor.endpoint.model_id,  # Use model_id, not full "provider/model" string
        "messages": messages,
        "temperature": actor.endpoint.temperature,
        "stream": True,
    }
    # logprobs=True is a GRPO training convenience that breaks eval against some
    # backends (observed: SGLang + DeepSeek V3.2 long prompts → immediate EOS).
    # Off by default; opt in via extra_params["logprobs"]=True when you need
    # importance sampling. extra_params gets folded in below via extra_body,
    # but the OpenAI SDK treats top-level `logprobs` as a typed field that
    # overrides extra_body, so set it at the typed level.
    if (
        hasattr(actor.endpoint, "extra_params")
        and actor.endpoint.extra_params
        and actor.endpoint.extra_params.get("logprobs") is True
    ):
        params["logprobs"] = True

    if actor.endpoint.max_completion_tokens is not None:
        params["max_completion_tokens"] = actor.endpoint.max_completion_tokens
    else:
        params["max_tokens"] = actor.endpoint.max_tokens

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = (
            actor.endpoint.tool_choice if actor.endpoint.tool_choice is not None else "auto"
        )
        if actor.endpoint.parallel_tool_calls is not None:
            params["parallel_tool_calls"] = actor.endpoint.parallel_tool_calls

    # Handle thinking/reasoning based on model's thinking_format
    # Different providers use different param formats to enable thinking
    from ..models import get_model

    model_metadata = get_model(actor.endpoint.provider, actor.endpoint.model_id)
    thinking_format = model_metadata.thinking_format if model_metadata else None

    if thinking_format == "zai":
        # Z.ai/GLM uses thinking: { type: "enabled" | "disabled" }
        # Must explicitly set since z.ai defaults to thinking enabled
        thinking_enabled = actor.endpoint.reasoning_effort is not None
        params["thinking"] = {"type": "enabled" if thinking_enabled else "disabled"}
    elif thinking_format == "qwen" or thinking_format == "dashscope":
        # Qwen/DashScope uses enable_thinking: boolean
        # Required to get reasoning_content from models like kimi-k2.5, qwen, deepseek-r1
        params["enable_thinking"] = actor.endpoint.reasoning_effort is not None
    elif thinking_format == "chat_template":
        # Baseten/some opencode models use chat_template_args
        if actor.endpoint.reasoning_effort is not None:
            params["chat_template_args"] = {"enable_thinking": True}
    elif actor.endpoint.reasoning_effort is not None:
        # OpenAI-style reasoning_effort (default)
        params["reasoning_effort"] = actor.endpoint.reasoning_effort

    # Request usage info in streaming response (final chunk contains usage)
    params["stream_options"] = {"include_usage": True}

    request_options: dict[str, Any] = {}
    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        request_options["extra_body"] = actor.endpoint.extra_params.copy()

    # Wide event logging for API request
    from .base import log_api_request

    _tool_names = [t.function.name for t in actor.tools] if actor.tools else []
    log_api_request(
        provider="openai",
        model=actor.endpoint.model,
        api_base=actor.endpoint.api_base,
        messages=params["messages"],
        tools=_tool_names,
        temperature=actor.endpoint.temperature,
        max_tokens=params.get("max_tokens") or params.get("max_completion_tokens"),
        tool_choice=params.get("tool_choice"),
        parallel_tool_calls=params.get("parallel_tool_calls"),
    )

    # Tiger Style: Minimal validation to catch common bugs before API call
    import json
    from typing import cast

    messages = cast(list, params["messages"])
    for i, msg in enumerate(messages):
        content = msg.get("content")
        assert isinstance(content, (str, list, type(None))), (
            f"Message {i} content must be str/list/None, got {type(content)}"
        )

        # Vision messages: content is list of parts with 'type' field
        if isinstance(content, list):
            for j, part in enumerate(content):
                # Common bug: nested message dict instead of vision part
                if "role" in part or ("content" in part and "type" not in part):
                    sanitized = sanitize_request_for_logging(params)
                    logger.error(
                        f"Invalid message format - nested role/content in message {i} part {j}\n"
                        f"Full request:\n{json.dumps(sanitized, indent=2)}"
                    )
                    raise AssertionError(
                        f"Message {i} content[{j}] has nested role/content fields.\n"
                        f"This usually means you accidentally put a message dict inside content.\n"
                        f"For vision: content should be [{{'type': 'text', 'text': '...'}}, ...]\n"
                        f"Got: {part}"
                    )

    # Capture timing for span
    import time
    from datetime import datetime

    request_start = time.perf_counter()
    request_started_at = datetime.now().isoformat()

    request_duration_ms = 0.0  # Initialize in case of error
    ttft_ms: float | None = None

    try:
        stream = await client.chat.completions.create(**params, **request_options)
        completion, ttft_ms = await aggregate_stream(stream, on_chunk, request_start)

        request_duration_ms = (time.perf_counter() - request_start) * 1000

        # Wide event for successful response
        from .base import log_api_response

        log_api_response(
            provider="openai",
            model=actor.endpoint.model,
            attempt=1,
            success=True,
            input_tokens=completion.usage.input_tokens if completion.usage else None,
            output_tokens=completion.usage.output_tokens if completion.usage else None,
            cache_read_tokens=completion.usage.cache_read_tokens if completion.usage else None,
            reasoning_tokens=completion.usage.reasoning_tokens if completion.usage else None,
            stop_reason=completion.choices[0].stop_reason if completion.choices else None,
            has_tool_calls=bool(completion.choices[0].message.get_tool_calls())
            if completion.choices
            else False,
        )

    except Exception as e:
        from openai import BadRequestError, RateLimitError

        from ..store import log_crash

        sanitized = sanitize_request_for_logging(params)

        # Tiger Style: Fail fast on 400 errors (invalid requests)
        # These indicate bugs in our code, not transient issues
        if isinstance(e, BadRequestError):
            crash_file = log_crash(e, "openai", actor.endpoint.model)
            # Tiger Style: Assertion to fail fast and surface the bug
            raise AssertionError(
                f"API returned 400 Bad Request: {e}\nCrash details written to: {crash_file}"
            ) from e

        # Tiger Style: Rate limits are operational errors, not bugs
        # Wrap in ProviderError so evaluation layer can exclude from accuracy
        if isinstance(e, RateLimitError):
            from .base import ProviderError

            error_msg = str(e)
            # Extract quota info if available
            if "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                logger.warning(
                    f"Rate limit exceeded for {actor.endpoint.model}\n"
                    f"  This is an operational limit, not a bug.\n"
                    f"  Solutions:\n"
                    f"    1. Reduce max_concurrent in your config (try 1-2 for Gemini)\n"
                    f"    2. Add delays between requests\n"
                    f"    3. Use a model with higher quota"
                )
            else:
                logger.warning(f"Rate limit error: {error_msg}")

            raise ProviderError(
                f"OpenAI rate limit exceeded after retries: {e}",
                original_error=e,
                attempts=actor.endpoint.max_retries,
                provider="openai",
            ) from e

        # For other transient errors (network issues, 5xx, etc), wrap as ProviderError
        from openai import APIConnectionError, APITimeoutError, InternalServerError

        # Transient network/API errors that should trigger retry
        if isinstance(
            e, (APIConnectionError, APITimeoutError, InternalServerError, httpx.RemoteProtocolError)
        ):
            fallback_succeeded = False
            if actor.endpoint.provider == "sglang":
                logger.warning(
                    "OpenAI SDK stream failed for %s; retrying via raw SSE compatibility path",
                    actor.endpoint.model,
                    extra={"exception": str(e), "request_params": sanitized},
                )
                try:
                    completion, ttft_ms = await _rollout_openai_raw_sse(
                        actor=actor,
                        params=params,
                        request_options=request_options,
                        on_chunk=on_chunk,
                        request_start=request_start,
                    )
                    request_duration_ms = (time.perf_counter() - request_start) * 1000
                except Exception:
                    logger.exception(
                        "Raw SSE compatibility fallback also failed for %s",
                        actor.endpoint.model,
                        extra={"exception": str(e), "request_params": sanitized},
                    )
                else:
                    from .base import log_api_response

                    log_api_response(
                        provider="openai",
                        model=actor.endpoint.model,
                        attempt=1,
                        success=True,
                        input_tokens=completion.usage.input_tokens if completion.usage else None,
                        output_tokens=completion.usage.output_tokens if completion.usage else None,
                        cache_read_tokens=completion.usage.cache_read_tokens
                        if completion.usage
                        else None,
                        reasoning_tokens=completion.usage.reasoning_tokens
                        if completion.usage
                        else None,
                        stop_reason=completion.choices[0].stop_reason
                        if completion.choices
                        else None,
                        has_tool_calls=bool(completion.choices[0].message.get_tool_calls())
                        if completion.choices
                        else False,
                    )
                    fallback_succeeded = True
            if fallback_succeeded:
                completion = replace(completion, model=actor.endpoint.model)
            else:
                from .base import ProviderError

                msg_list = params.get("messages", [])
                msg_count = len(cast(list, msg_list)) if isinstance(msg_list, list) else 0
                logger.exception(
                    f"OpenAI API call failed: {e}\n"
                    f"  Model: {actor.endpoint.model}\n"
                    f"  Messages: {msg_count} messages",
                    extra={
                        "exception": str(e),
                        "request_params": sanitized,
                        "model": actor.endpoint.model,
                    },
                )
                raise ProviderError(
                    f"OpenAI API error after retries: {e}",
                    original_error=e,
                    attempts=actor.endpoint.max_retries,
                    provider="openai",
                ) from e

        # For other errors, log and re-raise as-is (likely bugs)
        msg_list = params.get("messages", [])
        msg_count = len(cast(list, msg_list)) if isinstance(msg_list, list) else 0
        logger.exception(
            f"OpenAI API call failed: {e}\n"
            f"  Model: {actor.endpoint.model}\n"
            f"  Messages: {msg_count} messages",
            extra={
                "exception": str(e),
                "request_params": sanitized,
                "model": actor.endpoint.model,
            },
        )
        raise

    assert completion is not None
    completion = replace(completion, model=actor.endpoint.model)

    # Calculate cost if model pricing is available
    from ..models import get_model

    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model_id)
    if model_meta and model_meta.cost:
        cost = calculate_cost_from_usage(completion.usage, model_meta.cost)
        usage_with_cost = replace(completion.usage, cost=cost)
        completion = replace(completion, usage=usage_with_cost)

    # Emit span for cost/latency tracking
    from .base import persist_span

    session_id = kwargs.get("session_id")
    session_store = kwargs.get("session_store")
    await persist_span(
        session_id=session_id,
        started_at=request_started_at,
        duration_ms=request_duration_ms,
        provider=actor.endpoint.provider,
        model=actor.endpoint.model,
        api_base=actor.endpoint.api_base,
        usage=completion.usage,
        request_id=completion.id,
        finish_reason=completion.choices[0].finish_reason if completion.choices else None,
        ttft_ms=ttft_ms,
        session_store=session_store,
    )

    assert completion.choices is not None
    assert len(completion.choices) > 0
    final_message = completion.choices[0].message
    assert final_message is not None

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
