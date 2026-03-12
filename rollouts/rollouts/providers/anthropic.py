"""Anthropic Claude provider implementation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import trio
from anthropic import AsyncAnthropic

from ..agents import Actor
from ..core import (
    Message,
    Tool,
    ToolCall,
    Usage,
)
from ..dtypes import (
    ChatCompletion,
    Choice,
    FirstToken,
    ImageContent,
    LLMCallStart,
    RetryEnd,
    RetryStart,
    StreamDone,
    StreamError,
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
from .base import (
    _prepare_messages_for_llm,
    add_cache_control_to_last_content,
    calculate_cost_from_usage,
)

logger = logging.getLogger(__name__)


def _format_rate_limit_error(exc: Exception) -> str:
    """Format rate limit error with human-readable time until reset.

    Parses the retry-after header or anthropic-ratelimit headers to show
    something like "Rate limited (42m until reset)" instead of raw error JSON.
    """
    import anthropic

    if not isinstance(exc, anthropic.RateLimitError):
        # Not a rate limit error, return truncated message
        return str(exc)[:100]

    # Try to get retry-after from response headers
    retry_after_seconds: int | None = None

    if hasattr(exc, "response") and exc.response is not None:
        headers = exc.response.headers

        # Check retry-after header (in seconds)
        retry_after = headers.get("retry-after")
        if retry_after:
            try:
                retry_after_seconds = int(retry_after)
            except ValueError:
                pass

        # Fallback: check anthropic-ratelimit-unified-reset (Unix timestamp)
        if retry_after_seconds is None:
            reset_timestamp = headers.get("anthropic-ratelimit-unified-reset")
            if reset_timestamp:
                try:
                    reset_ts = int(reset_timestamp)
                    retry_after_seconds = max(0, reset_ts - int(time.time()))
                except ValueError:
                    pass

    # Format the message
    if retry_after_seconds is not None:
        if retry_after_seconds >= 3600:
            hours = retry_after_seconds // 3600
            mins = (retry_after_seconds % 3600) // 60
            time_str = f"{hours}h {mins}m" if mins > 0 else f"{hours}h"
        elif retry_after_seconds >= 60:
            mins = retry_after_seconds // 60
            time_str = f"{mins}m"
        else:
            time_str = f"{retry_after_seconds}s"

        return f"Rate limited ({time_str} until reset)"

    # Fallback to generic message
    return "Rate limited"


def _apply_inline_thinking_template(
    thinking_content: str, content: str, inline_thinking: str
) -> str:
    """Apply inline thinking template to combine thinking and content."""
    assert "{thinking}" in inline_thinking
    assert "{content}" in inline_thinking
    return inline_thinking.format(thinking=thinking_content, content=content)


def _message_to_anthropic(m: Message, inline_thinking: str | None = None) -> dict[str, Any]:
    """Convert a `Message` into Anthropic's streaming-compatible schema.

    Handles new ContentBlock-based message structure:
    - Extracts text from TextContent blocks
    - Extracts thinking from ThinkingContent blocks
    - Converts ToolCallContent to Anthropic tool_use format
    - Handles ImageContent for vision messages
    """

    assert m is not None
    assert isinstance(m, Message)
    assert m.role is not None
    assert isinstance(m.role, str)

    # Validate message content - catch empty messages early
    # Tiger Style: Use assertions for programmer errors (bugs in our code)
    # Note: Empty string content IS valid for tool results (tool returned nothing)
    if m.content is None or (isinstance(m.content, list) and len(m.content) == 0):
        import logging

        logger = logging.getLogger(__name__)
        logger.error(f"❌ Empty message content detected! Role: {m.role}")
        logger.error("   This usually means prepare_messages() is using the wrong dataset field.")
        logger.error(f"   Message object: {m}")
        raise AssertionError(
            f"Message has empty content (role={m.role}). "
            f"Check that prepare_messages() is using the correct dataset field name. "
            f"Common issue: using 'prompt' when dataset has 'problem_description'."
        )

    msg: dict[str, Any] = {"role": m.role}

    # Tiger Style: Explicit control flow - handle each content type
    # Handle string content (simple text messages)
    if isinstance(m.content, str):
        msg["content"] = m.content
        return msg

    # Handle ContentBlock list (structured messages with text/thinking/tools/images)
    assert isinstance(m.content, list), (
        f"content must be str or list[ContentBlock], got {type(m.content)}"
    )

    content_blocks = []

    for block in m.content:
        if isinstance(block, TextContent):
            content_blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            # If thinking signature is missing/empty (e.g., from aborted stream),
            # convert to text block to avoid API rejection per Anthropic API requirements
            if not block.thinking_signature or block.thinking_signature.strip() == "":
                content_blocks.append({
                    "type": "text",
                    "text": f"<thinking>\n{block.thinking}\n</thinking>",
                })
            else:
                thinking_block = {"type": "thinking", "thinking": block.thinking}
                thinking_block["signature"] = block.thinking_signature
                content_blocks.append(thinking_block)
        elif isinstance(block, ToolCallContent):
            content_blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.arguments,
            })
        elif isinstance(block, ImageContent):
            # Anthropic vision format
            if block.data.startswith("http"):
                # URL-based image
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "url",
                        "url": block.data,
                    },
                })
            else:
                # Base64-encoded image
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                })

    # If we have multiple blocks or any non-text blocks, use array format
    # Otherwise use string format for simple text messages
    if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
        msg["content"] = content_blocks[0]["text"]
    else:
        msg["content"] = content_blocks

    return msg


def _merge_consecutive_api_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge consecutive messages with the same role and filter empty assistant messages.

    The Anthropic API silently merges consecutive same-role messages server-side,
    which can cause issues with tool_result ordering. By merging explicitly,
    we ensure the content array order is correct.

    Also filters out assistant messages with empty content, as the Anthropic API
    rejects these (except for the optional final assistant message).
    When an assistant message with tool_use is filtered, we also remove any
    corresponding tool_result messages to avoid orphaned tool results.

    Args:
        messages: List of API-format messages (dict with role/content)

    Returns:
        Messages with consecutive same-role entries merged and empty assistants filtered
    """
    if not messages:
        return []

    # First pass: collect tool_use IDs from assistant messages that will be kept
    valid_tool_use_ids = set()
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            is_empty = (
                content == ""
                or content == []
                or (
                    isinstance(content, list)
                    and all(
                        (
                            isinstance(b, dict)
                            and b.get("type") == "text"
                            and b.get("text", "") == ""
                        )
                        for b in content
                    )
                )
            )
            if not is_empty and isinstance(content, list):
                # Collect tool_use IDs from non-empty assistant messages
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_use":
                        valid_tool_use_ids.add(block.get("id"))

    result: list[dict[str, Any]] = []

    for msg in messages:
        # Skip empty assistant messages (Anthropic API rejects them)
        if msg["role"] == "assistant":
            content = msg["content"]
            is_empty = (
                content == ""
                or content == []
                or (
                    isinstance(content, list)
                    and all(
                        (
                            isinstance(b, dict)
                            and b.get("type") == "text"
                            and b.get("text", "") == ""
                        )
                        for b in content
                    )
                )
            )
            if is_empty:
                continue  # Skip empty assistant messages

        # Filter out orphaned tool_results from user messages
        if msg["role"] == "user" and isinstance(msg["content"], list):
            filtered_content = []
            for block in msg["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id not in valid_tool_use_ids:
                        continue  # Skip orphaned tool_result
                filtered_content.append(block)

            # Skip user message if all content was filtered out
            if not filtered_content:
                continue

            msg = {"role": msg["role"], "content": filtered_content}

        if not result or result[-1]["role"] != msg["role"]:
            # Different role or first message - add as-is (copy to avoid mutation)
            result.append({"role": msg["role"], "content": msg["content"]})
        else:
            # Same role as previous - merge content
            prev = result[-1]
            prev_content = prev["content"]
            curr_content = msg["content"]

            # Normalize to lists
            if isinstance(prev_content, str):
                prev_content = [{"type": "text", "text": prev_content}]
            if isinstance(curr_content, str):
                curr_content = [{"type": "text", "text": curr_content}]

            # Merge content blocks
            prev["content"] = prev_content + curr_content

    return result


def _tool_to_anthropic(tool: Tool) -> dict[str, Any]:
    """Convert framework `Tool` definitions into Anthropic's schema."""
    return {
        "name": tool.function.name,
        "description": tool.function.description,
        "input_schema": {
            "type": tool.function.parameters.type,
            "properties": tool.function.parameters.properties,
            "required": tool.function.required,
        },
    }


async def aggregate_anthropic_stream(
    stream: object,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    request_start: float | None = None,
) -> tuple[ChatCompletion, float | None]:
    """Aggregate Anthropic SDK stream events into a `ChatCompletion` with granular events.

    Emits granular streaming events following pi-ai pattern:
    - start: Stream begins
    - text_start/delta/end: Text content lifecycle
    - thinking_start/delta/end: Extended thinking lifecycle
    - toolcall_start/delta/end: Tool call lifecycle with partial JSON parsing
    - done: Stream completes successfully
    - error: Stream encounters error

    Args:
        stream: Anthropic SDK stream object
        on_chunk: Callback for streaming events
        request_start: perf_counter timestamp when request was initiated (for TTFT calc)

    Returns:
        Tuple of (ChatCompletion, ttft_ms) where ttft_ms is time to first token in ms,
        or None if request_start wasn't provided.
    """

    # Emit start event
    await on_chunk(StreamStart())

    ttft_ms: float | None = None  # Time to first token

    # Update debug context for interrupt diagnostics
    try:
        from ..frontends.runner import get_debug_context

        debug_ctx = get_debug_context()
        debug_ctx.set_streaming()
    except ImportError:
        debug_ctx = None

    accumulated_content = ""
    thinking_content = ""
    thinking_signature = None
    tool_calls: list[ToolCall] = []
    tool_call_errors: dict[str, tuple[str, str]] = {}  # id -> (error_msg, raw_arguments)
    message_id = None
    created_at = int(time.time())
    finish_reason = "stop"

    # Track content blocks by index and their types
    # content_index -> {type: "text" | "thinking" | "tool_use", started: bool, accumulated: str}
    content_blocks: dict[int, dict[str, Any]] = {}

    # Tool-specific tracking
    tool_json_accumulator: dict[int, str] = {}
    tool_metadata: dict[int, dict[str, str]] = {}

    # Stream stall detection - log if no events for >10s
    last_event_time = time.time()
    stall_warning_threshold = 10.0  # seconds
    stall_warned = False

    # TODO: Add doom loop detection
    # Article quote: "Rarely, model responses run into 'doom loops', i.e., the model
    # re-generates part of its response endlessly, until it reaches the max_tokens limit."
    #
    # Problem: Model gets stuck repeating the same text pattern, wasting tokens and time.
    # This is rare but can corrupt eval results if not detected.
    #
    # Fix: Track recent output and detect repetition:
    #     REPETITION_WINDOW = 200
    #     if len(accumulated_content) > REPETITION_WINDOW * 3:
    #         suffix = accumulated_content[-REPETITION_WINDOW:]
    #         if accumulated_content[:-REPETITION_WINDOW].count(suffix) >= 2:
    #             raise StreamError(error="Doom loop detected - repetitive output pattern")

    async for event in stream:
        now = time.time()
        gap = now - last_event_time
        if gap > stall_warning_threshold and not stall_warned:
            logger.warning(f"Stream stalled for {gap:.1f}s waiting for next event")
            stall_warned = True
        elif gap <= stall_warning_threshold:
            stall_warned = False  # Reset warning if we got an event
        last_event_time = now

        # Update debug context for interrupt diagnostics
        if debug_ctx:
            debug_ctx.on_stream_event()

        event_type = event.type

        if event_type == "message_start":
            message_id = event.message.id
            created_at = int(time.time())

        elif event_type == "content_block_start":
            block = event.content_block
            index = event.index

            # Capture TTFT on first content block
            if ttft_ms is None and request_start is not None:
                ttft_ms = (time.perf_counter() - request_start) * 1000
                await on_chunk(FirstToken(ttft_ms=ttft_ms))

            # Initialize content block tracking
            content_blocks[index] = {
                "type": block.type,
                "started": False,
                "accumulated": "",
            }

            if block.type == "text":
                await on_chunk(TextStart(content_index=index))
                content_blocks[index]["started"] = True

            elif block.type == "thinking":
                await on_chunk(ThinkingStart(content_index=index))
                content_blocks[index]["started"] = True

            elif block.type == "tool_use":
                tool_metadata[index] = {"id": block.id, "name": block.name}
                tool_json_accumulator[index] = ""
                await on_chunk(
                    ToolCallStart(
                        content_index=index,
                        tool_call_id=block.id,
                        tool_name=block.name,
                    )
                )
                content_blocks[index]["started"] = True

        elif event_type == "content_block_delta":
            block = event.delta
            index = event.index

            if block.type == "text_delta":
                text = block.text
                accumulated_content += text
                content_blocks[index]["accumulated"] += text
                await on_chunk(TextDelta(content_index=index, delta=text))

            elif block.type == "thinking_delta":
                thinking_text = block.thinking
                thinking_content += thinking_text
                content_blocks[index]["accumulated"] += thinking_text
                await on_chunk(ThinkingDelta(content_index=index, delta=thinking_text))

            elif block.type == "signature_delta":
                # Accumulate thinking signature across deltas
                if thinking_signature is None:
                    thinking_signature = ""
                thinking_signature += block.signature

            elif block.type == "input_json_delta":
                tool_json_accumulator[index] += block.partial_json
                partial_args = parse_streaming_json(tool_json_accumulator[index])
                await on_chunk(
                    ToolCallDelta(
                        content_index=index,
                        tool_call_id=tool_metadata[index]["id"],
                        delta=block.partial_json,
                        partial_args=partial_args,
                    )
                )

        elif event_type == "content_block_stop":
            index = event.index
            block = event.content_block

            if block.type == "text":
                await on_chunk(
                    TextEnd(
                        content_index=index,
                        content=content_blocks[index]["accumulated"],
                    )
                )

            elif block.type == "thinking":
                await on_chunk(
                    ThinkingEnd(
                        content_index=index,
                        content=content_blocks[index]["accumulated"],
                    )
                )

            elif block.type == "tool_use":
                raw_json = tool_json_accumulator.get(index, "")
                try:
                    tool_input = json.loads(raw_json) if raw_json else {}
                    tool_call = ToolCall(
                        id=tool_metadata[index]["id"],
                        name=tool_metadata[index]["name"],
                        args=tool_input,
                    )
                    tool_calls.append(tool_call)

                    await on_chunk(
                        ToolCallEnd(
                            content_index=index,
                            tool_call=tool_call,
                        )
                    )

                except json.JSONDecodeError as e:
                    error_msg = f"Invalid JSON: {str(e)}"
                    tool_id = tool_metadata[index]["id"]
                    await on_chunk(
                        ToolCallError(
                            content_index=index,
                            tool_call_id=tool_id,
                            tool_name=tool_metadata[index]["name"],
                            error=error_msg,
                            raw_arguments=tool_json_accumulator[index],
                        )
                    )
                    # Still create a ToolCall so it appears in the message and agent loop
                    # can return an error to the model (like verifiers pattern)
                    tool_call = ToolCall(
                        id=tool_id,
                        name=tool_metadata[index]["name"],
                        args={},
                    )
                    tool_calls.append(tool_call)
                    tool_call_errors[tool_id] = (error_msg, tool_json_accumulator[index])

        elif event_type == "message_delta":
            if hasattr(event, "delta") and hasattr(event.delta, "stop_reason"):
                finish_reason = event.delta.stop_reason or "stop"

        elif event_type == "ping":
            # Ping events are informational, not emitting as StreamEvent
            pass

        elif event_type == "error":
            error_msg = f"{event.error.type}: {event.error.message}"
            await on_chunk(StreamError(error=error_msg))
            raise Exception(f"Anthropic stream error: {error_msg}")

    # Emit done event
    await on_chunk(StreamDone(finish_reason=finish_reason))

    # Build final message with ContentBlocks from accumulated streaming state

    final_content_blocks: list = []

    # Reconstruct content blocks in order from the content_blocks tracking dict
    # Sort by index to maintain order
    for index in sorted(content_blocks.keys()):
        block_info = content_blocks[index]
        block_type = block_info["type"]
        accumulated = block_info["accumulated"]

        if block_type == "text" and accumulated:
            final_content_blocks.append(TextContent(text=accumulated))
        elif block_type == "thinking" and accumulated:
            final_content_blocks.append(
                ThinkingContent(
                    thinking=accumulated,
                    thinking_signature=thinking_signature,
                )
            )
        elif block_type == "tool_use" and index in tool_metadata:
            # Get the final tool call from tool_calls list
            # Match by id
            for tc in tool_calls:
                if tc.id == tool_metadata[index]["id"]:
                    # Check if this tool call had a parse error
                    if tc.id in tool_call_errors:
                        error_msg, raw_args = tool_call_errors[tc.id]
                        final_content_blocks.append(
                            ToolCallContent(
                                id=tc.id,
                                name=tc.name,
                                arguments={},
                                parse_error=error_msg,
                                raw_arguments=raw_args,
                            )
                        )
                    else:
                        final_content_blocks.append(
                            ToolCallContent(
                                id=tc.id,
                                name=tc.name,
                                arguments=dict(tc.args),
                            )
                        )
                    break

    # Validate we actually received content from the stream
    # TODO: Also detect truncated responses (not just empty)
    # Article quote: "Some providers return empty or cut-off responses, although the
    # max_tokens are not reached. This includes AtlasCloud, Mancer, Fireworks (via OpenRouter)."
    #
    # Problem: We catch empty responses, but not truncated ones where the response was
    # cut off mid-generation without hitting max_tokens. Check finish_reason and response
    # patterns to detect this.
    #
    # Fix: Add truncation detection:
    #     if finish_reason not in ("stop", "end_turn", "tool_use") and not tool_calls:
    #         logger.warning(f"Unexpected finish_reason: {finish_reason} - possible truncation")
    if not final_content_blocks:
        raise ValueError(
            "aggregate_anthropic_stream produced empty message. "
            "No content blocks were received from the Anthropic stream. "
            f"content_blocks dict had {len(content_blocks)} entries. "
            "This may indicate the API call returned no content or streaming failed."
        )

    final_message = Message(
        role="assistant",
        content=final_content_blocks,
    )

    final_anthropic_message = await stream.get_final_message()

    # Extract cache token counts from Anthropic's usage
    anthropic_usage = final_anthropic_message.usage
    cache_read = getattr(anthropic_usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(anthropic_usage, "cache_creation_input_tokens", 0) or 0

    # Build Usage with cache tokens
    # input_tokens = non-cached input (total input minus cache read)
    usage = Usage(
        input_tokens=anthropic_usage.input_tokens - cache_read,
        output_tokens=anthropic_usage.output_tokens,
        cache_read_tokens=cache_read,
        cache_write_tokens=cache_write,
    )

    completion = ChatCompletion(
        id=message_id or f"msg_{int(time.time() * 1000)}",
        object="chat.completion",
        created=created_at,
        model="",
        usage=usage,
        choices=[Choice(0, final_message, finish_reason)],
    )

    return completion, ttft_ms


# TODO: Remove this function and all OAuth token refresh logic in rollout_anthropic().
# OAuth is no longer used for SDK driver. See cli.py for related cleanup TODOs.
async def _get_fresh_oauth_token() -> str | None:
    """Get a fresh OAuth token, refreshing if needed. Returns None if not using OAuth."""
    try:
        from ..frontends.tui.oauth import get_oauth_client

        client = get_oauth_client()
        return await client.get_valid_access_token()
    except Exception as e:
        logger.warning(f"Failed to refresh OAuth token: {e}")
        return None


def _create_anthropic_client(
    oauth_token: str | None,
    api_key: str,
    api_base: str | None,
    max_retries: int,
    timeout: float,
) -> AsyncAnthropic:
    """Create an Anthropic client with the appropriate auth."""
    # TODO: Refactor to use TypedDict for type-safe kwargs unpacking.
    # Currently ty reports invalid-argument-type because dict[str, Any] loses
    # per-key type info when unpacked with **kwargs. Example fix:
    #
    #   class ClientKwargs(TypedDict, total=False):
    #       api_key: str | None
    #       auth_token: str | None
    #       base_url: str | None
    #       max_retries: int
    #       timeout: float
    #
    # Then: client_kwargs: ClientKwargs = {...}
    if oauth_token:
        client_kwargs: dict[str, Any] = {
            "auth_token": oauth_token,
            "max_retries": max_retries,
            "timeout": timeout,
        }
    else:
        client_kwargs = {
            "api_key": api_key,
            "max_retries": max_retries,
            "timeout": timeout,
        }

    if api_base:
        # Anthropic SDK adds /v1 automatically, so remove it if present
        base_url = api_base.rstrip("/v1").rstrip("/")
        client_kwargs["base_url"] = base_url

    client = AsyncAnthropic(**client_kwargs)

    if oauth_token:
        # Prevent SDK from sending X-Api-Key header alongside OAuth Bearer token.
        # The SDK auto-reads ANTHROPIC_API_KEY from env, which causes both headers
        # to be sent, resulting in API key billing instead of OAuth billing.
        client.api_key = None

    return client


async def rollout_anthropic(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    user_message_for_thinking: str | None = None,
    turn_idx: int = 0,
    inline_thinking: str | None = None,
    cancel_scope: trio.CancelScope | None = None,
    **kwargs: Any,
) -> Actor:
    """Call Anthropic's API using streaming and update the actor.

    Args:
        cancel_scope: Optional Trio cancel scope for graceful cancellation.
            When cancelled, raises ProviderError instead of blocking on retries.

    Note: **kwargs accepts but ignores provider-specific params (e.g., openai reasoning params)
    """

    # Get fresh OAuth token if using OAuth (handles mid-session expiry)
    oauth_token = actor.endpoint.oauth_token
    if oauth_token:
        fresh_token = await _get_fresh_oauth_token()
        if fresh_token:
            oauth_token = fresh_token
        # If refresh failed, continue with existing token - it might still work

    client = _create_anthropic_client(
        oauth_token=oauth_token,
        api_key=actor.endpoint.api_key,
        api_base=actor.endpoint.api_base,
        max_retries=actor.endpoint.max_retries,
        timeout=actor.endpoint.timeout,
    )

    # Ensure client is closed even on exception. This prevents httpx connection cleanup
    # from racing with trio_asyncio teardown, which causes "Task got bad yield" errors.
    # Alternative fix: isolate trio_asyncio.open_loop() to only SSH operations, so httpx
    # runs in pure trio. But that's a larger refactor - this try/finally is sufficient.
    try:
        # Transform messages for cross-provider compatibility (like pi-ai does)
        from ..agents.transform_messages import transform_messages

        transformed_messages = transform_messages(
            actor.trajectory.messages,
            target_provider=actor.endpoint.provider,
            target_api="anthropic-messages",
        )

        # Strip details before sending to LLM
        llm_messages = _prepare_messages_for_llm(transformed_messages)

        system_prompt = None
        messages = []

        for m in llm_messages:
            if m.role == "system":
                # Extract text from ContentBlocks
                text_blocks = [b for b in m.content if isinstance(b, TextContent)]
                system_prompt = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            elif m.role == "tool":
                # Extract text from tool result content (handle both string and ContentBlock list)
                if isinstance(m.content, str):
                    tool_result_text = m.content
                elif isinstance(m.content, list):
                    text_blocks = [b for b in m.content if isinstance(b, TextContent)]
                    tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
                else:
                    tool_result_text = ""
                messages.append({
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": tool_result_text,
                        }
                    ],
                })
            else:
                messages.append(_message_to_anthropic(m, inline_thinking))

        if user_message_for_thinking and turn_idx > 0:
            messages.append({"role": "user", "content": user_message_for_thinking})

        if messages and messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "Begin."})

        # Merge consecutive same-role messages to avoid API rejection
        # The Anthropic API silently merges them server-side, but this can cause
        # tool_result ordering issues. Merge explicitly to control the order.
        messages = _merge_consecutive_api_messages(messages)

        messages_with_cache = add_cache_control_to_last_content(messages)

        params: dict[str, Any] = {
            "max_tokens": actor.endpoint.max_tokens,
            "messages": messages_with_cache,
            "model": actor.endpoint.model_id,  # Use model_id, not full "provider/model" string
            "temperature": actor.endpoint.temperature,
        }

        # For OAuth tokens or Claude Code API keys, we MUST include Claude Code identity prefix
        # This is required by Anthropic's API for OAuth authentication and Claude Code restricted API keys
        requires_claude_code_identity = (
            actor.endpoint.oauth_token or actor.endpoint.is_claude_code_api_key
        )
        if requires_claude_code_identity:
            claude_code_identity = "You are Claude Code, Anthropic's official CLI for Claude."
            if system_prompt:
                # Prepend Claude Code identity to existing system prompt
                params["system"] = [
                    {
                        "type": "text",
                        "text": claude_code_identity,
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}},
                ]
            else:
                # Just the Claude Code identity
                params["system"] = [
                    {
                        "type": "text",
                        "text": claude_code_identity,
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
        elif system_prompt:
            params["system"] = system_prompt

        if actor.tools:
            params["tools"] = [_tool_to_anthropic(t) for t in actor.tools]

        if actor.endpoint.thinking is not None:
            params["thinking"] = actor.endpoint.thinking

        # Wide event logging for API request (structured for queryability)
        # This single log line captures everything needed to debug API issues
        from .base import log_api_request

        _tool_names = [t.function.name for t in actor.tools] if actor.tools else []
        log_api_request(
            provider="anthropic",
            model=actor.endpoint.model,
            api_base=actor.endpoint.api_base,
            messages=params["messages"],
            system_prompt=system_prompt,
            tools=_tool_names,
            temperature=actor.endpoint.temperature,
            max_tokens=actor.endpoint.max_tokens,
            thinking_enabled=actor.endpoint.thinking is not None,
            turn_idx=turn_idx,
        )

        max_retries = 10
        base_delay = 2
        completion = None
        ttft_ms: float | None = None
        retrying = False  # Track if we emitted a RetryStart (to emit RetryEnd on success)

        # Capture timing for span
        import time
        from datetime import datetime

        request_start = time.perf_counter()
        request_started_at = datetime.now().isoformat()

        for attempt in range(max_retries + 1):
            try:
                # Emit LLMCallStart before making the API call
                from .base import log_api_attempt

                log_api_attempt(
                    provider="anthropic",
                    model=actor.endpoint.model,
                    attempt=attempt + 1,
                    max_attempts=max_retries + 1,
                )
                await on_chunk(LLMCallStart())

                # Build extra headers - include oauth beta header if using oauth or Claude Code API key
                extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
                if oauth_token or actor.endpoint.is_claude_code_api_key:
                    extra_headers["anthropic-beta"] = "oauth-2025-04-20,prompt-caching-2024-07-31"

                async with client.messages.stream(
                    **params,
                    extra_headers=extra_headers,
                ) as stream:
                    completion, ttft_ms = await aggregate_anthropic_stream(
                        stream, on_chunk, request_start
                    )
                    # If we were retrying and succeeded, emit RetryEnd
                    if retrying:
                        await on_chunk(RetryEnd(success=True, attempt=attempt + 1))

                    # Wide event for successful response
                    from .base import log_api_response

                    log_api_response(
                        provider="anthropic",
                        model=actor.endpoint.model,
                        attempt=attempt + 1,
                        success=True,
                        input_tokens=completion.usage.input_tokens if completion.usage else None,
                        output_tokens=completion.usage.output_tokens if completion.usage else None,
                        cache_read_tokens=completion.usage.cache_read_tokens
                        if completion.usage
                        else None,
                        cache_write_tokens=completion.usage.cache_write_tokens
                        if completion.usage
                        else None,
                        stop_reason=completion.choices[0].stop_reason
                        if completion.choices
                        else None,
                        has_tool_calls=bool(completion.choices[0].message.get_tool_calls())
                        if completion.choices
                        else False,
                    )
                    break

            except Exception as e:
                # Tiger Style: Fail fast on 400 errors (invalid requests)
                # These indicate bugs in our code or invalid configuration, not transient issues
                import anthropic

                from ..store import log_crash
                from .base import ContextTooLongError

                if isinstance(e, anthropic.BadRequestError):
                    error_str = str(e)
                    # Check for context length errors - these are recoverable, not crashes
                    if "prompt is too long" in error_str or "too many tokens" in error_str.lower():
                        # Try to extract token counts from error message
                        import re

                        match = re.search(r"(\d+)\s*tokens?\s*>\s*(\d+)", error_str)
                        current_tokens = int(match.group(1)) if match else None
                        max_tokens = int(match.group(2)) if match else None
                        raise ContextTooLongError(
                            f"Context too long: {current_tokens:,} tokens (max: {max_tokens:,})",
                            current_tokens=current_tokens,
                            max_tokens=max_tokens,
                        ) from e

                    # Other 400 errors are likely bugs - log and fail
                    crash_file = log_crash(e, "anthropic", actor.endpoint.model, messages=messages)
                    # Fail immediately - don't retry configuration errors
                    raise AssertionError(
                        f"API returned 400 Bad Request: {e}\nCrash details written to: {crash_file}"
                    ) from e

                # For OAuth: try to refresh token and retry once on auth errors
                if isinstance(e, anthropic.AuthenticationError):
                    if oauth_token and attempt == 0:
                        # Emit retry event for OAuth refresh
                        await on_chunk(
                            RetryStart(
                                attempt=1,
                                max_attempts=2,
                                delay_seconds=0,
                                error_message="OAuth token rejected, attempting refresh",
                                provider="anthropic",
                            )
                        )
                        fresh_token = await _get_fresh_oauth_token()
                        if fresh_token and fresh_token != oauth_token:
                            oauth_token = fresh_token
                            # Recreate client with new token
                            await client.close()
                            client = _create_anthropic_client(
                                oauth_token=oauth_token,
                                api_key=actor.endpoint.api_key,
                                api_base=actor.endpoint.api_base,
                                max_retries=actor.endpoint.max_retries,
                                timeout=actor.endpoint.timeout,
                            )
                            continue
                    from .base import AuthenticationError

                    raise AuthenticationError(
                        f"Authentication failed: {e}\nCheck your API key or OAuth token."
                    ) from e

                # Fail fast on 404 errors - model doesn't exist, retrying won't help
                if isinstance(e, anthropic.NotFoundError):
                    raise AssertionError(
                        f"Model not found: {actor.endpoint.model}. "
                        f"Check model name at https://docs.anthropic.com/en/docs/about-claude/models"
                    ) from e

                # Fail fast on programming errors - these are bugs in our code, not transient issues
                if isinstance(e, ValueError | AttributeError | TypeError | KeyError):
                    raise

                # Transient error - emit retry event and wait
                if attempt < max_retries:
                    # Check if cancelled before starting retry wait
                    if cancel_scope is not None and cancel_scope.cancel_called:
                        from .base import ProviderError

                        raise ProviderError(
                            f"Cancelled during retry: {_format_rate_limit_error(e)}",
                            original_error=e,
                            attempts=attempt + 1,
                            provider="anthropic",
                        ) from e

                    delay = base_delay * (2**attempt)
                    error_msg = _format_rate_limit_error(e)
                    await on_chunk(
                        RetryStart(
                            attempt=attempt + 1,
                            max_attempts=max_retries + 1,
                            delay_seconds=delay,
                            error_message=error_msg,
                            provider="anthropic",
                        )
                    )
                    retrying = True

                    # Sleep with cancellation support
                    if cancel_scope is not None:
                        with trio.move_on_after(delay) as sleep_scope:
                            await trio.sleep(delay)
                        if cancel_scope.cancel_called:
                            from .base import ProviderError

                            raise ProviderError(
                                f"Cancelled during retry wait: {error_msg}",
                                original_error=e,
                                attempts=attempt + 1,
                                provider="anthropic",
                            ) from e
                    else:
                        await trio.sleep(delay)
                    continue

                # All retries exhausted - emit RetryEnd and raise ProviderError
                from .base import ProviderError, log_api_response

                error_msg = _format_rate_limit_error(e)
                await on_chunk(
                    RetryEnd(
                        success=False,
                        attempt=max_retries + 1,
                        final_error=error_msg,
                    )
                )

                log_api_response(
                    provider="anthropic",
                    model=actor.endpoint.model,
                    attempt=max_retries + 1,
                    success=False,
                    error_type=type(e).__name__,
                    error_message=str(e),
                )
                raise ProviderError(
                    f"Anthropic API failed after {max_retries + 1} attempts: {e}",
                    original_error=e,
                    attempts=max_retries + 1,
                    provider="anthropic",
                ) from e

        if completion is None:
            from .base import ProviderError

            raise ProviderError(
                "Failed to get completion after all retries",
                attempts=max_retries + 1,
                provider="anthropic",
            )

        completion = replace(completion, model=actor.endpoint.model)

        # Calculate cost if model pricing is available
        from ..models import get_model

        model_meta = get_model(actor.endpoint.provider, actor.endpoint.model_id)
        if model_meta and model_meta.cost:
            cost = calculate_cost_from_usage(completion.usage, model_meta.cost)
            usage_with_cost = replace(completion.usage, cost=cost)
            completion = replace(completion, usage=usage_with_cost)

        # Emit span for cost/latency tracking
        request_duration_ms = (time.perf_counter() - request_start) * 1000
        from .base import persist_span

        session_id = kwargs.get("session_id")
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
        )

        final_message = completion.choices[0].message

        # Enrich message with provider/api/model metadata for cross-provider handoff
        final_message = replace(
            final_message,
            provider=actor.endpoint.provider,
            api="anthropic-messages",
            model=actor.endpoint.model,
        )

        new_trajectory = replace(
            actor.trajectory,
            messages=actor.trajectory.messages + [final_message],
            completions=actor.trajectory.completions + [completion],
        )

        return replace(actor, trajectory=new_trajectory)
    finally:
        await client.close()
