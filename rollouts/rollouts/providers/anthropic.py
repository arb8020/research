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

from rollouts.dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    ImageContent,
    LLMCallStart,
    Message,
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
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallError,
    ToolCallStart,
    Usage,
    parse_streaming_json,
)

from .base import (
    _prepare_messages_for_llm,
    add_cache_control_to_last_content,
    calculate_cost_from_usage,
    sanitize_request_for_logging,
)

logger = logging.getLogger(__name__)


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
    """Merge consecutive messages with the same role.

    The Anthropic API silently merges consecutive same-role messages server-side,
    which can cause issues with tool_result ordering. By merging explicitly,
    we ensure the content array order is correct.

    Args:
        messages: List of API-format messages (dict with role/content)

    Returns:
        Messages with consecutive same-role entries merged
    """
    if not messages:
        return []

    result: list[dict[str, Any]] = []

    for msg in messages:
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
    stream: object, on_chunk: Callable[[StreamEvent], Awaitable[None]]
) -> ChatCompletion:
    """Aggregate Anthropic SDK stream events into a `ChatCompletion` with granular events.

    Emits granular streaming events following pi-ai pattern:
    - start: Stream begins
    - text_start/delta/end: Text content lifecycle
    - thinking_start/delta/end: Extended thinking lifecycle
    - toolcall_start/delta/end: Tool call lifecycle with partial JSON parsing
    - done: Stream completes successfully
    - error: Stream encounters error
    """

    # Emit start event
    await on_chunk(StreamStart())

    # Update debug context for interrupt diagnostics
    try:
        from rollouts.frontends.runner import get_debug_context

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

    return completion


async def _get_fresh_oauth_token() -> str | None:
    """Get a fresh OAuth token, refreshing if needed. Returns None if not using OAuth.

    Raises:
        OAuthExpiredError: If token refresh fails (re-login required).
    """
    from rollouts.frontends.tui.oauth import get_oauth_client

    client = get_oauth_client()
    # Let OAuthExpiredError propagate up - caller should handle re-login
    return await client.get_valid_access_token()


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
) -> Actor:
    """Call Anthropic's API using streaming and update the actor.

    Note: **kwargs accepts but ignores provider-specific params (e.g., openai reasoning params)
    """

    # Get fresh OAuth token if using OAuth (handles mid-session expiry)
    # OAuthExpiredError will propagate up if refresh fails - caller handles re-login
    oauth_token = actor.endpoint.oauth_token
    if oauth_token:
        fresh_token = await _get_fresh_oauth_token()
        if fresh_token:
            oauth_token = fresh_token

    client = _create_anthropic_client(
        oauth_token=oauth_token,
        api_key=actor.endpoint.api_key,
        api_base=actor.endpoint.api_base,
        max_retries=actor.endpoint.max_retries,
        timeout=actor.endpoint.timeout,
    )

    # Transform messages for cross-provider compatibility (like pi-ai does)
    from rollouts.transform_messages import transform_messages

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
        "model": actor.endpoint.model,
        "temperature": actor.endpoint.temperature,
    }

    # For OAuth tokens, we MUST include Claude Code identity prefix
    # This is required by Anthropic's API for OAuth authentication
    if actor.endpoint.oauth_token:
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

    # Debug logging - show what we're sending (only at DEBUG level)
    logger.debug(f"\n{'=' * 60}")
    logger.debug("Anthropic API Request:")
    logger.debug(f"{'=' * 60}")
    logger.debug(f"Model: {actor.endpoint.model}")
    logger.debug(f"Max tokens: {actor.endpoint.max_tokens}")
    logger.debug(f"Temperature: {actor.endpoint.temperature}")
    logger.debug(f"API base: {actor.endpoint.api_base}")
    logger.debug(f"Messages count: {len(params['messages'])}")
    if system_prompt:
        logger.debug(f"System prompt length: {len(system_prompt)} chars")
        logger.debug(f"System prompt preview: {system_prompt[:200]}...")
    for i, msg in enumerate(params["messages"]):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if isinstance(content, str):
            logger.debug(f"Message {i} ({role}): {len(content)} chars - {content[:100]}...")
        elif isinstance(content, list):
            logger.debug(f"Message {i} ({role}): {len(content)} content blocks")
    logger.debug(f"{'=' * 60}\n")

    max_retries = 10
    base_delay = 2
    completion = None

    for attempt in range(max_retries + 1):
        try:
            # Emit LLMCallStart before making the API call
            await on_chunk(LLMCallStart())

            # Build extra headers - include oauth beta header if using oauth
            extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
            if oauth_token:
                extra_headers["anthropic-beta"] = "oauth-2025-04-20,prompt-caching-2024-07-31"

            async with client.messages.stream(  # type: ignore[missing-argument]
                **params,
                extra_headers=extra_headers,
            ) as stream:
                completion = await aggregate_anthropic_stream(stream, on_chunk)
                break

        except Exception as e:
            # Let OAuthExpiredError propagate immediately - handled by TUI for re-login
            from rollouts.frontends.tui.oauth import OAuthExpiredError

            if isinstance(e, OAuthExpiredError):
                raise

            # Tiger Style: Fail fast on 400 errors (invalid requests)
            # These indicate bugs in our code or invalid configuration, not transient issues
            import anthropic

            from rollouts.store import log_crash

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
                    print("🔄 OAuth token rejected, attempting refresh...")
                    from rollouts.frontends.tui.oauth import OAuthExpiredError

                    try:
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
                            print("🔐 OAuth token refreshed, retrying...")
                            continue
                    except OAuthExpiredError:
                        # Re-raise to propagate up to caller for re-login handling
                        raise
                raise RuntimeError(
                    f"Authentication failed: {e}\nCheck your API key or OAuth token."
                ) from e

            # Fail fast on ValueError - these are programming errors (e.g., empty message)
            if isinstance(e, ValueError):
                raise

            print(f"🔄 Anthropic API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
            if attempt == 0:  # Log detailed error info on first attempt
                print(f"   Endpoint model: {actor.endpoint.model}")
                print(f"   Endpoint api_base: {actor.endpoint.api_base}")

            if attempt < max_retries:
                delay = base_delay * (2**attempt)
                print(f"   Retrying in {delay}s...")
                await trio.sleep(delay)
                continue

            # All retries exhausted - raise ProviderError (excluded from accuracy)
            from .base import ProviderError

            sanitized = sanitize_request_for_logging(params)
            logger.exception(
                "Anthropic API call failed after all retries",
                extra={
                    "exception": str(e),
                    "request_params": sanitized,
                    "model": actor.endpoint.model,
                    "max_retries": max_retries,
                },
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
    from rollouts.models import get_model

    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model)
    if model_meta and model_meta.cost:
        cost = calculate_cost_from_usage(completion.usage, model_meta.cost)
        usage_with_cost = replace(completion.usage, cost=cost)
        completion = replace(completion, usage=usage_with_cost)

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

    await client.close()
    return replace(actor, trajectory=new_trajectory)
