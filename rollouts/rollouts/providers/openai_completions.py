"""OpenAI Chat Completions API provider implementation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import trio
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam

from rollouts.dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    ImageContent,
    Message,
    StreamDone,
    StreamEvent,
    StreamStart,
    TextContent,
    TextDelta,
    TextEnd,
    TextStart,
    ThinkingContent,
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

from .base import _prepare_messages_for_llm, calculate_cost_from_usage, sanitize_request_for_logging

logger = logging.getLogger(__name__)


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
    tool_calls = m.get_tool_calls()
    if not m.content and not tool_calls and m.role != "tool":
        logger.error(f"❌ Empty message content detected! Role: {m.role}")
        logger.error("   This usually means prepare_messages() is using the wrong dataset field.")
        logger.error(f"   Message object: {m}")
        assert False, (
            f"Message has empty content (role={m.role}). "
            f"Check that prepare_messages() is using the correct dataset field name. "
            f"Common issue: using 'prompt' when dataset has 'problem_description'."
        )

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
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": url}
                    })
                elif isinstance(block, ThinkingContent):
                    # Convert thinking to text for OpenAI (they don't support thinking)
                    content_parts.append({"type": "text", "text": f"<thinking>\n{block.thinking}\n</thinking>"})
            
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
                content_parts.append({"type": "text", "text": f"<thinking>\n{block.thinking}\n</thinking>"})
            
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


def _parse_usage(u: CompletionUsage) -> Usage:
    """Parse OpenAI CompletionUsage into our Usage dataclass with cache/reasoning tokens."""
    assert u is not None
    assert hasattr(u, "prompt_tokens")
    assert hasattr(u, "completion_tokens")

    # Extract cache tokens from prompt_tokens_details
    cached_tokens = 0
    if hasattr(u, "prompt_tokens_details") and u.prompt_tokens_details:
        cached_tokens = getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0

    # Extract reasoning tokens from completion_tokens_details
    reasoning_tokens = 0
    if hasattr(u, "completion_tokens_details") and u.completion_tokens_details:
        reasoning_tokens = getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0

    # Build Usage with granular token breakdown
    # input_tokens = prompt tokens minus cached (non-cached input)
    # output_tokens = completion tokens minus reasoning
    result = Usage(
        input_tokens=(u.prompt_tokens or 0) - cached_tokens,
        output_tokens=(u.completion_tokens or 0) - reasoning_tokens,
        reasoning_tokens=reasoning_tokens,
        cache_read_tokens=cached_tokens,
    )

    assert result.prompt_tokens >= 0
    assert result.completion_tokens >= 0
    return result


def _parse_completion(resp: Any) -> ChatCompletion:
    """Convert an OpenAI SDK response into the framework `ChatCompletion`."""
    assert resp is not None
    assert hasattr(resp, 'choices')
    assert hasattr(resp, 'id')
    assert hasattr(resp, 'object')
    assert hasattr(resp, 'created')
    assert hasattr(resp, 'model')
    assert hasattr(resp, 'usage')

    choices = []
    for c in resp.choices:
        assert c is not None
        assert hasattr(c, 'message')
        tool_calls = []
        if hasattr(c.message, "tool_calls") and c.message.tool_calls:
            for tc in c.message.tool_calls:
                assert tc is not None
                assert hasattr(tc, 'id')
                assert hasattr(tc, 'function')
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
        choices.append(Choice(c.index, msg, c.finish_reason))

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
) -> ChatCompletion:
    """Aggregate streaming chunks into a complete `ChatCompletion` with granular events.

    Emits granular streaming events following pi-ai pattern:
    - start: Stream begins
    - text_start/delta/end: Text content lifecycle
    - toolcall_start/delta/end: Tool call lifecycle with partial JSON parsing
    - done: Stream completes successfully
    - error: Stream encounters error
    """
    assert stream is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # Emit start event
    await on_chunk(StreamStart())

    accumulated_content = ""
    finish_reason = None
    response_id = None
    created = None

    # Track content blocks by index (text is content_index 0, tool calls start at 1)
    content_index = 0
    text_started = False

    # Track tool calls: idx -> {id, name, arguments, content_index, started}
    call_buf: dict[int, dict[str, Any]] = {}
    next_auto_index = 0

    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta

        if response_id is None:
            response_id = chunk.id
            created = chunk.created

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

    # Emit text_end if we started text
    if text_started:
        await on_chunk(TextEnd(content_index=content_index, content=accumulated_content))

    # Emit tool_end events and build final tool_calls list
    tool_calls: list[ToolCall] = []
    for idx, tc_buf in sorted(call_buf.items()):
        if tc_buf["name"]:
            try:
                args = (
                    json.loads(tc_buf["arguments"])
                    if tc_buf["arguments"]
                    else {}
                )
                tool_call = ToolCall(id=tc_buf["id"], name=tc_buf["name"], args=args)

                await on_chunk(
                    ToolCallEnd(
                        content_index=tc_buf["content_index"],
                        tool_call=tool_call,
                    )
                )

                tool_calls.append(tool_call)

            except json.JSONDecodeError as e:
                await on_chunk(
                    ToolCallError(
                        content_index=tc_buf["content_index"],
                        tool_call_id=tc_buf["id"],
                        tool_name=tc_buf["name"],
                        error=f"Invalid JSON arguments: {str(e)}",
                        raw_arguments=tc_buf["arguments"],
                    )
                )

    # Emit done event
    await on_chunk(StreamDone(finish_reason=finish_reason or "stop"))

    # Build final message with ContentBlocks
    content_blocks: list = []
    if accumulated_content:
        content_blocks.append(TextContent(text=accumulated_content))
    for tc in tool_calls:
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

    completion = ChatCompletion(
        id=response_id or "unknown",
        object="chat.completion",
        created=created or 0,
        model="",
        usage=Usage(),  # Will be populated from stream or final response
        choices=[Choice(0, final_message, finish_reason or "stop")],
    )

    assert completion is not None
    assert completion.choices is not None
    assert len(completion.choices) > 0
    return completion


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

    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "temperature": actor.endpoint.temperature,
        "stream": True,
    }

    if actor.endpoint.max_completion_tokens is not None:
        params["max_completion_tokens"] = actor.endpoint.max_completion_tokens
    else:
        params["max_tokens"] = actor.endpoint.max_tokens

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"

    if actor.endpoint.reasoning_effort is not None:
        params["reasoning_effort"] = actor.endpoint.reasoning_effort

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    # Tiger Style: Minimal validation to catch common bugs before API call
    import json
    from typing import cast

    messages = cast(list, params["messages"])
    for i, msg in enumerate(messages):
        content = msg.get("content")
        assert isinstance(content, (str, list, type(None))), \
            f"Message {i} content must be str/list/None, got {type(content)}"

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
                    assert False, (
                        f"Message {i} content[{j}] has nested role/content fields.\n"
                        f"This usually means you accidentally put a message dict inside content.\n"
                        f"For vision: content should be [{{'type': 'text', 'text': '...'}}, ...]\n"
                        f"Got: {part}"
                    )

    try:
        stream = await client.chat.completions.create(**params)
        completion = await aggregate_stream(stream, on_chunk)

    except Exception as e:
        # Log error with sanitized request details
        import json

        from openai import BadRequestError, RateLimitError

        sanitized = sanitize_request_for_logging(params)

        # Tiger Style: Fail fast on 400 errors (invalid requests)
        # These indicate bugs in our code, not transient issues
        if isinstance(e, BadRequestError):
            logger.error(
                f"OpenAI API 400 Bad Request - Invalid argument in request\n"
                f"Full sanitized request:\n{json.dumps(sanitized, indent=2)}"
            )
            # Tiger Style: Assertion to fail fast and surface the bug
            assert False, f"API returned 400 Bad Request: {e}\nThis indicates a bug in request construction. See logs above for full request."

        # Tiger Style: Rate limits are operational errors, not bugs
        # Log helpful context but let caller decide whether to retry or fail
        if isinstance(e, RateLimitError):
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
            # Re-raise - let evaluation layer handle gracefully
            raise

        # For other errors (network issues, etc), just log and raise
        msg_list = params.get('messages', [])
        msg_count = len(cast(list, msg_list)) if isinstance(msg_list, list) else 0
        logger.error(
            f"OpenAI API call failed: {e}\n"
            f"  Model: {actor.endpoint.model}\n"
            f"  Messages: {msg_count} messages",
            extra={
                "exception": str(e),
                "request_params": sanitized,
                "model": actor.endpoint.model,
            }
        )
        raise

    assert completion is not None
    completion = replace(completion, model=actor.endpoint.model)

    # Calculate cost if model pricing is available
    from rollouts.models import get_model

    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model)
    if model_meta and model_meta.cost:
        cost = calculate_cost_from_usage(completion.usage, model_meta.cost)
        usage_with_cost = replace(completion.usage, cost=cost)
        completion = replace(completion, usage=usage_with_cost)

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
