"""Provider-specific rollout helpers for the agents module."""

from __future__ import annotations

import copy
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import httpx
import trio
from anthropic import AsyncAnthropic
from dacite import from_dict
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletionMessageParam
from shared.retry import async_retry

logger = logging.getLogger(__name__)


class NonRetryableError(Exception):
    """Exception for errors that should not be retried."""
    pass


# Tiger Style: Pure helper functions for error classification and formatting
# Keep control flow in parent, push computation to leaf functions
from enum import Enum


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

    assert result.endswith("/chat/completions"), f"result must end with /chat/completions, got {result}"
    return result


def _build_vllm_params(actor: Actor) -> dict:
    """Build vLLM API parameters. Pure function."""
    assert actor is not None
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert actor.trajectory.messages is not None

    messages = [_message_to_openai(m) for m in actor.trajectory.messages]
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
    assert isinstance(raw_tool_calls, list), f"raw_tool_calls must be list, got {type(raw_tool_calls)}"
    assert len(raw_tool_calls) > 0, "raw_tool_calls cannot be empty"

    conformed = []
    for tc in raw_tool_calls:
        conformed.append(
            {
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": json.loads(tc["function"]["arguments"]) if tc["function"]["arguments"] else {},
            }
        )

    assert len(conformed) == len(raw_tool_calls), "conformed length must match input length"
    return conformed


async def _execute_vllm_request(
    api_base: str,
    params: dict,
    headers: dict,
    max_retries: int,
    backoff_base: int,
    timeout: float
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
                print(_format_context_length_error(params.get('max_tokens', 8192)))
                raise NonRetryableError(f"Context length exceeded: {response.text}")

            if error_type == VLLMErrorType.INVALID_PARAM:
                print(_format_invalid_param_error(list(params.keys())))
                raise NonRetryableError(f"Invalid parameter: {response.text}")

            response.raise_for_status()

    result = await _call_with_retry()
    assert result is not None, "result cannot be None"
    assert isinstance(result, dict), f"result must be dict, got {type(result)}"
    return result


from .dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    Message,
    StreamChunk,
    StreamEvent,
    StreamStart,
    TextStart,
    TextDelta,
    TextEnd,
    ThinkingStart,
    ThinkingDelta,
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallError,
    StreamDone,
    StreamError,
    Tool,
    ToolCall,
    Usage,
    parse_streaming_json,
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
                                    "image_url": {"url": url_preview}
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
            msg["content"]
            if isinstance(msg.get("content"), list)
            else [msg.get("content")]
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


def _message_to_openai(m: Message) -> ChatCompletionMessageParam:
    """Convert framework `Message` objects to the OpenAI SDK schema."""
    assert m is not None
    assert isinstance(m, Message)
    assert m.role is not None
    assert isinstance(m.role, str)
    assert len(m.role) > 0

    # Validate message content - catch empty messages early
    # Tiger Style: Use assertions for programmer errors (bugs in our code)
    if not m.content and not (hasattr(m, 'tool_calls') and m.tool_calls) and m.role != "tool":
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Empty message content detected! Role: {m.role}")
        logger.error("   This usually means prepare_messages() is using the wrong dataset field.")
        logger.error(f"   Message object: {m}")
        assert False, (
            f"Message has empty content (role={m.role}). "
            f"Check that prepare_messages() is using the correct dataset field name. "
            f"Common issue: using 'prompt' when dataset has 'problem_description'."
        )

    msg: dict[str, Any] = {"role": m.role}

    if m.content is not None:
        msg["content"] = m.content

    if m.tool_calls and m.role == "assistant":
        assert isinstance(m.tool_calls, list)
        msg["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.args),
                },
            }
            for tc in m.tool_calls
        ]

    if m.role == "tool":
        assert m.tool_call_id is not None
        msg["tool_call_id"] = m.tool_call_id
        msg["content"] = m.content

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
    assert u is not None
    assert hasattr(u, 'prompt_tokens')
    assert hasattr(u, 'completion_tokens')
    assert hasattr(u, 'total_tokens')
    result = Usage(u.prompt_tokens, u.completion_tokens, u.total_tokens)
    assert result is not None
    assert result.prompt_tokens >= 0
    assert result.completion_tokens >= 0
    assert result.total_tokens >= 0
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

        msg = Message(
            role=c.message.role,
            content=c.message.content,
            tool_calls=tool_calls,
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

    # Build final message and completion
    final_message = Message(role="assistant", content=accumulated_content or "", tool_calls=tool_calls)

    assert final_message is not None
    assert isinstance(final_message, Message)
    assert isinstance(tool_calls, list)

    completion = ChatCompletion(
        id=response_id or "unknown",
        object="chat.completion",
        created=created or 0,
        model="",
        usage=Usage(0, 0, 0),
        choices=[Choice(0, final_message, finish_reason or "stop")],
    )

    assert completion is not None
    assert completion.choices is not None
    assert len(completion.choices) > 0
    return completion


def verbose(level: int) -> bool:
    """Simple verbose checker - just return True for now."""
    return True


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
    completion = await _execute_vllm_request(api_base, params, headers, max_api_retries, backoff_base, timeout)
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

    client = AsyncOpenAI(
        api_key=actor.endpoint.api_key,
        base_url=actor.endpoint.api_base,
        max_retries=actor.endpoint.max_retries,
        timeout=actor.endpoint.timeout,
    )

    messages = [_message_to_openai(m) for m in actor.trajectory.messages]

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


async def aggregate_openai_responses_stream(
    stream: AsyncIterator,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
) -> tuple[Message, dict[str, Any]]:
    """Aggregate OpenAI Responses API streaming chunks into a complete Message.

    This handles the o1/o3 reasoning models which use a different API than chat completions.
    Key differences:
    - Uses response.output_item.added/done events instead of choices[]
    - Reasoning content comes through response.reasoning_summary_text.delta
    - Different event structure for tool calls and text

    Returns:
        tuple of (final_message, usage_dict)
    """
    assert stream is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # Emit start event
    await on_chunk(StreamStart())

    # Track content blocks
    content_blocks: list[dict[str, Any]] = []
    current_item: dict[str, Any] | None = None
    current_block_index = -1
    finish_reason = "stop"

    # Usage tracking
    usage_data: dict[str, Any] = {}

    async for event in stream:
        event_type = getattr(event, "type", None)
        if event_type and event_type.startswith("response.reasoning"):
            logger.debug(f"Reasoning event: {event_type}")

        # Handle output item start
        if event_type == "response.output_item.added":
            item = event.item
            item_type = getattr(item, "type", None)

            if item_type == "reasoning":
                # Start thinking block
                current_item = {"type": "reasoning", "summary": []}
                current_block_index = len(content_blocks)
                content_blocks.append({"type": "thinking", "thinking": ""})
                await on_chunk(ThinkingStart(content_index=current_block_index))

            elif item_type == "message":
                # Start text block
                current_item = {"type": "message", "content": []}
                current_block_index = len(content_blocks)
                content_blocks.append({"type": "text", "text": ""})
                await on_chunk(TextStart(content_index=current_block_index))

            elif item_type == "function_call":
                # Start tool call block
                call_id = getattr(item, "call_id", "") + "|" + getattr(item, "id", "")
                name = getattr(item, "name", "")
                current_item = {"type": "function_call", "call_id": call_id, "name": name, "arguments": ""}
                current_block_index = len(content_blocks)
                content_blocks.append({
                    "type": "toolCall",
                    "id": call_id,
                    "name": name,
                    "arguments": "",
                })
                await on_chunk(ToolCallStart(
                    content_index=current_block_index,
                    tool_call_id=call_id,
                    tool_name=name,
                ))

        # Handle reasoning summary deltas
        elif event_type == "response.reasoning_summary_part.added":
            if current_item and current_item.get("type") == "reasoning":
                current_item.setdefault("summary", []).append({"text": ""})
                logger.debug(f"Added new summary part, total parts: {len(current_item['summary'])}")

        elif event_type == "response.reasoning_summary_text.delta":
            if current_item and current_item.get("type") == "reasoning" and current_block_index >= 0:
                delta = event.delta
                current_item.setdefault("summary", [])
                if current_item["summary"]:
                    current_item["summary"][-1]["text"] += delta
                content_blocks[current_block_index]["thinking"] += delta
                await on_chunk(ThinkingDelta(
                    content_index=current_block_index,
                    delta=delta,
                ))

        elif event_type == "response.reasoning_summary_part.done":
            # Add newlines between summary parts
            if current_item and current_item.get("type") == "reasoning" and current_block_index >= 0:
                content_blocks[current_block_index]["thinking"] += "\n\n"
                await on_chunk(ThinkingDelta(
                    content_index=current_block_index,
                    delta="\n\n",
                ))

        # Handle text output deltas
        elif event_type == "response.content_part.added":
            if current_item and current_item.get("type") == "message":
                current_item.setdefault("content", []).append(event.part)

        elif event_type == "response.output_text.delta":
            if current_item and current_item.get("type") == "message" and current_block_index >= 0:
                delta = event.delta
                content_blocks[current_block_index]["text"] += delta
                await on_chunk(TextDelta(
                    content_index=current_block_index,
                    delta=delta,
                ))

        elif event_type == "response.refusal.delta":
            if current_item and current_item.get("type") == "message" and current_block_index >= 0:
                delta = event.delta
                content_blocks[current_block_index]["text"] += delta
                await on_chunk(TextDelta(
                    content_index=current_block_index,
                    delta=delta,
                ))

        # Handle function call argument deltas
        elif event_type == "response.function_call_arguments.delta":
            if current_item and current_item.get("type") == "function_call" and current_block_index >= 0:
                delta = event.delta
                current_item["arguments"] += delta
                content_blocks[current_block_index]["arguments"] += delta
                partial_args = parse_streaming_json(content_blocks[current_block_index]["arguments"])
                await on_chunk(ToolCallDelta(
                    content_index=current_block_index,
                    tool_call_id=content_blocks[current_block_index]["id"],
                    delta=delta,
                    partial_args=partial_args,
                ))

        # Handle output item completion
        elif event_type == "response.output_item.done":
            item = event.item
            item_type = getattr(item, "type", None)

            if item_type == "reasoning" and current_block_index >= 0:
                # Finalize thinking block - store the full item as signature for re-submission
                thinking_content = content_blocks[current_block_index]["thinking"]
                # Store the raw item as JSON so we can re-submit it exactly to the API
                import json as json_module

                # Build the reasoning item dict using the data we accumulated during streaming
                # The item from the SDK doesn't have the summary we built, so use current_item
                # Only include non-null fields as API rejects null fields (except summary which is required)
                summary = current_item.get("summary", []) if current_item else []
                item_dict = {
                    "type": "reasoning",
                    "id": getattr(item, "id", ""),
                    "summary": summary,  # Required field, even if empty
                }

                content = getattr(item, "content", None)
                if content is not None:
                    item_dict["content"] = content

                encrypted_content = getattr(item, "encrypted_content", None)
                if encrypted_content is not None:
                    item_dict["encrypted_content"] = encrypted_content

                status = getattr(item, "status", None)
                if status is not None:
                    item_dict["status"] = status

                logger.debug(f"Storing reasoning item with {len(summary)} summary parts, fields: {list(item_dict.keys())}")

                # Store the serialized item
                content_blocks[current_block_index]["thinkingSignature"] = json_module.dumps(item_dict)

                await on_chunk(ThinkingEnd(
                    content_index=current_block_index,
                    content=thinking_content,
                ))
                current_item = None

            elif item_type == "message" and current_block_index >= 0:
                # Finalize text block
                text_content = content_blocks[current_block_index]["text"]
                await on_chunk(TextEnd(
                    content_index=current_block_index,
                    content=text_content,
                ))
                current_item = None

            elif item_type == "function_call" and current_block_index >= 0:
                # Finalize tool call block
                try:
                    args = json.loads(getattr(item, "arguments", "{}"))
                    tool_call = ToolCall(
                        id=content_blocks[current_block_index]["id"],
                        name=content_blocks[current_block_index]["name"],
                        args=args,
                    )
                    await on_chunk(ToolCallEnd(
                        content_index=current_block_index,
                        tool_call=tool_call,
                    ))
                except json.JSONDecodeError as e:
                    await on_chunk(ToolCallError(
                        content_index=current_block_index,
                        tool_call_id=content_blocks[current_block_index]["id"],
                        tool_name=content_blocks[current_block_index]["name"],
                        error=f"Invalid JSON arguments: {str(e)}",
                        raw_arguments=content_blocks[current_block_index]["arguments"],
                    ))
                current_item = None

        # Handle completion
        elif event_type == "response.completed":
            response = event.response
            if hasattr(response, "usage") and response.usage:
                cached_tokens = 0
                if hasattr(response.usage, "input_tokens_details"):
                    details = response.usage.input_tokens_details
                    if hasattr(details, "cached_tokens"):
                        cached_tokens = details.cached_tokens or 0

                usage_data = {
                    "input_tokens": (getattr(response.usage, "input_tokens", 0) or 0) - cached_tokens,
                    "output_tokens": getattr(response.usage, "output_tokens", 0) or 0,
                    "cache_read_tokens": cached_tokens,
                    "cache_write_tokens": 0,
                }

            # Map status to finish reason
            status = getattr(response, "status", "completed")
            if status == "completed":
                finish_reason = "stop"
            elif status == "incomplete":
                finish_reason = "length"
            else:  # failed, cancelled
                finish_reason = "stop"  # Default to stop

            # Override if we have tool calls
            has_tool_calls = any(b.get("type") == "toolCall" for b in content_blocks)
            if has_tool_calls:
                finish_reason = "tool_calls"

        # Handle errors
        elif event_type == "error":
            error_msg = getattr(event, "message", "Unknown error")
            await on_chunk(StreamError(error=error_msg))
            raise Exception(error_msg)

        elif event_type == "response.failed":
            await on_chunk(StreamError(error="Response failed"))
            raise Exception("Response failed")

    # Emit done event
    await on_chunk(StreamDone(finish_reason=finish_reason))

    # Build final message
    # Build final message with ContentBlocks
    from .dtypes import TextContent, ThinkingContent, ToolCallContent

    final_content_blocks: list = []

    for block in content_blocks:
        if block["type"] == "text":
            final_content_blocks.append(TextContent(text=block["text"]))
        elif block["type"] == "thinking":
            thinking_sig = block.get("thinkingSignature")
            final_content_blocks.append(
                ThinkingContent(
                    thinking=block["thinking"],
                    thinking_signature=thinking_sig,
                )
            )
        elif block["type"] == "toolCall":
            try:
                args = json.loads(block["arguments"]) if block["arguments"] else {}
                final_content_blocks.append(
                    ToolCallContent(
                        id=block["id"],
                        name=block["name"],
                        arguments=args,
                    )
                )
            except json.JSONDecodeError:
                pass

    final_message = Message(
        role="assistant",
        content=final_content_blocks,
    )

    logger.debug(f"Built final message with {len(final_content_blocks)} content blocks")

    return final_message, usage_data


def _messages_to_openai_responses(messages: list[Message]) -> list[dict[str, Any]]:
    """Convert rollouts Messages to OpenAI Responses API format.

    Handles new ContentBlock-based message structure:
    - Extracts text from TextContent blocks
    - Extracts thinking from ThinkingContent blocks (with signature for re-use)
    - Converts ToolCallContent to function_call format

    Key differences from chat completions:
    - User messages use input_text content type
    - Assistant messages become separate message/function_call objects
    - Tool calls are separate function_call objects, not properties on messages
    - Tool results become function_call_output objects
    """
    from .dtypes import TextContent, ThinkingContent, ToolCallContent, ImageContent

    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            # Extract text from ContentBlocks
            text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
            user_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            result.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}]
            })

        elif msg.role == "assistant":
            # Assistant messages become separate objects
            output: list[dict[str, Any]] = []

            # Process ContentBlocks
            for block in msg.content:
                if isinstance(block, ThinkingContent) and block.thinking_signature:
                    # Reuse existing reasoning item
                    logger.debug(f"Found thinking_signature, re-submitting reasoning item")
                    try:
                        reasoning_item = json.loads(block.thinking_signature)
                        output.append(reasoning_item)
                        logger.debug(f"Added reasoning item: {reasoning_item.get('type', 'unknown')}, id={reasoning_item.get('id', 'unknown')}")
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse thinking_signature: {e}")
                        pass
                elif isinstance(block, TextContent):
                    # Add text content as message object
                    output.append({
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": block.text, "annotations": []}],
                        "status": "completed",
                        "id": f"msg_{int(time.time())}_{id(msg)}",
                    })
                elif isinstance(block, ToolCallContent):
                    # Tool call IDs in responses API use format: call_id|id
                    # Split if already in that format, otherwise use as call_id
                    if "|" in block.id:
                        call_id, func_id = block.id.split("|", 1)
                    else:
                        call_id = block.id
                        func_id = f"fc_{int(time.time())}"

                    output.append({
                        "type": "function_call",
                        "id": func_id,
                        "call_id": call_id,
                        "name": block.name,
                        "arguments": json.dumps(block.arguments),
                    })

            # Add all output objects
            if output:
                result.extend(output)

        elif msg.role == "tool":
            # Extract text from ContentBlocks for tool result
            text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
            tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""

            # Extract call_id from tool_call_id (format: call_id|id or just call_id)
            call_id = msg.tool_call_id
            if "|" in call_id:
                call_id = call_id.split("|")[0]

            result.append({
                "type": "function_call_output",
                "call_id": call_id,
                "output": tool_result_text,
            })

    return result


async def rollout_openai_responses(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Make an OpenAI Responses API call (for o1/o3 reasoning models) with streaming.

    This uses the Responses API which is different from the Chat Completions API.
    It's specifically designed for reasoning models that produce extended thinking.

    Note: **kwargs accepts but ignores provider-specific params (e.g., anthropic thinking params)
    """
    assert actor is not None
    assert isinstance(actor, Actor)
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    client = AsyncOpenAI(
        api_key=actor.endpoint.api_key,
        base_url=actor.endpoint.api_base,
        max_retries=actor.endpoint.max_retries,
        timeout=actor.endpoint.timeout,
    )

    # Transform messages for cross-provider compatibility (like pi-ai does)
    from .transform_messages import transform_messages
    transformed_messages = transform_messages(
        actor.trajectory.messages,
        target_provider=actor.endpoint.provider,
        target_api="openai-responses"
    )

    # Convert messages to OpenAI Responses format
    # Note: The Responses API uses a completely different message format than chat completions
    messages = _messages_to_openai_responses(transformed_messages)

    params = {
        "model": actor.endpoint.model,
        "input": messages,  # Note: Responses API uses 'input' not 'messages'
        "stream": True,
    }

    # Add max_output_tokens (not max_completion_tokens for Responses API)
    if actor.endpoint.max_completion_tokens is not None:
        params["max_output_tokens"] = actor.endpoint.max_completion_tokens
    elif actor.endpoint.max_tokens is not None:
        params["max_output_tokens"] = actor.endpoint.max_tokens

    # Temperature is supported in Responses API (but not for some reasoning models like GPT-5-Codex)
    # Skip temperature for GPT-5 models which don't support it
    model_name = actor.endpoint.model.lower()
    if hasattr(actor.endpoint, "temperature") and actor.endpoint.temperature is not None:
        if not model_name.startswith("gpt-5"):
            params["temperature"] = actor.endpoint.temperature

    # Add reasoning config for reasoning models
    # Check if model supports reasoning
    from rollouts.models import get_model
    try:
        model_metadata = get_model(actor.endpoint.provider, actor.endpoint.model)
        is_reasoning_model = model_metadata and model_metadata.reasoning
    except (KeyError, ValueError):
        is_reasoning_model = False

    # Only set reasoning config if user explicitly requested it
    # GPT-5 models will still emit reasoning items, but without summaries unless requested
    if actor.endpoint.reasoning_effort is not None and is_reasoning_model:
        params["reasoning"] = {
            "effort": actor.endpoint.reasoning_effort,
            "summary": "auto",  # Request summary so we can re-submit it
        }
        params["include"] = ["reasoning.encrypted_content"]

    # Tools for Responses API (different format than chat completions!)
    if actor.tools:
        # Responses API expects flat structure: {type, name, description, parameters}
        # Not nested like chat completions: {type, function: {name, description, parameters}}
        params["tools"] = [
            {
                "type": "function",
                "name": t.function.name,
                "description": t.function.description,
                "parameters": {
                    "type": t.function.parameters.type,
                    "properties": t.function.parameters.properties,
                    "required": t.function.required,
                },
                "strict": None,
            }
            for t in actor.tools
        ]

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    try:
        # Use the responses.create endpoint
        stream = await client.responses.create(**params)
        final_message, usage_data = await aggregate_openai_responses_stream(stream, on_chunk)

    except Exception as e:
        import json
        from openai import BadRequestError, RateLimitError

        sanitized = sanitize_request_for_logging(params)

        if isinstance(e, BadRequestError):
            logger.error(
                f"OpenAI Responses API 400 Bad Request - Invalid argument in request\n"
                f"Full sanitized request:\n{json.dumps(sanitized, indent=2)}"
            )
            assert False, f"API returned 400 Bad Request: {e}\nThis indicates a bug in request construction. See logs above for full request."

        if isinstance(e, RateLimitError):
            logger.warning(f"Rate limit exceeded for {actor.endpoint.model}")
            raise

        logger.error(
            f"OpenAI Responses API call failed: {e}\n"
            f"  Model: {actor.endpoint.model}",
            extra={
                "exception": str(e),
                "request_params": sanitized,
                "model": actor.endpoint.model,
            }
        )
        raise

    # Build completion object
    usage = Usage(
        prompt_tokens=usage_data.get("input_tokens", 0),
        completion_tokens=usage_data.get("output_tokens", 0),
        total_tokens=usage_data.get("input_tokens", 0) + usage_data.get("output_tokens", 0),
    )

    # Enrich message with provider/api/model metadata for cross-provider handoff
    final_message = replace(
        final_message,
        provider=actor.endpoint.provider,
        api="openai-responses",
        model=actor.endpoint.model,
        usage=usage,
        stop_reason="stop",  # Responses API doesn't provide finish_reason in stream
    )

    completion = ChatCompletion(
        id="responses-" + str(int(time.time())),
        object="chat.completion",
        created=int(time.time()),
        model=actor.endpoint.model,
        usage=usage,
        choices=[Choice(0, final_message, "stop")],
    )

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    result_actor = replace(actor, trajectory=new_trajectory)
    assert result_actor is not None
    assert result_actor.trajectory is not None
    return result_actor


async def aggregate_google_stream(
    stream: AsyncIterator,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
) -> tuple[Message, dict[str, Any]]:
    """Aggregate Google Generative AI streaming chunks into a complete Message.

    Handles Gemini models' streaming format with support for:
    - Text content streaming
    - Thinking/reasoning content (thought=True flag)
    - Tool calls (functionCall)

    Returns:
        tuple of (final_message, usage_dict)
    """
    assert stream is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # Emit start event
    await on_chunk(StreamStart())

    # Track content blocks
    content_blocks: list[dict[str, Any]] = []
    current_block: dict[str, Any] | None = None
    finish_reason = "stop"

    # Usage tracking
    usage_data: dict[str, Any] = {}

    # Tool call counter for generating unique IDs
    tool_call_counter = 0

    async for chunk in stream:
        # Access the candidate from the chunk
        candidate = None
        if hasattr(chunk, "candidates") and chunk.candidates:
            candidate = chunk.candidates[0]

        # Process content parts
        if candidate and hasattr(candidate, "content") and candidate.content:
            parts = getattr(candidate.content, "parts", None)
            if parts:
                for part in parts:
                    # Handle text content
                    if hasattr(part, "text") and part.text is not None:
                        is_thinking = getattr(part, "thought", False) == True

                        # Check if we need to start a new block
                        if (
                            not current_block
                            or (is_thinking and current_block.get("type") != "thinking")
                            or (not is_thinking and current_block.get("type") != "text")
                        ):
                            # Finalize previous block if exists
                            if current_block:
                                block_index = len(content_blocks) - 1
                                if current_block["type"] == "text":
                                    await on_chunk(TextEnd(
                                        content_index=block_index,
                                        content=current_block["text"],
                                    ))
                                else:
                                    await on_chunk(ThinkingEnd(
                                        content_index=block_index,
                                        content=current_block["thinking"],
                                    ))

                            # Start new block
                            if is_thinking:
                                current_block = {
                                    "type": "thinking",
                                    "thinking": "",
                                    "thinkingSignature": getattr(part, "thoughtSignature", None),
                                }
                                content_blocks.append(current_block)
                                await on_chunk(ThinkingStart(content_index=len(content_blocks) - 1))
                            else:
                                current_block = {"type": "text", "text": ""}
                                content_blocks.append(current_block)
                                await on_chunk(TextStart(content_index=len(content_blocks) - 1))

                        # Add delta to current block
                        block_index = len(content_blocks) - 1
                        if current_block["type"] == "thinking":
                            current_block["thinking"] += part.text
                            if getattr(part, "thoughtSignature", None):
                                current_block["thinkingSignature"] = part.thoughtSignature
                            await on_chunk(ThinkingDelta(
                                content_index=block_index,
                                delta=part.text,
                            ))
                        else:
                            current_block["text"] += part.text
                            await on_chunk(TextDelta(
                                content_index=block_index,
                                delta=part.text,
                            ))

                    # Handle function calls
                    if hasattr(part, "functionCall") and part.functionCall:
                        # Finalize previous block if exists
                        if current_block:
                            block_index = len(content_blocks) - 1
                            if current_block["type"] == "text":
                                await on_chunk(TextEnd(
                                    content_index=block_index,
                                    content=current_block["text"],
                                ))
                            else:
                                await on_chunk(ThinkingEnd(
                                    content_index=block_index,
                                    content=current_block["thinking"],
                                ))
                            current_block = None

                        # Generate unique ID
                        fc = part.functionCall
                        provided_id = getattr(fc, "id", None)
                        needs_new_id = (
                            not provided_id
                            or any(b.get("type") == "toolCall" and b.get("id") == provided_id for b in content_blocks)
                        )
                        if needs_new_id:
                            tool_call_counter += 1
                            tool_call_id = f"{fc.name}_{int(time.time())}_{tool_call_counter}"
                        else:
                            tool_call_id = provided_id

                        # Create tool call block
                        tool_call_block = {
                            "type": "toolCall",
                            "id": tool_call_id,
                            "name": getattr(fc, "name", ""),
                            "arguments": dict(getattr(fc, "args", {})),
                        }
                        if getattr(part, "thoughtSignature", None):
                            tool_call_block["thoughtSignature"] = part.thoughtSignature

                        content_blocks.append(tool_call_block)
                        block_index = len(content_blocks) - 1

                        # Emit tool call events
                        await on_chunk(ToolCallStart(
                            content_index=block_index,
                            tool_call_id=tool_call_id,
                            tool_name=tool_call_block["name"],
                        ))
                        await on_chunk(ToolCallDelta(
                            content_index=block_index,
                            tool_call_id=tool_call_id,
                            delta=json.dumps(tool_call_block["arguments"]),
                            partial_args=tool_call_block["arguments"],
                        ))

                        tool_call = ToolCall(
                            id=tool_call_id,
                            name=tool_call_block["name"],
                            args=tool_call_block["arguments"],
                        )
                        await on_chunk(ToolCallEnd(
                            content_index=block_index,
                            tool_call=tool_call,
                        ))

        # Handle finish reason
        if candidate and hasattr(candidate, "finishReason") and candidate.finishReason:
            # Map Google's FinishReason to our finish_reason
            finish_reason_value = candidate.finishReason
            if hasattr(finish_reason_value, "name"):
                finish_reason_name = finish_reason_value.name
            else:
                finish_reason_name = str(finish_reason_value)

            if finish_reason_name == "STOP":
                finish_reason = "stop"
            elif finish_reason_name == "MAX_TOKENS":
                finish_reason = "length"
            else:
                finish_reason = "stop"  # Default for other reasons

            # Override if we have tool calls
            has_tool_calls = any(b.get("type") == "toolCall" for b in content_blocks)
            if has_tool_calls:
                finish_reason = "tool_calls"

        # Handle usage metadata
        if hasattr(chunk, "usageMetadata") and chunk.usageMetadata:
            metadata = chunk.usageMetadata
            usage_data = {
                "input_tokens": getattr(metadata, "promptTokenCount", 0) or 0,
                "output_tokens": (
                    (getattr(metadata, "candidatesTokenCount", 0) or 0) +
                    (getattr(metadata, "thoughtsTokenCount", 0) or 0)
                ),
                "cache_read_tokens": getattr(metadata, "cachedContentTokenCount", 0) or 0,
                "cache_write_tokens": 0,
            }

    # Finalize current block if exists
    if current_block:
        block_index = len(content_blocks) - 1
        if current_block["type"] == "text":
            await on_chunk(TextEnd(
                content_index=block_index,
                content=current_block["text"],
            ))
        else:
            await on_chunk(ThinkingEnd(
                content_index=block_index,
                content=current_block["thinking"],
            ))

    # Emit done event
    await on_chunk(StreamDone(finish_reason=finish_reason))

    # Build final message with ContentBlocks
    from .dtypes import TextContent, ThinkingContent, ToolCallContent

    final_content_blocks: list = []

    for block in content_blocks:
        if block["type"] == "text":
            final_content_blocks.append(TextContent(text=block["text"]))
        elif block["type"] == "thinking":
            final_content_blocks.append(
                ThinkingContent(
                    thinking=block["thinking"],
                    thinking_signature=block.get("thinkingSignature"),
                )
            )
        elif block["type"] == "toolCall":
            final_content_blocks.append(
                ToolCallContent(
                    id=block["id"],
                    name=block["name"],
                    arguments=block["arguments"],
                    thought_signature=block.get("thoughtSignature"),
                )
            )

    final_message = Message(
        role="assistant",
        content=final_content_blocks,
    )

    return final_message, usage_data


async def rollout_google(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Make a Google Generative AI (Gemini) API call with streaming.

    Note: **kwargs accepts but ignores provider-specific params (e.g., anthropic thinking params)
    Note: Uses trio-asyncio to bridge trio event loop with asyncio-based google-genai SDK
    """
    assert actor is not None
    assert isinstance(actor, Actor)
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "Google Generative AI SDK not installed. Install with: pip install google-genai"
        )

    try:
        import trio_asyncio
    except ImportError:
        raise ImportError(
            "trio-asyncio not installed. Install with: pip install trio-asyncio"
        )

    # Get API key
    api_key = actor.endpoint.api_key
    if not api_key:
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "Gemini API key is required. Set GEMINI_API_KEY environment variable "
                "or provide it in endpoint.api_key"
            )

    # Transform messages for cross-provider compatibility (like pi-ai does)
    from .transform_messages import transform_messages
    transformed_messages = transform_messages(
        actor.trajectory.messages,
        target_provider=actor.endpoint.provider,
        target_api="google-generative-ai"
    )

    # Prepare message conversion outside asyncio context
    # Convert messages to Google format
    from .dtypes import TextContent, ToolCallContent

    contents = []
    for m in transformed_messages:
        if m.role == "user":
            # Extract text from ContentBlocks
            text_blocks = [b for b in m.content if isinstance(b, TextContent)]
            user_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=user_text)]
            ))
        elif m.role == "assistant":
            parts = []
            # Process ContentBlocks
            for block in m.content:
                if isinstance(block, TextContent):
                    parts.append(types.Part(text=block.text))
                elif isinstance(block, ToolCallContent):
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=block.name,
                            args=block.arguments,
                        )
                    ))
            if parts:
                contents.append(types.Content(
                    role="model",
                    parts=parts
                ))
        elif m.role == "tool":
            # Extract text from ContentBlocks for tool result
            text_blocks = [b for b in m.content if isinstance(b, TextContent)]
            tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            contents.append(types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=m.tool_name if m.tool_name else "unknown",
                        response={"result": tool_result_text}
                    )
                )]
            ))

    # Build config
    config = types.GenerateContentConfig(
        temperature=actor.endpoint.temperature if hasattr(actor.endpoint, "temperature") else None,
        max_output_tokens=actor.endpoint.max_tokens if hasattr(actor.endpoint, "max_tokens") else None,
    )

    # Add tools if present
    if actor.tools:
        tools = []
        function_declarations = []
        for tool in actor.tools:
            function_declarations.append(types.FunctionDeclaration(
                name=tool.function.name,
                description=tool.function.description,
                parameters={
                    "type": tool.function.parameters.type,
                    "properties": tool.function.parameters.properties,
                    "required": tool.function.required,
                },
            ))
        tools.append(types.Tool(function_declarations=function_declarations))
        config.tools = tools

    # The actual API call in asyncio context
    async def _call_google_in_asyncio():
        # Create client inside asyncio context
        client = genai.Client(api_key=api_key)

        try:
            # Generate streaming response
            stream = await client.aio.models.generate_content_stream(
                model=actor.endpoint.model,
                contents=contents,
                config=config,
            )

            final_message, usage_data = await aggregate_google_stream(stream, on_chunk)
            return final_message, usage_data

        except Exception as e:
            import json

            logger.error(
                f"Google Generative AI API call failed: {e}\n"
                f"  Model: {actor.endpoint.model}",
                extra={
                    "exception": str(e),
                    "model": actor.endpoint.model,
                }
            )
            raise

    # Run the asyncio function from trio context using trio-asyncio loop
    async with trio_asyncio.open_loop() as loop:
        final_message, usage_data = await trio_asyncio.aio_as_trio(_call_google_in_asyncio)()

    # Build completion object
    usage = Usage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        cache_read_tokens=usage_data.get("cache_read_tokens", 0),
    )

    # Enrich message with provider/api/model metadata for cross-provider handoff
    final_message = replace(
        final_message,
        provider=actor.endpoint.provider,
        api="google-generative-ai",
        model=actor.endpoint.model,
        usage=usage,
        stop_reason="stop",
    )

    completion = ChatCompletion(
        id="google-" + str(int(time.time())),
        object="chat.completion",
        created=int(time.time()),
        model=actor.endpoint.model,
        usage=usage,
        choices=[Choice(0, final_message, "stop")],
    )

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    result_actor = replace(actor, trajectory=new_trajectory)
    assert result_actor is not None
    assert result_actor.trajectory is not None
    return result_actor


def _apply_inline_thinking_template(
    thinking_content: str, content: str, inline_thinking: str
) -> str:
    """Apply inline thinking template to combine thinking and content."""
    assert "{thinking}" in inline_thinking
    assert "{content}" in inline_thinking
    return inline_thinking.format(thinking=thinking_content, content=content)


def _message_to_anthropic(
    m: Message, inline_thinking: str | None = None
) -> dict[str, Any]:
    """Convert a `Message` into Anthropic's streaming-compatible schema.

    Handles new ContentBlock-based message structure:
    - Extracts text from TextContent blocks
    - Extracts thinking from ThinkingContent blocks
    - Converts ToolCallContent to Anthropic tool_use format
    - Handles ImageContent for vision messages
    """
    from .dtypes import TextContent, ThinkingContent, ToolCallContent, ImageContent

    assert m is not None
    assert isinstance(m, Message)
    assert m.role is not None
    assert isinstance(m.role, str)

    # Validate message content - catch empty messages early
    # Tiger Style: Use assertions for programmer errors (bugs in our code)
    if not m.content:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Empty message content detected! Role: {m.role}")
        logger.error("   This usually means prepare_messages() is using the wrong dataset field.")
        logger.error(f"   Message object: {m}")
        assert False, (
            f"Message has empty content (role={m.role}). "
            f"Check that prepare_messages() is using the correct dataset field name. "
            f"Common issue: using 'prompt' when dataset has 'problem_description'."
        )

    msg: dict[str, Any] = {"role": m.role}

    # Build content blocks from ContentBlock list
    content_blocks = []

    for block in m.content:
        if isinstance(block, TextContent):
            content_blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            thinking_block = {"type": "thinking", "thinking": block.thinking}
            if block.thinking_signature:
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
                    }
                })
            else:
                # Base64-encoded image
                content_blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    }
                })

    # If we have multiple blocks or any non-text blocks, use array format
    # Otherwise use string format for simple text messages
    if len(content_blocks) == 1 and content_blocks[0].get("type") == "text":
        msg["content"] = content_blocks[0]["text"]
    else:
        msg["content"] = content_blocks

    return msg


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
    stream, on_chunk: Callable[[StreamEvent], Awaitable[None]]
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

    accumulated_content = ""
    thinking_content = ""
    thinking_signature = None
    tool_calls: list[ToolCall] = []
    message_id = None
    created_at = int(time.time())
    finish_reason = "stop"

    # Track content blocks by index and their types
    # content_index -> {type: "text" | "thinking" | "tool_use", started: bool, accumulated: str}
    content_blocks: dict[int, dict[str, Any]] = {}

    # Tool-specific tracking
    tool_json_accumulator: dict[int, str] = {}
    tool_metadata: dict[int, dict[str, str]] = {}

    async for event in stream:
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
                if hasattr(block, "signature") and block.signature:
                    thinking_signature = block.signature
                await on_chunk(ThinkingDelta(content_index=index, delta=thinking_text))

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
                    await on_chunk(
                        ToolCallError(
                            content_index=index,
                            tool_call_id=tool_metadata[index]["id"],
                            tool_name=tool_metadata[index]["name"],
                            error=f"Invalid JSON: {str(e)}",
                            raw_arguments=tool_json_accumulator[index],
                        )
                    )

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

    # Build final message with ContentBlocks
    from .dtypes import TextContent, ThinkingContent, ToolCallContent

    content_blocks: list = []

    # Add thinking content if present
    if thinking_content:
        content_blocks.append(
            ThinkingContent(
                thinking=thinking_content,
                thinking_signature=thinking_signature,
            )
        )

    # Add text content if present
    if accumulated_content:
        content_blocks.append(TextContent(text=accumulated_content))

    # Add tool calls as ToolCallContent blocks
    for tc in tool_calls:
        content_blocks.append(
            ToolCallContent(
                id=tc.id,
                name=tc.name,
                arguments=tc.args,
            )
        )

    final_message = Message(
        role="assistant",
        content=content_blocks,
    )

    final_anthropic_message = await stream.get_final_message()

    usage = Usage(
        input_tokens=final_anthropic_message.usage.input_tokens,
        output_tokens=final_anthropic_message.usage.output_tokens,
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

    client_kwargs: dict[str, Any] = {
        "api_key": actor.endpoint.api_key,
        "max_retries": actor.endpoint.max_retries,
        "timeout": actor.endpoint.timeout,
    }
    if actor.endpoint.api_base:
        # Anthropic SDK adds /v1 automatically, so remove it if present
        base_url = actor.endpoint.api_base.rstrip('/v1').rstrip('/')
        client_kwargs["base_url"] = base_url
    client = AsyncAnthropic(**client_kwargs)

    # Transform messages for cross-provider compatibility (like pi-ai does)
    from .transform_messages import transform_messages
    transformed_messages = transform_messages(
        actor.trajectory.messages,
        target_provider=actor.endpoint.provider,
        target_api="anthropic-messages"
    )

    system_prompt = None
    messages = []

    from .dtypes import TextContent

    for m in transformed_messages:
        if m.role == "system":
            # Extract text from ContentBlocks
            text_blocks = [b for b in m.content if isinstance(b, TextContent)]
            system_prompt = "\n".join(b.text for b in text_blocks) if text_blocks else ""
        elif m.role == "tool":
            # Extract text from ContentBlocks for tool result
            text_blocks = [b for b in m.content if isinstance(b, TextContent)]
            tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": tool_result_text,
                        }
                    ],
                }
            )
        else:
            messages.append(_message_to_anthropic(m, inline_thinking))

    if user_message_for_thinking and turn_idx > 0:
        messages.append({"role": "user", "content": user_message_for_thinking})

    if messages and messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "Begin."})

    messages_with_cache = add_cache_control_to_last_content(messages)

    params: dict[str, Any] = {
        "max_tokens": actor.endpoint.max_tokens,
        "messages": messages_with_cache,
        "model": actor.endpoint.model,
        "temperature": actor.endpoint.temperature,
    }

    if system_prompt:
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
    for i, msg in enumerate(params['messages']):
        role = msg.get('role', 'unknown')
        content = msg.get('content', '')
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
            async with client.messages.stream(  # type: ignore[missing-argument]
                **params,
                extra_headers={"anthropic-beta": "prompt-caching-2024-07-31"},
            ) as stream:
                completion = await aggregate_anthropic_stream(stream, on_chunk)
                break

        except Exception as e:
            print(
                f"🔄 Anthropic API error (attempt {attempt + 1}/{max_retries + 1}): {str(e)}"
            )
            if attempt == 0:  # Log detailed error info on first attempt
                print(f"   Endpoint model: {actor.endpoint.model}")
                print(f"   Endpoint api_base: {actor.endpoint.api_base}")

            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                print(f"   Retrying in {delay}s...")
                await trio.sleep(delay)
                continue
            # Log error with sanitized request details
            sanitized = sanitize_request_for_logging(params)
            logger.error(
                "Anthropic API call failed after all retries",
                extra={
                    "exception": str(e),
                    "request_params": sanitized,
                    "model": actor.endpoint.model,
                    "max_retries": max_retries,
                }
            )
            raise

    if completion is None:
        raise Exception("Failed to get completion after all retries")

    completion = replace(completion, model=actor.endpoint.model)
    final_message = completion.choices[0].message

    # Enrich message with provider/api/model metadata for cross-provider handoff
    final_message = replace(
        final_message,
        provider=actor.endpoint.provider,
        api="anthropic-messages",
        model=actor.endpoint.model,
        usage=completion.usage if completion.usage else None,
        stop_reason=completion.choices[0].finish_reason,
    )

    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [final_message],
        completions=actor.trajectory.completions + [completion],
    )

    await client.close()
    return replace(actor, trajectory=new_trajectory)


# Provider registry - maps API types to their streaming functions
# This enables unified provider abstraction inspired by pi-ai
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rollouts.dtypes import ProviderStreamFunction
    from rollouts.models import ApiType

_PROVIDER_REGISTRY: dict[str, "ProviderStreamFunction"] = {
    "openai-completions": rollout_openai,
    "openai-responses": rollout_openai_responses,
    "anthropic-messages": rollout_anthropic,
    "google-generative-ai": rollout_google,
}


def get_provider_function(provider: str, model_id: str | None = None) -> "ProviderStreamFunction":
    """Get the streaming function for a provider/model combination.

    Uses API type abstraction - multiple providers can share the same implementation.
    For example, OpenAI, Groq, Cerebras, xAI all use openai-completions API.

    Args:
        provider: Provider identifier (e.g., "openai", "groq", "anthropic")
        model_id: Optional model ID for API type detection (needed for OpenAI o1/o3 models)

    Returns:
        Provider streaming function

    Raises:
        AssertionError: If provider/API type not found

    Examples:
        get_provider_function("openai", "gpt-4o") -> rollout_openai
        get_provider_function("groq", "llama-3.3-70b") -> rollout_openai (same function!)
        get_provider_function("anthropic", "claude-3-5-sonnet") -> rollout_anthropic
    """
    from rollouts.models import get_api_type

    api_type = get_api_type(provider, model_id)
    func = _PROVIDER_REGISTRY.get(api_type)
    assert func is not None, f"No provider function registered for API type: {api_type}"
    return func
