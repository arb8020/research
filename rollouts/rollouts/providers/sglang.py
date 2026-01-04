"""vLLM/SGLang provider implementation.

Five modes available:
- rollout_sglang: Non-streaming, uses httpx directly
- rollout_sglang_streaming: Streaming via OpenAI SDK, with ToolCallError handling
- rollout_sglang_token_level: Token-level TI/TO via SGLang's /generate endpoint
- rollout_vllm_token_level: Token-level TI/TO via vLLM's /v1/completions endpoint

Token-level modes (TI/TO) avoid retokenization issues that cause RL training collapse.
They pass token IDs directly to the inference server, store generated token_ids in
Choice, and decode to text for the agent interface.

See radixark/miles for reference token-level implementation:
https://github.com/radixark/miles/blob/main/miles/rollout/sglang_rollout.py
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any

import httpx
from dacite import from_dict
from openai import AsyncOpenAI

from .._retry import async_retry
from ..dtypes import Actor, ChatCompletion, StreamEvent, ToolCallError
from .base import (
    NonRetryableError,
    VLLMErrorType,
    _classify_vllm_error,
    _format_context_length_error,
    _format_invalid_param_error,
    _prepare_messages_for_llm,
)
from .openai_completions import _message_to_openai, _tool_to_openai, aggregate_stream

logger = logging.getLogger(__name__)


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


@dataclass
class ConformResult:
    """Result of conforming tool calls - includes both successes and errors."""

    tool_calls: list[dict]
    errors: list[dict]  # [{id, name, error, raw_arguments}, ...]


def _conform_tool_calls(raw_tool_calls: list) -> ConformResult:
    """Convert raw OpenAI tool calls to conformed format with error handling.

    Instead of crashing on invalid JSON, captures errors and returns them
    alongside successfully parsed tool calls.

    Returns:
        ConformResult with tool_calls (successful) and errors (failed parses)
    """
    assert isinstance(raw_tool_calls, list), (
        f"raw_tool_calls must be list, got {type(raw_tool_calls)}"
    )
    assert len(raw_tool_calls) > 0, "raw_tool_calls cannot be empty"

    conformed = []
    errors = []

    for tc in raw_tool_calls:
        tool_id = tc["id"]
        tool_name = tc["function"]["name"]
        raw_args = tc["function"]["arguments"]

        try:
            if raw_args:
                args = json.loads(raw_args)
                # Verify it's a dict (object), not a primitive
                if not isinstance(args, dict):
                    raise ValueError(f"Tool args must be object, got {type(args).__name__}")
            else:
                args = {}

            conformed.append({
                "id": tool_id,
                "name": tool_name,
                "args": args,
            })
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse tool call args for {tool_name}: {e}")
            errors.append({
                "id": tool_id,
                "name": tool_name,
                "error": str(e),
                "raw_arguments": raw_args,
            })

    return ConformResult(tool_calls=conformed, errors=errors)


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
    async def _call_with_retry() -> httpx.Response:
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

    # Process tool calls with error handling
    message = completion["choices"][0]["message"]
    raw_tool_calls = message.get("tool_calls")

    if raw_tool_calls:
        conform_result = _conform_tool_calls(raw_tool_calls)
        message["tool_calls"] = conform_result.tool_calls

        # Emit ToolCallError events for any parse failures
        for i, error in enumerate(conform_result.errors):
            await on_chunk(
                ToolCallError(
                    content_index=i,
                    tool_call_id=error["id"],
                    tool_name=error["name"],
                    error=error["error"],
                    raw_arguments=error["raw_arguments"],
                )
            )
    else:
        message["tool_calls"] = []

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


async def rollout_sglang_streaming(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Invoke SGLang server with streaming via OpenAI SDK.

    This is the preferred method for SGLang - uses streaming to get:
    - Real-time token output
    - Proper ToolCallError handling for malformed tool args
    - Consistent behavior with other streaming providers

    Args:
        actor: Current actor state with endpoint and trajectory
        on_chunk: Callback for streaming events
        **kwargs: Additional arguments (ignored, for API consistency)

    Returns:
        Updated actor with new message appended to trajectory
    """
    assert actor is not None
    assert isinstance(actor, Actor)
    assert actor.endpoint is not None
    assert actor.trajectory is not None
    assert actor.endpoint.api_base, "SGLang requires api_base to be set"
    assert on_chunk is not None
    assert callable(on_chunk)

    # Normalize base URL for OpenAI SDK (needs /v1 suffix, not /v1/chat/completions)
    api_base = actor.endpoint.api_base.rstrip("/")
    if api_base.endswith("/v1/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    elif not api_base.endswith("/v1"):
        api_base = api_base + "/v1"

    # Create OpenAI client pointing to SGLang server
    client = AsyncOpenAI(
        api_key="not-needed",  # SGLang doesn't require API key
        base_url=api_base,
        max_retries=actor.endpoint.max_retries,
        timeout=actor.endpoint.timeout,
    )

    # Prepare messages
    llm_messages = _prepare_messages_for_llm(actor.trajectory.messages)
    messages = [_message_to_openai(m) for m in llm_messages]

    # Build params
    params = {
        "model": actor.endpoint.model,
        "messages": messages,
        "temperature": actor.endpoint.temperature,
        "stream": True,
    }

    if actor.endpoint.max_tokens:
        params["max_tokens"] = actor.endpoint.max_tokens

    if actor.tools:
        params["tools"] = [_tool_to_openai(t) for t in actor.tools]
        params["tool_choice"] = "auto"

    if hasattr(actor.endpoint, "extra_params") and actor.endpoint.extra_params:
        params.update(actor.endpoint.extra_params)

    # Execute streaming request - reuse aggregate_stream from openai_completions
    # This handles ToolCallError gracefully when JSON parsing fails
    try:
        stream = await client.chat.completions.create(**params)
        completion = await aggregate_stream(stream, on_chunk)
    except NonRetryableError:
        # Context length, invalid params - re-raise as-is
        raise
    except Exception as e:
        from .base import ProviderError

        logger.exception(f"SGLang streaming request failed: {e}")
        raise ProviderError(
            f"SGLang API error: {e}",
            original_error=e,
            attempts=actor.endpoint.max_retries,
            provider="sglang",
        ) from e

    # Update trajectory with completion
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

    result_actor = replace(actor, trajectory=new_trajectory)
    assert result_actor is not None
    assert result_actor.trajectory is not None
    return result_actor


async def rollout_sglang_token_level(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Token-level SGLang rollout using /generate endpoint.

    Uses TI/TO (Tokens-In/Tokens-Out) to avoid retokenization issues that
    cause RL training collapse. Tokenizes messages, calls /generate with
    input_ids, stores output token_ids directly.

    Args:
        actor: Current actor state with endpoint and trajectory
        on_chunk: Callback for streaming events (not used for token-level)
        **kwargs: Must include:
            - tokenizer: HuggingFace tokenizer for the model
            - suffix_ids (optional): Pre-computed suffix tokens for multi-turn

    Returns:
        Updated actor with new message and token_ids stored in Choice

    Reference: radixark/miles/rollout/sglang_rollout.py
    """
    import time
    from dataclasses import replace

    from ..dtypes import (
        ChatCompletion,
        Choice,
        Logprob,
        Logprobs,
        Message,
        TextContent,
        ToolCallContent,
        Usage,
    )
    from ..inference.backends import (
        compute_suffix_ids,
        generate_sglang,
        log_token_mismatch,
        tokenize_chat,
    )

    assert actor is not None
    assert actor.endpoint is not None
    assert actor.trajectory is not None

    # Get tokenizer from kwargs (required)
    tokenizer = kwargs.get("tokenizer")
    assert tokenizer is not None, "tokenizer is required for token-level rollout"

    # Get or compute suffix_ids for multi-turn
    suffix_ids = kwargs.get("suffix_ids")
    if suffix_ids is None:
        suffix_ids = compute_suffix_ids(tokenizer)

    # Build input_ids from trajectory
    # For multi-turn, we use stored token_ids from previous completions
    input_ids = _build_input_ids_from_trajectory(
        actor.trajectory,
        tokenizer,
        suffix_ids,
    )

    # Normalize base URL (strip /v1/chat/completions if present)
    api_base = actor.endpoint.api_base.rstrip("/")
    if api_base.endswith("/v1/chat/completions"):
        api_base = api_base[: -len("/v1/chat/completions")]
    elif api_base.endswith("/v1"):
        api_base = api_base[: -len("/v1")]

    # Call /generate with token IDs
    result = await generate_sglang(
        base_url=api_base,
        input_ids=input_ids,
        max_tokens=actor.endpoint.max_tokens,
        temperature=actor.endpoint.temperature,
        timeout=actor.endpoint.timeout,
    )

    # Decode generated tokens to text
    generated_text = tokenizer.decode(result.output_ids, skip_special_tokens=True)

    # Parse tool calls from generated text if tools are available
    tool_calls = []
    if actor.tools:
        tool_call_parser = kwargs.get("tool_call_parser", "hermes")
        tools_for_parser = (
            [_tool_to_openai(t) for t in actor.tools] if tool_call_parser != "hermes" else None
        )
        tool_calls = parse_tool_calls(
            generated_text, tools=tools_for_parser, parser=tool_call_parser
        )

    # Build Message from decoded text
    if tool_calls:
        content = [
            TextContent(text=generated_text),
            *[
                ToolCallContent(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["args"],
                )
                for tc in tool_calls
            ],
        ]
    else:
        content = generated_text

    message = Message(role="assistant", content=content)

    # Build Logprobs from result
    logprob_content = []
    for i, (token_id, lp) in enumerate(zip(result.output_ids, result.logprobs, strict=False)):
        token_str = tokenizer.decode([token_id])
        token_bytes = list(token_str.encode("utf-8"))
        top_lps = []
        if result.top_logprobs and i < len(result.top_logprobs):
            top_lps = list(result.top_logprobs[i].values())
        logprob_content.append(
            Logprob(
                token=token_str,
                logprob=lp,
                bytes=token_bytes,
                top_logprobs=top_lps,
            )
        )

    # Build Choice with token_ids
    choice = Choice(
        index=0,
        message=message,
        finish_reason=result.finish_reason,
        logprobs=Logprobs(content=logprob_content),
        token_ids=result.output_ids,  # Store the actual token IDs!
    )

    # Build ChatCompletion
    completion = ChatCompletion(
        id=f"chatcmpl-sglang-tito-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=actor.endpoint.model,
        usage=Usage(
            input_tokens=len(input_ids),
            output_tokens=len(result.output_ids),
        ),
        choices=[choice],
    )

    # Debug: check if our tokens match what chat template would produce
    all_messages = actor.trajectory.messages + [message]
    reference_ids = tokenize_chat(tokenizer, [_msg_to_dict(m) for m in all_messages])
    our_full_ids = input_ids + list(result.output_ids)
    log_token_mismatch(our_full_ids, reference_ids, tokenizer, context="rollout_sglang_token_level")

    # Update trajectory
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [message],
        completions=actor.trajectory.completions + [completion],
    )

    return replace(actor, trajectory=new_trajectory)


async def rollout_vllm_token_level(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    **kwargs: Any,
) -> Actor:
    """Token-level vLLM rollout using /v1/completions with prompt_token_ids.

    Uses TI/TO (Tokens-In/Tokens-Out) to avoid retokenization issues that
    cause RL training collapse. Same pattern as rollout_sglang_token_level
    but hits vLLM's /v1/completions endpoint instead.

    Args:
        actor: Current actor state with endpoint and trajectory
        on_chunk: Callback for streaming events (not used for token-level)
        **kwargs: Must include:
            - tokenizer: HuggingFace tokenizer for the model
            - suffix_ids (optional): Pre-computed suffix tokens for multi-turn

    Returns:
        Updated actor with new message and token_ids stored in Choice
    """
    import time

    from ..dtypes import (
        ChatCompletion,
        Choice,
        Logprob,
        Logprobs,
        Message,
        TextContent,
        ToolCallContent,
        Usage,
    )
    from ..inference.backends import (
        compute_suffix_ids,
        generate_vllm,
        log_token_mismatch,
        tokenize_chat,
    )

    assert actor is not None
    assert actor.endpoint is not None
    assert actor.trajectory is not None

    # Get tokenizer from kwargs (required)
    tokenizer = kwargs.get("tokenizer")
    assert tokenizer is not None, "tokenizer is required for token-level rollout"

    # Get or compute suffix_ids for multi-turn
    suffix_ids = kwargs.get("suffix_ids")
    if suffix_ids is None:
        suffix_ids = compute_suffix_ids(tokenizer)

    # Build input_ids from trajectory (reuse same helper as SGLang)
    input_ids = _build_input_ids_from_trajectory(
        actor.trajectory,
        tokenizer,
        suffix_ids,
    )

    # Normalize base URL
    api_base = actor.endpoint.api_base.rstrip("/")
    if api_base.endswith("/v1/chat/completions"):
        api_base = api_base[: -len("/chat/completions")]
    elif not api_base.endswith("/v1"):
        api_base = api_base + "/v1"
    # Strip trailing /v1 since generate_vllm adds it
    if api_base.endswith("/v1"):
        api_base = api_base[: -len("/v1")]

    # Call /v1/completions with token IDs
    result = await generate_vllm(
        base_url=api_base,
        input_ids=input_ids,
        max_tokens=actor.endpoint.max_tokens,
        temperature=actor.endpoint.temperature,
        timeout=actor.endpoint.timeout,
    )

    # Decode generated tokens to text
    generated_text = tokenizer.decode(result.output_ids, skip_special_tokens=True)

    # Parse tool calls from generated text if tools are available
    tool_calls = []
    if actor.tools:
        tool_call_parser = kwargs.get("tool_call_parser", "hermes")
        tools_for_parser = (
            [_tool_to_openai(t) for t in actor.tools] if tool_call_parser != "hermes" else None
        )
        tool_calls = parse_tool_calls(
            generated_text, tools=tools_for_parser, parser=tool_call_parser
        )

    # Build Message from decoded text
    if tool_calls:
        content = [
            TextContent(text=generated_text),
            *[
                ToolCallContent(
                    id=tc["id"],
                    name=tc["name"],
                    arguments=tc["args"],
                )
                for tc in tool_calls
            ],
        ]
    else:
        content = generated_text

    message = Message(role="assistant", content=content)

    # Build Logprobs from result
    logprob_content = []
    for i, (token_id, lp) in enumerate(zip(result.output_ids, result.logprobs, strict=False)):
        token_str = tokenizer.decode([token_id])
        token_bytes = list(token_str.encode("utf-8"))
        top_lps = []
        if result.top_logprobs and i < len(result.top_logprobs):
            # vLLM returns dict with string keys, convert if needed
            top_lp_dict = result.top_logprobs[i]
            if isinstance(top_lp_dict, dict):
                top_lps = list(top_lp_dict.values())
        logprob_content.append(
            Logprob(
                token=token_str,
                logprob=lp,
                bytes=token_bytes,
                top_logprobs=top_lps,
            )
        )

    # Build Choice with token_ids
    choice = Choice(
        index=0,
        message=message,
        finish_reason=result.finish_reason,
        logprobs=Logprobs(content=logprob_content),
        token_ids=result.output_ids,  # Store the actual token IDs!
    )

    # Build ChatCompletion
    completion = ChatCompletion(
        id=f"chatcmpl-vllm-tito-{int(time.time())}",
        object="chat.completion",
        created=int(time.time()),
        model=actor.endpoint.model,
        usage=Usage(
            input_tokens=len(input_ids),
            output_tokens=len(result.output_ids),
        ),
        choices=[choice],
    )

    # Debug: check if our tokens match what chat template would produce
    all_messages = actor.trajectory.messages + [message]
    reference_ids = tokenize_chat(tokenizer, [_msg_to_dict(m) for m in all_messages])
    our_full_ids = input_ids + list(result.output_ids)
    log_token_mismatch(our_full_ids, reference_ids, tokenizer, context="rollout_vllm_token_level")

    # Update trajectory
    new_trajectory = replace(
        actor.trajectory,
        messages=actor.trajectory.messages + [message],
        completions=actor.trajectory.completions + [completion],
    )

    return replace(actor, trajectory=new_trajectory)


def _build_input_ids_from_trajectory(
    trajectory: Any,
    tokenizer: Any,
    suffix_ids: list[int],
) -> list[int]:
    """Build input_ids from trajectory, using stored token_ids when available.

    For multi-turn, we concatenate:
    - Previous turns' input_ids + output token_ids + suffix_ids
    - Current turn's prompt tokens

    This avoids retokenization of previous assistant responses.
    """
    from ..inference.backends import (
        append_suffix_with_overlap,
        tokenize_chat,
    )

    # Check if we have stored token_ids from previous completions
    if trajectory.completions:
        # Build from stored tokens
        # Find messages that are user/system (prompts) vs assistant (completions)
        # and use stored token_ids for assistant messages when available
        all_ids: list[int] = []

        for i, msg in enumerate(trajectory.messages):
            msg_dict = _msg_to_dict(msg)

            if msg_dict["role"] == "assistant":
                # Check if we have stored token_ids for this completion
                completion_idx = sum(
                    1 for m in trajectory.messages[:i] if _msg_to_dict(m)["role"] == "assistant"
                )
                if completion_idx < len(trajectory.completions):
                    completion = trajectory.completions[completion_idx]
                    if completion.choices and completion.choices[0].token_ids:
                        # Use stored token_ids
                        stored_ids = list(completion.choices[0].token_ids)
                        all_ids = append_suffix_with_overlap(all_ids, suffix_ids)
                        all_ids.extend(stored_ids)
                        all_ids = append_suffix_with_overlap(all_ids, suffix_ids)
                        continue

            # No stored token_ids, tokenize this message
            if i == 0:
                msg_ids = tokenize_chat(tokenizer, [msg_dict])
            else:
                # Use prefix trick for delimiter
                from ..inference.backends import tokenize_message_with_delimiter

                msg_ids = tokenize_message_with_delimiter(tokenizer, msg_dict)

            all_ids.extend(msg_ids)

        return all_ids

    else:
        # First turn - just tokenize all messages
        msg_dicts = [_msg_to_dict(m) for m in trajectory.messages]
        return tokenize_chat(tokenizer, msg_dicts, add_generation_prompt=True)


def _msg_to_dict(msg: Any) -> dict[str, str]:
    """Convert Message to dict for tokenization."""
    from ..dtypes import TextContent, ThinkingContent

    content = msg.content
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        # Extract text from content blocks
        text_parts = []
        for block in content:
            if isinstance(block, TextContent):
                text_parts.append(block.text)
            elif isinstance(block, ThinkingContent):
                text_parts.append(block.thinking)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "thinking":
                    text_parts.append(block.get("thinking", ""))
        text = "".join(text_parts)
    else:
        text = str(content) if content else ""

    return {"role": msg.role, "content": text}


def parse_tool_calls(
    text: str,
    tools: list[dict] | None = None,
    parser: str = "hermes",
) -> list[dict]:
    """Parse tool calls from generated text.

    Args:
        text: Generated text that may contain tool calls
        tools: List of tool definitions (required for "sglang" parser)
        parser: Parser to use:
            - "hermes": Hermes XML format (default, works with Qwen2.5, Llama, etc.)
            - "sglang": SGLang's native FunctionCallParser (supports qwen25, llama, etc.)
            - "qwen25", "llama", etc.: Shorthand for sglang parser with specific model

    Returns:
        List of parsed tool calls: [{"id": str, "name": str, "args": dict}, ...]
    """
    if parser == "hermes":
        return _parse_tool_calls_hermes(text)
    elif parser == "sglang" or parser in ("qwen25", "llama", "mistral"):
        # Use SGLang's native parser
        sglang_parser_type = parser if parser != "sglang" else "qwen25"
        return _parse_tool_calls_sglang(text, tools or [], sglang_parser_type)
    else:
        logger.warning(f"Unknown tool call parser: {parser}, falling back to hermes")
        return _parse_tool_calls_hermes(text)


def _parse_tool_calls_hermes(text: str) -> list[dict]:
    """Parse tool calls using Hermes XML format.

    Hermes models (used by Qwen2.5, Llama, etc.) output tool calls as:
        <tool_call>
        {"name": "function_name", "arguments": {"arg": "value"}}
        </tool_call>

    Multiple tool calls appear as separate XML blocks:
        <tool_call>{...}</tool_call><tool_call>{...}</tool_call>

    Reference: https://github.com/NousResearch/Hermes-Function-Calling
    """
    import re
    import time

    tool_calls = []

    # Match <tool_call>...</tool_call> blocks
    # Use DOTALL to match newlines within the block
    pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
    matches = re.findall(pattern, text, re.DOTALL)

    for i, match in enumerate(matches):
        try:
            # Parse the JSON content
            data = json.loads(match)

            # Hermes format: {"name": str, "arguments": dict}
            name = data.get("name", "")
            arguments = data.get("arguments", {})

            # Generate unique ID for the tool call
            tool_call_id = f"call_{int(time.time())}_{i}"

            tool_calls.append({
                "id": tool_call_id,
                "name": name,
                "args": arguments if isinstance(arguments, dict) else {},
            })
        except json.JSONDecodeError as e:
            # Log but don't crash - model might have malformed output
            logger.warning(f"Failed to parse tool call JSON: {e}. Content: {match[:100]}")
            continue

    return tool_calls


def _parse_tool_calls_sglang(
    text: str,
    tools: list[dict],
    parser_type: str = "qwen25",
) -> list[dict]:
    """Parse tool calls using SGLang's native FunctionCallParser.

    Supports multiple parser types: qwen25, llama, mistral, etc.
    See: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/function_call

    Args:
        text: Generated text that may contain tool calls
        tools: List of tool definitions in OpenAI format
        parser_type: SGLang parser type (qwen25, llama, mistral, etc.)

    Returns:
        List of parsed tool calls: [{"id": str, "name": str, "args": dict}, ...]
    """
    import time

    try:
        from sglang.srt.function_call.function_call_parser import FunctionCallParser
        from sglang.srt.managers.io_struct import Function, Tool
    except ImportError:
        logger.warning("sglang not installed, falling back to hermes parser")
        return _parse_tool_calls_hermes(text)

    if not tools:
        return []

    # Convert tools to SGLang format
    tools_list = []
    for tool in tools:
        if "function" in tool:
            tools_list.append(
                Tool(
                    function=Function(
                        name=tool["function"]["name"],
                        description=tool["function"].get("description", ""),
                        parameters=tool["function"].get("parameters", {}),
                    ),
                    type=tool.get("type", "function"),
                )
            )

    if not tools_list:
        return []

    # Parse using SGLang
    parser = FunctionCallParser(tools=tools_list, tool_call_parser=parser_type)
    normal_text, calls = parser.parse_non_stream(text)

    # Convert to our format
    tool_calls = []
    for i, call in enumerate(calls):
        call_dict = call.model_dump() if hasattr(call, "model_dump") else call
        tool_call_id = f"call_{int(time.time())}_{i}"

        # SGLang returns {"name": str, "parameters": str (JSON)}
        name = call_dict.get("name", "")
        params_str = call_dict.get("parameters", "{}")

        try:
            args = json.loads(params_str) if isinstance(params_str, str) else params_str
        except json.JSONDecodeError:
            args = {}

        tool_calls.append({
            "id": tool_call_id,
            "name": name,
            "args": args if isinstance(args, dict) else {},
        })

    return tool_calls


# Keep old name for backwards compatibility
def _parse_tool_calls_from_text(text: str) -> list[dict]:
    """Deprecated: Use parse_tool_calls() instead."""
    return _parse_tool_calls_hermes(text)
