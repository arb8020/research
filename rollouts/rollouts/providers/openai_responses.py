"""OpenAI Responses API provider implementation (o1/o3 reasoning models)."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

import trio
from openai import AsyncOpenAI

from rollouts.dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    ImageContent,
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
)

from .base import _prepare_messages_for_llm, calculate_cost_from_usage


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
                # Push the full part object which includes 'type' field required for re-submission
                # The part has structure like {"type": "summary_text", "text": "..."}
                part_dict = {"type": getattr(event.part, "type", "summary_text"), "text": ""}
                current_item.setdefault("summary", []).append(part_dict)
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
                # Finalize text block - save message ID for re-submission
                text_content = content_blocks[current_block_index]["text"]
                content_blocks[current_block_index]["textSignature"] = getattr(item, "id", None)
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
                # Extract cache tokens from input_tokens_details
                cached_tokens = 0
                if hasattr(response.usage, "input_tokens_details"):
                    details = response.usage.input_tokens_details
                    if hasattr(details, "cached_tokens"):
                        cached_tokens = details.cached_tokens or 0

                # Extract reasoning tokens from output_tokens_details
                reasoning_tokens = 0
                if hasattr(response.usage, "output_tokens_details"):
                    details = response.usage.output_tokens_details
                    if hasattr(details, "reasoning_tokens"):
                        reasoning_tokens = details.reasoning_tokens or 0

                output_tokens = getattr(response.usage, "output_tokens", 0) or 0
                usage_data = {
                    "input_tokens": (getattr(response.usage, "input_tokens", 0) or 0) - cached_tokens,
                    "output_tokens": output_tokens - reasoning_tokens,
                    "reasoning_tokens": reasoning_tokens,
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

    final_content_blocks: list = []

    for block in content_blocks:
        if block["type"] == "text":
            final_content_blocks.append(TextContent(
                text=block["text"],
                text_signature=block.get("textSignature"),  # Message ID for re-submission
            ))
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

    result: list[dict[str, Any]] = []

    for msg in messages:
        if msg.role == "user":
            # Tiger Style: Explicit control flow - handle each content type
            # Handle string content (simple text messages)
            if isinstance(msg.content, str):
                user_text = msg.content
            # Handle ContentBlock list (structured messages)
            elif isinstance(msg.content, list):
                text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
                user_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            else:
                assert False, f"User message content must be str or list[ContentBlock], got {type(msg.content)}"

            result.append({
                "role": "user",
                "content": [{"type": "input_text", "text": user_text}]
            })

        elif msg.role == "assistant":
            # Assistant messages become separate objects
            output: list[dict[str, Any]] = []

            # Tiger Style: Explicit control flow - handle each content type
            # Handle string content (simple text response)
            if isinstance(msg.content, str):
                output.append({
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": msg.content, "annotations": []}],
                    "status": "completed",
                    "id": f"msg_{int(time.time())}_{id(msg)}",
                })
            # Handle ContentBlock list (structured response with thinking/tools)
            elif isinstance(msg.content, list):
                # Process ContentBlocks
                for block in msg.content:
                    if isinstance(block, ThinkingContent) and block.thinking_signature:
                        # Reuse existing reasoning item only if it has encrypted_content
                        # Without encrypted_content, the API rejects the reasoning item
                        # See: https://community.openai.com/t/need-reasoning-false-option-for-gpt-5/1351588/7
                        try:
                            reasoning_item = json.loads(block.thinking_signature)
                            if reasoning_item.get("encrypted_content"):
                                output.append(reasoning_item)
                                logger.debug(f"Added reasoning item with encrypted_content: id={reasoning_item.get('id', 'unknown')}")
                            else:
                                logger.debug(f"Skipping reasoning item without encrypted_content: id={reasoning_item.get('id', 'unknown')}")
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse thinking_signature: {e}")
                            pass
                    elif isinstance(block, TextContent):
                        # Add text content as message object
                        # Use actual message ID from API if available (for proper re-submission)
                        msg_id = block.text_signature or f"msg_{int(time.time())}_{id(msg)}"
                        output.append({
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": block.text, "annotations": []}],
                            "status": "completed",
                            "id": msg_id,
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
            # Tiger Style: Explicit control flow - handle both string and ContentBlock content
            if isinstance(msg.content, str):
                tool_result_text = msg.content
            elif isinstance(msg.content, list):
                text_blocks = [b for b in msg.content if isinstance(b, TextContent)]
                tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            else:
                tool_result_text = ""

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

    client_kwargs = {
        "api_key": actor.endpoint.api_key,
        "max_retries": actor.endpoint.max_retries,
        "timeout": actor.endpoint.timeout,
    }
    # Only set base_url if it's provided (non-empty)
    if actor.endpoint.api_base:
        client_kwargs["base_url"] = actor.endpoint.api_base
    client = AsyncOpenAI(**client_kwargs)

    # Transform messages for cross-provider compatibility (like pi-ai does)
    from rollouts.transform_messages import transform_messages
    transformed_messages = transform_messages(
        actor.trajectory.messages,
        target_provider=actor.endpoint.provider,
        target_api="openai-responses"
    )

    # Strip details before sending to LLM
    llm_messages = _prepare_messages_for_llm(transformed_messages)

    # Convert messages to OpenAI Responses format
    # Note: The Responses API uses a completely different message format than chat completions
    messages = _messages_to_openai_responses(llm_messages)

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

    # GPT-5 models always produce reasoning, so we must always request encrypted_content
    # to be able to re-submit the reasoning items in subsequent turns
    # See: https://community.openai.com/t/need-reasoning-false-option-for-gpt-5/1351588/7
    if is_reasoning_model and model_name.startswith("gpt-5"):
        if actor.endpoint.reasoning_effort is not None:
            params["reasoning"] = {
                "effort": actor.endpoint.reasoning_effort,
                "summary": "auto",
            }
        else:
            # Default to low effort to minimize cost when not explicitly requested
            params["reasoning"] = {
                "effort": "low",
                "summary": "auto",
            }
        params["include"] = ["reasoning.encrypted_content"]
    elif actor.endpoint.reasoning_effort is not None and is_reasoning_model:
        # Non-GPT-5 reasoning models with explicit reasoning_effort
        params["reasoning"] = {
            "effort": actor.endpoint.reasoning_effort,
            "summary": "auto",
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

    # Build completion object with granular token breakdown
    usage = Usage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        reasoning_tokens=usage_data.get("reasoning_tokens", 0),
        cache_read_tokens=usage_data.get("cache_read_tokens", 0),
        cache_write_tokens=usage_data.get("cache_write_tokens", 0),
    )

    # Enrich message with provider/api/model metadata for cross-provider handoff
    final_message = replace(
        final_message,
        provider=actor.endpoint.provider,
        api="openai-responses",
        model=actor.endpoint.model,
    )

    # Calculate cost if model pricing is available
    from rollouts.models import get_model

    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model)
    if model_meta and model_meta.cost:
        cost = calculate_cost_from_usage(usage, model_meta.cost)
        usage = replace(usage, cost=cost)

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
