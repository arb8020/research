"""Google Generative AI (Gemini) provider implementation."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from typing import Any

from rollouts.dtypes import (
    Actor,
    ChatCompletion,
    Choice,
    Message,
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
    Tool,
    ToolCall,
    ToolCallContent,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    Usage,
)

from .base import _prepare_messages_for_llm, calculate_cost_from_usage


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
            cached_tokens = getattr(metadata, "cachedContentTokenCount", 0) or 0
            prompt_tokens = getattr(metadata, "promptTokenCount", 0) or 0
            usage_data = {
                "input_tokens": prompt_tokens - cached_tokens,  # Non-cached input
                "output_tokens": getattr(metadata, "candidatesTokenCount", 0) or 0,
                "reasoning_tokens": getattr(metadata, "thoughtsTokenCount", 0) or 0,
                "cache_read_tokens": cached_tokens,
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

    contents = []
    for m in transformed_messages:
        if m.role == "user":
            # Tiger Style: Explicit control flow - handle both string and ContentBlock content
            if isinstance(m.content, str):
                user_text = m.content
            elif isinstance(m.content, list):
                text_blocks = [b for b in m.content if isinstance(b, TextContent)]
                user_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            else:
                user_text = ""
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=user_text)]
            ))
        elif m.role == "assistant":
            parts = []
            # Tiger Style: Explicit control flow - handle both string and ContentBlock content
            if isinstance(m.content, str):
                parts.append(types.Part(text=m.content))
            elif isinstance(m.content, list):
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
            # Tiger Style: Explicit control flow - handle both string and ContentBlock content
            if isinstance(m.content, str):
                tool_result_text = m.content
            elif isinstance(m.content, list):
                text_blocks = [b for b in m.content if isinstance(b, TextContent)]
                tool_result_text = "\n".join(b.text for b in text_blocks) if text_blocks else ""
            else:
                tool_result_text = ""
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

    # Build completion object with granular token breakdown
    usage = Usage(
        input_tokens=usage_data.get("input_tokens", 0),
        output_tokens=usage_data.get("output_tokens", 0),
        reasoning_tokens=usage_data.get("reasoning_tokens", 0),
        cache_read_tokens=usage_data.get("cache_read_tokens", 0),
        cache_write_tokens=usage_data.get("cache_write_tokens", 0),
    )

    # Calculate cost if model pricing is available
    from rollouts.models import get_model

    model_meta = get_model(actor.endpoint.provider, actor.endpoint.model)
    if model_meta and model_meta.cost:
        cost = calculate_cost_from_usage(usage, model_meta.cost)
        usage = replace(usage, cost=cost)

    # Enrich message with provider/api/model metadata for cross-provider handoff
    final_message = replace(
        final_message,
        provider=actor.endpoint.provider,
        api="google-generative-ai",
        model=actor.endpoint.model,
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
