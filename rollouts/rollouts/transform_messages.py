"""Message transformation for cross-provider compatibility.

Tiger Style implementation inspired by pi-ai's transform-messages.ts.

When switching providers mid-conversation (e.g., Claude → GPT-4o), messages must be
transformed because providers have incompatible formats for thinking/reasoning content
and tool calls.

Two-pass transformation:
1. Convert thinking blocks to provider-appropriate format
2. Filter orphaned tool calls (calls without corresponding results)
"""

from __future__ import annotations

from typing import Any

from .dtypes import (
    ContentBlock,
    ImageContent,
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)


def transform_messages(
    messages: list[Message],
    from_provider: str | None = None,
    from_api: str | None = None,
    to_provider: str | None = None,
    to_api: str | None = None,
    # Alternate parameter names for provider code
    target_provider: str | None = None,
    target_api: str | None = None,
) -> list[Message]:
    """Transform messages from one provider format to another.

    Tiger Style: Pure function, explicit parameters, two-pass algorithm.

    Supports two calling conventions:
    1. Explicit: transform_messages(msgs, from_provider="anthropic", from_api="...", to_provider="openai", to_api="...")
    2. Auto-detect source: transform_messages(msgs, target_provider="openai", target_api="...")

    Args:
        messages: Message history to transform
        from_provider: Source provider (e.g., "anthropic", "openai") - optional, inferred from messages if not provided
        from_api: Source API type (e.g., "anthropic-messages", "openai-completions") - optional
        to_provider: Target provider - can also use target_provider
        to_api: Target API type - can also use target_api
        target_provider: Alternate name for to_provider (used by provider code)
        target_api: Alternate name for to_api (used by provider code)

    Returns:
        Transformed messages compatible with target provider

    Examples:
        # Explicit source/target
        transformed = transform_messages(
            messages=history,
            from_provider="anthropic",
            from_api="anthropic-messages",
            to_provider="openai",
            to_api="openai-completions",
        )

        # Auto-detect source (used by providers)
        transformed = transform_messages(
            messages=history,
            target_provider="openai",
            target_api="openai-completions",
        )
    """
    assert messages is not None
    assert isinstance(messages, list)

    # Handle alternate parameter names
    if target_provider is not None:
        to_provider = target_provider
    if target_api is not None:
        to_api = target_api

    # Infer source provider/api from first message with metadata
    if from_provider is None or from_api is None:
        for msg in messages:
            if hasattr(msg, 'provider') and msg.provider:
                from_provider = msg.provider
                from_api = msg.api or from_api
                break

    # If still not found, use target as fallback (no transformation needed)
    if from_provider is None:
        from_provider = to_provider or "unknown"
    if from_api is None:
        from_api = to_api or "unknown"

    assert to_provider is not None, "Must specify to_provider or target_provider"
    assert to_api is not None, "Must specify to_api or target_api"

    # Pass 1: Convert thinking blocks when switching providers
    transformed_messages = _transform_thinking_blocks(messages, from_provider, from_api, to_provider, to_api)
    assert transformed_messages is not None
    assert isinstance(transformed_messages, list)
    assert len(transformed_messages) == len(messages)

    # Pass 2: Filter orphaned tool calls
    filtered_messages = _filter_orphaned_tool_calls(transformed_messages)
    assert filtered_messages is not None
    assert isinstance(filtered_messages, list)
    assert len(filtered_messages) <= len(transformed_messages)

    return filtered_messages


def _transform_thinking_blocks(
    messages: list[Message],
    from_provider: str,
    from_api: str,
    to_provider: str,
    to_api: str,
) -> list[Message]:
    """Pass 1: Transform thinking blocks between provider formats.

    Tiger Style: Single responsibility - only transforms thinking, doesn't filter.

    When switching providers, thinking blocks must be converted:
    - Anthropic thinking → <thinking> tags in text
    - OpenAI reasoning → <thinking> tags in text
    - Other direction: Keep as text (target provider might not support thinking)

    Args:
        messages: Messages to transform
        from_provider: Source provider
        from_api: Source API type
        to_provider: Target provider
        to_api: Target API type

    Returns:
        Messages with transformed thinking blocks
    """
    assert messages is not None
    assert isinstance(messages, list)

    result = []
    for msg in messages:
        assert msg is not None
        assert isinstance(msg, Message)

        # User and tool result messages pass through unchanged
        if msg.role == "user" or msg.role == "tool":
            result.append(msg)
            continue

        # Assistant messages need transformation check
        if msg.role == "assistant":
            # Check if message is from same provider/API
            msg_provider = msg.provider or from_provider
            msg_api = msg.api or from_api

            if msg_provider == to_provider and msg_api == to_api:
                # Same provider/API - no transformation needed
                result.append(msg)
                continue

            # Different provider - transform thinking blocks
            transformed_content = _transform_content_blocks(msg.content, msg_provider, msg_api, to_provider, to_api)

            # Build new message with transformed content
            from dataclasses import replace
            transformed_msg = replace(
                msg,
                content=transformed_content,
                provider=to_provider,
                api=to_api,
            )
            result.append(transformed_msg)
        else:
            # Unknown role - pass through
            result.append(msg)

    assert len(result) == len(messages)
    return result


def _transform_content_blocks(
    content: str | list[dict[str, Any]] | list[ContentBlock] | None,
    from_provider: str,
    from_api: str,
    to_provider: str,
    to_api: str,
) -> str | list[dict[str, Any]] | list[ContentBlock] | None:
    """Transform content blocks between provider formats.

    Tiger Style: Handle all content types explicitly.

    Args:
        content: Message content (str, legacy dict list, or ContentBlock list)
        from_provider: Source provider
        from_api: Source API type
        to_provider: Target provider
        to_api: Target API type

    Returns:
        Transformed content
    """
    # String content - pass through unchanged
    if isinstance(content, str) or content is None:
        return content

    # Legacy dict format - pass through (for backward compatibility)
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return content

    # ContentBlock list - transform thinking blocks
    if isinstance(content, list):
        transformed_blocks: list[ContentBlock] = []

        for block in content:
            if isinstance(block, ThinkingContent):
                # Convert thinking block to text with <thinking> tags
                # This makes thinking visible to providers that don't support it
                text_block = TextContent(
                    type="text",
                    text=f"<thinking>\n{block.thinking}\n</thinking>",
                )
                transformed_blocks.append(text_block)
            elif isinstance(block, TextContent):
                # Text blocks pass through
                transformed_blocks.append(block)
            elif isinstance(block, ToolCallContent):
                # Tool calls pass through (filtered in pass 2)
                transformed_blocks.append(block)
            elif isinstance(block, ImageContent):
                # Images pass through
                transformed_blocks.append(block)
            else:
                # Unknown block type - pass through
                transformed_blocks.append(block)

        return transformed_blocks

    # Unknown content type - pass through
    return content


def _filter_orphaned_tool_calls(messages: list[Message]) -> list[Message]:
    """Pass 2: Remove tool calls that don't have corresponding tool results.

    Tiger Style: Single responsibility - only filters, doesn't transform.

    When messages are replayed to a new provider, orphaned tool calls (calls without
    results) can confuse the model. Keep only tool calls that have matching results.

    Exception: Keep all tool calls in the final message (may be ongoing turn).

    Args:
        messages: Messages with potentially orphaned tool calls

    Returns:
        Messages with orphaned tool calls removed
    """
    assert messages is not None
    assert isinstance(messages, list)

    result = []

    for index, msg in enumerate(messages):
        assert msg is not None
        assert isinstance(msg, Message)

        # Non-assistant messages pass through
        if msg.role != "assistant":
            result.append(msg)
            continue

        # Last message - keep all tool calls (may be ongoing)
        is_last_message = (index == len(messages) - 1)
        if is_last_message:
            result.append(msg)
            continue

        # Extract tool call IDs from this message
        tool_call_ids = _extract_tool_call_ids(msg.content)

        # No tool calls - pass through
        if len(tool_call_ids) == 0:
            result.append(msg)
            continue

        # Scan forward to find matching tool results
        matched_tool_call_ids = _find_matching_tool_results(messages, index)

        # Filter content blocks to keep only matched tool calls
        filtered_content = _filter_content_by_tool_calls(msg.content, matched_tool_call_ids)

        # Build new message with filtered content
        from dataclasses import replace
        filtered_msg = replace(msg, content=filtered_content)
        result.append(filtered_msg)

    return result


def _extract_tool_call_ids(content: str | list[dict[str, Any]] | list[ContentBlock] | None) -> list[str]:
    """Extract tool call IDs from message content.

    Tiger Style: Pure function, explicit return type.

    Args:
        content: Message content

    Returns:
        List of tool call IDs found in content
    """
    if content is None or isinstance(content, str):
        return []

    # Legacy dict format
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return []

    # ContentBlock list
    if isinstance(content, list):
        tool_call_ids = []
        for block in content:
            if isinstance(block, ToolCallContent):
                tool_call_ids.append(block.id)
        return tool_call_ids

    return []


def _find_matching_tool_results(messages: list[Message], current_index: int) -> set[str]:
    """Find tool result IDs that match tool calls from current message.

    Tiger Style: Pure function, explicit algorithm.

    Scans forward from current message until hitting another assistant message.
    Collects all tool result IDs found.

    Args:
        messages: All messages
        current_index: Index of current assistant message

    Returns:
        Set of tool call IDs that have corresponding results
    """
    assert messages is not None
    assert isinstance(messages, list)
    assert current_index >= 0
    assert current_index < len(messages)

    matched_ids: set[str] = set()

    # Scan forward from next message
    for i in range(current_index + 1, len(messages)):
        next_msg = messages[i]
        assert next_msg is not None

        # Stop at next assistant message
        if next_msg.role == "assistant":
            break

        # Check for tool results
        if next_msg.role == "tool":
            tool_call_id = next_msg.tool_call_id
            if tool_call_id:
                matched_ids.add(tool_call_id)

    return matched_ids


def _filter_content_by_tool_calls(
    content: str | list[dict[str, Any]] | list[ContentBlock] | None,
    matched_tool_call_ids: set[str],
) -> str | list[dict[str, Any]] | list[ContentBlock] | None:
    """Filter content blocks to keep only matched tool calls.

    Tiger Style: Pure function, preserves non-tool-call blocks.

    Args:
        content: Message content
        matched_tool_call_ids: Set of tool call IDs to keep

    Returns:
        Filtered content (removes orphaned tool calls)
    """
    # String content - pass through
    if isinstance(content, str) or content is None:
        return content

    # Legacy dict format - pass through
    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
        return content

    # ContentBlock list - filter tool calls
    if isinstance(content, list):
        filtered_blocks: list[ContentBlock] = []

        for block in content:
            if isinstance(block, ToolCallContent):
                # Keep only if matched
                if block.id in matched_tool_call_ids:
                    filtered_blocks.append(block)
            else:
                # Keep all non-tool-call blocks
                filtered_blocks.append(block)

        return filtered_blocks

    # Unknown type - pass through
    return content
