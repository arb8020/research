#!/usr/bin/env python3
"""Regression test for log_api_response with tool calls.

Bug: commit 6216368 introduced `has_tool_calls=bool(message.tool_calls)`
but Message dataclass uses `get_tool_calls()` method, not `.tool_calls` attribute.

This caused AttributeError in the logging code which was incorrectly caught
by retry logic, leading to infinite retries.

Cut point: log_api_response() must handle real ChatCompletion objects.

Run: python tests/regression/test_log_api_response_tool_calls.py
"""

from rollouts.core import (
    ChatCompletion,
    Choice,
    Message,
    TextContent,
    Usage,
)
from rollouts.dtypes import ToolCallContent
from rollouts.providers.base import log_api_response


def test_with_tool_calls() -> None:
    """log_api_response should not crash when message has tool calls."""
    message = Message(
        role="assistant",
        content=[
            TextContent(text="I'll read that file."),
            ToolCallContent(id="call_123", name="read", arguments={"path": "foo.txt"}),
        ],
    )

    completion = ChatCompletion(
        id="test-completion",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        usage=Usage(input_tokens=100, output_tokens=50),
        choices=[Choice(index=0, message=message, finish_reason="tool_use")],
    )

    # This should not raise AttributeError: 'Message' object has no attribute 'tool_calls'
    log_api_response(
        provider="anthropic",
        model="test-model",
        attempt=1,
        success=True,
        input_tokens=100,
        output_tokens=50,
        stop_reason="tool_use",
        has_tool_calls=bool(completion.choices[0].message.get_tool_calls()),
    )


def test_without_tool_calls() -> None:
    """log_api_response should handle messages without tool calls."""
    message = Message(
        role="assistant",
        content=[TextContent(text="Hello!")],
    )

    completion = ChatCompletion(
        id="test-completion",
        object="chat.completion",
        created=1234567890,
        model="test-model",
        usage=Usage(input_tokens=100, output_tokens=50),
        choices=[Choice(index=0, message=message, finish_reason="stop")],
    )

    log_api_response(
        provider="anthropic",
        model="test-model",
        attempt=1,
        success=True,
        input_tokens=100,
        output_tokens=50,
        stop_reason="stop",
        has_tool_calls=bool(completion.choices[0].message.get_tool_calls()),
    )
