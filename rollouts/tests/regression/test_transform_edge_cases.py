#!/usr/bin/env python3
"""Regression tests for edge cases in message transformation.

These tests verify specific edge cases and corner cases that might break
when refactoring transformation logic.

Separated from main integration tests to keep core tests focused.
"""

import pytest
import trio

from rollouts import (
    Message,
    ToolCallContent,
    transform_messages,
)


@pytest.mark.trio
async def test_last_message_tool_calls_preserved():
    """Test that tool calls in the last message are preserved.

    The last message might be mid-turn (tool calls pending execution).
    These should NOT be filtered.

    Edge case: When switching providers mid-turn, we want to preserve
    pending tool calls so the new provider can execute them.
    """
    print("\n=== Testing Last Message Tool Calls ===\n")

    messages = [
        Message(role="user", content="Calculate stuff"),
        Message(
            role="assistant",
            content=[
                ToolCallContent(id="call_pending", name="add", arguments={"a": 1, "b": 2}),
            ],
            provider="openai",
            api="openai-completions",
        ),
        # No tool result yet - this is the last message
    ]

    print("1. Transforming messages with pending tool call...")
    transformed = transform_messages(
        messages=messages,
        from_provider="openai",
        from_api="openai-completions",
        to_provider="anthropic",
        to_api="anthropic-messages",
    )

    # Check that pending tool call was kept
    assistant_msg = transformed[1]
    assert assistant_msg.role == "assistant"

    if isinstance(assistant_msg.content, list):
        tool_calls = [b for b in assistant_msg.content if isinstance(b, ToolCallContent)]
        assert len(tool_calls) == 1, "Should keep pending tool call in last message"
        assert tool_calls[0].id == "call_pending"
        print("  ✓ Pending tool call preserved")

    print("\n=== Last Message Test PASSED ===\n")


@pytest.mark.trio
async def test_regression_transform_edge_cases():
    """Run all regression tests for transformation edge cases."""
    await test_last_message_tool_calls_preserved()


if __name__ == "__main__":
    async def main():
        print("\n" + "="*70)
        print("REGRESSION TEST: Message Transformation Edge Cases")
        print("="*70)
        try:
            await test_last_message_tool_calls_preserved()
        except Exception as e:
            print(f"❌ Regression test failed: {e}")
            import traceback
            traceback.print_exc()
        print("\n" + "="*70 + "\n")

    trio.run(main)

