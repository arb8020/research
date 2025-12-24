#!/usr/bin/env python3
"""Regression test for thinking block handling on session resume.

Reproduces: https://github.com/anthropics/claude-code/issues/12316

The bug: When consecutive assistant messages exist (from user interrupts during
tool execution), the Anthropic API silently merges them. This causes thinking
blocks from different responses to be combined, triggering:
    400 - "thinking blocks in the latest assistant message cannot be modified"

The fix: Insert synthetic "[interrupted]" tool results between consecutive
assistant messages to prevent the merge.
"""

import os

import pytest

from rollouts.dtypes import (
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
    Trajectory,
)
from rollouts.transform_messages import transform_messages


def build_interrupted_session_messages(
    thinking_signature: str,
) -> list[Message]:
    """Build a message sequence that simulates user interrupt during tool execution.

    This pattern occurs when:
    1. Assistant responds with thinking + tool_use
    2. User presses Escape (interrupt)
    3. New LLM call happens with thinking + tool_use
    4. Original tool call never got a result

    Result: consecutive assistant messages, which Anthropic merges and rejects.
    """
    return [
        Message(role="user", content="Run a command for me"),
        # First assistant response - has tool call
        Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="Let me run this command.",
                    thinking_signature=thinking_signature,
                ),
                TextContent(type="text", text="Running the command..."),
                ToolCallContent(
                    type="toolCall",
                    id="tool_interrupted_1",
                    name="bash",
                    arguments={"command": "sleep 60"},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
        ),
        # CONSECUTIVE assistant (user interrupted, new LLM call)
        # This is the problematic pattern - no tool_result between assistants
        Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="User wants something else now.",
                    thinking_signature=thinking_signature,
                ),
                ToolCallContent(
                    type="toolCall",
                    id="tool_interrupted_2",
                    name="bash",
                    arguments={"command": "echo hello"},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
        ),
        # Another consecutive assistant
        Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="Running the new command.",
                    thinking_signature=thinking_signature,
                ),
                ToolCallContent(
                    type="toolCall",
                    id="tool_with_result",
                    name="bash",
                    arguments={"command": "echo world"},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
        ),
        # Only the last tool call got a result
        Message(
            role="tool",
            content="world",
            tool_call_id="tool_with_result",
        ),
        # User continues
        Message(role="user", content="Thanks, what's next?"),
    ]


class TestSyntheticToolResultInsertion:
    """Test that synthetic tool results are inserted for orphaned tool calls."""

    def test_consecutive_assistants_get_synthetic_results(self) -> None:
        """Verify transform_messages inserts synthetic results between consecutive assistants."""
        messages = build_interrupted_session_messages("fake_sig_for_unit_test")

        # Before fix: 6 messages with consecutive assistants
        # [user, assistant, assistant, assistant, tool, user]
        assert len(messages) == 6

        # Count consecutive assistant pairs before transform
        consecutive_before = 0
        for i in range(1, len(messages)):
            if messages[i].role == "assistant" and messages[i - 1].role == "assistant":
                consecutive_before += 1
        assert consecutive_before == 2, "Should have 2 consecutive assistant pairs"

        # Transform
        transformed = transform_messages(
            messages, target_provider="anthropic", target_api="anthropic-messages"
        )

        # After fix: synthetic tool results inserted
        assert len(transformed) > len(messages), "Should have more messages after inserting synthetic results"

        # Count consecutive assistant pairs after transform
        consecutive_after = 0
        for i in range(1, len(transformed)):
            if transformed[i].role == "assistant" and transformed[i - 1].role == "assistant":
                consecutive_after += 1
        assert consecutive_after == 0, "Should have NO consecutive assistant pairs after transform"

        # Verify synthetic results were inserted
        synthetic_count = sum(
            1
            for m in transformed
            if m.role == "tool" and "[interrupted" in str(m.content)
        )
        assert synthetic_count == 2, "Should have 2 synthetic tool results"


@pytest.mark.trio
async def test_resume_with_consecutive_assistants() -> None:
    """Integration test: Resume a session that has consecutive assistant messages.

    This reproduces the actual crash scenario:
    1. Get a real thinking signature from the API
    2. Build a message history with consecutive assistants (simulating interrupt)
    3. Attempt to continue the conversation
    4. Pre-fix: Fails with "thinking blocks cannot be modified"
    5. Post-fix: Succeeds because synthetic tool results prevent consecutive merge
    """
    from rollouts import Actor, Endpoint, rollout, stdout_handler

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    endpoint = Endpoint(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key=api_key,
        temperature=1.0,
        thinking={"type": "enabled", "budget_tokens": 1024},
    )

    # Step 1: Get a real thinking signature from the API
    print("\n1. Getting real thinking signature from API...")
    initial_actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content="Say hi")]),
        endpoint=endpoint,
        tools=[],
    )
    result1 = await rollout(initial_actor, on_chunk=stdout_handler)

    # Extract the thinking signature
    assistant_msg = result1.trajectory.messages[-1]
    assert assistant_msg.role == "assistant"
    assert isinstance(assistant_msg.content, list)

    thinking_blocks = [
        b for b in assistant_msg.content if isinstance(b, ThinkingContent)
    ]
    assert len(thinking_blocks) >= 1, "Expected thinking block"
    real_signature = thinking_blocks[0].thinking_signature
    assert real_signature, "Expected valid signature"

    print(f"   Got signature: {real_signature[:50]}...")

    # Step 2: Build problematic message sequence (consecutive assistants)
    print("\n2. Building message history with consecutive assistants...")
    problematic_messages = build_interrupted_session_messages(real_signature)

    # Step 3: Try to continue this conversation
    print("\n3. Attempting to resume (should succeed with synthetic results)...")

    actor2 = Actor(
        trajectory=Trajectory(messages=problematic_messages),
        endpoint=endpoint,
        tools=[],
    )

    # This would fail without the fix:
    # anthropic.BadRequestError: 400 - thinking blocks cannot be modified
    result2 = await rollout(actor2, on_chunk=stdout_handler)

    assert result2.trajectory.messages[-1].role == "assistant"
    print("\n✓ Successfully resumed session with consecutive assistants!")
