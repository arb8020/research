#!/usr/bin/env python3
"""Test for thinking block handling on session resume.

Reproduces the bug: https://github.com/anthropics/claude-code/issues/12316

## Background

The Anthropic API requires that thinking blocks in ALL assistant messages must:
1. Be passed back unmodified (including the signature)
2. The signature must match exactly what was originally returned

When resuming a session, if we modify or corrupt thinking blocks in any way,
the API returns:
    400 - "thinking or redacted_thinking blocks in the latest assistant message
           cannot be modified. These blocks must remain as they were in the
           original response."

## Root Cause Analysis (Dec 2025)

Investigation of a crash with error `messages.5.content.3: thinking blocks cannot
be modified` revealed several potential issues:

1. **Consecutive Same-Role Messages**: Our internal message history can have
   consecutive assistant messages (e.g., when tool calls complete quickly or
   streaming is interrupted). The Anthropic API may merge these, causing
   thinking blocks to be seen as "modified".

2. **Missing Synthetic Tool Results**: When a user message interrupts a tool
   flow (assistant with tool_use but no tool_result), we need to insert
   synthetic tool results to maintain proper message sequencing.

3. **Cross-Provider Thinking Block Handling**: When switching providers, thinking
   blocks should be converted to text blocks with <thinking> tags (see pi-mono's
   transform-messages.ts for reference implementation).

## Reference Implementation: pi-mono

See /tmp/pi-mono/packages/ai/src/providers/transorm-messages.ts:
- Lines 88-104: Insert synthetic tool results for orphaned tool calls
- Lines 118-135: Handle user messages interrupting tool flows
- Lines 49-54: Convert thinking to text when switching providers

## This Test Suite

This test verifies that:
1. Thinking blocks are preserved correctly through serialize/deserialize
2. The _message_to_anthropic function produces valid output
3. Session resume doesn't corrupt thinking signatures
"""

import os

import pytest

from rollouts.dtypes import (
    Message,
    TextContent,
    ThinkingContent,
    Trajectory,
)
from rollouts.providers.anthropic import _message_to_anthropic


class TestThinkingBlockPreservation:
    """Test that thinking blocks are preserved correctly."""

    def test_thinking_signature_preserved_in_serialize_deserialize(self) -> None:
        """Verify thinking_signature survives JSON round-trip."""
        original_msg = Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="Let me think about this...",
                    thinking_signature="EoAECkYIChgCKkDXqncF97Q+p944x4EUEi26E6j1H2iwM5P+yqXe67sXLPygERV5zhgIi09qnHHdZr1xJSzkEatN1fwe8BOn1nRB",
                ),
                TextContent(type="text", text="The answer is 42."),
            ],
            provider="anthropic",
            api="anthropic-messages",
        )

        # Serialize and deserialize
        json_str = original_msg.to_json()
        restored_msg = Message.from_json(json_str)

        # Verify thinking signature is preserved
        assert isinstance(restored_msg.content, list)
        thinking_block = restored_msg.content[0]
        assert isinstance(thinking_block, ThinkingContent)
        assert (
            thinking_block.thinking_signature
            == "EoAECkYIChgCKkDXqncF97Q+p944x4EUEi26E6j1H2iwM5P+yqXe67sXLPygERV5zhgIi09qnHHdZr1xJSzkEatN1fwe8BOn1nRB"
        )
        assert thinking_block.thinking == "Let me think about this..."

    def test_message_to_anthropic_includes_signature(self) -> None:
        """Verify _message_to_anthropic includes the signature field."""
        msg = Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="Reasoning here...",
                    thinking_signature="test_signature_abc123",
                ),
                TextContent(type="text", text="Response text."),
            ],
            provider="anthropic",
            api="anthropic-messages",
        )

        anthropic_msg = _message_to_anthropic(msg)

        assert anthropic_msg["role"] == "assistant"
        assert isinstance(anthropic_msg["content"], list)

        # Find the thinking block
        thinking_blocks = [b for b in anthropic_msg["content"] if b.get("type") == "thinking"]
        assert len(thinking_blocks) == 1

        thinking_block = thinking_blocks[0]
        assert thinking_block["type"] == "thinking"
        assert thinking_block["thinking"] == "Reasoning here..."
        assert thinking_block["signature"] == "test_signature_abc123"

    def test_message_to_anthropic_converts_missing_signature_to_text(self) -> None:
        """Verify thinking blocks without signature become text blocks.

        This is important for aborted streams where the signature wasn't received.
        Anthropic rejects thinking blocks without valid signatures.
        """
        msg = Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="Incomplete thinking...",
                    thinking_signature=None,  # Missing signature
                ),
                TextContent(type="text", text="Response."),
            ],
            provider="anthropic",
            api="anthropic-messages",
        )

        anthropic_msg = _message_to_anthropic(msg)

        # Should have converted thinking to text block
        content = anthropic_msg["content"]
        thinking_blocks = [b for b in content if b.get("type") == "thinking"]
        text_blocks = [b for b in content if b.get("type") == "text"]

        # No raw thinking blocks (they become text)
        assert len(thinking_blocks) == 0
        # Two text blocks: converted thinking + original text
        assert len(text_blocks) == 2
        # First text block should contain <thinking> tags
        assert "<thinking>" in text_blocks[0]["text"]
        assert "Incomplete thinking..." in text_blocks[0]["text"]


class TestSessionResumeWithThinking:
    """Test session resume scenarios with thinking blocks."""

    def test_resume_session_preserves_all_thinking_blocks(self) -> None:
        """Verify all thinking blocks in a session history are preserved.

        The API checks ALL assistant messages, not just the last one.
        """
        # Simulate a session with multiple assistant messages with thinking
        messages = [
            Message(role="user", content="First question"),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(
                        type="thinking",
                        thinking="First reasoning...",
                        thinking_signature="sig1_abc123",
                    ),
                    TextContent(type="text", text="First answer."),
                ],
                provider="anthropic",
                api="anthropic-messages",
            ),
            Message(role="user", content="Follow-up question"),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(
                        type="thinking",
                        thinking="Second reasoning...",
                        thinking_signature="sig2_xyz789",
                    ),
                    TextContent(type="text", text="Second answer."),
                ],
                provider="anthropic",
                api="anthropic-messages",
            ),
            Message(role="user", content="Continue..."),
        ]

        # Convert all to Anthropic format (simulating what rollout_anthropic does)
        anthropic_messages = []
        for msg in messages:
            if msg.role == "user":
                anthropic_messages.append({"role": "user", "content": msg.content})
            elif msg.role == "assistant":
                anthropic_messages.append(_message_to_anthropic(msg))

        # Verify both assistant messages have correct thinking blocks with signatures
        assistant_msgs = [m for m in anthropic_messages if m["role"] == "assistant"]
        assert len(assistant_msgs) == 2

        for i, amsg in enumerate(assistant_msgs):
            content = amsg["content"]
            thinking_blocks = [b for b in content if b.get("type") == "thinking"]
            assert len(thinking_blocks) == 1, f"Assistant message {i} should have 1 thinking block"
            assert "signature" in thinking_blocks[0], (
                f"Assistant message {i} thinking block should have signature"
            )
            assert thinking_blocks[0]["signature"] is not None

    def test_consecutive_assistant_messages_both_preserved(self) -> None:
        """Test consecutive assistant messages preserve thinking blocks.

        This can happen when tool calls complete quickly or streaming is interrupted.
        """
        # This can happen when tool calls complete quickly or when streaming is interrupted
        messages = [
            Message(role="user", content="Question"),
            Message(
                role="assistant",
                content=[
                    ThinkingContent(
                        type="thinking",
                        thinking="First thought...",
                        thinking_signature="sig_first",
                    ),
                    TextContent(type="text", text="Starting to answer..."),
                ],
                provider="anthropic",
                api="anthropic-messages",
            ),
            # Consecutive assistant message (e.g., tool result handling)
            Message(
                role="assistant",
                content=[
                    ThinkingContent(
                        type="thinking",
                        thinking="Continuing...",
                        thinking_signature="sig_second",
                    ),
                    TextContent(type="text", text="Full answer."),
                ],
                provider="anthropic",
                api="anthropic-messages",
            ),
        ]

        # Both should convert correctly
        anthropic_msgs = [_message_to_anthropic(m) for m in messages if m.role == "assistant"]

        assert len(anthropic_msgs) == 2
        for _i, amsg in enumerate(anthropic_msgs):
            thinking_blocks = [b for b in amsg["content"] if b.get("type") == "thinking"]
            assert len(thinking_blocks) == 1
            assert thinking_blocks[0]["signature"] is not None


@pytest.mark.trio
async def test_real_api_thinking_block_round_trip() -> None:
    """Integration test: verify thinking blocks work with real API.

    This test:
    1. Makes a real API call with thinking enabled
    2. Extracts the thinking block with signature
    3. Simulates sending it back in a follow-up (what happens on resume)

    Skip if no API key.
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

    initial_msg = Message(role="user", content="What is 2+2?")
    actor = Actor(
        trajectory=Trajectory(messages=[initial_msg]),
        endpoint=endpoint,
        tools=[],
    )

    # Get response with thinking
    result = await rollout(actor, on_chunk=stdout_handler)

    assistant_msg = result.trajectory.messages[-1]
    assert assistant_msg.role == "assistant"
    assert isinstance(assistant_msg.content, list)

    # Find thinking block
    thinking_blocks = [b for b in assistant_msg.content if isinstance(b, ThinkingContent)]
    assert len(thinking_blocks) >= 1, "Response should include thinking block"

    thinking_block = thinking_blocks[0]
    assert thinking_block.thinking_signature is not None, "Thinking block should have signature"
    assert len(thinking_block.thinking_signature) > 0

    print(
        f"\n✓ Received thinking block with signature length: {len(thinking_block.thinking_signature)}"
    )

    # Verify the block can be converted back to Anthropic format
    anthropic_msg = _message_to_anthropic(assistant_msg)
    converted_thinking = [b for b in anthropic_msg["content"] if b.get("type") == "thinking"]
    assert len(converted_thinking) >= 1
    assert converted_thinking[0]["signature"] == thinking_block.thinking_signature

    print("✓ Thinking block correctly converted for API")
