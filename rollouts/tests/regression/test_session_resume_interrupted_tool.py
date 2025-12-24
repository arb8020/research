#!/usr/bin/env python3
"""Regression test for session resume with interrupted tool calls.

Reproduces: Session resume API error - tool_result in wrong position

The bug: When resuming a session that was interrupted during tool execution,
the API rejects with:
    messages.0.content.1: unexpected `tool_use_id` found in `tool_result` blocks

This indicates the tool_result is ending up in the first message's content array,
meaning consecutive user messages are being merged incorrectly.

Scenario:
1. Start session, ask to run `sleep 60`
2. Model responds with tool_use (bash command)
3. User presses Escape to interrupt during execution
4. Session saved with: [system, user, assistant(tool_use), tool("[interrupted]")]
5. Resume session, type new message
6. ERROR: messages.0.content.1 has tool_result without matching tool_use
"""

import json

from rollouts.dtypes import (
    Message,
    TextContent,
    ThinkingContent,
    ToolCallContent,
)
from rollouts.transform_messages import transform_messages
from rollouts.providers.anthropic import (
    _message_to_anthropic,
    _merge_consecutive_api_messages,
)


def test_session_resume_message_order():
    """Test that messages are in correct order after transform and merge.

    Session file structure (from real repro):
    0: system
    1: user ("sleep 60")
    2: assistant (with tool_use)
    3: tool ("[interrupted]")

    After resume, user adds new message:
    4: user ("what happened?")

    Expected API message order:
    0: user ("sleep 60")
    1: assistant (tool_use)
    2: user (tool_result + "what happened?")  <- merged correctly

    Bug: tool_result ends up in message 0 somehow.
    """
    # Simulate session messages (from the actual session file)
    session_messages = [
        Message(
            role="system",
            content="You are a coding assistant with access to file and shell tools.",
        ),
        Message(role="user", content="sleep 60"),
        Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="The user wants me to execute a sleep command for 60 seconds.",
                    thinking_signature="EoMCCkYI...",  # Truncated for test
                ),
                ToolCallContent(
                    type="toolCall",
                    id="toolu_014dRxC2urffovBmdbr4qwr6",
                    name="bash",
                    arguments={"command": "sleep 60", "timeout": 120},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
            model="claude-opus-4-5-20251101",
        ),
        Message(
            role="tool",
            content="[interrupted]",
            tool_call_id="toolu_014dRxC2urffovBmdbr4qwr6",
        ),
    ]

    # User types new message after resume
    new_user_message = Message(role="user", content="what happened?")
    all_messages = session_messages + [new_user_message]

    # Transform messages
    transformed = transform_messages(
        all_messages,
        target_provider="anthropic",
        target_api="anthropic-messages",
    )

    # Convert to API format (simulating rollout_anthropic logic)
    api_messages = []
    for m in transformed:
        if m.role == "system":
            continue  # System prompt handled separately
        elif m.role == "tool":
            if isinstance(m.content, str):
                tool_result_text = m.content
            else:
                tool_result_text = ""
            api_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": tool_result_text,
                    }
                ],
            })
        else:
            api_messages.append(_message_to_anthropic(m))

    # Merge consecutive same-role messages
    merged = _merge_consecutive_api_messages(api_messages)

    # Debug output
    print("\n=== Transformed messages ===")
    for i, m in enumerate(transformed):
        print(f"{i}: role={m.role}, tool_call_id={getattr(m, 'tool_call_id', None)}")

    print("\n=== API messages before merge ===")
    for i, m in enumerate(api_messages):
        print(f"{i}: role={m['role']}, content_types={[c.get('type') if isinstance(c, dict) else 'text' for c in (m['content'] if isinstance(m['content'], list) else [m['content']])]}")

    print("\n=== API messages after merge ===")
    for i, m in enumerate(merged):
        content = m["content"]
        if isinstance(content, list):
            types = [c.get("type") if isinstance(c, dict) else "text" for c in content]
        else:
            types = ["text"]
        print(f"{i}: role={m['role']}, content_types={types}")

    # Assertions
    assert len(merged) >= 3, f"Should have at least 3 messages, got {len(merged)}"

    # First message should be user with text content only (not tool_result)
    first_msg = merged[0]
    assert first_msg["role"] == "user", f"First message should be user, got {first_msg['role']}"

    # Check first message doesn't contain tool_result
    first_content = first_msg["content"]
    if isinstance(first_content, list):
        for block in first_content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                raise AssertionError(
                    f"First user message should NOT contain tool_result!\n"
                    f"This is the bug: {json.dumps(merged, indent=2, default=str)}"
                )

    # Second message should be assistant with tool_use
    second_msg = merged[1]
    assert second_msg["role"] == "assistant", f"Second message should be assistant, got {second_msg['role']}"

    # Third message should be user with tool_result
    third_msg = merged[2]
    assert third_msg["role"] == "user", f"Third message should be user, got {third_msg['role']}"

    # Third message should contain tool_result AND the new user text
    third_content = third_msg["content"]
    assert isinstance(third_content, list), "Third message content should be a list"

    has_tool_result = any(
        isinstance(c, dict) and c.get("type") == "tool_result"
        for c in third_content
    )
    has_text = any(
        isinstance(c, dict) and c.get("type") == "text" and "what happened" in c.get("text", "")
        for c in third_content
    )

    assert has_tool_result, "Third message should contain tool_result"
    assert has_text, "Third message should contain the new user text"

    print("\n✅ All assertions passed!")


def test_tool_result_position_in_merged_messages():
    """Verify tool_result appears after corresponding tool_use, not before.

    The Anthropic API requires:
    - tool_result must be in a user message
    - tool_result must come AFTER the assistant message with matching tool_use

    This test verifies the order is maintained after merging.
    """
    session_messages = [
        Message(role="user", content="run a command"),
        Message(
            role="assistant",
            content=[
                TextContent(type="text", text="I'll run that for you."),
                ToolCallContent(
                    type="toolCall",
                    id="tool_abc123",
                    name="bash",
                    arguments={"command": "echo hello"},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
        ),
        Message(role="tool", content="hello", tool_call_id="tool_abc123"),
        Message(role="user", content="thanks!"),
    ]

    transformed = transform_messages(
        session_messages,
        target_provider="anthropic",
        target_api="anthropic-messages",
    )

    # Convert to API format
    api_messages = []
    for m in transformed:
        if m.role == "tool":
            api_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": m.content if isinstance(m.content, str) else "",
                    }
                ],
            })
        else:
            api_messages.append(_message_to_anthropic(m))

    merged = _merge_consecutive_api_messages(api_messages)

    # Find positions
    tool_use_idx = None
    tool_result_idx = None

    for i, msg in enumerate(merged):
        content = msg["content"]
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "tool_use" and block.get("id") == "tool_abc123":
                        tool_use_idx = i
                    if block.get("type") == "tool_result" and block.get("tool_use_id") == "tool_abc123":
                        tool_result_idx = i

    assert tool_use_idx is not None, "Should find tool_use"
    assert tool_result_idx is not None, "Should find tool_result"
    assert tool_result_idx > tool_use_idx, (
        f"tool_result (at {tool_result_idx}) should come after tool_use (at {tool_use_idx})"
    )

    print(f"\n✅ tool_use at {tool_use_idx}, tool_result at {tool_result_idx} - order correct!")


def test_interrupt_then_continue_scenario():
    """Test the exact scenario from the bug:

    1. User starts session, types "sleep 60"
    2. Model responds with thinking + tool_use
    3. User presses Escape during tool execution
    4. Session has: [system, user, assistant(tool_use), tool("[interrupted]")]
    5. User types new message to continue
    6. API call should have correct message order

    The bug: After interrupt handling adds the tool result, we may end up with
    messages in wrong order, causing API rejection.
    """
    # Exact message sequence from session 20251223_172747_a0c54a
    messages = [
        Message(
            role="system",
            content="You are a coding assistant...",
        ),
        Message(role="user", content="sleep 60"),
        Message(
            role="assistant",
            content=[
                ThinkingContent(
                    type="thinking",
                    thinking="The user wants me to execute a sleep command.",
                    thinking_signature="EqMCCkYI...",
                ),
                ToolCallContent(
                    type="toolCall",
                    id="toolu_01FhjgAgBh8zeZdGpNf6nAzo",
                    name="bash",
                    arguments={"command": "sleep 60", "timeout": 120},
                ),
            ],
            provider="anthropic",
            api="anthropic-messages",
            model="claude-opus-4-5-20251101",
        ),
        Message(
            role="tool",
            content="[interrupted]",
            tool_call_id="toolu_01FhjgAgBh8zeZdGpNf6nAzo",
        ),
    ]

    # After interrupt, user types new message
    messages.append(Message(role="user", content="what happened?"))

    # Transform
    transformed = transform_messages(
        messages, target_provider="anthropic", target_api="anthropic-messages"
    )

    # Convert to API format (exact logic from rollout_anthropic)
    api_messages = []
    for m in transformed:
        if m.role == "system":
            continue
        elif m.role == "tool":
            content = m.content if isinstance(m.content, str) else ""
            api_messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": m.tool_call_id,
                        "content": content,
                    }
                ],
            })
        else:
            api_messages.append(_message_to_anthropic(m))

    # Insert "Begin." if first message is not user (line 597-598 of anthropic.py)
    if api_messages and api_messages[0]["role"] != "user":
        api_messages.insert(0, {"role": "user", "content": "Begin."})

    # Merge consecutive same-role messages
    merged = _merge_consecutive_api_messages(api_messages)

    print("\n=== API messages after full pipeline ===")
    for i, m in enumerate(merged):
        print(f"{i}: role={m['role']}")
        content = m["content"]
        if isinstance(content, list):
            for j, c in enumerate(content):
                if isinstance(c, dict):
                    print(f"    [{j}] type={c.get('type')}", end="")
                    if c.get("type") == "tool_result":
                        print(f" tool_use_id={c.get('tool_use_id')}", end="")
                    print()
        else:
            print(f"    text: {str(content)[:50]}")

    # THE BUG CHECK: First message should NOT have tool_result
    first_msg = merged[0]
    if isinstance(first_msg["content"], list):
        for j, block in enumerate(first_msg["content"]):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                raise AssertionError(
                    f"BUG FOUND: First message has tool_result at position {j}!\n"
                    f"This is exactly the error: messages.0.content.{j}: "
                    f"unexpected tool_use_id found in tool_result blocks\n"
                    f"Full first message: {first_msg}"
                )

    print("\n✅ No bug detected - tool_result is not in first message")


if __name__ == "__main__":
    print("Running test_session_resume_message_order...")
    test_session_resume_message_order()

    print("\nRunning test_tool_result_position_in_merged_messages...")
    test_tool_result_position_in_merged_messages()

    print("\nRunning test_interrupt_then_continue_scenario...")
    test_interrupt_then_continue_scenario()

    print("\n" + "=" * 50)
    print("All tests passed!")
