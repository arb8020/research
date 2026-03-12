"""Test ClaudeCodeEnvironment by running a simple task.

Run this test:
    cd ~/research/rollouts
    uv run python tests/test_claude_code_env.py

Or with pytest:
    uv run python -m pytest tests/test_claude_code_env.py -v -s

Prerequisites:
    - Claude Code CLI installed: npm install -g @anthropic-ai/claude-code
    - ANTHROPIC_API_KEY set in environment
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from rollouts.environments.claude_code import (
    ClaudeCodeEnvironment,
    ClaudeCodeTrajectory,
    run_claude_code,
)

pytestmark = pytest.mark.live


def _require_claude_runtime() -> None:
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("Claude Code CLI not installed")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")


@pytest.mark.trio
async def test_simple_task():
    """Test running a simple task with Claude Code."""
    _require_claude_runtime()
    print("\n=== Test: Simple Task ===")

    # Create a temporary directory with a simple file
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple Python file with a bug
        test_file = Path(tmpdir) / "calculator.py"
        test_file.write_text("""\
def add(a, b):
    return a - b  # Bug: should be a + b

def multiply(a, b):
    return a * b

if __name__ == "__main__":
    print(f"2 + 3 = {add(2, 3)}")
    print(f"2 * 3 = {multiply(2, 3)}")
""")

        print(f"Created test file: {test_file}")
        print(f"Working dir: {tmpdir}")

        # Run Claude Code on the task
        trajectory = await run_claude_code(
            task="There's a bug in calculator.py - the add function returns wrong results. Fix it.",
            working_dir=tmpdir,
            model="claude-sonnet-4-20250514",
            max_turns=10,
            timeout_seconds=120.0,
        )

        print("\n--- Results ---")
        print(f"Stop reason: {trajectory.stop_reason}")
        print(f"Num turns: {trajectory.num_turns}")
        print(f"Duration: {trajectory.duration_ms}ms")
        print(f"Tool calls: {len(trajectory.get_tool_calls())}")

        if trajectory.error:
            print(f"Error: {trajectory.error}")

        print("\n--- Final Text ---")
        print(trajectory.get_final_text()[:500])

        print("\n--- Tool Calls ---")
        for tc in trajectory.get_tool_calls():
            print(f"  - {tc.get('name')}: {json.dumps(tc.get('input', {}))[:100]}...")

        # Check if the file was fixed
        fixed_content = test_file.read_text()
        if "a + b" in fixed_content:
            print("\n✅ Bug was fixed!")
        else:
            print("\n❌ Bug was NOT fixed")
            print(f"Current content:\n{fixed_content}")

        return trajectory


@pytest.mark.trio
async def test_environment_protocol():
    """Test that ClaudeCodeEnvironment satisfies the Environment protocol."""
    print("\n=== Test: Environment Protocol ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = ClaudeCodeEnvironment(
            working_dir=Path(tmpdir),
            model="claude-sonnet-4-20250514",
            max_turns=5,
        )

        # Test get_tools
        tools = env.get_tools()
        assert len(tools) == 1
        assert tools[0].function.name == "run_agent"
        print(f"✅ get_tools() returns: {[t.function.name for t in tools]}")

        # Test requires_confirmation
        from rollouts.core import ToolCall

        tc = ToolCall(id="test", name="run_agent", args={"task": "test"})
        assert not env.requires_confirmation(tc)
        print("✅ requires_confirmation() returns False")

        # Test get_status_info
        status = env.get_status_info()
        assert status is not None
        assert "env" in status
        print(f"✅ get_status_info() returns: {status}")

        # Test serialize/deserialize
        data = await env.serialize()
        env2 = await ClaudeCodeEnvironment.deserialize(data)
        assert env2.model == env.model
        assert env2.max_turns == env.max_turns
        print("✅ serialize/deserialize roundtrip works")

        print("\n✅ All protocol tests passed!")


@pytest.mark.trio
@pytest.mark.skip(reason="Requires Claude Code CLI to be properly installed and configured")
async def test_streaming_callback():
    """Test that streaming callback receives messages."""
    _require_claude_runtime()
    print("\n=== Test: Streaming Callback ===")

    messages_received = []

    async def on_message(msg: dict):
        messages_received.append(msg)
        msg_type = msg.get("type", "unknown")
        print(f"  📨 {msg_type}", end="")
        if msg_type == "assistant":
            content = msg.get("message", {}).get("content", [])
            for block in content:
                if block.get("type") == "text":
                    text = block.get("text", "")[:50]
                    print(f": {text}...")
                    break
            else:
                print()
        else:
            print()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple file
        (Path(tmpdir) / "hello.txt").write_text("Hello, World!")

        env = ClaudeCodeEnvironment(
            working_dir=Path(tmpdir),
            model="claude-sonnet-4-20250514",
            max_turns=3,
        )

        trajectory = await env.run_task(
            "Read hello.txt and tell me what it says.",
            on_message=on_message,
        )

        print(f"\nReceived {len(messages_received)} messages")
        print(f"Stop reason: {trajectory.stop_reason}")

        # Should have received at least init and result
        assert len(messages_received) > 0, "Should receive at least some messages"
        print("\n✅ Streaming callback test passed!")


def test_trajectory_parsing():
    """Test ClaudeCodeTrajectory parsing without running Claude Code."""
    print("\n=== Test: Trajectory Parsing ===")

    # Simulate the NDJSON output from Claude Code
    messages = [
        {"type": "system", "subtype": "init", "session_id": "test-123", "tools": ["Read", "Write"]},
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "I'll read the file."},
                    {
                        "type": "tool_use",
                        "id": "tool_1",
                        "name": "Read",
                        "input": {"path": "test.txt"},
                    },
                ]
            },
        },
        {
            "type": "user",
            "message": {
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool_1", "content": "Hello World"},
                ]
            },
        },
        {
            "type": "assistant",
            "message": {
                "content": [
                    {"type": "text", "text": "The file contains 'Hello World'."},
                ]
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "num_turns": 2,
            "is_error": False,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    ]

    trajectory = ClaudeCodeTrajectory(
        messages=tuple(messages),
        stop_reason="success",
        num_turns=2,
        duration_ms=1000,
        tokens_in=100,
        tokens_out=50,
    )

    # Test helper methods
    assistant_msgs = trajectory.get_assistant_messages()
    assert len(assistant_msgs) == 2
    print(f"✅ get_assistant_messages() returns {len(assistant_msgs)} messages")

    tool_calls = trajectory.get_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "Read"
    print(f"✅ get_tool_calls() returns {len(tool_calls)} calls")

    final_text = trajectory.get_final_text()
    assert "Hello World" in final_text
    print(f"✅ get_final_text() returns: {final_text[:50]}...")

    print("\n✅ Trajectory parsing test passed!")
