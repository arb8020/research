"""Test CodexEnvironment by running a simple task.

Run this test:
    cd ~/research/rollouts
    uv run python tests/test_codex_env.py

Prerequisites:
    - Codex CLI installed
    - OPENAI_API_KEY set in environment
"""

import json
import os
import tempfile
from pathlib import Path

import pytest

from rollouts.environments.codex import (
    CodexEnvironment,
    CodexTrajectory,
    run_codex,
)

pytestmark = pytest.mark.live


def _require_codex_runtime() -> None:
    import shutil

    if shutil.which("codex") is None:
        pytest.skip("Codex CLI not installed")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")


def test_trajectory_parsing():
    """Test CodexTrajectory parsing without running Codex."""
    print("\n=== Test: Trajectory Parsing ===")

    # Simulate the JSONL output from Codex
    events = [
        {"type": "thread.started", "thread_id": "test-123"},
        {"type": "turn.started"},
        {"type": "item.started", "item": {"id": "item_0", "type": "agent_message"}},
        {
            "type": "item.completed",
            "item": {"id": "item_0", "type": "agent_message", "text": "I'll read the file."},
        },
        {
            "type": "item.started",
            "item": {"id": "item_1", "type": "tool_call", "name": "read_file"},
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "tool_call",
                "name": "read_file",
                "arguments": {"path": "test.txt"},
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_2",
                "type": "agent_message",
                "text": "The file contains 'Hello World'.",
            },
        },
        {"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 50}},
    ]

    trajectory = CodexTrajectory(
        events=tuple(events),
        stop_reason="success",
        num_turns=1,
        duration_ms=1000,
        tokens_in=100,
        tokens_out=50,
        thread_id="test-123",
    )

    # Test helper methods
    items = trajectory.get_completed_items()
    assert len(items) == 3
    print(f"✅ get_completed_items() returns {len(items)} items")

    messages = trajectory.get_agent_messages()
    assert len(messages) == 2
    print(f"✅ get_agent_messages() returns {len(messages)} messages")

    tool_calls = trajectory.get_tool_calls()
    assert len(tool_calls) == 1
    assert tool_calls[0]["name"] == "read_file"
    print(f"✅ get_tool_calls() returns {len(tool_calls)} calls")

    final_text = trajectory.get_final_text()
    assert "Hello World" in final_text
    print(f"✅ get_final_text() returns: {final_text[:50]}...")

    print("\n✅ Trajectory parsing test passed!")


@pytest.mark.trio
async def test_environment_protocol():
    """Test that CodexEnvironment satisfies the Environment protocol."""
    print("\n=== Test: Environment Protocol ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        env = CodexEnvironment(
            working_dir=Path(tmpdir),
            model="o3",
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
        assert status["env"] == "codex"
        print(f"✅ get_status_info() returns: {status}")

        # Test serialize/deserialize
        data = await env.serialize()
        env2 = await CodexEnvironment.deserialize(data)
        assert env2.model == env.model
        print("✅ serialize/deserialize roundtrip works")

        print("\n✅ All protocol tests passed!")


@pytest.mark.trio
async def test_simple_task():
    """Test running a simple task with Codex."""
    _require_codex_runtime()
    print("\n=== Test: Simple Task ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple file
        test_file = Path(tmpdir) / "hello.txt"
        test_file.write_text("Hello, World!")

        print(f"Created test file: {test_file}")
        print(f"Working dir: {tmpdir}")

        messages_received = []

        async def on_event(event: dict):
            messages_received.append(event)
            event_type = event.get("type", "unknown")
            print(f"  📨 {event_type}", end="")
            if event_type == "item.completed":
                item = event.get("item", {})
                if item.get("type") == "agent_message":
                    text = item.get("text", "")[:50]
                    print(f": {text}...")
                else:
                    print(f": {item.get('type')}")
            else:
                print()

        trajectory = await run_codex(
            task="Read hello.txt and tell me what it says. Be brief.",
            working_dir=tmpdir,
            model="o3",
            timeout_seconds=120.0,
        )

        print("\n--- Results ---")
        print(f"Stop reason: {trajectory.stop_reason}")
        print(f"Num turns: {trajectory.num_turns}")
        print(f"Duration: {trajectory.duration_ms}ms")
        print(f"Tokens in: {trajectory.tokens_in}")
        print(f"Tokens out: {trajectory.tokens_out}")
        print(f"Thread ID: {trajectory.thread_id}")

        if trajectory.error:
            print(f"Error: {trajectory.error}")

        print("\n--- Final Text ---")
        print(trajectory.get_final_text()[:500])

        print("\n--- Tool Calls ---")
        for tc in trajectory.get_tool_calls():
            print(f"  - {tc.get('name')}: {json.dumps(tc.get('arguments', {}))[:100]}...")

        return trajectory
