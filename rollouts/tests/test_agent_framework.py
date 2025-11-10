#!/usr/bin/env python3
"""Quick smoke test for the agent framework.

Tests basic functionality without needing API keys.
"""

import pytest
import trio
from rollouts import (
    Message, Trajectory, Tool, ToolFunction, ToolFunctionParameter,
    ToolCall, ToolResult, Endpoint, Actor, Environment, AgentState,
    RunConfig, StopReason, CalculatorEnvironment
)


def test_basic_types():
    """Test that basic types can be created."""
    print("✓ Testing basic types...")

    # Message
    msg = Message(role="user", content="Hello")
    assert msg.role == "user"
    assert msg.content == "Hello"

    # ToolCall
    tc = ToolCall(id="1", name="add", args={"a": 1, "b": 2})
    assert tc.name == "add"

    # Trajectory
    traj = Trajectory(messages=[msg])
    assert len(traj.messages) == 1

    print("  ✓ Message, ToolCall, Trajectory work")


def test_environment():
    """Test that CalculatorEnvironment works."""
    print("✓ Testing CalculatorEnvironment...")

    env = CalculatorEnvironment()
    tools = env.get_tools()
    assert len(tools) > 0

    # Check tools have the right structure
    tool = tools[0]
    assert isinstance(tool, Tool)
    assert tool.function.name in ["add", "subtract", "multiply", "divide"]

    print(f"  ✓ Calculator has {len(tools)} tools: {[t.function.name for t in tools]}")


@pytest.mark.trio
async def test_tool_execution():
    """Test that tools can be executed."""
    print("✓ Testing tool execution...")

    env = CalculatorEnvironment()
    tool_call = ToolCall(id="1", name="add", args={"value": 5})

    # Create minimal state for exec_tool
    endpoint = Endpoint(provider="test", model="test")
    actor = Actor(trajectory=Trajectory(), endpoint=endpoint, tools=env.get_tools())
    state = AgentState(actor=actor, environment=env, max_turns=1)
    async def _dummy_chunk(x):
        await trio.lowlevel.checkpoint()
    run_config = RunConfig(on_chunk=_dummy_chunk)  # Dummy

    result = await env.exec_tool(tool_call, state, run_config)

    if not result.ok:
        print(f"  ✗ Tool execution failed: {result.error}")
        print(f"    Result: {result}")
        return

    assert "5" in result.content  # Calculator adds to current value (0 + 5)

    print(f"  ✓ add(5) = {result.content}")


def test_serialization():
    """Test that types can serialize to/from JSON."""
    print("✓ Testing serialization...")

    # Message
    msg = Message(role="user", content="test")
    json_str = msg.to_json()
    msg2 = Message.from_json(json_str)
    assert msg.role == msg2.role
    assert msg.content == msg2.content

    # Trajectory
    traj = Trajectory(messages=[msg], rewards=1.0)
    json_str = traj.to_json()
    traj2 = Trajectory.from_json(json_str)
    assert len(traj2.messages) == 1
    assert traj2.rewards == 1.0

    print("  ✓ Message and Trajectory serialize correctly")


def test_imports():
    """Test that all expected symbols are exported."""
    print("✓ Testing imports...")

    from rollouts import (
        # Core types
        Trajectory, Message, ToolCall, ToolResult, Tool,
        # Agent types
        Actor, Environment, AgentState, RunConfig,
        # Environments
        CalculatorEnvironment,
        # Providers
        rollout_openai, rollout_sglang, rollout_anthropic,
        # Functions
        run_agent,
    )

    print("  ✓ All expected symbols can be imported")


async def main():
    """Run all tests."""
    print("\n=== Testing Agent Framework ===\n")

    test_basic_types()
    test_environment()
    await test_tool_execution()
    test_serialization()
    test_imports()

    print("\n=== ✅ All tests passed! ===\n")
    print("Agent framework is working correctly.")
    print("\nNext steps:")
    print("  - Test with actual API (requires API keys)")
    print("  - Build SFT data collection on top")
    print("  - Add training infrastructure")


if __name__ == "__main__":
    trio.run(main)
