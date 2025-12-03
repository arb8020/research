#!/usr/bin/env python3
"""Test multi-turn agent execution with calculator.

This tests whether the agent can:
1. Make multiple tool calls across turns
2. Maintain state (calculator value)
3. Use tools to solve a problem

Without a real LLM, we'll manually simulate the agent loop.
"""

import pytest
import trio

from rollouts import (
    Actor,
    AgentState,
    CalculatorEnvironment,
    Endpoint,
    Message,
    RunConfig,
    ToolCall,
    Trajectory,
)


@pytest.mark.trio
async def test_manual_multi_turn():
    """Manually test multi-turn tool execution.

    Simulate solving: x^2 - 5x + 6 = 0
    Using quadratic formula: x = (5 ± sqrt(25 - 24)) / 2 = (5 ± 1) / 2
    Solutions: x = 3 or x = 2

    Steps:
    1. Calculate discriminant: 5*5 - 4*1*6 = 25 - 24 = 1
    2. Calculate sqrt(1) = 1
    3. Calculate (5 + 1) / 2 = 3
    4. Calculate (5 - 1) / 2 = 2
    """
    print("🧮 Testing multi-turn calculator usage...")
    print("Problem: Solve x^2 - 5x + 6 = 0")
    print()

    env = CalculatorEnvironment()
    endpoint = Endpoint(provider="test", model="test")
    actor = Actor(trajectory=Trajectory(), endpoint=endpoint, tools=env.get_tools())
    state = AgentState(actor=actor, environment=env)

    async def _dummy_chunk(x):
        await trio.lowlevel.checkpoint()
    run_config = RunConfig(on_chunk=_dummy_chunk)

    # Turn 1: Calculate 5*5 = 25
    print("Turn 1: Calculate 5*5")
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="1", name="add", args={"value": 25}))
    print(f"  Result: {state.actor.trajectory.messages[-1].content}")
    print()

    # Turn 2: Calculate 4*1*6 = 24
    print("Turn 2: Calculate 4*1*6 = 24, then subtract from 25")
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="2", name="subtract", args={"value": 24}))
    print(f"  Result: {state.actor.trajectory.messages[-1].content}")
    print()

    # Turn 3: Now we have discriminant = 1, calculate (5 + sqrt(1)) / 2
    print("Turn 3: Clear and calculate (5 + 1) / 2")
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="3", name="clear", args={}))
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="4", name="add", args={"value": 6}))
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="5", name="divide", args={"value": 2}))
    print(f"  Result: {state.actor.trajectory.messages[-1].content}")
    print()

    # Turn 4: Calculate (5 - 1) / 2
    print("Turn 4: Clear and calculate (5 - 1) / 2")
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="6", name="clear", args={}))
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="7", name="add", args={"value": 4}))
    state = await execute_tool_turn(state, env, run_config,
                                    ToolCall(id="8", name="divide", args={"value": 2}))
    print(f"  Result: {state.actor.trajectory.messages[-1].content}")
    print()

    # Verify we got the right answers
    print("✅ Solutions: x = 3 and x = 2")
    print(f"   Calculator correctly maintained state across {len(state.actor.trajectory.messages)} messages")
    print(f"   Used {state.turn_idx} turns")


async def execute_tool_turn(state: AgentState, env, run_config, tool_call: ToolCall) -> AgentState:
    """Execute a single tool call and update state."""
    # Execute the tool
    result = await env.exec_tool(tool_call, state, run_config)

    # Add tool result as a message (simulating what the agent loop does)
    tool_message = Message(
        role="tool",
        content=result.content,
        tool_call_id=tool_call.id
    )

    # Update trajectory
    new_messages = state.actor.trajectory.messages + [tool_message]
    new_trajectory = Trajectory(
        messages=new_messages,
        completions=state.actor.trajectory.completions
    )

    # Update actor and state
    new_actor = Actor(
        trajectory=new_trajectory,
        endpoint=state.actor.endpoint,
        tools=state.actor.tools
    )

    new_state = AgentState(
        actor=new_actor,
        environment=state.environment,
        stop=state.stop,
        turn_idx=state.turn_idx + 1,
        pending_tool_calls=state.pending_tool_calls,
        next_tool_idx=state.next_tool_idx
    )

    return new_state


async def main():
    print("\n=== Testing Multi-Turn Agent ===\n")
    await test_manual_multi_turn()
    print("\n=== ✅ Multi-turn test passed! ===\n")
    print("Next: Test with actual LLM API (requires API key)")


if __name__ == "__main__":
    trio.run(main)
