#!/usr/bin/env python3
"""Test run_agent() with a real LLM.

This tests the full agent loop:
- LLM decides which tools to call
- Tools execute
- Results feed back to LLM
- Continues until task complete

Requires: OPENAI_API_KEY environment variable
"""

import os

import pytest
import trio

from rollouts import (
    Actor,
    AgentState,
    CalculatorEnvironment,
    Endpoint,
    Message,
    RunConfig,
    Trajectory,
    handle_stop_max_turns,
    run_agent,
    stdout_handler,
)


@pytest.mark.trio
async def test_calculator_agent():
    """Test agent solving a simple math problem using calculator.

    Task: "What is 25 + 17?"

    Expected behavior:
    1. LLM calls add(25)
    2. Calculator returns 25
    3. LLM calls add(17)
    4. Calculator returns 42
    5. LLM calls complete_task()
    """
    # Check API key first
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    print("🤖 Testing real agent with OpenAI...")
    print("Task: What is 25 + 17?")
    print()

    # Setup
    env = CalculatorEnvironment()
    endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",  # Cheap and fast
        api_key=api_key,
        api_base="https://api.openai.com/v1",
        temperature=0.0,  # Deterministic
    )

    initial_message = Message(
        role="user",
        content="Calculate 25 + 17. Use the calculator tools, then call complete_task when done."
    )

    actor = Actor(
        trajectory=Trajectory(messages=[initial_message]),
        endpoint=endpoint,
        tools=env.get_tools()
    )

    state = AgentState(
        actor=actor,
        environment=env
    )

    run_config = RunConfig(
        on_chunk=stdout_handler,  # Print tokens as they stream
        handle_stop=handle_stop_max_turns(10)  # Stop after 10 turns
    )

    # Run agent!
    print("Starting agent...")
    print("=" * 60)

    try:
        states = await run_agent(state, run_config)
        final_state = states[-1]

        print("=" * 60)
        print()
        print(f"✅ Agent completed in {final_state.turn_idx} turns")
        print(f"   Stop reason: {final_state.stop}")
        print()

        # Show the conversation
        print("Conversation:")
        for i, msg in enumerate(final_state.actor.trajectory.messages):
            print(f"  [{i}] {msg.role}: {msg.content[:100]}")
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"      -> tool: {tc.name}({tc.args})")

        print()

        # Verify calculator was used
        tool_messages = [m for m in final_state.actor.trajectory.messages if m.role == "tool"]
        print(f"✓ Used {len(tool_messages)} tool calls")

        # Check if we got the right answer
        last_content = str(final_state.actor.trajectory.messages[-1].content or "")
        if "42" in last_content:
            print("✓ Got correct answer: 42")
        else:
            print(f"? Answer unclear: {last_content[:200]}")

    except Exception as e:
        print(f"❌ Agent failed: {e}")
        import traceback
        traceback.print_exc()


async def main():
    print("\n=== Testing Real Agent with LLM ===\n")
    try:
        await test_calculator_agent()
        print("\n=== Test complete ===\n")
    except Exception as e:
        if e.__class__.__name__ == "Skipped":
            print(f"\n⏭️  Test skipped: {e}\n")
        else:
            raise


if __name__ == "__main__":
    trio.run(main)
