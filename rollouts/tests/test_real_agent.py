#!/usr/bin/env python3
"""Test run_agent() with a real LLM.

This tests the full agent loop:
- LLM decides which tools to call
- Tools execute
- Results feed back to LLM
- Continues until task complete

Requires: OPENAI_API_KEY environment variable
"""

import asyncio
import os
from rollouts import (
    Message, Trajectory, Endpoint, Actor, AgentState, RunConfig,
    CalculatorEnvironment, run_agent, stdout_handler
)


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
    print("🤖 Testing real agent with OpenAI...")
    print("Task: What is 25 + 17?")
    print()

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not set, skipping test")
        return

    # Setup
    env = CalculatorEnvironment()
    endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",  # Cheap and fast
        api_key=os.getenv("OPENAI_API_KEY", ""),
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
        environment=env,
        max_turns=10
    )

    run_config = RunConfig(
        on_chunk=stdout_handler  # Print tokens as they stream
    )

    # Run agent!
    print("Starting agent...")
    print("=" * 60)

    try:
        final_state = await run_agent(state, run_config)

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
    await test_calculator_agent()
    print("\n=== Test complete ===\n")


if __name__ == "__main__":
    asyncio.run(main())
