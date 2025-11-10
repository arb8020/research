#!/usr/bin/env python3
"""Simple test of refactored streaming code."""

import trio
import os
from rollouts.dtypes import Endpoint, Actor, AgentState, Message, Trajectory
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.agents import run_agent, RunConfig, stdout_handler

async def main():
    # Check for API keys (try Anthropic first, fallback to OpenAI)
    use_anthropic = os.getenv("ANTHROPIC_API_KEY") is not None
    use_openai = os.getenv("OPENAI_API_KEY") is not None

    if not use_anthropic and not use_openai:
        print("❌ Set ANTHROPIC_API_KEY or OPENAI_API_KEY to test")
        return

    if use_anthropic:
        provider_name = "Anthropic (Claude Sonnet 4.5)"
        print("🧪 Testing refactored streaming with Anthropic...")
    else:
        provider_name = "OpenAI (GPT-4o-mini)"
        print("🧪 Testing refactored streaming with OpenAI...")

    print("=" * 50)

    # Simple messages
    messages = [
        Message(role="user", content="What is 5 + 3? Use the add function.")
    ]

    trajectory = Trajectory(messages=messages)

    if use_anthropic:
        endpoint = Endpoint(
            provider="anthropic",
            model="claude-sonnet-4-5-20250929",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0.1
        )
    else:
        endpoint = Endpoint(
            provider="openai",
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0.1
        )
    
    actor = Actor(trajectory=trajectory, endpoint=endpoint)
    env = CalculatorEnvironment()
    
    state = AgentState(
        actor=actor,
        environment=env,
        max_turns=3
    )
    
    config = RunConfig(on_chunk=stdout_handler)
    
    print("\n🚀 Running agent with refactored streaming...")
    print("-" * 50)
    
    try:
        states = await run_agent(state, config)
        final_state = states[-1]
        
        print("\n" + "=" * 50)
        print("✅ SUCCESS! Refactored streaming works!")
        print("=" * 50)
        print(f"Turns: {final_state.turn_idx}")
        print(f"Calculator value: {final_state.environment.current_value}")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 50)
        print(f"❌ FAILED: {e}")
        print("=" * 50)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = trio.run(main)
    exit(exit_code)
