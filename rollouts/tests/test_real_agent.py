#!/usr/bin/env python3
"""Test run_agent() with real LLM providers.

This tests the full agent loop with each supported provider:
- OpenAI (openai-completions API)
- OpenAI o1 (openai-responses API - reasoning models)
- Anthropic (anthropic-messages API)
- Groq (openai-completions API, different provider)
- Google Gemini (google-generative-ai API)

Each test:
- LLM decides which tools to call
- Tools execute
- Results feed back to LLM
- Continues until task complete

Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY, GROQ_API_KEY, and/or GEMINI_API_KEY environment variables
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


async def run_calculator_test(provider: str, model: str, api_key: str, api_base: str = ""):
    """Shared test logic for all providers.

    Task: "What is 25 + 17?"

    Expected behavior:
    1. LLM calls add(25)
    2. Calculator returns 25
    3. LLM calls add(17)
    4. Calculator returns 42
    5. LLM calls complete_task()
    """
    print(f"🤖 Testing {provider} with model {model}...")
    print("Task: What is 25 + 17?")
    print()

    env = CalculatorEnvironment()
    endpoint = Endpoint(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
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

    state = AgentState(actor=actor, environment=env)
    run_config = RunConfig(
        on_chunk=stdout_handler,
        handle_stop=handle_stop_max_turns(10)
    )

    print("Starting agent...")
    print("=" * 60)

    states = await run_agent(state, run_config)
    final_state = states[-1]

    print("=" * 60)
    print()
    print(f"✅ {provider} completed in {final_state.turn_idx} turns")
    print(f"   Stop reason: {final_state.stop}")
    print()

    # Verify calculator was used
    tool_messages = [m for m in final_state.actor.trajectory.messages if m.role == "tool"]
    assert len(tool_messages) > 0, f"{provider} didn't use any tools!"
    print(f"✓ Used {len(tool_messages)} tool calls")


@pytest.mark.trio
async def test_openai_agent():
    """Test OpenAI provider (openai-completions API)"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    await run_calculator_test(
        provider="openai",
        model="gpt-4o-mini",
        api_key=api_key,
        api_base="https://api.openai.com/v1"
    )


@pytest.mark.trio
async def test_anthropic_agent():
    """Test Anthropic provider (anthropic-messages API)"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    await run_calculator_test(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key=api_key,
        api_base="https://api.anthropic.com"
    )


@pytest.mark.trio
async def test_groq_agent():
    """Test Groq provider (openai-completions API - proves provider abstraction works)"""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        pytest.skip("GROQ_API_KEY not set")

    await run_calculator_test(
        provider="groq",
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        api_base="https://api.groq.com/openai/v1"
    )


@pytest.mark.trio
async def test_openai_responses_api():
    """Test OpenAI Responses API with gpt-5.1-codex-mini (GPT-5 Codex model)

    GPT-5.1-Codex models are designed for agentic coding tasks and only support
    the Responses API (not chat completions).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")

    await run_calculator_test(
        provider="openai",
        model="gpt-5.1-codex-mini",
        api_key=api_key,
        api_base="https://api.openai.com/v1"
    )


@pytest.mark.trio
@pytest.mark.skip(reason="Gemini not calling tools in streaming mode - needs investigation")
async def test_google_gemini_agent():
    """Test Google Gemini provider (google-generative-ai API)

    Uses trio-asyncio to bridge trio and asyncio event loops.

    NOTE: Infrastructure works (trio-asyncio bridge functional), but model doesn't
    call tools. Possible causes:
    1. Tool configuration format issue
    2. Gemini model doesn't support tools in streaming mode
    3. Need system prompt or toolChoice configuration

    See: https://github.com/googleapis/python-genai/issues/106
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set")

    await run_calculator_test(
        provider="google",
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        api_base=""  # Google uses default base URL
    )


if __name__ == "__main__":
    import sys
    async def main():
        print("\n=== Testing Real Agents with LLM Providers ===\n")

        tests = [
            ("OpenAI", test_openai_agent),
            ("OpenAI Responses API (gpt-4.1)", test_openai_responses_api),
            ("Anthropic", test_anthropic_agent),
            ("Groq", test_groq_agent),
            ("Google Gemini", test_google_gemini_agent),
        ]

        for name, test_func in tests:
            try:
                print(f"\n--- {name} ---")
                await test_func()
            except Exception as e:
                if "skip" in str(e).lower():
                    print(f"⏭️  Skipped: {e}")
                else:
                    print(f"❌ Failed: {e}")
                    import traceback
                    traceback.print_exc()

        print("\n=== Tests complete ===\n")

    trio.run(main)
