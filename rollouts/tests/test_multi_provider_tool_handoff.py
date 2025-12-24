#!/usr/bin/env python3
"""Integration test for multi-provider tool call handoff.

Tests the scenario where we switch providers mid-conversation while using tools.
This exercises:
- Tool message construction with tool_call_id
- Message transformation between providers
- The full agent loop across provider boundaries

Requires: OPENAI_API_KEY, ANTHROPIC_API_KEY environment variables

TODO: test_openai_to_anthropic_tool_handoff fails with "empty stream" from Anthropic.
      This appears to be Anthropic returning no content after receiving a transformed
      conversation from OpenAI. Needs investigation - possibly related to how we
      transform tool results or the model believing the task is complete.
      See: rollouts/providers/anthropic.py:aggregate_anthropic_stream
"""

import os

import pytest
import trio
from rollouts.transform_messages import transform_messages

from rollouts import (
    Actor,
    AgentState,
    CalculatorEnvironment,
    Endpoint,
    Message,
    RunConfig,
    Trajectory,
    get_api_type,
    handle_stop_max_turns,
    run_agent,
    stdout_handler,
)


def get_endpoint(provider: str, model: str, api_key: str, api_base: str = "") -> Endpoint:
    """Create an endpoint for a provider."""
    return Endpoint(
        provider=provider,
        model=model,
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,
    )


async def run_single_turn(
    actor: Actor,
    env: CalculatorEnvironment,
    user_message: str,
) -> tuple[Actor, CalculatorEnvironment]:
    """Run a single agent turn (may include multiple tool calls).

    Returns the updated actor and environment.
    """
    # Add user message
    new_messages = actor.trajectory.messages + [Message(role="user", content=user_message)]
    actor = Actor(
        trajectory=Trajectory(messages=new_messages),
        endpoint=actor.endpoint,
        tools=env.get_tools(),
    )

    state = AgentState(actor=actor, environment=env)
    run_config = RunConfig(
        on_chunk=stdout_handler,
        handle_stop=handle_stop_max_turns(5),
    )

    states = await run_agent(state, run_config)
    final_state = states[-1]
    # Environment may have been updated during tool execution
    final_env = final_state.environment
    assert isinstance(final_env, CalculatorEnvironment)
    return final_state.actor, final_env


def switch_provider(
    actor: Actor,
    new_endpoint: Endpoint,
    env: CalculatorEnvironment,
) -> Actor:
    """Switch to a new provider, transforming messages appropriately."""
    old_api = get_api_type(actor.endpoint.provider, actor.endpoint.model)
    new_api = get_api_type(new_endpoint.provider, new_endpoint.model)

    transformed = transform_messages(
        messages=actor.trajectory.messages,
        from_provider=actor.endpoint.provider,
        from_api=old_api,
        to_provider=new_endpoint.provider,
        to_api=new_api,
    )

    return Actor(
        trajectory=Trajectory(messages=transformed),
        endpoint=new_endpoint,
        tools=env.get_tools(),
    )


def verify_tool_call_ids(messages: list[Message], context: str) -> None:
    """Verify all tool messages have valid tool_call_ids."""
    tool_messages = [m for m in messages if m.role == "tool"]
    print(f"\n{context}: {len(tool_messages)} tool messages")
    for i, msg in enumerate(tool_messages):
        print(f"  [{i}] tool_call_id={msg.tool_call_id!r}")
        assert msg.tool_call_id is not None, f"Tool message {i} has None tool_call_id"
        assert msg.tool_call_id != "", f"Tool message {i} has empty tool_call_id"


@pytest.mark.trio
async def test_openai_to_anthropic_tool_handoff():
    """Test switching from OpenAI to Anthropic mid-conversation with tool calls.

    This is the core test - verifies tool_call_id integrity across provider switch.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    print("\n" + "=" * 70)
    print("TEST: OpenAI → Anthropic Tool Handoff")
    print("=" * 70)

    env = CalculatorEnvironment()

    # Step 1: Start with OpenAI - make a tool call
    print("\n--- Step 1: OpenAI makes tool call ---")
    openai_endpoint = get_endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_key,
        api_base="https://api.openai.com/v1",
    )

    actor = Actor(
        trajectory=Trajectory(messages=[]),
        endpoint=openai_endpoint,
        tools=env.get_tools(),
    )

    actor, env = await run_single_turn(actor, env, "Use the add tool to add 10 to the calculator.")

    # Verify OpenAI made tool calls with valid IDs
    verify_tool_call_ids(actor.trajectory.messages, "After OpenAI")
    openai_tool_count = len([m for m in actor.trajectory.messages if m.role == "tool"])
    assert openai_tool_count > 0, "OpenAI should have made at least one tool call"

    # Step 2: Switch to Anthropic - continue conversation
    print("\n--- Step 2: Switch to Anthropic ---")
    anthropic_endpoint = get_endpoint(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key=anthropic_key,
        api_base="https://api.anthropic.com",
    )

    # Transform messages for new provider
    actor = switch_provider(actor, anthropic_endpoint, env)

    # Verify tool_call_ids survived the transformation
    verify_tool_call_ids(actor.trajectory.messages, "After transform to Anthropic")

    # Make another tool call with Anthropic
    actor, env = await run_single_turn(actor, env, "Now use the add tool to add 5 more.")

    # Verify all tool_call_ids are still valid
    verify_tool_call_ids(actor.trajectory.messages, "After Anthropic tool call")

    total_tool_count = len([m for m in actor.trajectory.messages if m.role == "tool"])
    assert total_tool_count > openai_tool_count, "Anthropic should have added more tool calls"

    print("\n" + "=" * 70)
    print("✅ OpenAI → Anthropic handoff PASSED")
    print("=" * 70)


@pytest.mark.trio
async def test_anthropic_to_openai_tool_handoff():
    """Test switching from Anthropic to OpenAI mid-conversation with tool calls.

    Tests the reverse direction to ensure bidirectional compatibility.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    print("\n" + "=" * 70)
    print("TEST: Anthropic → OpenAI Tool Handoff")
    print("=" * 70)

    env = CalculatorEnvironment()

    # Step 1: Start with Anthropic
    print("\n--- Step 1: Anthropic makes tool call ---")
    anthropic_endpoint = get_endpoint(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key=anthropic_key,
        api_base="https://api.anthropic.com",
    )

    actor = Actor(
        trajectory=Trajectory(messages=[]),
        endpoint=anthropic_endpoint,
        tools=env.get_tools(),
    )

    actor, env = await run_single_turn(actor, env, "Use the multiply tool to multiply by 3.")

    verify_tool_call_ids(actor.trajectory.messages, "After Anthropic")
    anthropic_tool_count = len([m for m in actor.trajectory.messages if m.role == "tool"])
    assert anthropic_tool_count > 0, "Anthropic should have made at least one tool call"

    # Step 2: Switch to OpenAI
    print("\n--- Step 2: Switch to OpenAI ---")
    openai_endpoint = get_endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_key,
        api_base="https://api.openai.com/v1",
    )

    actor = switch_provider(actor, openai_endpoint, env)
    verify_tool_call_ids(actor.trajectory.messages, "After transform to OpenAI")

    # Make another tool call with OpenAI
    actor, env = await run_single_turn(actor, env, "Now use the add tool to add 7.")

    verify_tool_call_ids(actor.trajectory.messages, "After OpenAI tool call")

    total_tool_count = len([m for m in actor.trajectory.messages if m.role == "tool"])
    assert total_tool_count > anthropic_tool_count, "OpenAI should have added more tool calls"

    print("\n" + "=" * 70)
    print("✅ Anthropic → OpenAI handoff PASSED")
    print("=" * 70)


@pytest.mark.trio
async def test_tool_call_id_format_after_transform():
    """Test that tool_call_id format is valid after transformation.

    Some providers use different ID formats. Verify IDs are usable after transform.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")

    if not openai_key:
        pytest.skip("OPENAI_API_KEY not set")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    print("\n" + "=" * 70)
    print("TEST: tool_call_id Format Integrity")
    print("=" * 70)

    env = CalculatorEnvironment()

    # Start with OpenAI to get OpenAI-format IDs
    openai_endpoint = get_endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_key,
        api_base="https://api.openai.com/v1",
    )

    actor = Actor(
        trajectory=Trajectory(messages=[]),
        endpoint=openai_endpoint,
        tools=env.get_tools(),
    )

    actor, env = await run_single_turn(
        actor, env, "Use add to add 1, then add to add 2, then add to add 3."
    )

    # Collect IDs before transform
    tool_msgs_before = [m for m in actor.trajectory.messages if m.role == "tool"]
    ids_before = [(m.tool_call_id, m.content) for m in tool_msgs_before]

    print(f"\nOpenAI IDs ({len(ids_before)} tool messages):")
    for tid, content in ids_before:
        print(f"  {tid!r} -> {str(content)[:50]!r}")

    # Transform to Anthropic
    anthropic_endpoint = get_endpoint(
        provider="anthropic",
        model="claude-3-5-haiku-20241022",
        api_key=anthropic_key,
    )
    actor = switch_provider(actor, anthropic_endpoint, env)

    # Check IDs after transform
    tool_msgs_after = [m for m in actor.trajectory.messages if m.role == "tool"]
    ids_after = [(m.tool_call_id, m.content) for m in tool_msgs_after]

    print(f"\nAfter transform to Anthropic ({len(ids_after)} tool messages):")
    for tid, content in ids_after:
        print(f"  {tid!r} -> {str(content)[:50]!r}")

    # Verify count preserved
    assert len(ids_after) == len(ids_before), (
        f"Tool message count changed: {len(ids_before)} -> {len(ids_after)}"
    )

    # Verify all IDs are valid strings
    for i, (tid, _) in enumerate(ids_after):
        assert tid is not None, f"ID {i} is None after transform"
        assert isinstance(tid, str), f"ID {i} is not a string: {type(tid)}"
        assert len(tid) > 0, f"ID {i} is empty string"

    print("\n" + "=" * 70)
    print("✅ tool_call_id format integrity PASSED")
    print("=" * 70)


if __name__ == "__main__":

    async def main():
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Multi-Provider Tool Call Handoff")
        print("=" * 70)

        tests = [
            ("OpenAI → Anthropic", test_openai_to_anthropic_tool_handoff),
            ("Anthropic → OpenAI", test_anthropic_to_openai_tool_handoff),
            ("tool_call_id Format", test_tool_call_id_format_after_transform),
        ]

        passed = 0
        failed = 0
        skipped = 0

        for name, test_func in tests:
            try:
                await test_func()
                passed += 1
            except Exception as e:
                if "skip" in str(e).lower():
                    print(f"⏭️  Skipped {name}: {e}")
                    skipped += 1
                else:
                    print(f"❌ Failed {name}: {e}")
                    import traceback

                    traceback.print_exc()
                    failed += 1

        print("\n" + "=" * 70)
        print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * 70 + "\n")

    trio.run(main)
