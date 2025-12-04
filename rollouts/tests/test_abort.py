#!/usr/bin/env python3
"""Test abort/cancellation support in run_agent().

These tests verify that:
1. Cancellation actually cancels in-flight HTTP requests (not just polling)
2. Final state is set to StopReason.ABORTED when cancelled
3. Aborted state is checkpointed properly

Requires: OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable for HTTP cancellation tests
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
    StopReason,
    Trajectory,
    handle_stop_max_turns,
    run_agent,
)


def create_test_state(provider: str = "openai", model: str = "gpt-4o-mini") -> AgentState:
    """Create a test agent state for abort testing."""
    api_key = os.getenv("OPENAI_API_KEY") if provider == "openai" else os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip(f"Missing {provider.upper()}_API_KEY environment variable")

    env = CalculatorEnvironment()
    endpoint = Endpoint(
        provider=provider,
        model=model,
        api_key=api_key,
        temperature=0.0,
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

    return AgentState(actor=actor, environment=env)


@pytest.mark.trio
async def test_abort_cancels_http_request():
    """Verify abort actually cancels in-flight HTTP request.

    This test verifies that when cancel_scope.cancel() is called during
    an HTTP request, the request is actually cancelled (not just checked
    at polling points).
    """
    state = create_test_state()
    cancel_scope = trio.CancelScope()
    aborted = False

    async with trio.open_nursery() as nursery:
        async def agent_task():
            nonlocal aborted
            run_config = RunConfig(
                on_chunk=lambda e: None,  # Silent handler
                cancel_scope=cancel_scope,
                handle_stop=handle_stop_max_turns(10),
            )
            try:
                with cancel_scope:
                    await run_agent(state, run_config)
            except trio.Cancelled:
                aborted = True
                raise

        nursery.start_soon(agent_task)

        # Cancel almost immediately - should interrupt HTTP request
        await trio.sleep(0.1)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()  # Clean up nursery

    assert aborted, "Agent should have been cancelled"


@pytest.mark.trio
async def test_abort_sets_stop_reason():
    """Verify final state has StopReason.ABORTED when cancelled.

    This test verifies that when cancellation occurs, the final state
    in the states list has stop=StopReason.ABORTED.
    """
    state = create_test_state()
    states = []
    cancel_scope = trio.CancelScope()

    async with trio.open_nursery() as nursery:
        async def agent_task():
            nonlocal states
            run_config = RunConfig(
                on_chunk=lambda e: None,  # Silent handler
                cancel_scope=cancel_scope,
                handle_stop=handle_stop_max_turns(10),
            )
            try:
                with cancel_scope:
                    states = await run_agent(state, run_config)
            except trio.Cancelled:
                pass  # Expected - run_agent handles this internally

        nursery.start_soon(agent_task)
        # Give it a bit of time to start making HTTP request
        await trio.sleep(0.5)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()

    assert len(states) > 0, "Should have at least initial state"
    assert states[-1].stop == StopReason.ABORTED, f"Expected StopReason.ABORTED, got {states[-1].stop}"


@pytest.mark.trio
async def test_abort_with_anthropic():
    """Test abort with Anthropic provider (different HTTP client)."""
    state = create_test_state(provider="anthropic", model="claude-3-5-sonnet-20241022")
    cancel_scope = trio.CancelScope()
    aborted = False

    async with trio.open_nursery() as nursery:
        async def agent_task():
            nonlocal aborted
            run_config = RunConfig(
                on_chunk=lambda e: None,
                cancel_scope=cancel_scope,
                handle_stop=handle_stop_max_turns(10),
            )
            try:
                with cancel_scope:
                    await run_agent(state, run_config)
            except trio.Cancelled:
                aborted = True
                raise

        nursery.start_soon(agent_task)
        await trio.sleep(0.1)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()

    assert aborted, "Agent should have been cancelled"


@pytest.mark.trio
async def test_abort_checkpoints_state():
    """Verify that aborted state is checkpointed.

    This test verifies that when cancellation occurs, the aborted state
    is properly checkpointed via the checkpoint event system.
    """
    state = create_test_state()
    checkpointed_events = []
    cancel_scope = trio.CancelScope()

    async def checkpoint_handler(event):
        """Capture checkpoint events."""
        from rollouts import StreamChunk
        if isinstance(event, StreamChunk):
            checkpointed_events.append(event.type)

    async with trio.open_nursery() as nursery:
        async def agent_task():
            run_config = RunConfig(
                on_chunk=checkpoint_handler,
                cancel_scope=cancel_scope,
                handle_stop=handle_stop_max_turns(10),
            )
            try:
                with cancel_scope:
                    await run_agent(state, run_config)
            except trio.Cancelled:
                pass  # Expected

        nursery.start_soon(agent_task)
        await trio.sleep(0.5)
        cancel_scope.cancel()
        nursery.cancel_scope.cancel()

    # Should have checkpointed the final (aborted) state
    assert "final" in checkpointed_events, f"Should have checkpointed final state on abort, got: {checkpointed_events}"

