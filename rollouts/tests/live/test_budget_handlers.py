#!/usr/bin/env python3
"""Regression tests for token and cost budget handlers.

Tests that budget handlers correctly stop agent execution when limits are exceeded.
Uses real API calls - no mocking.

Requires ANTHROPIC_API_KEY environment variable.
"""

import os

import pytest
import trio

from rollouts.agents import (
    Actor,
    AgentState,
    RunConfig,
    StopReason,
    compose_handlers,
    handle_stop_cost_budget,
    handle_stop_max_turns,
    handle_stop_token_budget,
    run_agent,
)
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.environments import CalculatorEnvironment

pytestmark = pytest.mark.live


def _get_api_key() -> str:
    """Get Anthropic API key or skip test."""
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    return key


async def _silent_chunk(event: object) -> None:
    """Discard stream events."""
    pass


def test_token_budget_stops_agent() -> None:
    """Token budget handler stops agent when token limit exceeded.

    Regression: handle_stop was setting stop reason but run_agent wasn't
    appending the stopped state to the returned states list.
    """

    async def run_test() -> None:
        api_key = _get_api_key()
        endpoint = Endpoint.create(
            "anthropic", "claude-sonnet-4-20250514", api_key=api_key, max_tokens=500
        )
        env = CalculatorEnvironment()

        actor = Actor(
            trajectory=Trajectory(messages=[Message(role="user", content="Calculate 123 * 456")]),
            endpoint=endpoint,
            tools=env.get_tools(),
        )

        state = AgentState(
            actor=actor,
            environment=env,
            turn_idx=0,
            stop=None,
        )

        # Low token budget - should stop after first completion
        run_config = RunConfig(
            on_chunk=_silent_chunk,
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_token_budget(100),  # Very low
            ]),
        )

        states = await run_agent(state, run_config)
        final = states[-1]

        # Verify we got completions and used tokens
        completions = final.actor.trajectory.completions
        total_tokens = sum(c.usage.total_tokens if c.usage else 0 for c in completions)

        assert len(completions) >= 1, "Should have at least one completion"
        assert total_tokens > 100, f"Should exceed budget, got {total_tokens}"
        assert final.stop == StopReason.BUDGET_EXCEEDED, (
            f"Expected BUDGET_EXCEEDED, got {final.stop}"
        )

    trio.run(run_test)


def test_cost_budget_stops_agent() -> None:
    """Cost budget handler stops agent when cost limit exceeded.

    Regression: get_model was called with endpoint.model instead of
    endpoint.model_id, so cost was never populated (always $0).
    """

    async def run_test() -> None:
        api_key = _get_api_key()
        endpoint = Endpoint.create(
            "anthropic", "claude-sonnet-4-20250514", api_key=api_key, max_tokens=500
        )
        env = CalculatorEnvironment()

        actor = Actor(
            trajectory=Trajectory(
                messages=[Message(role="user", content="Calculate 2+2, then 3+3, then 4+4")]
            ),
            endpoint=endpoint,
            tools=env.get_tools(),
        )

        state = AgentState(
            actor=actor,
            environment=env,
            turn_idx=0,
            stop=None,
        )

        # Low cost budget - should stop after 1-2 completions
        run_config = RunConfig(
            on_chunk=_silent_chunk,
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_cost_budget(0.005),  # $0.005
            ]),
        )

        states = await run_agent(state, run_config)
        final = states[-1]

        # Verify cost is populated and exceeded budget
        completions = final.actor.trajectory.completions
        total_cost = sum(c.usage.cost.total if c.usage and c.usage.cost else 0 for c in completions)

        assert len(completions) >= 1, "Should have at least one completion"
        assert total_cost > 0, "Cost should be populated (was $0 before fix)"
        assert total_cost > 0.005, f"Should exceed budget, got ${total_cost}"
        assert final.stop == StopReason.BUDGET_EXCEEDED, (
            f"Expected BUDGET_EXCEEDED, got {final.stop}"
        )

    trio.run(run_test)


def test_cost_is_populated() -> None:
    """Verify completion usage includes cost data.

    Regression: get_model(provider, endpoint.model) failed because
    endpoint.model is "anthropic/claude-..." but registry key is just
    "claude-...". Fixed to use endpoint.model_id.
    """

    async def run_test() -> None:
        api_key = _get_api_key()
        endpoint = Endpoint.create(
            "anthropic", "claude-sonnet-4-20250514", api_key=api_key, max_tokens=100
        )
        env = CalculatorEnvironment()

        actor = Actor(
            trajectory=Trajectory(messages=[Message(role="user", content="Calculate 2 + 2")]),
            endpoint=endpoint,
            tools=env.get_tools(),
        )

        state = AgentState(
            actor=actor,
            environment=env,
            turn_idx=0,
            stop=None,
        )

        run_config = RunConfig(
            on_chunk=_silent_chunk,
            handle_stop=handle_stop_max_turns(2),
        )

        states = await run_agent(state, run_config)
        final = states[-1]

        # Check that cost is populated on completions
        for i, c in enumerate(final.actor.trajectory.completions):
            assert c.usage is not None, f"Completion {i} missing usage"
            assert c.usage.cost is not None, f"Completion {i} missing cost"
            assert c.usage.cost.total > 0, (
                f"Completion {i} cost is ${c.usage.cost.total}, expected > 0"
            )

    trio.run(run_test)
