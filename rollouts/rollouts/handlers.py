# Stop condition handlers and run configuration utilities
#
# These are composable functions for controlling agent behavior:
# - Stop conditions (max turns, token budget, cost budget, empty message)
# - Turn warnings
# - Tool reminders
#
# Design: Each handler is a pure function (AgentState) -> AgentState.
# Compose them with compose_handlers() for complex stopping logic.

from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING

from .dtypes import (
    AgentState,
    Message,
    RunConfig,
    StopReason,
)

if TYPE_CHECKING:
    pass


# ── Stop Handlers ────────────────────────────────────────────────────────────


def handle_stop_max_turns(max_turns: int) -> Callable[[AgentState], AgentState]:
    """Stop when max turns reached.

    Args:
        max_turns: Maximum number of turns before stopping

    Returns:
        Handler function that stops when turn_idx >= max_turns

    Example:
        run_config = RunConfig(
            handle_stop=handle_stop_max_turns(5),  # Stop after 5 turns
        )
    """
    assert max_turns > 0, "max_turns must be positive"

    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)
        assert state.turn_idx >= 0

        if state.turn_idx >= max_turns:
            result_state = replace(state, stop=StopReason.MAX_TURNS)
            assert result_state.stop is not None
            return result_state
        return state

    return handler


def handle_stop_token_budget(max_tokens: int) -> Callable[[AgentState], AgentState]:
    """Stop when total tokens exceeds budget.

    Example:
        RunConfig(handle_stop=handle_stop_token_budget(100000))
    """

    def handler(state: AgentState) -> AgentState:
        total_tokens = sum(len(msg.content or "") for msg in state.actor.trajectory.messages)
        if total_tokens >= max_tokens:
            return replace(state, stop=StopReason.MAX_TURNS)  # TODO: Add BUDGET_EXCEEDED
        return state

    return handler


def handle_stop_cost_budget(
    max_cost_usd: float, cost_fn: Callable[[AgentState], float]
) -> Callable[[AgentState], AgentState]:
    """Stop when estimated cost exceeds budget.

    Args:
        max_cost_usd: Maximum cost in USD
        cost_fn: Function that estimates cost from state

    Example:
        def estimate_cost(state):
            # Count tokens, multiply by model pricing
            return tokens * 0.00001

        RunConfig(handle_stop=handle_stop_cost_budget(5.0, estimate_cost))
    """

    def handler(state: AgentState) -> AgentState:
        current_cost = cost_fn(state)
        if current_cost >= max_cost_usd:
            return replace(state, stop=StopReason.MAX_TURNS)  # TODO: Add BUDGET_EXCEEDED
        return state

    return handler


def handle_stop_on_empty_message() -> Callable[[AgentState], AgentState]:
    """Stop when assistant returns empty message (no content, no tool calls).

    This handles cases where the model signals completion by returning an empty
    response (e.g., Claude's end_turn with no content).

    Returns:
        Handler function that stops on empty assistant messages

    Example:
        run_config = RunConfig(
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_on_empty_message(),
            ]),
        )
    """

    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)

        # Check if last message is an empty assistant message
        if state.actor.trajectory.messages:
            last_msg = state.actor.trajectory.messages[-1]
            if (
                last_msg.role == "assistant"
                and not last_msg.content
                and not last_msg.get_tool_calls()
            ):
                result_state = replace(state, stop=StopReason.MAX_TURNS)
                assert result_state.stop is not None
                return result_state

        return state

    return handler


def compose_handlers(
    handlers: list[Callable[[AgentState], AgentState]],
) -> Callable[[AgentState], AgentState]:
    """Compose multiple stop handlers into a single handler.

    Handlers are applied in order. If any handler sets a stop reason, that state
    is returned immediately without calling subsequent handlers.

    Args:
        handlers: List of stop handler functions

    Returns:
        Composed handler function

    Example:
        run_config = RunConfig(
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_on_empty_message(),
            ]),
        )
    """
    assert handlers, "handlers list cannot be empty"
    assert all(callable(h) for h in handlers), "all handlers must be callable"

    def composed_handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)

        current_state = state
        for handler in handlers:
            current_state = handler(current_state)
            # If any handler sets stop, return immediately
            if current_state.stop:
                return current_state

        return current_state

    return composed_handler


# ── Turn Warning Handlers ────────────────────────────────────────────────────


def inject_turn_warning(max_turns: int, warning_at: int = 2) -> Callable[[AgentState], AgentState]:
    """Inject warning when N turns remaining.

    Args:
        max_turns: Total turns available
        warning_at: Warn when this many turns remaining (default: 2)

    Returns:
        Handler function that injects warning message

    Example:
        run_config = RunConfig(
            on_step_start=inject_turn_warning(max_turns=5, warning_at=2),
        )
    """
    assert max_turns > 0
    assert warning_at > 0
    assert warning_at < max_turns

    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)
        assert state.turn_idx >= 0

        turns_left = max_turns - state.turn_idx
        if turns_left == warning_at:
            warning = Message(
                role="user",
                content=f"⚠️ You have {warning_at} turns remaining. Please complete your task quickly.",
            )
            # TODO(replace-chains): This nested replace pattern appears ~5 times in codebase:
            # replace(state, actor=replace(state.actor, trajectory=...))
            # A helper like `update_trajectory(state, messages)` would reduce repetition.
            # See also: agents.py:382, runner.py:316, dtypes.py:1144 (triple nested)
            # Low priority - it's one repeated idiom, not blocking.
            new_trajectory = replace(
                state.actor.trajectory, messages=state.actor.trajectory.messages + [warning]
            )
            result_state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))
            assert result_state is not None
            return result_state
        return state

    return handler


# ── Tool Handlers ────────────────────────────────────────────────────────────


async def inject_tool_reminder(state: AgentState, run_config: "RunConfig") -> AgentState:
    """Remind the agent to use tools when no tool call was made."""
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None

    reminder = Message(
        role="user",
        content="Please use the available tools to complete the task. What calculation would you like to perform?",
    )
    new_trajectory = replace(
        state.actor.trajectory, messages=state.actor.trajectory.messages + [reminder]
    )
    result_state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))
    assert result_state is not None
    return result_state
