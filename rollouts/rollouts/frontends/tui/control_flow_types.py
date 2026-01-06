"""
Control flow types for InteractiveAgentRunner.

These types make the control flow explicit by returning what happened
rather than using flags and mutation.

See REFACTOR_DESIGN.md for the full design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ...dtypes import AgentState


# ─── Input Phase Results ───────────────────────────────────────────────────────
# What happened when we waited for user input?


@dataclass(frozen=True)
class InputExit:
    """User wants to exit entirely (Ctrl+C)."""

    pass


@dataclass(frozen=True)
class InputContinue:
    """Slash command handled, loop back for more input.

    Used when a command just displays info (e.g., /model with no args)
    or makes a change that doesn't affect agent state.
    """

    message: str | None = None


@dataclass(frozen=True)
class InputNewState:
    """Slash command changed state (model/env/session).

    The caller should use the new state going forward.
    Used by /model, /thinking, /slice, /env.
    """

    state: AgentState
    message: str | None = None


@dataclass(frozen=True)
class InputMessage:
    """User entered a message to send to LLM."""

    text: str


# Union of all input results
InputResult = InputExit | InputContinue | InputNewState | InputMessage


# ─── Agent Phase Results ───────────────────────────────────────────────────────
# What happened when we ran the agent?


@dataclass(frozen=True)
class AgentCompleted:
    """Agent finished normally (task complete or no tools).

    The agent loop exited with TASK_COMPLETED or because handle_no_tool
    indicated we should wait for more input.
    """

    states: list[AgentState]


@dataclass(frozen=True)
class AgentInterrupted:
    """User pressed Escape to interrupt the agent.

    The agent was cancelled mid-run. May have a partial response
    that was being streamed when interrupted.
    """

    states: list[AgentState]
    partial_response: str | None = None


@dataclass(frozen=True)
class AgentExited:
    """User pressed Ctrl+C to exit entirely.

    The outer cancel scope was cancelled, meaning the user wants
    to exit the TUI completely.
    """

    states: list[AgentState]


@dataclass(frozen=True)
class AgentError:
    """Recoverable error during agent run.

    The agent encountered an error that we can recover from by
    showing a message and letting the user try again.

    error_kind values:
    - "context_too_long": Context exceeded model's limit
    - "oauth_expired": OAuth token expired, needs re-auth
    """

    states: list[AgentState]
    error: Exception
    error_kind: str


# Union of all agent outcomes
AgentOutcome = AgentCompleted | AgentInterrupted | AgentExited | AgentError


# ─── Helper to get final state from any outcome ────────────────────────────────


def get_final_state(outcome: AgentOutcome) -> AgentState | None:
    """Extract the final state from any AgentOutcome."""
    states = outcome.states
    return states[-1] if states else None
