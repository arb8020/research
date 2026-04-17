"""Helpers for reconciling live agent state with persisted sessions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..core import (
    Environment,
    EnvironmentConfig,
    Trajectory,
    TrajectoryEnvironment,
    TrajectorySession,
)
from .types import AgentState

if TYPE_CHECKING:
    from ..store import SessionStore


# TODO(session-harness-decoupling): the session here is the durable log of what
# happened — messages, tool calls, environment config. The harness (run_agent)
# is the loop that drives turns. These are partially decoupled already:
# ensure_persisted_session writes messages per-turn, and the session can be
# reconstructed from the store on wake.
#
# The remaining coupling: environment state is stored as a serialized blob on
# the session, meaning the session knows about environment internals. The cleaner
# model (per the Anthropic managed-agents article) is that the session stores
# only the environment *type* and *config* (enough to reprovision), and the
# environment is reprovisioned fresh on wake rather than deserialized from state.
#
# Concretely: environment_to_session_config currently stores only class name,
# which is correct. But state_to_persisted_trajectory threads environment_bundle.state
# through, meaning deserialized mutable state can end up in the session log.
# On wake, callers should call environment.initialize() rather than
# environment.deserialize(session.environment.state).
#
# This also applies to the external agent path: once run_external_agent has a
# session, ensure_persisted_session should work identically for both paths.
def environment_to_session_config(
    environment: Environment | None,
    confirm_tools: bool = False,
) -> EnvironmentConfig:
    """Convert a live environment into durable session config."""
    config = {"confirm_tools": confirm_tools}
    if environment is None:
        return EnvironmentConfig(type="none", config=config)

    # Use class name as type; richer mutable state is stored in environment_state.
    return EnvironmentConfig(type=type(environment).__name__, config=config)


def state_to_persisted_trajectory(state: AgentState) -> Trajectory:
    """Project live agent state into the canonical persisted trajectory shape."""
    source = state.actor.trajectory
    environment_bundle = source.environment
    if state.environment is not None:
        environment = TrajectoryEnvironment.from_session_parts(
            environment_to_session_config(state.environment, state.confirm_tools),
            environment_bundle.state if environment_bundle is not None else None,
        )
    elif environment_bundle is not None:
        environment = environment_bundle
    else:
        environment = TrajectoryEnvironment.from_session_parts(
            environment_to_session_config(None, state.confirm_tools)
        )

    return Trajectory(
        completions=list(source.completions),
        messages=list(source.messages),
        metadata=dict(source.metadata),
        session=TrajectorySession(
            session_id=source.session.session_id,
            parent_id=state.parent_session_id or source.session.parent_id,
            branch_point=state.branch_point
            if state.branch_point is not None
            else source.session.branch_point,
            endpoint=state.actor.endpoint,
            stop_reason=source.session.stop_reason,
            created_at=source.session.created_at,
            updated_at=source.session.updated_at,
            tags=dict(source.session.tags),
            vcs=source.session.vcs,
        ),
        environment=environment,
    )


async def ensure_persisted_session(
    state: AgentState,
    session_store: SessionStore | None,
) -> AgentState:
    """Ensure the current state has a persisted session and synced message history.

    Session refactor (1b): after ensuring persistence, state.leaf_id is set
    to the session's current tail so subsequent appends from the loop carry
    explicit parent_id.
    """
    if session_store is None:
        return state

    if not state.session_id:
        session_trajectory = state_to_persisted_trajectory(state)
        session, err = await session_store.save_trajectory(session_trajectory)
        if err is not None or session is None:
            raise RuntimeError(err or "Failed to create session")
        # After save_trajectory, the last message's id is the leaf.
        leaf_id: str | None = None
        if session.messages:
            leaf_id = session.messages[-1].id
        return replace(state, session_id=session.session_id, leaf_id=leaf_id)

    session, err = await session_store.get(state.session_id)
    if err is not None or session is None:
        raise RuntimeError(err or f"Session not found: {state.session_id}")

    persisted_count = len(session.messages)
    current_messages = state.actor.trajectory.messages
    leaf_id = session.messages[-1].id if session.messages else None

    if len(current_messages) > persisted_count:
        for msg in current_messages[persisted_count:]:
            # Thread leaf_id through: each append extends the prior one.
            msg_with_parent = msg if msg.parent_id is not None else replace(msg, parent_id=leaf_id)
            stored = await session_store.append_message(state.session_id, msg_with_parent)
            leaf_id = stored.id

    return replace(state, leaf_id=leaf_id)
