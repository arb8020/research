"""Tests for session switching (used by /slice and /env slash commands).

Regression test for: switch_session() not updating runner state properly
after /slice creates a child session.
"""

from dataclasses import dataclass, field

import trio

from rollouts.agents import Actor, AgentState
from rollouts.dtypes import Endpoint, EnvironmentConfig, Message, Trajectory
from rollouts.slice import run_slice_command
from rollouts.store import FileSessionStore


@dataclass
class MockRunner:
    """Mock runner with switch_session implementation matching the real one."""

    endpoint: Endpoint
    session_store: FileSessionStore
    session_id: str | None = None
    initial_trajectory: Trajectory = field(default_factory=lambda: Trajectory(messages=[]))
    _current_trajectory: Trajectory | None = None
    _session_switched: bool = False  # New flag
    environment: None = None

    async def switch_session(self, new_session_id: str) -> bool:
        """Switch to a different session - must update endpoint and set flag."""
        if not self.session_store:
            return False

        session, err = await self.session_store.get(new_session_id)
        if err or not session:
            return False

        self.session_id = new_session_id

        # Critical: update endpoint from the loaded session
        self.endpoint = session.endpoint

        # Critical: update trajectory
        self.initial_trajectory = Trajectory(messages=session.messages)
        self._current_trajectory = self.initial_trajectory

        # Critical: signal main loop to rebuild state
        self._session_switched = True

        return True

    def rebuild_state_after_switch(self, user_input: str) -> AgentState:
        """Simulate what the main loop does after detecting _session_switched."""
        if not self._session_switched:
            raise ValueError("_session_switched not set")

        self._session_switched = False

        new_trajectory = Trajectory(
            messages=list(self.initial_trajectory.messages)
            + [Message(role="user", content=user_input)]
        )
        return AgentState(
            actor=Actor(
                trajectory=new_trajectory,
                endpoint=self.endpoint,
                tools=[],
            ),
            environment=None,
            session_id=self.session_id,
        )


def test_switch_session_updates_endpoint() -> None:
    """Verify switch_session updates endpoint from the child session."""

    async def _test() -> None:
        session_store = FileSessionStore()

        # Create parent session with anthropic endpoint
        parent_endpoint = Endpoint(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
        )

        parent = await session_store.create(
            endpoint=parent_endpoint,
            environment=EnvironmentConfig(type="none"),
        )

        # Add messages
        await session_store.append_message(parent.session_id, Message(role="user", content="Hello"))
        await session_store.append_message(
            parent.session_id, Message(role="assistant", content="Hi!")
        )

        # Reload to get messages
        parent, _ = await session_store.get(parent.session_id)

        # Create child via slice (simulating /slice command)
        child, err = await run_slice_command(
            session=parent,
            spec="0:2",
            endpoint=parent_endpoint,
            session_store=session_store,
        )
        assert not err, f"Slice failed: {err}"

        # Create runner with DIFFERENT endpoint (simulates /model before /slice)
        runner = MockRunner(
            endpoint=Endpoint(provider="openai", model="gpt-4o", max_tokens=4000),
            session_store=session_store,
            session_id=parent.session_id,
        )

        # Before switch - runner has openai endpoint
        assert runner.endpoint.provider == "openai"
        assert runner.endpoint.model == "gpt-4o"
        assert not runner._session_switched

        # Switch to child session
        success = await runner.switch_session(child.session_id)
        assert success, "switch_session failed"

        # After switch - runner should have child's endpoint AND flag set
        assert runner.endpoint.provider == child.endpoint.provider
        assert runner.endpoint.model == child.endpoint.model
        assert runner._session_switched, "_session_switched flag not set!"

    trio.run(_test)


def test_switch_session_sets_flag() -> None:
    """Verify switch_session sets _session_switched flag."""

    async def _test() -> None:
        session_store = FileSessionStore()
        endpoint = Endpoint(provider="anthropic", model="claude-sonnet-4-20250514")

        parent = await session_store.create(
            endpoint=endpoint,
            environment=EnvironmentConfig(type="none"),
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="Hi"))
        parent, _ = await session_store.get(parent.session_id)

        child, _ = await run_slice_command(
            session=parent,
            spec="0:1",
            endpoint=endpoint,
            session_store=session_store,
        )

        runner = MockRunner(
            endpoint=endpoint,
            session_store=session_store,
            session_id=parent.session_id,
        )

        assert not runner._session_switched, "Flag should start False"

        await runner.switch_session(child.session_id)

        assert runner._session_switched, "Flag should be True after switch"

    trio.run(_test)


def test_rebuild_state_uses_new_endpoint() -> None:
    """Verify the main loop rebuild uses the updated endpoint after switch."""

    async def _test() -> None:
        session_store = FileSessionStore()

        parent_endpoint = Endpoint(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
        )

        parent = await session_store.create(
            endpoint=parent_endpoint,
            environment=EnvironmentConfig(type="none"),
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="Hello"))
        parent, _ = await session_store.get(parent.session_id)

        child, _ = await run_slice_command(
            session=parent,
            spec="0:1",
            endpoint=parent_endpoint,
            session_store=session_store,
        )

        # Runner starts with different endpoint
        runner = MockRunner(
            endpoint=Endpoint(provider="openai", model="gpt-4o"),
            session_store=session_store,
            session_id=parent.session_id,
        )

        # Switch to child
        await runner.switch_session(child.session_id)

        # Simulate main loop rebuilding state
        new_state = runner.rebuild_state_after_switch("Next message")

        # The rebuilt state should use the child's endpoint
        assert new_state.actor.endpoint.provider == "anthropic"
        assert new_state.actor.endpoint.model == "claude-sonnet-4-20250514"

        # And the flag should be consumed
        assert not runner._session_switched

    trio.run(_test)


def test_switch_session_updates_trajectory() -> None:
    """Verify switch_session updates trajectory from the child session."""

    async def _test() -> None:
        session_store = FileSessionStore()
        endpoint = Endpoint(provider="anthropic", model="claude-sonnet-4-20250514")

        # Create parent with 4 messages
        parent = await session_store.create(
            endpoint=endpoint,
            environment=EnvironmentConfig(type="none"),
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="1"))
        await session_store.append_message(
            parent.session_id, Message(role="assistant", content="2")
        )
        await session_store.append_message(parent.session_id, Message(role="user", content="3"))
        await session_store.append_message(
            parent.session_id, Message(role="assistant", content="4")
        )

        parent, _ = await session_store.get(parent.session_id)
        assert len(parent.messages) == 4

        # Slice to get first 2 messages
        child, err = await run_slice_command(
            session=parent,
            spec="0:2",
            endpoint=endpoint,
            session_store=session_store,
        )
        assert not err

        # Create runner
        runner = MockRunner(
            endpoint=endpoint,
            session_store=session_store,
            session_id=parent.session_id,
            initial_trajectory=Trajectory(messages=parent.messages),
        )

        # Before switch - 4 messages
        assert len(runner.initial_trajectory.messages) == 4

        # Switch to child
        await runner.switch_session(child.session_id)

        # After switch - should have child's messages
        child_loaded, _ = await session_store.get(child.session_id)
        assert len(runner.initial_trajectory.messages) == len(child_loaded.messages)

    trio.run(_test)
