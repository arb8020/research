import json
from dataclasses import dataclass

import trio

from rollouts.agents import Actor, AgentState
from rollouts.agents.session_runtime import (
    ensure_persisted_session,
    state_to_persisted_trajectory,
)
from rollouts.cli import cmd_send
from rollouts.core import (
    Endpoint,
    EnvironmentConfig,
    EnvironmentResumeMode,
    Message,
    PendingInput,
    SessionHandle,
    SessionStatus,
    Trajectory,
    TrajectoryAnnotations,
    TrajectoryEnvironment,
    TrajectorySession,
)
from rollouts.store import FileSessionStore


def test_session_handle_to_trajectory_preserves_session_metadata() -> None:
    session = SessionHandle.from_trajectory(
        Trajectory(
            messages=[Message(role="user", content="hello")],
            annotations=TrajectoryAnnotations(reward=1.5),
            session=TrajectorySession(
                session_id="sess_123",
                parent_id="parent_456",
                branch_point=7,
                endpoint=Endpoint.from_legacy(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
                status=SessionStatus.WAITING.value,
                created_at="2026-03-08T12:00:00",
                updated_at="2026-03-08T12:30:00",
                tags={"task": "refactor"},
                vcs={"type": "git", "revision": "abc123"},
            ),
            environment=TrajectoryEnvironment(
                kind="coding",
                config={"confirm_tools": True},
                state={"cwd": "/tmp/project"},
                resume_mode=EnvironmentResumeMode.COLD,
            ),
        )
    )

    trajectory = session.to_trajectory()

    assert trajectory.messages == session.messages
    assert trajectory.annotations.reward == 1.5
    assert trajectory.session.session_id == "sess_123"
    assert trajectory.session.parent_id == "parent_456"
    assert trajectory.session.branch_point == 7
    assert trajectory.session.endpoint is not None
    assert trajectory.session.endpoint.provider == "anthropic"
    assert trajectory.session.status == SessionStatus.WAITING.value
    assert trajectory.session.tags == {"task": "refactor"}
    assert trajectory.environment is not None
    assert trajectory.environment.kind == "coding"
    assert trajectory.environment.config == {"confirm_tools": True}
    assert trajectory.environment.state == {"cwd": "/tmp/project"}
    assert trajectory.environment.resume_mode == EnvironmentResumeMode.COLD


def test_session_handle_from_trajectory_uses_nested_trajectory_bundles() -> None:
    trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        annotations=TrajectoryAnnotations(reward={"score": 0.75}),
        session=TrajectorySession(
            session_id="sess_nested",
            parent_id="parent_nested",
            branch_point=3,
            endpoint=Endpoint.from_legacy(provider="openai", model="gpt-4o"),
            status=SessionStatus.COMPLETED.value,
            created_at="2026-03-08T13:00:00",
            updated_at="2026-03-08T13:15:00",
            tags={"mode": "detached"},
            vcs={"type": "git", "revision": "def456"},
        ),
        environment=TrajectoryEnvironment(
            kind="terminal_bench",
            config={"task_id": "hello-world"},
            state={"_env_ref": "live-ref-placeholder"},
            resume_mode=EnvironmentResumeMode.WARM,
        ),
    )

    endpoint = Endpoint.from_legacy(provider="openai", model="gpt-4o")
    session = SessionHandle.from_trajectory(trajectory, endpoint=endpoint)

    assert session.session_id == "sess_nested"
    assert session.parent_id == "parent_nested"
    assert session.branch_point == 3
    assert session.status == SessionStatus.COMPLETED
    assert session.reward == {"score": 0.75}
    assert session.endpoint.provider == "openai"
    assert session.environment.type == "terminal_bench"
    assert session.environment.config == {"task_id": "hello-world"}
    assert session.environment_state == {"_env_ref": "live-ref-placeholder"}
    assert session.tags == {"mode": "detached"}
    assert session.vcs == {"type": "git", "revision": "def456"}


def test_trajectory_environment_from_session_parts_marks_live_refs_warm() -> None:
    environment = EnvironmentConfig(type="terminal_bench", config={"task_id": "hello-world"})

    bundle = TrajectoryEnvironment.from_session_parts(
        environment,
        {"_env_ref": "live-ref-placeholder"},
    )

    assert bundle.kind == "terminal_bench"
    assert bundle.config == {"task_id": "hello-world"}
    assert bundle.state == {"_env_ref": "live-ref-placeholder"}
    assert bundle.resume_mode == EnvironmentResumeMode.WARM


def test_file_session_store_get_trajectory_uses_canonical_adapter(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        endpoint = Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-20250514")

        session = await store.create(
            endpoint=endpoint,
            environment=EnvironmentConfig(type="coding", config={"confirm_tools": True}),
            parent_id="parent_store",
            branch_point=2,
            tags={"task": "store"},
            vcs={"type": "git", "revision": "abc123"},
        )
        await store.append_message(session.session_id, Message(role="user", content="hello"))
        await store.update(
            session.session_id,
            status=SessionStatus.WAITING,
            environment_state={"cwd": "/tmp/project"},
            reward=2.0,
        )

        trajectory, err = await store.get_trajectory(session.session_id)

        assert err is None
        assert trajectory is not None
        assert [msg.role for msg in trajectory.messages] == ["user"]
        assert trajectory.annotations.reward == 2.0
        assert trajectory.session.session_id == session.session_id
        assert trajectory.session.parent_id == "parent_store"
        assert trajectory.session.branch_point == 2
        assert trajectory.session.status == SessionStatus.WAITING.value
        assert trajectory.session.tags == {"task": "store"}
        assert trajectory.environment is not None
        assert trajectory.environment.kind == "coding"
        assert trajectory.environment.config == {"confirm_tools": True}
        assert trajectory.environment.state == {"cwd": "/tmp/project"}

    trio.run(_test)


def test_file_session_store_save_trajectory_round_trips_session_metadata(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        trajectory = Trajectory(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            annotations=TrajectoryAnnotations(reward=3.0),
            session=TrajectorySession(
                session_id=None,
                parent_id="parent_save",
                branch_point=2,
                endpoint=Endpoint.from_legacy(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
                status=SessionStatus.PENDING.value,
                tags={"origin": "trajectory"},
                vcs={"type": "git", "revision": "abc123"},
            ),
            environment=TrajectoryEnvironment(
                kind="coding",
                config={"confirm_tools": True},
                state={"cwd": "/tmp/project"},
                resume_mode=EnvironmentResumeMode.COLD,
            ),
        )

        saved, err = await store.save_trajectory(trajectory)

        assert err is None
        assert saved is not None
        assert saved.session_id
        assert saved.parent_id == "parent_save"
        assert saved.branch_point == 2
        assert saved.endpoint.provider == "anthropic"
        assert saved.environment.type == "coding"
        assert saved.environment_state == {"cwd": "/tmp/project"}
        assert saved.reward == 3.0
        assert saved.tags == {"origin": "trajectory"}
        assert [msg.role for msg in saved.messages] == ["user", "assistant"]

        loaded, load_err = await store.get_trajectory(saved.session_id)
        assert load_err is None
        assert loaded is not None
        assert loaded.session.endpoint is not None
        assert loaded.session.endpoint.provider == "anthropic"
        assert loaded.session.parent_id == "parent_save"
        assert loaded.environment is not None
        assert loaded.environment.kind == "coding"

    trio.run(_test)


def test_file_session_store_round_trips_control_state(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        session = await store.create(
            endpoint=Endpoint.from_legacy(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            ),
            environment=EnvironmentConfig(type="none"),
        )

        await store.write_pending_input(
            session.session_id,
            {"type": "no_tools", "last_message": "Need your input"},
        )
        _, err = await store.enqueue_message(
            session.session_id,
            Message(role="user", content="queued follow-up"),
        )
        assert err is None

        loaded, err = await store.get(session.session_id)
        assert err is None
        assert loaded is not None
        assert loaded.pending_input == PendingInput(
            kind="no_tools",
            metadata={"last_message": "Need your input"},
        )
        assert [msg.content for msg in loaded.queued_messages] == ["queued follow-up"]

        pending = await store.read_pending_input(session.session_id)
        assert pending == {"type": "no_tools", "last_message": "Need your input"}

        queued, err = await store.consume_queued_message(session.session_id)
        assert err is None
        assert queued is not None
        assert queued.content == "queued follow-up"

        loaded_after_consume, err = await store.get(session.session_id)
        assert err is None
        assert loaded_after_consume is not None
        assert loaded_after_consume.pending_input == loaded.pending_input
        assert loaded_after_consume.queued_messages == ()

    trio.run(_test)


def test_file_session_store_optionally_mirrors_atif(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(
            base_dir=tmp_path / "sessions",
            atif_filename="trajectory.json",
            atif_output_path=tmp_path / "trajectory.json",
        )
        session = await store.create(
            endpoint=Endpoint.from_legacy(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            ),
            environment=EnvironmentConfig(type="coding", config={"confirm_tools": True}),
        )

        await store.append_message(session.session_id, Message(role="user", content="hello"))
        await store.append_message(
            session.session_id,
            Message(role="assistant", content="hi there"),
        )

        atif_path = tmp_path / "sessions" / session.session_id / "trajectory.json"
        assert atif_path.exists()

        payload = json.loads(atif_path.read_text())
        assert payload["schema_version"] == "ATIF-v1.6"
        assert payload["session_id"] == session.session_id
        assert payload["agent"]["name"] == "rollouts"
        assert [step["source"] for step in payload["steps"]] == ["user", "agent"]

        root_payload = json.loads((tmp_path / "trajectory.json").read_text())
        assert root_payload["session_id"] == session.session_id

    trio.run(_test)


def test_cmd_send_queues_message_on_session_handle(tmp_path) -> None:
    @dataclass
    class StubConfig:
        session: str | None = None
        bootstrap_input: str | None = None
        detached: bool = False

    store = FileSessionStore(base_dir=tmp_path / "sessions")

    async def _setup() -> str:
        session = await store.create(
            endpoint=Endpoint.from_legacy(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            ),
            environment=EnvironmentConfig(type="none"),
        )
        await store.write_pending_input(
            session.session_id,
            {"type": "no_tools", "last_message": "Waiting"},
        )
        return session.session_id

    session_id = trio.run(_setup)
    config = StubConfig()

    result = cmd_send(config, store, session_id, "resume with this")

    assert result == -1
    assert config.session == session_id
    assert config.detached is True
    assert config.bootstrap_input is None

    async def _verify() -> None:
        loaded, err = await store.get(session_id)
        assert err is None
        assert loaded is not None
        assert loaded.pending_input is None
        assert [msg.content for msg in loaded.queued_messages] == ["resume with this"]
        assert loaded.status == SessionStatus.PENDING

    trio.run(_verify)


def test_state_to_persisted_trajectory_preserves_runtime_annotations() -> None:
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(
                messages=[Message(role="user", content="hello")],
                rewards=2.5,
                group=3,
                replica=1,
                advantages=0.75,
                metadata={"task": "refactor"},
                annotations=TrajectoryAnnotations(reward={"score": 0.9}),
            ),
            endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-20250514"),
            tools=[],
        ),
        environment=None,
        parent_session_id="parent_live",
        branch_point=4,
        confirm_tools=True,
    )

    trajectory = state_to_persisted_trajectory(state)

    assert trajectory.messages == [Message(role="user", content="hello")]
    assert trajectory.rewards == 2.5
    assert trajectory.group == 3
    assert trajectory.replica == 1
    assert trajectory.advantages == 0.75
    assert trajectory.metadata == {"task": "refactor"}
    assert trajectory.annotations.reward == {"score": 0.9}
    assert trajectory.session.parent_id == "parent_live"
    assert trajectory.session.branch_point == 4
    assert trajectory.session.endpoint is not None
    assert trajectory.environment is not None
    assert trajectory.environment.kind == "none"
    assert trajectory.environment.config == {"confirm_tools": True}


def test_ensure_persisted_session_creates_new_session(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        state = AgentState(
            actor=Actor(
                trajectory=Trajectory(messages=[Message(role="user", content="hello")]),
                endpoint=Endpoint.from_legacy(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
                tools=[],
            ),
            environment=None,
            confirm_tools=True,
        )

        ensured = await ensure_persisted_session(state, store)

        assert ensured.session_id is not None
        session, err = await store.get(ensured.session_id)
        assert err is None
        assert session is not None
        assert [msg.role for msg in session.messages] == ["user"]
        assert session.environment.type == "none"
        assert session.environment.config == {"confirm_tools": True}

    trio.run(_test)


def test_ensure_persisted_session_appends_gap_messages(tmp_path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        session = await store.create(
            endpoint=Endpoint.from_legacy(provider="anthropic", model="claude-sonnet-4-20250514"),
            environment=EnvironmentConfig(type="none", config={"confirm_tools": False}),
        )
        await store.append_message(session.session_id, Message(role="user", content="hello"))

        state = AgentState(
            actor=Actor(
                trajectory=Trajectory(
                    messages=[
                        Message(role="user", content="hello"),
                        Message(role="assistant", content="hi"),
                    ]
                ),
                endpoint=session.endpoint,
                tools=[],
            ),
            environment=None,
            session_id=session.session_id,
        )

        ensured = await ensure_persisted_session(state, store)

        assert ensured.session_id == session.session_id
        loaded, err = await store.get(session.session_id)
        assert err is None
        assert loaded is not None
        assert [msg.role for msg in loaded.messages] == ["user", "assistant"]

    trio.run(_test)
