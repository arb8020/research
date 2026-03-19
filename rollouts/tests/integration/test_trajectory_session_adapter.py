import json
from pathlib import Path

import trio

from rollouts.agents import Actor, AgentState
from rollouts.agents.session_runtime import (
    ensure_persisted_session,
    state_to_persisted_trajectory,
)
from rollouts.core import (
    Endpoint,
    EnvironmentConfig,
    EnvironmentResumeMode,
    Message,
    StopReason,
    Trajectory,
    TrajectoryEnvironment,
    TrajectorySession,
)
from rollouts.store import FileSessionStore


def test_trajectory_convenience_accessors_preserve_session_metadata() -> None:
    trajectory = Trajectory(
        messages=[Message(role="user", content="hello")],
        session=TrajectorySession(
            session_id="sess_123",
            parent_id="parent_456",
            branch_point=7,
            endpoint=Endpoint.from_legacy(
                provider="anthropic",
                model="claude-sonnet-4-20250514",
            ),
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

    assert trajectory.session_id == "sess_123"
    assert trajectory.parent_id == "parent_456"
    assert trajectory.branch_point == 7
    assert trajectory.endpoint.provider == "anthropic"
    assert trajectory.status == "pending"
    assert trajectory.tags == {"task": "refactor"}
    assert trajectory.environment is not None
    assert trajectory.environment.kind == "coding"
    assert trajectory.environment.config == {"confirm_tools": True}
    assert trajectory.environment.state == {"cwd": "/tmp/project"}
    assert trajectory.environment.resume_mode == EnvironmentResumeMode.COLD
    assert trajectory.environment_config().type == "coding"
    assert trajectory.environment_state() == {"cwd": "/tmp/project"}


def test_trajectory_nested_bundles_stay_accessible() -> None:
    trajectory = Trajectory(
        messages=[Message(role="assistant", content="done")],
        session=TrajectorySession(
            session_id="sess_nested",
            parent_id="parent_nested",
            branch_point=3,
            endpoint=Endpoint.from_legacy(provider="openai", model="gpt-4o"),
            stop_reason=StopReason.TASK_COMPLETED,
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

    assert trajectory.session_id == "sess_nested"
    assert trajectory.parent_id == "parent_nested"
    assert trajectory.branch_point == 3
    assert trajectory.status == "completed"
    assert trajectory.endpoint.provider == "openai"
    assert trajectory.environment_config().type == "terminal_bench"
    assert trajectory.environment_config().config == {"task_id": "hello-world"}
    assert trajectory.environment_state() == {"_env_ref": "live-ref-placeholder"}
    assert trajectory.tags == {"mode": "detached"}
    assert trajectory.vcs == {"type": "git", "revision": "def456"}


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


def test_file_session_store_get_trajectory_uses_canonical_adapter(tmp_path: Path) -> None:
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
            stop_reason=None,
            environment_state={"cwd": "/tmp/project"},
        )

        trajectory, err = await store.get_trajectory(session.session_id)

        assert err is None
        assert trajectory is not None
        assert [msg.role for msg in trajectory.messages] == ["user"]
        assert trajectory.session.session_id == session.session_id
        assert trajectory.session.parent_id == "parent_store"
        assert trajectory.session.branch_point == 2
        assert trajectory.session.stop_reason is None
        assert trajectory.session.tags == {"task": "store"}
        assert trajectory.environment is not None
        assert trajectory.environment.kind == "coding"
        assert trajectory.environment.config == {"confirm_tools": True}
        assert trajectory.environment.state == {"cwd": "/tmp/project"}

    trio.run(_test)


def test_file_session_store_save_trajectory_round_trips_session_metadata(tmp_path: Path) -> None:
    async def _test() -> None:
        store = FileSessionStore(base_dir=tmp_path / "sessions")
        trajectory = Trajectory(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content="hi"),
            ],
            session=TrajectorySession(
                session_id=None,
                parent_id="parent_save",
                branch_point=2,
                endpoint=Endpoint.from_legacy(
                    provider="anthropic",
                    model="claude-sonnet-4-20250514",
                ),
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
        assert saved.environment_config().type == "coding"
        assert saved.environment_state() == {"cwd": "/tmp/project"}
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


def test_file_session_store_optionally_mirrors_atif(tmp_path: Path) -> None:
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


def test_state_to_persisted_trajectory_preserves_runtime_checkpoint_state() -> None:
    state = AgentState(
        actor=Actor(
            trajectory=Trajectory(
                messages=[Message(role="user", content="hello")],
                metadata={"task": "refactor"},
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
    assert trajectory.metadata == {"task": "refactor"}
    assert trajectory.session.parent_id == "parent_live"
    assert trajectory.session.branch_point == 4
    assert trajectory.session.endpoint is not None
    assert trajectory.environment is not None
    assert trajectory.environment.kind == "none"
    assert trajectory.environment.config == {"confirm_tools": True}


def test_trajectory_from_dict_ignores_removed_reward_fields() -> None:
    trajectory = Trajectory.from_dict({
        "messages": [{"role": "assistant", "content": "done"}],
        "metadata": {"source": "legacy"},
        "rewards": 2.5,
        "group": 3,
        "replica": 1,
        "advantages": 0.75,
        "annotations": {"reward": {"score": 0.9}},
        "session": {
            "session_id": "sess_legacy",
            "status": "pending",
        },
    })

    assert trajectory.metadata == {"source": "legacy"}
    assert trajectory.session.session_id == "sess_legacy"
    assert trajectory.status == "pending"


def test_ensure_persisted_session_creates_new_session(tmp_path: Path) -> None:
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
        assert session.environment_config().type == "none"
        assert session.environment_config().config == {"confirm_tools": True}

    trio.run(_test)


def test_ensure_persisted_session_appends_gap_messages(tmp_path: Path) -> None:
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
