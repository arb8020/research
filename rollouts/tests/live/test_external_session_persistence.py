"""Live test: per-turn session persistence for external agent path.

Verifies that messages written by trajectory_from_claude_code appear in the
session store as they arrive, not just after the run completes.

Requires: claude CLI installed and ANTHROPIC_API_KEY set.
"""

import json
import os
from pathlib import Path

import pytest

from rollouts.agents import RunConfig, stdout_handler
from rollouts.environments.local_workspace_resource import LocalWorkspaceResource
from rollouts.eval.external_attempts import trajectory_from_claude_code
from rollouts.store import FileSessionStore, generate_session_id

pytestmark = pytest.mark.live


@pytest.mark.trio
async def test_trajectory_from_claude_code_writes_messages_to_session_store(
    tmp_path: Path,
) -> None:
    """Messages appear in the session store before trajectory_from_claude_code returns."""
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    store_dir = tmp_path / "sessions"
    store = FileSessionStore(base_dir=store_dir)

    # Caller creates the session before launching the agent.
    session_id = generate_session_id()
    session_dir = store_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(json.dumps({"session_id": session_id, "tags": {}}))

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = LocalWorkspaceResource.from_existing(workspace_dir)

    run_config = RunConfig(
        on_chunk=stdout_handler,
        session_store=store,
        session_id=session_id,
    )

    artifact = await trajectory_from_claude_code(
        "Print the word HELLO and nothing else.",
        "live-test-1",
        {},
        workspace=workspace,
        run_config=run_config,
        model="claude-haiku-4-5",
        timeout_seconds=120.0,
    )

    # The run completed — now verify messages are in the store.
    trajectory, err = await store.get_trajectory(session_id)
    assert err is None, f"Failed to load session: {err}"
    assert trajectory is not None
    assert len(trajectory.messages) > 0, "No messages written to session store"

    # Messages in store should match messages in artifact.
    assert len(trajectory.messages) == len(artifact.trajectory.messages), (
        f"Store has {len(trajectory.messages)} messages "
        f"but artifact has {len(artifact.trajectory.messages)}"
    )

    # At least one assistant message.
    roles = [m.role for m in trajectory.messages]
    assert "assistant" in roles, f"No assistant message in store, got roles: {roles}"


@pytest.mark.trio
async def test_trajectory_from_claude_code_stores_cli_session_id(
    tmp_path: Path,
) -> None:
    """CLI native session ID is stored in the harness session tags on first discovery."""
    import shutil

    if shutil.which("claude") is None:
        pytest.skip("claude CLI not installed")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    store_dir = tmp_path / "sessions"
    store = FileSessionStore(base_dir=store_dir)

    session_id = generate_session_id()
    session_dir = store_dir / session_id
    session_dir.mkdir(parents=True)
    (session_dir / "session.json").write_text(json.dumps({"session_id": session_id, "tags": {}}))

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    workspace = LocalWorkspaceResource.from_existing(workspace_dir)

    run_config = RunConfig(
        on_chunk=stdout_handler,
        session_store=store,
        session_id=session_id,
    )

    await trajectory_from_claude_code(
        "Print the word HELLO and nothing else.",
        "live-test-2",
        {},
        workspace=workspace,
        run_config=run_config,
        model="claude-haiku-4-5",
        timeout_seconds=120.0,
    )

    trajectory, err = await store.get_trajectory(session_id)
    assert err is None
    assert trajectory is not None

    # CLI session ID should be in the session tags.
    tags = trajectory.session.tags or {}
    assert "cli_session_id" in tags, f"cli_session_id not found in session tags: {tags}"
    assert tags["cli_session_id"], "cli_session_id tag is empty"
