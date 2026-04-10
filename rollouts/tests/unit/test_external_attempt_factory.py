from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from rollouts.core import Message, Trajectory
from rollouts.environments.local_workspace_resource import LocalWorkspaceResource
from rollouts.eval import external_attempts
from rollouts.eval.external_attempts import (
    ExternalAttemptArtifact,
    RemoteRuntimePreparation,
    _remote_acp_uv_prepare_command,
    _remote_codex_acp_auth_payload,
    make_external_attempt_executor,
    make_external_trajectory_adapter,
)
from rollouts.training.types import Status


@pytest.mark.trio
async def test_make_external_attempt_executor_builds_codex_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_codex_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        workspace: object,
        *,
        model: str,
        sandbox: str,
        run_config: object,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
        observed["workspace"] = workspace
        observed["model"] = model
        observed["sandbox"] = sandbox
        observed["run_config"] = run_config
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "codex"},
            status=Status.COMPLETED,
        )

    monkeypatch.setattr(external_attempts, "trajectory_from_codex", _fake_codex_adapter)

    executor = make_external_attempt_executor(
        "codex",
        prompt_builder=lambda sample: f"prompt:{sample['text']}",
        model="gpt-5.1-codex-mini",
        sandbox="workspace-write",
    )

    run_config = object()
    attempt = await executor({"text": "hello"}, "sample-1", None, run_config)

    assert attempt.response == "ok"
    assert attempt.metadata["runtime"] == "codex"
    assert observed == {
        "prompt": "prompt:hello",
        "sample_id": "sample-1",
        "sample_data": {"text": "hello"},
        "model": "gpt-5.1-codex-mini",
        "sandbox": "workspace-write",
        "run_config": run_config,
    }
    assert isinstance(observed["workspace"], LocalWorkspaceResource)


def test_make_external_attempt_executor_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="Unsupported external runtime"):
        make_external_attempt_executor(
            cast(external_attempts.ExternalRuntime, "not-a-runtime"),
            prompt_builder=lambda sample: str(sample),
        )


@pytest.mark.trio
async def test_make_external_trajectory_adapter_passes_projected_cwd_and_run_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_claude_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        workspace: object,
        *,
        model: str,
        run_config: object,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
        observed["workspace"] = workspace
        observed["run_config"] = run_config
        observed["model"] = model
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "claude_code"},
            status=Status.COMPLETED,
        )

    monkeypatch.setattr(external_attempts, "trajectory_from_claude_code", _fake_claude_adapter)

    adapter = make_external_trajectory_adapter(
        "claude_code",
        model="sonnet",
    )

    run_config = object()
    workspace = LocalWorkspaceResource.from_existing(tmp_path)
    artifact = await adapter(
        "prompt:hello",
        "sample-1",
        {"text": "hello"},
        workspace,
        run_config,
    )

    assert artifact.metadata["runtime"] == "claude_code"
    assert observed == {
        "prompt": "prompt:hello",
        "sample_id": "sample-1",
        "sample_data": {"text": "hello"},
        "workspace": workspace,
        "run_config": run_config,
        "model": "sonnet",
    }


@pytest.mark.trio
async def test_make_external_trajectory_adapter_supports_claude_acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_claude_acp_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        workspace: object,
        *,
        model: str,
        run_config: object,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
        observed["workspace"] = workspace
        observed["run_config"] = run_config
        observed["model"] = model
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "claude_acp"},
            status=Status.COMPLETED,
        )

    monkeypatch.setattr(external_attempts, "trajectory_from_claude_acp", _fake_claude_acp_adapter)

    adapter = make_external_trajectory_adapter(
        "claude_acp",
        model="claude-agent-acp",
    )

    run_config = object()
    workspace = LocalWorkspaceResource.from_existing(tmp_path)

    artifact = await adapter(
        "prompt:hello",
        "sample-1",
        {"text": "hello"},
        workspace,
        run_config,
    )

    assert artifact.metadata["runtime"] == "claude_acp"
    assert observed == {
        "prompt": "prompt:hello",
        "sample_id": "sample-1",
        "sample_data": {"text": "hello"},
        "workspace": workspace,
        "run_config": run_config,
        "model": "claude-agent-acp",
    }


@pytest.mark.trio
async def test_make_external_trajectory_adapter_supports_codex_acp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_codex_acp_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        workspace: object,
        *,
        run_config: object,
        model: str,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
        observed["workspace"] = workspace
        observed["run_config"] = run_config
        observed["model"] = model
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "codex_acp"},
            status=Status.COMPLETED,
        )

    monkeypatch.setattr(external_attempts, "trajectory_from_codex_acp", _fake_codex_acp_adapter)

    adapter = make_external_trajectory_adapter(
        "codex_acp",
        model="codex-acp",
    )

    run_config = object()
    workspace = LocalWorkspaceResource.from_existing(tmp_path)
    artifact = await adapter(
        "prompt:hello",
        "sample-1",
        {"text": "hello"},
        workspace,
        run_config,
    )

    assert artifact.metadata["runtime"] == "codex_acp"
    assert observed == {
        "prompt": "prompt:hello",
        "sample_id": "sample-1",
        "sample_data": {"text": "hello"},
        "workspace": workspace,
        "run_config": run_config,
        "model": "codex-acp",
    }


@pytest.mark.trio
async def test_make_external_trajectory_adapter_falls_back_to_cwd_for_cwd_only_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}

    async def _fake_openhands_adapter(
        prompt: str,
        sample_id: str,
        sample_data: dict[str, str],
        *,
        cwd: str,
        run_config: object,
        model: str | None = None,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
        observed["cwd"] = cwd
        observed["run_config"] = run_config
        observed["model"] = model
        return ExternalAttemptArtifact(
            trajectory=Trajectory(messages=[Message(role="assistant", content="ok")]),
            metadata={"runtime": "openhands"},
            status=Status.COMPLETED,
        )

    monkeypatch.setattr(external_attempts, "trajectory_from_openhands", _fake_openhands_adapter)

    adapter = make_external_trajectory_adapter(
        "openhands",
        model="gpt-4o",
    )

    run_config = object()
    workspace = LocalWorkspaceResource.from_existing(tmp_path)
    artifact = await adapter(
        "prompt:hello",
        "sample-1",
        {"text": "hello"},
        workspace,
        run_config,
    )

    assert artifact.metadata["runtime"] == "openhands"
    assert observed == {
        "prompt": "prompt:hello",
        "sample_id": "sample-1",
        "sample_data": {"text": "hello"},
        "cwd": str(tmp_path),
        "run_config": run_config,
        "model": "gpt-4o",
    }


def test_remote_runtime_preparation_keeps_uv_config_explicit() -> None:
    preparation = RemoteRuntimePreparation(
        mode="uv",
        uv_config_toml='exclude-newer = "7 days"\n',
        npmrc="min-release-age=8\nignore-scripts=true\n",
    )

    command = _remote_acp_uv_prepare_command(
        "claude_acp",
        uv_config_path="/tmp/rollouts-external-runtime/claude_acp/uv.toml",
        npmrc_path="/tmp/rollouts-external-runtime/claude_acp/.npmrc",
    )

    assert preparation.uv_config_toml == 'exclude-newer = "7 days"\n'
    assert preparation.npmrc == "min-release-age=8\nignore-scripts=true\n"
    assert "--config-file /tmp/rollouts-external-runtime/claude_acp/uv.toml" in command
    assert "NPM_CONFIG_USERCONFIG=/tmp/rollouts-external-runtime/claude_acp/.npmrc" in command
    assert "npx -y @agentclientprotocol/claude-agent-acp --help" not in command


def test_remote_codex_acp_auth_payload_materializes_apikey_mode() -> None:
    payload = _remote_codex_acp_auth_payload("sk-test")

    assert payload == {
        "auth_mode": "apikey",
        "OPENAI_API_KEY": "sk-test",
        "tokens": None,
        "last_refresh": None,
    }
