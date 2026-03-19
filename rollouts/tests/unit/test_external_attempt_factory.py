from __future__ import annotations

from typing import cast

import pytest

from rollouts.core import Message, Trajectory
from rollouts.eval import external_attempts
from rollouts.eval.external_attempts import ExternalAttemptArtifact, make_external_attempt_executor
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
        *,
        model: str,
        sandbox: str,
        run_config: object,
    ) -> ExternalAttemptArtifact:
        observed["prompt"] = prompt
        observed["sample_id"] = sample_id
        observed["sample_data"] = sample_data
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


def test_make_external_attempt_executor_rejects_unknown_runtime() -> None:
    with pytest.raises(ValueError, match="Unsupported external runtime"):
        make_external_attempt_executor(
            cast(external_attempts.ExternalRuntime, "not-a-runtime"),
            prompt_builder=lambda sample: str(sample),
        )
