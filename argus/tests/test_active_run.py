from pathlib import Path

import pytest

from argus.active_run import ActiveRun
from argus.model import RunKind, RunStatus


def test_active_run_tracks_attempt_lifecycle_and_projection() -> None:
    active_run = ActiveRun.create(
        run_id="run_20260324-123000",
        kind=RunKind.TRAINING,
        entrypoint="rollouts/configs/trusted/rl/qwen3_0_6b_megatron_modal.py",
        run_dir=Path("/tmp/results/rl/run_20260324-123000"),
        name="run_20260324-123000",
    )

    assert active_run.run_id == "run_20260324-123000"
    assert active_run.status == RunStatus.PENDING
    assert active_run.snapshot.current_attempt is not None
    assert active_run.snapshot.current_attempt.attempt_id == active_run.attempt_id

    active_run.bind_allocation(provider="modal", node_id="sb-12345")
    assert active_run.snapshot.allocation is not None
    assert active_run.snapshot.allocation.provider == "modal"
    assert active_run.snapshot.allocation.node_id == "sb-12345"

    active_run.update_stage("modal_repo_sync")
    assert active_run.stage == "modal_repo_sync"

    active_run.mark_running(stage="modal_training")
    assert active_run.status == RunStatus.RUNNING
    assert active_run.stage == "modal_training"

    active_run.record_artifact(
        name="training_jsonl",
        path="/workspace/research/rollouts/results/rl/run_20260324-123000/training.jsonl",
        kind="jsonl",
    )
    assert len(active_run.snapshot.artifacts) == 1
    assert active_run.snapshot.artifacts[0].name == "training_jsonl"

    active_run.mark_succeeded(exit_code=0)
    assert active_run.status == RunStatus.SUCCEEDED
    assert active_run.snapshot.current_attempt is not None
    assert active_run.snapshot.current_attempt.status == "succeeded"


def test_active_run_rejects_stage_updates_after_terminal_state() -> None:
    active_run = ActiveRun.create(
        run_id="run_20260324-123001",
        kind=RunKind.TRAINING,
        entrypoint="rollouts/configs/trusted/rl/qwen3_0_6b_megatron_modal.py",
        run_dir=Path("/tmp/results/rl/run_20260324-123001"),
    )

    active_run.mark_failed(error="boom", exit_code=1)
    assert active_run.status == RunStatus.FAILED

    with pytest.raises(AssertionError):
        active_run.update_stage("should_not_apply")
