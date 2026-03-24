from __future__ import annotations

from bifrost.modal_backend import _emit_projected_training_artifact_event


def test_emit_projected_training_artifact_event_projects_step_complete() -> None:
    emitted: list[tuple[str, dict[str, object]]] = []

    def emit(event: str, **data: object) -> None:
        emitted.append((event, data))

    _emit_projected_training_artifact_event(
        emit,
        {
            "event": "step_complete",
            "step": 3,
            "mean_reward": 0.5,
            "pg_loss": -0.1,
            "entropy": 1.2,
            "num_samples": 32,
            "num_groups": 4,
            "step_total_ms": 123.4,
            "rollout_step_count": 3.0,
            "gpu_allocated_gb": 1.0,
            "gpu_reserved_gb": 2.0,
            "ram_gb": 3.0,
        },
        projected_by="modal_parent",
    )

    assert emitted == [
        (
            "step_complete",
            {
                "projected_from_artifact": True,
                "projected_by": "modal_parent",
                "projection_source": "training.jsonl",
                "step": 3,
                "mean_reward": 0.5,
                "pg_loss": -0.1,
                "entropy": 1.2,
                "num_samples": 32,
                "num_groups": 4,
                "step_total_ms": 123.4,
                "rollout_step_count": 3.0,
                "gpu_allocated_gb": 1.0,
                "gpu_reserved_gb": 2.0,
                "ram_gb": 3.0,
            },
        )
    ]


def test_emit_projected_training_artifact_event_projects_train_step_complete() -> None:
    emitted: list[tuple[str, dict[str, object]]] = []

    def emit(event: str, **data: object) -> None:
        emitted.append((event, data))

    _emit_projected_training_artifact_event(
        emit,
        {
            "event": "train_step_complete",
            "step": 4,
            "mean_reward": 0.6,
            "pg_loss": -0.2,
            "entropy": 1.1,
            "loss": -0.2,
            "grad_norm": 0.9,
            "process_batch_ms": 500.0,
            "checkpoint_ms": 25.0,
            "weight_sync_ms": 750.0,
            "step_total_ms": 1275.0,
            "rollout_step_count": 4.0,
        },
        projected_by="modal_parent",
    )

    assert emitted == [
        (
            "train_step_complete",
            {
                "projected_from_artifact": True,
                "projected_by": "modal_parent",
                "projection_source": "training.jsonl",
                "step": 4,
                "mean_reward": 0.6,
                "pg_loss": -0.2,
                "entropy": 1.1,
                "loss": -0.2,
                "grad_norm": 0.9,
                "process_batch_ms": 500.0,
                "checkpoint_ms": 25.0,
                "weight_sync_ms": 750.0,
                "step_total_ms": 1275.0,
                "rollout_step_count": 4.0,
            },
        )
    ]


def test_emit_projected_training_artifact_event_ignores_other_events() -> None:
    emitted: list[tuple[str, dict[str, object]]] = []

    def emit(event: str, **data: object) -> None:
        emitted.append((event, data))

    _emit_projected_training_artifact_event(
        emit,
        {"event": "something_else", "step": 1},
        projected_by="modal_parent",
    )

    assert emitted == []
