"""Materialize run snapshots from append-only events."""

from __future__ import annotations

from dataclasses import replace

from .model import (
    AllocationRef,
    ArtifactRef,
    AttemptRecord,
    AttemptStatus,
    Event,
    EventKind,
    RunRecord,
    RunSnapshot,
    RunStatus,
)


def materialize_snapshot(run: RunRecord, events: list[Event]) -> RunSnapshot:
    """Build a current run snapshot from append-only events."""
    assert run.run_id, "run_id cannot be empty"

    attempts: list[AttemptRecord] = []
    artifacts: list[ArtifactRef] = []
    latest_metrics: dict[str, float] = {}
    allocation: AllocationRef | None = None
    current_stage: str | None = None
    current_attempt: AttemptRecord | None = None
    status = RunStatus.PENDING
    last_heartbeat_at = None
    cursor = -1

    for event in events:
        assert event.run_id == run.run_id, (
            f"Event run_id mismatch: expected {run.run_id}, got {event.run_id}"
        )
        cursor = max(cursor, event.seq)

        if event.kind == EventKind.RUN_CREATED:
            status = RunStatus.PENDING
            continue

        if event.kind == EventKind.ATTEMPT_CREATED:
            ordinal = int(event.payload["ordinal"])
            attempt = AttemptRecord(
                attempt_id=event.attempt_id or event.payload["attempt_id"],
                run_id=run.run_id,
                ordinal=ordinal,
            )
            attempts.append(attempt)
            current_attempt = attempt
            continue

        if current_attempt is not None and event.attempt_id == current_attempt.attempt_id:
            if event.kind == EventKind.ATTEMPT_STARTED:
                current_attempt = replace(
                    current_attempt,
                    status=AttemptStatus.RUNNING,
                    started_at=event.ts,
                )
                attempts[-1] = current_attempt
                status = RunStatus.RUNNING
                continue

            if event.kind == EventKind.ATTEMPT_FINISHED:
                current_attempt = replace(
                    current_attempt,
                    status=AttemptStatus.SUCCEEDED,
                    finished_at=event.ts,
                )
                attempts[-1] = current_attempt
                status = RunStatus.SUCCEEDED
                continue

            if event.kind == EventKind.ATTEMPT_FAILED:
                current_attempt = replace(
                    current_attempt,
                    status=AttemptStatus.FAILED,
                    finished_at=event.ts,
                )
                attempts[-1] = current_attempt
                status = RunStatus.FAILED
                continue

            if event.kind == EventKind.ATTEMPT_CANCELLED:
                current_attempt = replace(
                    current_attempt,
                    status=AttemptStatus.CANCELLED,
                    finished_at=event.ts,
                )
                attempts[-1] = current_attempt
                status = RunStatus.CANCELLED
                continue

        if event.kind == EventKind.ALLOCATION_BOUND:
            allocation = AllocationRef(
                provider=str(event.payload["provider"]),
                node_id=str(event.payload["node_id"]),
                image_ref=event.payload.get("image_ref"),
                workspace=event.payload.get("workspace"),
            )
            continue

        if event.kind == EventKind.STAGE_UPDATED:
            current_stage = str(event.payload["stage"])
            continue

        if event.kind == EventKind.METRICS_REPORTED:
            for key, value in event.payload.items():
                latest_metrics[str(key)] = float(value)
            continue

        if event.kind == EventKind.ARTIFACT_RECORDED:
            artifacts.append(
                ArtifactRef(
                    name=str(event.payload["name"]),
                    path=str(event.payload["path"]),
                    kind=str(event.payload["kind"]),
                )
            )
            continue

        if event.kind == EventKind.HEARTBEAT:
            last_heartbeat_at = event.ts
            continue

    return RunSnapshot(
        run=run,
        cursor=cursor,
        status=status,
        current_stage=current_stage,
        current_attempt=current_attempt,
        attempts=tuple(attempts),
        allocation=allocation,
        latest_metrics=latest_metrics,
        artifacts=tuple(artifacts),
        last_heartbeat_at=last_heartbeat_at,
    )
