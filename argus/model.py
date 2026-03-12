"""Core Argus datatypes.

Argus owns supervisor-level semantics:

- user intent (`RunSpec`)
- durable logical runs (`RunRecord`)
- concrete executions (`AttemptRecord`)
- append-only facts (`Event`)
- derived materialized state (`RunSnapshot`)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4


def utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    """Generate a stable-looking identifier."""
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}_{uuid4().hex[:8]}"


class RunKind(StrEnum):
    """High-level workload category."""

    TRAINING = "training"
    EVALUATION = "evaluation"
    KERNEL_BENCH = "kernel_bench"
    GENERIC = "generic"


class RunStatus(StrEnum):
    """Current logical run status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptStatus(StrEnum):
    """Current concrete attempt status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventKind(StrEnum):
    """Canonical event kinds for first-pass supervision."""

    RUN_CREATED = "run_created"
    ATTEMPT_CREATED = "attempt_created"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_FINISHED = "attempt_finished"
    ATTEMPT_FAILED = "attempt_failed"
    ATTEMPT_CANCELLED = "attempt_cancelled"
    ALLOCATION_BOUND = "allocation_bound"
    STAGE_UPDATED = "stage_updated"
    METRICS_REPORTED = "metrics_reported"
    ARTIFACT_RECORDED = "artifact_recorded"
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True)
class RunSpec:
    """User intent for a detached workload."""

    entrypoint: str
    kind: RunKind
    name: str | None = None
    tags: dict[str, str] = field(default_factory=dict)
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.entrypoint, "entrypoint cannot be empty"


@dataclass(frozen=True)
class RunRecord:
    """Durable logical run identity."""

    run_id: str
    spec: RunSpec
    created_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class AttemptRecord:
    """One concrete execution attempt for a run."""

    attempt_id: str
    run_id: str
    ordinal: int
    status: AttemptStatus = AttemptStatus.PENDING
    created_at: datetime = field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None


@dataclass(frozen=True)
class AllocationRef:
    """Bound compute allocation for an attempt."""

    provider: str
    node_id: str
    image_ref: str | None = None
    workspace: str | None = None

    def __post_init__(self) -> None:
        assert self.provider, "provider cannot be empty"
        assert self.node_id, "node_id cannot be empty"


@dataclass(frozen=True)
class ArtifactRef:
    """Artifact announced by the runtime."""

    name: str
    path: str
    kind: str

    def __post_init__(self) -> None:
        assert self.name, "artifact name cannot be empty"
        assert self.path, "artifact path cannot be empty"
        assert self.kind, "artifact kind cannot be empty"


@dataclass(frozen=True)
class Event:
    """Append-only supervisor fact."""

    seq: int
    run_id: str
    kind: EventKind
    ts: datetime
    attempt_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.seq >= 0, "seq must be non-negative"
        assert self.run_id, "run_id cannot be empty"


@dataclass(frozen=True)
class RunSnapshot:
    """Materialized current state for a logical run."""

    run: RunRecord
    cursor: int
    status: RunStatus = RunStatus.PENDING
    current_stage: str | None = None
    current_attempt: AttemptRecord | None = None
    attempts: tuple[AttemptRecord, ...] = ()
    allocation: AllocationRef | None = None
    latest_metrics: dict[str, float] = field(default_factory=dict)
    artifacts: tuple[ArtifactRef, ...] = ()
    last_heartbeat_at: datetime | None = None


def make_run(spec: RunSpec) -> RunRecord:
    """Create a new logical run record."""
    return RunRecord(run_id=new_id("run"), spec=spec)


def make_attempt(run_id: str, ordinal: int) -> AttemptRecord:
    """Create a new attempt for an existing run."""
    assert run_id, "run_id cannot be empty"
    assert ordinal > 0, "ordinal must be positive"
    return AttemptRecord(attempt_id=new_id("attempt"), run_id=run_id, ordinal=ordinal)
