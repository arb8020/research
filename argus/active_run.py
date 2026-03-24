"""Live run wrapper over Argus' append-only supervision model.

`RunRecord` is the durable logical identity. `ActiveRun` is the small mutable
owner for one in-process launcher execution: it appends facts to the journal and
keeps the latest materialized snapshot locally available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .journal import InMemoryEventJournal
from .model import EventKind, RunKind, RunRecord, RunSnapshot, RunSpec, RunStatus, make_attempt
from .projection import materialize_snapshot


@dataclass
class ActiveRun:
    """Mutable in-process owner for one logical run plus its current attempt."""

    record: RunRecord
    run_dir: Path
    journal: InMemoryEventJournal = field(default_factory=InMemoryEventJournal)
    snapshot: RunSnapshot = field(init=False)
    _attempt_id: str = field(init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        kind: RunKind,
        entrypoint: str,
        run_dir: Path,
        name: str | None = None,
        tags: dict[str, str] | None = None,
        args: tuple[str, ...] = (),
        env: dict[str, str] | None = None,
    ) -> ActiveRun:
        """Create one logical run with its first execution attempt."""
        spec = RunSpec(
            entrypoint=entrypoint,
            kind=kind,
            name=name,
            tags={} if tags is None else dict(tags),
            args=args,
            env={} if env is None else dict(env),
        )
        record = RunRecord(run_id=run_id, spec=spec)
        active = cls(record=record, run_dir=run_dir)
        active.journal.append(run_id=run_id, kind=EventKind.RUN_CREATED)
        attempt = make_attempt(run_id=run_id, ordinal=1)
        active._attempt_id = attempt.attempt_id
        active.journal.append(
            run_id=run_id,
            kind=EventKind.ATTEMPT_CREATED,
            attempt_id=attempt.attempt_id,
            payload={"attempt_id": attempt.attempt_id, "ordinal": attempt.ordinal},
        )
        active.snapshot = materialize_snapshot(record, active.journal.list_events(run_id))
        return active

    @property
    def run_id(self) -> str:
        return self.record.run_id

    @property
    def status(self) -> RunStatus:
        return self.snapshot.status

    @property
    def stage(self) -> str | None:
        return self.snapshot.current_stage

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    def _append(
        self,
        kind: EventKind,
        *,
        payload: dict[str, object] | None = None,
        attempt_scoped: bool = True,
    ) -> None:
        self.journal.append(
            run_id=self.run_id,
            kind=kind,
            attempt_id=self._attempt_id if attempt_scoped else None,
            payload={} if payload is None else payload,
        )
        self.snapshot = materialize_snapshot(self.record, self.journal.list_events(self.run_id))

    def bind_allocation(
        self,
        *,
        provider: str,
        node_id: str,
        image_ref: str | None = None,
        workspace: str | None = None,
    ) -> None:
        assert provider, "provider cannot be empty"
        assert node_id, "node_id cannot be empty"
        self._append(
            EventKind.ALLOCATION_BOUND,
            payload={
                "provider": provider,
                "node_id": node_id,
                "image_ref": image_ref,
                "workspace": workspace,
            },
        )

    def mark_running(self, *, stage: str | None = None) -> None:
        assert self.status in {RunStatus.PENDING, RunStatus.RUNNING}, (
            f"cannot mark running from terminal status {self.status}"
        )
        if self.status == RunStatus.PENDING:
            self._append(EventKind.ATTEMPT_STARTED)
        if stage is not None:
            self.update_stage(stage)

    def update_stage(self, stage: str) -> None:
        assert stage, "stage cannot be empty"
        assert self.status in {RunStatus.PENDING, RunStatus.RUNNING}, (
            f"cannot update stage from terminal status {self.status}"
        )
        self._append(EventKind.STAGE_UPDATED, payload={"stage": stage})

    def record_artifact(self, *, name: str, path: str, kind: str) -> None:
        assert name, "artifact name cannot be empty"
        assert path, "artifact path cannot be empty"
        assert kind, "artifact kind cannot be empty"
        self._append(
            EventKind.ARTIFACT_RECORDED,
            payload={"name": name, "path": path, "kind": kind},
        )

    def mark_succeeded(self, *, exit_code: int | None = None) -> None:
        assert self.status in {RunStatus.PENDING, RunStatus.RUNNING}, (
            f"cannot mark succeeded from terminal status {self.status}"
        )
        self._append(
            EventKind.ATTEMPT_FINISHED,
            payload={} if exit_code is None else {"exit_code": exit_code},
        )

    def mark_failed(self, *, error: str | None = None, exit_code: int | None = None) -> None:
        assert self.status in {RunStatus.PENDING, RunStatus.RUNNING}, (
            f"cannot mark failed from terminal status {self.status}"
        )
        payload: dict[str, object] = {}
        if error is not None:
            payload["error"] = error
        if exit_code is not None:
            payload["exit_code"] = exit_code
        self._append(EventKind.ATTEMPT_FAILED, payload=payload)

    def mark_cancelled(self, *, reason: str | None = None) -> None:
        assert self.status in {RunStatus.PENDING, RunStatus.RUNNING}, (
            f"cannot mark cancelled from terminal status {self.status}"
        )
        payload: dict[str, object] = {}
        if reason is not None:
            payload["reason"] = reason
        self._append(EventKind.ATTEMPT_CANCELLED, payload=payload)
