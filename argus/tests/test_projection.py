from argus.journal import InMemoryEventJournal
from argus.model import EventKind, RunKind, RunSpec, make_attempt, make_run
from argus.projection import materialize_snapshot


def test_materialize_snapshot_tracks_attempt_lifecycle() -> None:
    spec = RunSpec(
        entrypoint="rollouts/examples/rl/reverse_text/grpo_01_01.py", kind=RunKind.TRAINING
    )
    run = make_run(spec)
    attempt = make_attempt(run.run_id, ordinal=1)

    journal = InMemoryEventJournal()
    journal.append(run_id=run.run_id, kind=EventKind.RUN_CREATED)
    journal.append(
        run_id=run.run_id,
        kind=EventKind.ATTEMPT_CREATED,
        attempt_id=attempt.attempt_id,
        payload={"attempt_id": attempt.attempt_id, "ordinal": 1},
    )
    journal.append(
        run_id=run.run_id,
        kind=EventKind.ALLOCATION_BOUND,
        attempt_id=attempt.attempt_id,
        payload={"provider": "runpod", "node_id": "abc123", "workspace": "/workspace/research"},
    )
    journal.append(
        run_id=run.run_id,
        kind=EventKind.ATTEMPT_STARTED,
        attempt_id=attempt.attempt_id,
    )
    journal.append(
        run_id=run.run_id,
        kind=EventKind.STAGE_UPDATED,
        attempt_id=attempt.attempt_id,
        payload={"stage": "training"},
    )
    journal.append(
        run_id=run.run_id,
        kind=EventKind.METRICS_REPORTED,
        attempt_id=attempt.attempt_id,
        payload={"loss": 1.25, "reward": 0.7},
    )
    journal.append(
        run_id=run.run_id,
        kind=EventKind.ATTEMPT_FINISHED,
        attempt_id=attempt.attempt_id,
    )

    snapshot = materialize_snapshot(run, journal.list_events(run.run_id))

    assert snapshot.status == "succeeded"
    assert snapshot.current_stage == "training"
    assert snapshot.current_attempt is not None
    assert snapshot.current_attempt.attempt_id == attempt.attempt_id
    assert snapshot.current_attempt.status == "succeeded"
    assert snapshot.allocation is not None
    assert snapshot.allocation.provider == "runpod"
    assert snapshot.latest_metrics == {"loss": 1.25, "reward": 0.7}


def test_journal_since_filters_by_cursor() -> None:
    spec = RunSpec(entrypoint="eval.py", kind=RunKind.EVALUATION)
    run = make_run(spec)
    journal = InMemoryEventJournal()

    first = journal.append(run_id=run.run_id, kind=EventKind.RUN_CREATED)
    journal.append(run_id=run.run_id, kind=EventKind.HEARTBEAT)

    later = journal.since(run.run_id, first.seq)
    assert len(later) == 1
    assert later[0].kind == EventKind.HEARTBEAT
