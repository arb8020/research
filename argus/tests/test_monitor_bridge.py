from __future__ import annotations

import json
from pathlib import Path

from argus import monitor


def test_build_rollouts_monitor_argv_preserves_attach_and_transport_flags() -> None:
    class Args:
        output_dir = None
        latest = False
        attach = "run_123"
        runs = False
        probe = False
        tail = True
        tail_lines = 25
        debug = True
        debug_interval = 50
        keep_alive = True
        terminate = False
        cancel = None
        sync_only = True

    forwarded = monitor._build_rollouts_monitor_argv(Args())

    assert forwarded == [
        "--attach",
        "run_123",
        "--tail",
        "--tail-lines",
        "25",
        "--debug",
        "--debug-interval",
        "50",
        "--keep-alive",
        "--sync-only",
    ]


def test_monitor_main_delegates_normal_local_view_to_rollouts(
    monkeypatch: object, tmp_path: Path
) -> None:
    called: dict[str, list[str]] = {}

    def fake_delegate(args: object) -> int:
        called["argv"] = monitor._build_rollouts_monitor_argv(args)  # type: ignore[arg-type]
        return 17

    monkeypatch.setattr(monitor, "_delegate_to_rollouts_monitor", fake_delegate)

    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    result = monitor.monitor_main([str(run_dir)])

    assert result == 17
    assert called["argv"] == [str(run_dir)]


def test_monitor_main_keeps_launch_listing_in_argus(monkeypatch: object, tmp_path: Path) -> None:
    launches_dir = tmp_path / "launches"
    launches_dir.mkdir()
    monkeypatch.setattr(monitor, "LAUNCHES_DIR", launches_dir)

    payload = {
        "launcher_id": "launch_123",
        "pid": 999999,
        "provider": "local",
        "config_path": "/tmp/config.py",
    }
    (launches_dir / "launch_123.json").write_text(json.dumps(payload))

    delegated = {"called": False}

    def fake_delegate(args: object) -> int:
        delegated["called"] = True
        return 17

    monkeypatch.setattr(monitor, "_delegate_to_rollouts_monitor", fake_delegate)

    result = monitor.monitor_main(["--launches"])

    assert result == 0
    assert delegated["called"] is False


def test_monitor_main_wait_for_event_succeeds_without_delegating(
    monkeypatch: object, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        json.dumps({"event": "training_preflight_weight_sync_witness_ok", "ok": True}) + "\n"
    )

    delegated = {"called": False}

    def fake_delegate(args: object) -> int:
        delegated["called"] = True
        return 17

    monkeypatch.setattr(monitor, "_delegate_to_rollouts_monitor", fake_delegate)

    result = monitor.monitor_main([
        str(run_dir),
        "--wait-for-event",
        "training_preflight_weight_sync_witness_ok",
        "--timeout-seconds",
        "0.1",
    ])

    assert result == 0
    assert delegated["called"] is False


def test_monitor_main_wait_for_event_fails_on_nonzero_remote_exit(
    monkeypatch: object, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        json.dumps({"event": "remote_exit_observed", "exit_code": 1}) + "\n"
    )

    delegated = {"called": False}

    def fake_delegate(args: object) -> int:
        delegated["called"] = True
        return 17

    monkeypatch.setattr(monitor, "_delegate_to_rollouts_monitor", fake_delegate)

    result = monitor.monitor_main([
        str(run_dir),
        "--wait-for-event",
        "training_preflight_weight_sync_witness_ok",
        "--timeout-seconds",
        "0.1",
    ])

    assert result == 1
    assert delegated["called"] is False


def test_resolve_wait_run_dir_latest_prefers_newest_rollouts_results(
    monkeypatch: object, tmp_path: Path
) -> None:
    import os

    results_run = tmp_path / "results" / "run_old"
    rollouts_run = tmp_path / "rollouts" / "results" / "rl" / "run_new"
    results_run.mkdir(parents=True)
    rollouts_run.mkdir(parents=True)
    (results_run / "run.jsonl").write_text("")
    (rollouts_run / "run.jsonl").write_text("")
    os.utime(results_run, (1, 1))
    os.utime(rollouts_run, (10, 10))

    def fake_find_latest_run_dir(base_dir: str) -> Path | None:
        if base_dir == "results":
            return results_run
        if base_dir == "rollouts/results":
            return rollouts_run
        return None

    monkeypatch.setattr(monitor, "_find_latest_run_dir", fake_find_latest_run_dir)

    class Args:
        attach = None
        latest = True
        output_dir = None

    resolved = monitor._resolve_wait_run_dir(Args())

    assert resolved == rollouts_run
