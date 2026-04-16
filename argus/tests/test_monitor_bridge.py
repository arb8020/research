from __future__ import annotations

import json
from pathlib import Path

from argus import monitor


def test_monitor_main_local_view_errors(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()

    result = monitor.monitor_main([str(run_dir)])

    assert result == 2


def test_monitor_main_local_tail_uses_argus_tail(monkeypatch: object, tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text("")

    tailed: dict[str, object] = {}

    def fake_tail_run(run_dir_arg: Path, fmt: str = "pretty", timestamps: bool = False) -> int:
        tailed["run_dir"] = run_dir_arg
        tailed["fmt"] = fmt
        tailed["timestamps"] = timestamps
        return 23

    monkeypatch.setattr(monitor, "tail_run", fake_tail_run)

    result = monitor.monitor_main([str(run_dir), "--tail", "--format", "json"])

    assert result == 23
    assert tailed == {"run_dir": run_dir, "fmt": "json", "timestamps": False}


def test_monitor_main_local_latest_resolves(tmp_path: Path, monkeypatch: object) -> None:
    latest = tmp_path / "results" / "eval" / "run_123"
    latest.mkdir(parents=True)
    monkeypatch.setattr(monitor, "_find_latest_run_dir", lambda _base: latest)

    result = monitor.monitor_main(["--latest", str(tmp_path / "results")])

    assert result == 2


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

    result = monitor.monitor_main(["--launches"])

    assert result == 0


def test_monitor_main_wait_for_event_succeeds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        json.dumps({"event": "training_preflight_weight_sync_witness_ok", "ok": True}) + "\n"
    )

    result = monitor.monitor_main([
        str(run_dir),
        "--wait-for-event",
        "training_preflight_weight_sync_witness_ok",
        "--timeout-seconds",
        "0.1",
    ])

    assert result == 0


def test_monitor_main_wait_for_event_fails_on_nonzero_remote_exit(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        json.dumps({"event": "remote_exit_observed", "exit_code": 1}) + "\n"
    )

    result = monitor.monitor_main([
        str(run_dir),
        "--wait-for-event",
        "training_preflight_weight_sync_witness_ok",
        "--timeout-seconds",
        "0.1",
    ])

    assert result == 1


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
