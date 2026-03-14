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


def test_monitor_main_keeps_argus_snapshot_viewer_for_html_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    run_dir.mkdir()
    (run_dir / "run.jsonl").write_text(
        json.dumps({"ts": "2026-03-13T12:00:00", "event": "run_start", "stage": "boot"}) + "\n"
    )

    html_path = tmp_path / "snapshot.html"
    result = monitor.monitor_main([str(run_dir), "--html", str(html_path)])

    assert result == 0
    assert html_path.exists()
    assert "Argus Run Viewer" in html_path.read_text()
