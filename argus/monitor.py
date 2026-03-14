"""Argus monitor and transport wrapper.

Argus owns control-plane concerns:

- resolve run IDs
- connect to remote logs/sync transports
- keep local artifact streams fresh

Rollouts owns workload-aware rendering:

- RL/SFT/eval TUI semantics
- event/log interpretation beyond generic control-plane facts

This module therefore does two things:
1. expose a small Argus-native snapshot viewer for generic control-plane facts
2. delegate normal run watching/attach flows to the Rollouts monitor
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollouts.jobs import get_job, get_latest_job, list_jobs

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"
LAUNCHES_DIR = Path.home() / ".argus" / "launches"


@dataclass(frozen=True)
class LogEvent:
    ts: str
    source: str
    event: str
    payload: dict[str, Any]


@dataclass
class MonitorSnapshot:
    run_dir: Path
    log_files: tuple[Path, ...]
    event_count: int
    last_event: LogEvent | None
    stages: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    recent_events: list[LogEvent] = field(default_factory=list)


def _read_jsonl(path: Path, *, source: str) -> list[LogEvent]:
    if not path.exists():
        return []
    events: list[LogEvent] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        ts = str(raw.get("ts", ""))
        event = str(raw.get("event", "unknown"))
        payload = {k: v for k, v in raw.items() if k not in {"ts", "event", "source"}}
        events.append(LogEvent(ts=ts, source=source, event=event, payload=payload))
    return events


def _find_latest_run(base_dir: Path = RESULTS_DIR) -> Path | None:
    if not base_dir.is_dir():
        return None
    candidates: list[Path] = []
    for child in base_dir.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        candidates.append(child)
        for grandchild in child.iterdir():
            if grandchild.is_dir() and not grandchild.name.startswith("."):
                candidates.append(grandchild)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_run_dir(output_dir: str | None, latest: bool, attach: str | None) -> Path:
    if attach is not None:
        job = get_latest_job() if attach == "__latest__" else get_job(attach)
        assert job.log_path, f"Job {job.job_id} has no log_path"
        run_dir = Path(job.log_path)
        if not run_dir.is_absolute():
            run_dir = (REPO_ROOT / run_dir).resolve()
        assert run_dir.exists(), f"Run directory not found: {run_dir}"
        return run_dir

    if latest:
        run_dir = _find_latest_run()
        assert run_dir is not None, f"No run directories found under {RESULTS_DIR}"
        return run_dir

    if output_dir is None:
        raise AssertionError("Provide an output directory, --latest, or --attach.")

    run_dir = Path(output_dir)
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    assert run_dir.exists(), f"Run directory not found: {run_dir}"
    return run_dir


def _build_snapshot(run_dir: Path) -> MonitorSnapshot:
    run_log = run_dir / "run.jsonl"
    monitor_log = run_dir / "monitor.jsonl"
    events = _read_jsonl(run_log, source="run") + _read_jsonl(monitor_log, source="monitor")
    events.sort(key=lambda event: (event.ts, event.source, event.event))

    stages: list[str] = []
    metrics: dict[str, Any] = {}
    metadata: dict[str, Any] = {"run_dir": str(run_dir)}

    for event in events:
        if event.event in {"run_start", "submit_start"}:
            metadata.update(event.payload)
        if "stage" in event.payload:
            stage = str(event.payload["stage"])
            if not stages or stages[-1] != stage:
                stages.append(stage)
        for key, value in event.payload.items():
            if isinstance(value, (int, float, str, bool)) and (
                "metric" in key
                or "loss" in key
                or "reward" in key
                or "version" in key
                or key.endswith("_sec")
                or key.endswith("_steps")
            ):
                metrics[key] = value

    log_files = tuple(path for path in (run_log, monitor_log) if path.exists())
    return MonitorSnapshot(
        run_dir=run_dir,
        log_files=log_files,
        event_count=len(events),
        last_event=events[-1] if events else None,
        stages=stages[-10:],
        metrics=metrics,
        metadata=metadata,
        recent_events=events[-20:],
    )


def _render_snapshot_text(snapshot: MonitorSnapshot) -> str:
    lines = [
        f"run: {snapshot.run_dir.name}",
        f"path: {snapshot.run_dir}",
        f"log files: {', '.join(path.name for path in snapshot.log_files) or 'none'}",
        f"events: {snapshot.event_count}",
    ]
    if snapshot.last_event is not None:
        lines.append(
            f"last event: {snapshot.last_event.ts} {snapshot.last_event.source}:{snapshot.last_event.event}"
        )
    if snapshot.stages:
        lines.append(f"stages: {' -> '.join(snapshot.stages)}")
    if snapshot.metadata:
        lines.append("")
        lines.append("metadata:")
        for key in sorted(snapshot.metadata):
            lines.append(f"  {key}: {snapshot.metadata[key]}")
    if snapshot.metrics:
        lines.append("")
        lines.append("metrics:")
        for key in sorted(snapshot.metrics):
            lines.append(f"  {key}: {snapshot.metrics[key]}")
    if snapshot.recent_events:
        lines.append("")
        lines.append("recent events:")
        for event in snapshot.recent_events[-10:]:
            payload = ", ".join(f"{k}={v}" for k, v in sorted(event.payload.items()))
            suffix = f" {payload}" if payload else ""
            lines.append(f"  {event.ts} {event.source}:{event.event}{suffix}")
    return "\n".join(lines)


def _render_snapshot_html(snapshot: MonitorSnapshot) -> str:
    text = _render_snapshot_text(snapshot)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Argus Run Viewer - {html.escape(snapshot.run_dir.name)}</title>
  <style>
    body {{
      margin: 0;
      padding: 24px;
      background: #101418;
      color: #e7edf2;
      font: 14px/1.5 Menlo, Monaco, Consolas, monospace;
    }}
    .panel {{
      max-width: 1100px;
      margin: 0 auto;
      padding: 20px;
      background: #172028;
      border: 1px solid #2b3946;
      border-radius: 12px;
      white-space: pre-wrap;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 16px;
    }}
  </style>
</head>
<body>
  <div class="panel">
    <h1>Argus Run Viewer</h1>
    {html.escape(text)}
  </div>
</body>
</html>
"""


def _print_runs() -> int:
    jobs = list_jobs()
    if not jobs:
        print("No jobs found in ~/.rollouts/jobs.json")
        return 0
    print(f"{'JOB ID':<30} {'STATUS':<12} {'LOG PATH'}")
    print("-" * 96)
    for job in jobs:
        print(f"{job.job_id:<30} {job.status:<12} {job.log_path or '-'}")
    return 0


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _print_launches() -> int:
    if not LAUNCHES_DIR.exists():
        print("No active launcher records found in ~/.argus/launches")
        return 0

    records: list[dict[str, Any]] = []
    for path in sorted(LAUNCHES_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        pid = payload.get("pid")
        payload["alive"] = isinstance(pid, int) and _process_alive(pid)
        records.append(payload)

    if not records:
        print("No active launcher records found in ~/.argus/launches")
        return 0

    print(f"{'LAUNCHER ID':<36} {'PID':<8} {'ALIVE':<6} {'PROVIDER':<10} {'CONFIG'}")
    print("-" * 140)
    for record in records:
        print(
            f"{record.get('launcher_id', ''):<36} "
            f"{str(record.get('pid', '')):<8} "
            f"{str(record.get('alive', False)):<6} "
            f"{str(record.get('provider', '')):<10} "
            f"{record.get('config_path', '')}"
        )
    return 0


def _tail_snapshot(snapshot: MonitorSnapshot, tail_lines: int | None) -> int:
    limit = 20 if tail_lines is None else tail_lines
    for event in snapshot.recent_events[-limit:]:
        payload = json.dumps(event.payload, sort_keys=True)
        print(f"{event.ts} {event.source}:{event.event} {payload}")
    return 0


def _watch_snapshot(run_dir: Path, refresh_sec: float = 1.0) -> int:
    try:
        while True:
            snapshot = _build_snapshot(run_dir)
            sys.stdout.write("\x1b[2J\x1b[H")
            sys.stdout.write(_render_snapshot_text(snapshot))
            sys.stdout.write("\n")
            sys.stdout.flush()
            time.sleep(refresh_sec)
    except KeyboardInterrupt:
        return 0


def _use_argus_snapshot_viewer(args: argparse.Namespace) -> bool:
    """Return True when the user asked for the Argus-native generic viewer.

    The Argus-native viewer is intentionally narrow: it renders generic
    `run.jsonl` / `monitor.jsonl` control-plane facts and should not become a
    second workload-aware TUI stack.
    """
    return bool(args.html or args.watch or args.launches)


def _build_rollouts_monitor_argv(args: argparse.Namespace) -> list[str]:
    """Translate Argus monitor args into the Rollouts monitor CLI surface."""
    forwarded: list[str] = []

    if args.output_dir:
        forwarded.append(args.output_dir)
    if args.latest:
        forwarded.append("--latest")
    if args.attach is not None:
        forwarded.append("--attach")
        if args.attach != "__latest__":
            forwarded.append(args.attach)
    if args.runs:
        forwarded.append("--runs")
    if getattr(args, "probe", False):
        forwarded.append("--probe")
    if args.tail:
        forwarded.append("--tail")
    if args.tail_lines is not None:
        forwarded.extend(["--tail-lines", str(args.tail_lines)])
    if getattr(args, "debug", False):
        forwarded.append("--debug")
        if getattr(args, "debug_interval", None) is not None:
            forwarded.extend(["--debug-interval", str(args.debug_interval)])
    if getattr(args, "keep_alive", False):
        forwarded.append("--keep-alive")
    if getattr(args, "terminate", False):
        forwarded.append("--terminate")
    if getattr(args, "cancel", None):
        forwarded.extend(["--cancel", args.cancel])
    if getattr(args, "sync_only", False):
        forwarded.append("--sync-only")

    return forwarded


def _delegate_to_rollouts_monitor(args: argparse.Namespace) -> int:
    """Hand off monitoring/rendering to the Rollouts monitor implementation."""
    from rollouts.tui.monitor_cli import monitor_main as rollouts_monitor_main

    return rollouts_monitor_main(_build_rollouts_monitor_argv(args))


def monitor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="argus monitor",
        description="Argus run attach/sync wrapper with optional Rollouts viewer handoff",
    )
    parser.add_argument("output_dir", nargs="?", help="Run directory to inspect")
    parser.add_argument("--latest", action="store_true", help="Use the most recent run directory")
    parser.add_argument(
        "--attach",
        nargs="?",
        const="__latest__",
        metavar="RUN_ID",
        help="Resolve/sync a remote run from ~/.rollouts/jobs.json",
    )
    parser.add_argument("--runs", action="store_true", help="List known runs from local registry")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="With --runs, check broker liveness and LogsServer reachability",
    )
    parser.add_argument(
        "--launches",
        action="store_true",
        help="List active local Argus launcher records",
    )
    parser.add_argument(
        "--tail",
        action="store_true",
        help="Tail remote/local logs to stdout instead of launching the Rollouts TUI",
    )
    parser.add_argument(
        "--tail-lines",
        type=int,
        default=None,
        metavar="N",
        help="With --tail or --attach, print the last N lines and exit",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass debug rendering through to the Rollouts monitor",
    )
    parser.add_argument(
        "--debug-interval",
        type=int,
        default=100,
        metavar="N",
        help="Debug frame interval for the Rollouts monitor",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep instance running after an attached remote run completes",
    )
    parser.add_argument(
        "--terminate",
        action="store_true",
        help="Auto-terminate instance after an attached remote run completes",
    )
    parser.add_argument(
        "--cancel",
        metavar="RUN_ID",
        help="Cancel a running remote job by killing its tmux session",
    )
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="With --attach: sync logs locally without launching a viewer",
    )
    parser.add_argument(
        "--html",
        metavar="PATH",
        help="Write a static HTML viewer for the resolved run snapshot",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Continuously redraw a terminal view from local logs",
    )
    args = parser.parse_args(argv)

    if args.runs:
        if _use_argus_snapshot_viewer(args):
            return _print_runs()
        return _delegate_to_rollouts_monitor(args)

    if args.launches:
        return _print_launches()

    if args.attach is not None or args.cancel or args.sync_only:
        return _delegate_to_rollouts_monitor(args)

    if args.output_dir is not None or args.latest:
        if not _use_argus_snapshot_viewer(args):
            return _delegate_to_rollouts_monitor(args)

    run_dir = _resolve_run_dir(args.output_dir, args.latest, args.attach)
    snapshot = _build_snapshot(run_dir)

    if args.html:
        out_path = Path(args.html)
        if not out_path.is_absolute():
            out_path = (REPO_ROOT / out_path).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(_render_snapshot_html(snapshot))
        print(out_path)
        return 0

    if args.tail or args.tail_lines is not None:
        return _tail_snapshot(snapshot, args.tail_lines)

    if args.watch or sys.stdout.isatty():
        return _watch_snapshot(run_dir)

    print(_render_snapshot_text(snapshot))
    return 0
