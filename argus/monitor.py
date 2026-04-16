"""Argus monitor wrapper.

Argus owns control-plane concerns:

- resolve run identity
- attach/sync remote artifacts when needed
- expose launcher/control-plane status

Rollouts owns workload-aware viewing:

- RL/SFT/eval TUI semantics
- event/log interpretation
- local run-directory rendering

So `argus monitor` should stay thin: control-plane-only flags live here, and
all actual run viewing delegates to the Rollouts monitor implementation.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

LAUNCHES_DIR = Path.home() / ".argus" / "launches"
_DEFAULT_WAIT_FAILURE_EVENTS = frozenset({"run_failed"})


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
    if args.probe:
        forwarded.append("--probe")
    if args.tail:
        forwarded.append("--tail")
    if hasattr(args, "format") and args.format:
        forwarded.extend(["--format", args.format])
    if args.tail_lines is not None:
        forwarded.extend(["--tail-lines", str(args.tail_lines)])
    if args.debug:
        forwarded.append("--debug")
        forwarded.extend(["--debug-interval", str(args.debug_interval)])
    if args.keep_alive:
        forwarded.append("--keep-alive")
    if args.terminate:
        forwarded.append("--terminate")
    if args.cancel:
        forwarded.extend(["--cancel", args.cancel])
    if args.sync_only:
        forwarded.append("--sync-only")

    return forwarded


def _delegate_to_rollouts_monitor(args: argparse.Namespace) -> int:
    """Hand off monitoring/rendering to the Rollouts monitor implementation."""
    from rollouts.monitor.cli import monitor_main as rollouts_monitor_main

    return rollouts_monitor_main(_build_rollouts_monitor_argv(args))


def _find_latest_run_dir(base_dir: str) -> Path | None:
    """Resolve the most recent run directory under a candidate base dir."""
    from rollouts.monitor.cli import find_latest_run

    resolved = find_latest_run(base_dir)
    if resolved is None:
        return None
    return Path(resolved)


def _resolve_wait_run_dir(args: argparse.Namespace) -> Path:
    """Resolve a local run directory for wait-mode polling."""
    if args.attach is not None:
        raise ValueError("--wait-for-event does not support --attach yet")

    if args.latest:
        if args.output_dir is not None:
            resolved = _find_latest_run_dir(args.output_dir)
            if resolved is None:
                raise ValueError(f"No run directories found under {args.output_dir!r}")
            return resolved

        candidate_runs = [
            resolved
            for resolved in (
                _find_latest_run_dir("results"),
                _find_latest_run_dir("rollouts/results"),
            )
            if resolved is not None
        ]
        if not candidate_runs:
            raise ValueError("No run directories found under 'results' or 'rollouts/results'")
        candidate_runs.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        return candidate_runs[0]

    if args.output_dir is None:
        raise ValueError("--wait-for-event requires a run directory or --latest")

    return Path(args.output_dir)


def _event_is_failure(entry: dict[str, Any], fail_on_events: set[str]) -> bool:
    """Return whether a structured journal entry denotes terminal failure."""
    event = entry.get("event")
    if not isinstance(event, str):
        return False

    if event in fail_on_events:
        return True

    if event == "remote_exit_observed":
        exit_code = entry.get("exit_code")
        return isinstance(exit_code, int) and exit_code != 0

    return False


def _wait_for_run_event(
    run_dir: Path,
    *,
    target_event: str,
    timeout_seconds: float | None,
    poll_interval: float,
    fail_on_events: set[str],
) -> int:
    """Poll a local run journal until a target event or terminal failure occurs."""
    run_jsonl = run_dir / "run.jsonl"
    start = time.monotonic()
    offset = 0

    print(
        f"Waiting for event {target_event!r} in {run_jsonl}"
        + (f" (timeout {timeout_seconds:.1f}s)" if timeout_seconds is not None else "")
    )

    while True:
        if timeout_seconds is not None and (time.monotonic() - start) > timeout_seconds:
            print(
                f"Timed out waiting for {target_event!r} in {run_jsonl} after {timeout_seconds:.1f}s",
            )
            return 124

        if run_jsonl.exists():
            with open(run_jsonl, encoding="utf-8") as handle:
                handle.seek(offset)
                while True:
                    line = handle.readline()
                    if not line:
                        break
                    offset = handle.tell()
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event = entry.get("event")
                    if event == target_event:
                        print(json.dumps(entry, sort_keys=True))
                        return 0

                    if _event_is_failure(entry, fail_on_events):
                        print(json.dumps(entry, sort_keys=True))
                        return 1

        time.sleep(poll_interval)


def monitor_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="argus monitor",
        description="Argus attach/sync wrapper that hands off run viewing to rollouts monitor",
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
        "--format",
        choices=["pretty", "json"],
        default="pretty",
        help="Output format when tailing: pretty (human-readable, default) or json (raw JSONL)",
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
        "--wait-for-event",
        metavar="EVENT",
        help="Poll run.jsonl until this event appears, then exit",
    )
    parser.add_argument(
        "--fail-on-event",
        action="append",
        default=None,
        metavar="EVENT",
        help="Treat this event as terminal failure while waiting (repeatable)",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Timeout for --wait-for-event",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        metavar="SECONDS",
        help="Polling interval for --wait-for-event",
    )
    args = parser.parse_args(argv)

    if args.launches:
        return _print_launches()

    if args.wait_for_event is not None:
        fail_on_events = set(_DEFAULT_WAIT_FAILURE_EVENTS)
        if args.fail_on_event is not None:
            fail_on_events.update(args.fail_on_event)
        run_dir = _resolve_wait_run_dir(args)
        return _wait_for_run_event(
            run_dir,
            target_event=args.wait_for_event,
            timeout_seconds=args.timeout_seconds,
            poll_interval=args.poll_interval,
            fail_on_events=fail_on_events,
        )

    return _delegate_to_rollouts_monitor(args)
