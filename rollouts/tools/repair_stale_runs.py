"""Repair stale eval runs by appending a synthetic eval_end event.

A "stale" run is a results directory that:
  - has events.jsonl
  - has no report.json
  - last event is NOT eval_end
  - events.jsonl has not been modified in --stale-minutes (default 30)

Running this script is safe: it only appends a single line to events.jsonl
and never modifies existing content.  It is idempotent — re-running on an
already-repaired file is a no-op.

Usage:
    uv run python tools/repair_stale_runs.py ~/silares_stuff/charisma/results
    uv run python tools/repair_stale_runs.py ~/silares_stuff/charisma/results --dry-run
    uv run python tools/repair_stale_runs.py ~/silares_stuff/charisma/results --stale-minutes 60
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
from pathlib import Path


def last_event_type(events_file: Path) -> str | None:
    """Return the 'message' field of the last non-empty line, or None."""
    try:
        with open(events_file, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            line = line.strip()
            if line:
                try:
                    return json.loads(line).get("message")
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return None


def find_stale_runs(results_dir: Path, stale_seconds: float) -> list[Path]:
    stale = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        events_file = subdir / "events.jsonl"
        report_file = subdir / "report.json"
        if not events_file.exists():
            continue
        if report_file.exists():
            continue
        age = time.time() - events_file.stat().st_mtime
        if age < stale_seconds:
            continue
        last = last_event_type(events_file)
        if last == "eval_end":
            continue
        stale.append(subdir)
    return stale


def repair(subdir: Path, dry_run: bool) -> None:
    events_file = subdir / "events.jsonl"
    last = last_event_type(events_file)

    # Extract eval_name from eval_start if present
    eval_name = subdir.name
    try:
        for line in events_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if obj.get("message") == "eval_start":
                eval_name = obj.get("eval_name", eval_name)
                break
    except (OSError, json.JSONDecodeError):
        pass

    # Count completed samples for total
    total = 0
    try:
        for line in events_file.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("message") == "sample_end":
                    total += 1
            except json.JSONDecodeError:
                pass
    except OSError:
        pass

    synthetic_event = {
        "message": "eval_end",
        "timestamp": datetime.datetime.now(tz=datetime.timezone.utc).isoformat(),
        "logger": "rollouts.eval.events",
        "level": "INFO",
        "taskName": None,
        "eval_name": eval_name,
        "total": total,
        "interrupted": True,
        "synthetic": True,
    }

    age_hours = (time.time() - events_file.stat().st_mtime) / 3600
    print(
        f"{'[dry-run] ' if dry_run else ''}repair {subdir.name} "
        f"(last_event={last!r}, samples={total}, age={age_hours:.1f}h)"
    )

    if not dry_run:
        with open(events_file, "a") as f:
            f.write(json.dumps(synthetic_event) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "results_dirs", nargs="+", type=Path, help="Results directory/directories to scan"
    )
    parser.add_argument(
        "--stale-minutes",
        type=float,
        default=30,
        help="Minutes since last write to consider stale (default: 30)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would be done without modifying files"
    )
    args = parser.parse_args()

    stale_seconds = args.stale_minutes * 60
    total_repaired = 0

    for results_dir in args.results_dirs:
        if not results_dir.exists():
            print(f"warning: {results_dir} does not exist, skipping")
            continue
        stale = find_stale_runs(results_dir, stale_seconds)
        if not stale:
            print(f"{results_dir}: no stale runs found")
            continue
        print(f"{results_dir}: {len(stale)} stale run(s)")
        for subdir in stale:
            repair(subdir, dry_run=args.dry_run)
            total_repaired += 1

    if args.dry_run:
        print(f"\n{total_repaired} run(s) would be repaired (--dry-run)")
    else:
        print(f"\n{total_repaired} run(s) repaired")


if __name__ == "__main__":
    main()
