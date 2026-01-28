"""Monitor subcommand for rollouts CLI.

Usage:
    rollouts monitor results/rl/run_20250127/       # Watch specific run
    rollouts monitor --latest                        # Watch most recent run in results/
    rollouts monitor --latest results/sft/           # Most recent in a custom dir
    rollouts monitor --attach run_20250127-143052    # Attach to remote run by ID
    rollouts monitor --attach --latest               # Attach to most recent active run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ACTIVE_RUNS_PATH = Path("results/rl/.active_runs.json")


def find_latest_run(base_dir: str = "results") -> Path | None:
    """Find the most recent run directory across all experiment types.

    Searches results/, results/rl/, results/sft/, results/eval/, etc.
    """
    base = Path(base_dir)
    if not base.is_dir():
        return None

    # Collect candidate dirs: immediate children + one level deeper
    candidates: list[Path] = []
    for child in base.iterdir():
        if child.is_dir():
            if child.name.startswith("."):
                continue
            candidates.append(child)
            # Also check subdirs (e.g. results/rl/run_20250127/)
            for grandchild in child.iterdir():
                if grandchild.is_dir() and not grandchild.name.startswith("."):
                    candidates.append(grandchild)

    if not candidates:
        return None

    candidates.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return candidates[0]


def _load_active_runs() -> list[dict]:
    """Load .active_runs.json, return [] if missing."""
    if not ACTIVE_RUNS_PATH.exists():
        return []
    return json.loads(ACTIVE_RUNS_PATH.read_text())


def _find_active_run(run_id: str | None) -> dict:
    """Find an active run by run_id, or the latest one."""
    runs = _load_active_runs()
    assert runs, f"No active runs found in {ACTIVE_RUNS_PATH}"

    if run_id is None:
        return runs[-1]

    for run in reversed(runs):
        if run["run_id"] == run_id:
            return run

    raise AssertionError(f"No active run found with id {run_id!r}")


def _run_attached(run_id: str | None) -> int:
    """Attach to a remote training run via bifrost.

    1. Resolve run metadata from .active_runs.json by run_id
    2. Connect to remote node via bifrost
    3. Start background sync thread (download_files every 3s)
    4. Launch monitor TUI watching the local sync dir
    5. On quit: prompt to terminate or keep instance alive
    """
    import threading
    import time

    from dotenv import load_dotenv

    from bifrost import JobInfo, acquire_node, job_status

    from .rlmon import make_app

    load_dotenv()

    run = _find_active_run(run_id)
    resolved_run_id = run["run_id"]
    node_id = run["node_id"]
    remote_output_dir = run["remote_output_dir"]

    print(f"Attaching to run: {resolved_run_id}")
    print(f"Node: {node_id}")
    print(f"Remote: {remote_output_dir}")

    bifrost, instance = acquire_node(node_id=node_id)

    local_sync_dir = Path("results/rl") / resolved_run_id
    local_sync_dir.mkdir(parents=True, exist_ok=True)

    sync_files = [
        "metrics.jsonl",
        "rollouts.jsonl",
        "training.log",
        "error_log.jsonl",
        "events.jsonl",
        "config.json",
        "sglang.log",
        "vllm.log",
    ]

    stop_sync = threading.Event()

    def sync_loop() -> None:
        while not stop_sync.is_set():
            for filename in sync_files:
                try:
                    bifrost.download_files(
                        remote_path=f"{remote_output_dir}/{filename}",
                        local_path=str(local_sync_dir / filename),
                        recursive=False,
                    )
                except Exception:
                    pass
            stop_sync.wait(3.0)

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()

    # Give first sync a moment to populate files
    time.sleep(1.5)

    job_info = JobInfo(
        name="rl-training",
        tmux_session=run.get("tmux_session", "rl-training"),
        log_file=run.get("log_file"),
    )

    print(f"Watching: {local_sync_dir}")
    app = make_app(str(local_sync_dir))
    app.run()

    # TUI exited
    stop_sync.set()
    sync_thread.join(timeout=5.0)

    status = job_status(bifrost, job_info)
    if status == "running":
        print(f"\nTraining is still running on {node_id}.")
    else:
        print(f"\nTraining has completed on {node_id}.")

    # Final sync
    print("Final sync...")
    for filename in sync_files:
        try:
            result = bifrost.download_files(
                remote_path=f"{remote_output_dir}/{filename}",
                local_path=str(local_sync_dir / filename),
                recursive=False,
            )
            if result and result.success:
                print(f"  Synced: {resolved_run_id}/{filename}")
        except Exception:
            pass

    if instance:
        answer = input("Terminate instance? [y/n] ").strip().lower()
        if answer == "y":
            print(f"Terminating {node_id}...")
            instance.terminate()
            print("Terminated.")
        else:
            print(f"Instance kept alive: {node_id}")
            print(f"Reattach: rollouts monitor --attach {resolved_run_id}")

    return 0


def monitor_main(argv: list[str] | None = None) -> int:
    """Entry point for `rollouts monitor` subcommand."""
    parser = argparse.ArgumentParser(
        prog="rollouts monitor",
        description="btop-style experiment monitor (RL, SFT, eval, generic)",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to training/eval output directory",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch the most recent run directory",
    )
    parser.add_argument(
        "--attach",
        nargs="?",
        const="__latest__",
        metavar="RUN_ID",
        help="Attach to remote run by ID (e.g. run_20250127-143052). No value = latest.",
    )
    args = parser.parse_args(argv)

    # ── Attach mode ──
    if args.attach is not None:
        if args.attach == "__latest__" or args.latest:
            return _run_attached(run_id=None)
        else:
            return _run_attached(run_id=args.attach)

    # ── Local mode ──
    from .rlmon import make_app

    if args.latest:
        base = args.output_dir or "results"
        latest = find_latest_run(base)
        if latest is None:
            print(f"No run directories found in {base}", file=sys.stderr)
            return 1
        output_dir = latest
        print(f"Watching: {output_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parser.print_help()
        return 1

    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        return 1

    app = make_app(str(output_dir))
    app.run()
    return 0
