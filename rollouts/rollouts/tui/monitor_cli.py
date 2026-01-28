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
    """Attach to a remote training run via LogsServer.

    1. Resolve run metadata from .active_runs.json by run_id
    2. Connect to LogsServer via RemoteWorker (TCP, no SSH needed)
    3. Start background sync thread (tail commands every 2s)
    4. Launch monitor TUI watching the local sync dir
    5. On quit: final sync, optionally terminate instance
    """
    import threading
    import time

    from miniray import RemoteWorker

    from .rlmon import make_app

    run = _find_active_run(run_id)
    resolved_run_id = run["run_id"]
    logs_host = run["logs_host"]
    logs_port = run["logs_port"]
    node_id = run.get("node_id")  # Optional, for terminate prompt

    print(f"Attaching to run: {resolved_run_id}")
    print(f"LogsServer: {logs_host}:{logs_port}")

    worker = RemoteWorker(logs_host, logs_port)
    worker.connect()

    # Discover available files
    worker.send({"cmd": "list"})
    available = worker.recv()
    print(f"Files: {', '.join(available['files'])}")

    local_sync_dir = Path("results/rl") / resolved_run_id
    local_sync_dir.mkdir(parents=True, exist_ok=True)

    # Track byte offsets per file for incremental tail
    offsets: dict[str, int] = {}

    stop_sync = threading.Event()

    def sync_loop() -> None:
        while not stop_sync.is_set():
            try:
                worker.send({"cmd": "list"})
                resp = worker.recv()
                files = resp.get("files", [])

                for filename in files:
                    offset = offsets.get(filename, 0)
                    worker.send({"cmd": "tail", "file": filename, "offset": offset})
                    result = worker.recv()

                    if result.get("error"):
                        continue

                    new_lines = result.get("lines", [])
                    new_offset = result.get("offset", offset)

                    if new_lines:
                        local_path = local_sync_dir / filename
                        with open(local_path, "a") as f:
                            for line in new_lines:
                                f.write(line + "\n")

                    offsets[filename] = new_offset

            except (EOFError, BrokenPipeError, ConnectionResetError):
                print("[monitor] LogsServer connection lost")
                break

            stop_sync.wait(2.0)

    sync_thread = threading.Thread(target=sync_loop, daemon=True)
    sync_thread.start()

    # Give first sync a moment to populate files
    time.sleep(1.5)

    print(f"Watching: {local_sync_dir}")
    app = make_app(str(local_sync_dir))
    app.run()

    # TUI exited — final sync
    stop_sync.set()
    sync_thread.join(timeout=5.0)

    print("\nFinal sync...")
    try:
        worker.send({"cmd": "list"})
        resp = worker.recv()
        for filename in resp.get("files", []):
            offset = offsets.get(filename, 0)
            worker.send({"cmd": "tail", "file": filename, "offset": offset})
            result = worker.recv()
            new_lines = result.get("lines", [])
            if new_lines:
                local_path = local_sync_dir / filename
                with open(local_path, "a") as f:
                    for line in new_lines:
                        f.write(line + "\n")
                print(f"  Synced: {resolved_run_id}/{filename} (+{len(new_lines)} lines)")
    except (EOFError, BrokenPipeError, ConnectionResetError):
        print("  LogsServer disconnected, skipping final sync")

    worker.close()

    # Optional: terminate instance (requires bifrost + API key)
    if node_id:
        answer = input(f"\nTerminate instance {node_id}? [y/n] ").strip().lower()
        if answer == "y":
            from dotenv import load_dotenv

            from bifrost import acquire_node

            load_dotenv()
            bifrost, instance = acquire_node(node_id=node_id)
            assert instance is not None, f"Instance not found: {node_id}"
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
