#!/usr/bin/env python3
"""Remote runner that unifies training logs + sglang logs into JSONL stream.

Runs on the remote machine. Spawns training process and tails sglang log,
wrapping everything into a unified JSONL stream on stdout.

Usage (on remote):
    python -m rollouts.tui.remote_runner \
        --training-cmd "python examples/rl/calculator/grpo_01_01.py" \
        --sglang-log "/tmp/rollouts_rl/calculator_grpo/sglang_server.log"

Output format (JSONL to stdout):
    {"logger": "rollouts.training.rl_loop", "level": "INFO", "message": "Step 1/10", ...}
    {"logger": "sglang", "message": "[2025-12-16 19:44:03] Load weight end..."}
    {"logger": "rollouts.training.metrics", "step": 1, "reward": 0.5, ...}
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


def tail_file_as_jsonl(file_path: str, logger_name: str, stop_event: threading.Event) -> None:
    """Tail a file and emit each line as JSONL with logger tag.

    Waits for file to exist, then tails it continuously.
    """
    path = Path(file_path)

    # Wait for file to exist (sglang takes a while to start)
    while not path.exists() and not stop_event.is_set():
        time.sleep(0.5)

    if stop_event.is_set():
        return

    with open(path) as f:
        # Start at end of file
        f.seek(0, 2)

        while not stop_event.is_set():
            line = f.readline()
            if line:
                # Emit as JSONL
                record = {
                    "logger": logger_name,
                    "message": line.rstrip("\n"),
                }
                print(json.dumps(record), flush=True)
            else:
                time.sleep(0.1)


def run_training_with_json_logging(cmd: str, stop_event: threading.Event) -> int:
    """Run training command, passing through its JSONL output.

    Training should already emit JSONL via setup_logging(use_json=True).
    We just pass it through to stdout.

    Returns exit code.
    """
    # Set env to force JSON logging
    env = os.environ.copy()
    env["LOG_JSON"] = "true"
    env["PYTHONUNBUFFERED"] = "1"

    process = subprocess.Popen(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
    )

    # Stream output line by line
    assert process.stdout is not None
    for line in process.stdout:
        # Pass through (already JSONL from training)
        print(line, end="", flush=True)

    process.wait()
    stop_event.set()  # Signal tailers to stop
    return process.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run training with unified JSONL output")
    parser.add_argument("--training-cmd", required=True, help="Training command to run")
    parser.add_argument("--sglang-log", help="Path to sglang server log file to tail")
    parser.add_argument("--metrics-log", help="Path to metrics JSONL file to tail")
    args = parser.parse_args()

    stop_event = threading.Event()
    tailers: list[threading.Thread] = []

    # Start sglang log tailer if specified
    if args.sglang_log:
        t = threading.Thread(
            target=tail_file_as_jsonl,
            args=(args.sglang_log, "sglang", stop_event),
            daemon=True,
        )
        t.start()
        tailers.append(t)

    # Start metrics log tailer if specified
    if args.metrics_log:
        t = threading.Thread(
            target=tail_file_as_jsonl,
            args=(args.metrics_log, "metrics", stop_event),
            daemon=True,
        )
        t.start()
        tailers.append(t)

    # Run training (blocks until complete)
    exit_code = run_training_with_json_logging(args.training_cmd, stop_event)

    # Give tailers a moment to flush
    time.sleep(0.5)
    stop_event.set()

    # Emit completion marker
    print(json.dumps({
        "logger": "runner",
        "message": "Training complete",
        "exit_code": exit_code,
    }), flush=True)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
