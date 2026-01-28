"""Shared utilities for RL examples.

Provides run_remote() for deploying any RL training script to a remote GPU.

Two modes:
  --provision (no --tui): fire-and-forget. Submit job, save metadata, exit immediately.
  --provision --tui:      fire-and-forget + launch rollouts monitor --attach inline.
  (no --provision):       blocking. Stream logs until completion, sync results, terminate.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient, JobInfo
    from broker import ClientGPUInstance

ACTIVE_RUNS_PATH = Path("results/rl/.active_runs.json")


def _save_active_run(run_info: dict) -> None:
    """Append a run entry to .active_runs.json."""
    ACTIVE_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)

    runs: list[dict] = []
    if ACTIVE_RUNS_PATH.exists():
        runs = json.loads(ACTIVE_RUNS_PATH.read_text())

    runs.append(run_info)
    ACTIVE_RUNS_PATH.write_text(json.dumps(runs, indent=2) + "\n")


def _deploy_and_submit(
    script_path: str,
    node_id: str | None,
    gpu_count: int,
    gpu_type: str,
    use_json_logs: bool,
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace).
    """
    from dotenv import load_dotenv

    from bifrost import GPUQuery, ProcessSpec, acquire_node

    load_dotenv()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Acquire node
    if node_id:
        bifrost, instance = acquire_node(node_id=node_id)
        print(f"Connected to existing instance: {node_id}")
    else:
        print(f"Provisioning {gpu_count}x {gpu_type}...")
        bifrost, instance = acquire_node(
            provision=GPUQuery(type=gpu_type, count=gpu_count, min_cuda="12.8")
        )
        if instance:
            print(f"Instance: {instance.provider}:{instance.id}")

    # Deploy code
    repo_root = Path(__file__).parent.parent.parent
    script_rel_path = Path(script_path).relative_to(repo_root)

    print("Deploying code...")
    bootstrap = [
        "apt-get update && apt-get install -y tmux libnuma1 || true",
        "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        "cd rollouts && ~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12",
        "~/.local/bin/uv pip install torch transformers datasets accelerate sglang[all] curl_cffi peft",
    ]
    workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", bootstrap_cmd=bootstrap)
    print("Code deployed")

    # Create run output directory
    remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    bifrost.exec(f"mkdir -p {remote_output_dir}")
    training_log = f"{remote_output_dir}/training.log"

    # Build environment variables
    env_vars = {"PYTHONUNBUFFERED": "1", "ROLLOUTS_RUN_NAME": run_name}
    if use_json_logs:
        env_vars["ROLLOUTS_JSON_LOGS"] = "true"

    # Submit training job
    print(f"Starting training run: {run_name}")
    job = bifrost.submit(
        ProcessSpec(
            command="/root/.local/bin/uv",
            args=("run", "python", str(script_rel_path)),
            cwd=f"{workspace}/rollouts",
            env=env_vars,
        ),
        name="rl-training",
        log_file=training_log,
        workspace=f"{workspace}/rollouts",
    )

    print(f"Training started in tmux session: {job.tmux_session}")
    print(f"Log file: {job.log_file}")

    return bifrost, instance, job, run_name, remote_output_dir, workspace


def _block_until_complete(
    bifrost: BifrostClient,
    instance: ClientGPUInstance | None,
    job: JobInfo,
    run_name: str,
    remote_output_dir: str,
    use_tui: bool,
    tui_debug: bool,
    keep_alive: bool,
) -> None:
    """Block on job, stream logs, sync results, optionally terminate."""
    from bifrost import job_stream_until_complete

    try:
        if tui_debug:
            print("-" * 50)
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
            print("-" * 50)
            if not success:
                print(f"Training failed: {err} (exit code: {exit_code})")
        elif use_tui:
            import threading
            import time

            from rollouts.tui.monitor import TrainingMonitor

            monitor = TrainingMonitor()
            lines_queue: list[str] = []
            streaming_done = threading.Event()
            stream_result: tuple[bool, int | None, str | None] = (False, None, None)

            def collect_lines() -> None:
                nonlocal stream_result

                def queue_line(line: str) -> None:
                    lines_queue.append(line)

                stream_result = job_stream_until_complete(
                    bifrost, job, on_line=queue_line, timeout=7200, poll_interval=1.0
                )
                streaming_done.set()

            collector = threading.Thread(target=collect_lines, daemon=True)
            collector.start()

            monitor._running = True
            monitor.terminal.start(on_input=lambda x: None, on_resize=monitor._on_resize)

            try:
                while monitor._running and not streaming_done.is_set():
                    while lines_queue:
                        raw_line = lines_queue.pop(0)
                        monitor.feed_line(raw_line)

                    data = monitor.terminal.read_input()
                    if data:
                        monitor._handle_input(data)

                    if monitor._needs_redraw:
                        monitor._render()
                        monitor._needs_redraw = False

                    time.sleep(0.05)
            finally:
                monitor.terminal.stop()

            success, exit_code, err = stream_result
            if not success:
                print(f"Training failed: {err} (exit code: {exit_code})")
        else:
            print("-" * 50)
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
            print("-" * 50)
            if not success:
                print(f"Training failed: {err} (exit code: {exit_code})")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Syncing logs before exit...")

    finally:
        _sync_and_cleanup(bifrost, instance, run_name, remote_output_dir, keep_alive)


def _sync_and_cleanup(
    bifrost: BifrostClient,
    instance: ClientGPUInstance | None,
    run_name: str,
    remote_output_dir: str | None,
    keep_alive: bool,
) -> None:
    """Sync results from remote and optionally terminate instance."""
    print("\nSyncing results...")
    local_results = Path("results/rl")
    local_run_dir = local_results / run_name
    local_run_dir.mkdir(parents=True, exist_ok=True)

    if remote_output_dir:
        files_to_sync = [
            "training.log",
            "config.json",
            "rollouts.jsonl",
            "sglang.log",
            "vllm.log",
        ]

        for filename in files_to_sync:
            try:
                result = bifrost.download_files(
                    remote_path=f"{remote_output_dir}/{filename}",
                    local_path=str(local_run_dir / filename),
                    recursive=False,
                )
                if result and result.success:
                    print(f"  Synced: {run_name}/{filename}")
            except Exception:
                pass

    if instance:
        if not keep_alive:
            print(f"\nTerminating instance {instance.provider}:{instance.id}...")
            instance.terminate()
        else:
            print(f"\nInstance kept alive: {instance.provider}:{instance.id}")
            print(f"Reuse with: --node-id {instance.provider}:{instance.id}")


def run_remote(
    script_path: str,
    keep_alive: bool = False,
    node_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
    fire_and_forget: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
) -> None:
    """Run training script on remote GPU via bifrost.

    Two modes controlled by fire_and_forget:
      False (default): Block on job, stream logs, sync results, terminate.
      True:            Submit job, save metadata, exit immediately.
                       If use_tui=True, launches rollouts monitor --attach after submit.
    """
    use_json_logs = use_tui or tui_debug or fire_and_forget

    bifrost, instance, job, run_name, remote_output_dir, workspace = _deploy_and_submit(
        script_path=script_path,
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        use_json_logs=use_json_logs,
    )

    if not fire_and_forget:
        _block_until_complete(
            bifrost,
            instance,
            job,
            run_name,
            remote_output_dir,
            use_tui=use_tui,
            tui_debug=tui_debug,
            keep_alive=keep_alive,
        )
        return

    # ── Fire-and-forget mode ──
    assert instance is not None, "fire-and-forget requires a provisioned instance"
    node_id_str = f"{instance.provider}:{instance.id}"

    # Save run metadata for rollouts monitor --attach
    run_info = {
        "run_id": run_name,
        "node_id": node_id_str,
        "remote_output_dir": remote_output_dir,
        "tmux_session": job.tmux_session,
        "log_file": job.log_file,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    _save_active_run(run_info)

    print("\nTraining submitted (fire-and-forget).")
    print(f"  Run:       {run_name}")
    print(f"  Node:      {node_id_str}")
    print(f"  Remote:    {remote_output_dir}")
    print("\nAttach later:")
    print(f"  rollouts monitor --attach {run_name}")
    print("  rollouts monitor --attach --latest")

    if use_tui:
        # Launch rollouts monitor --attach inline
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "rollouts", "monitor", "--attach", run_name],
            check=False,
        )
