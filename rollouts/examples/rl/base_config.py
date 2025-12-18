"""Shared utilities for RL examples.

Provides run_remote() for deploying any RL training script to a remote GPU.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path


def run_remote(
    script_path: str,
    keep_alive: bool = False,
    gpu_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
) -> None:
    """Run training script on remote GPU via broker/bifrost.

    Uses tmux + log file pattern for reliable logging even on crashes.
    Based on kerbal/wafer_stuff patterns.

    Args:
        script_path: Absolute path to the training script (__file__)
        keep_alive: Keep GPU after completion
        gpu_id: Reuse existing GPU instance (format: "provider:instance_id")
        use_tui: Show TUI monitor for logs
        tui_debug: Print raw JSONL instead of TUI (for debugging)
    """
    from dotenv import load_dotenv

    from bifrost.client import BifrostClient
    from broker.client import GPUClient
    from kerbal.job_monitor import LogStreamConfig, stream_log_until_complete
    from kerbal.tmux import start_tmux_session

    load_dotenv()

    # Generate run name upfront so all logs go to the same place
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Get credentials
    runpod_key = os.getenv("RUNPOD_API_KEY")
    ssh_key_path = os.path.expanduser("~/.ssh/id_ed25519")

    assert runpod_key, "RUNPOD_API_KEY required for remote execution"

    client_gpu = GPUClient(
        credentials={"runpod": runpod_key},
        ssh_key_path=ssh_key_path,
    )

    # Get or provision GPU
    if gpu_id:
        provider, instance_id = gpu_id.split(":", 1)
        instance = client_gpu.get_instance(instance_id, provider)
        assert instance, f"Instance not found: {gpu_id}"
    else:
        print("Provisioning 2x GPU...")
        instance = client_gpu.create(
            client_gpu.gpu_type.contains("A100") | client_gpu.gpu_type.contains("4090"),
            gpu_count=2,
            cloud_type="secure",
            container_disk_gb=100,
            sort=lambda x: x.price_per_hour,
            min_cuda_version="12.8",  # Ensure compatible driver for PyTorch 2.x
        )
        print(f"Instance: {instance.provider}:{instance.id}")

    print("Waiting for SSH...")
    instance.wait_until_ssh_ready(timeout=600)

    key_path = client_gpu.get_ssh_key_path(instance.provider)
    bifrost = BifrostClient(instance.ssh_connection_string(), ssh_key_path=key_path)

    # Compute relative path from repo root
    repo_root = Path(__file__).parent.parent.parent  # examples/rl/base_config.py -> repo root
    script_rel_path = Path(script_path).relative_to(repo_root)

    try:
        # Deploy code with bootstrap
        print("Deploying code...")
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch transformers datasets accelerate sglang[all] curl_cffi",
        ]
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Create run output directory and set up training log
        remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
        bifrost.exec(f"mkdir -p {remote_output_dir}")
        training_log = f"{remote_output_dir}/training.log"

        # Build command with run name passed via env var
        print(f"Starting training run: {run_name}")
        env_vars = f"PYTHONUNBUFFERED=1 ROLLOUTS_RUN_NAME={run_name}"
        if use_tui or tui_debug:
            env_vars += " ROLLOUTS_JSON_LOGS=true"
        cmd = f"{env_vars} uv run python {script_rel_path}"
        print(f"Command: {cmd}")

        session_name = "rl-training"
        session, err = start_tmux_session(
            client=bifrost,
            session_name=session_name,
            command=cmd,
            workspace=f"{workspace}/rollouts",
            log_file=training_log,
            capture_exit_code=True,
        )
        if err:
            print(f"Failed to start tmux session: {err}")
            return

        print(f"Training started in tmux session: {session}")
        print(f"Log file: {training_log}")

        # Stream logs until complete
        monitor_config = LogStreamConfig(
            session_name=session_name,
            log_file=training_log,
            timeout_sec=7200,  # 2 hours
            poll_interval_sec=1.0,
        )

        if tui_debug:
            # Just print raw output (stream_log_until_complete prints to stdout)
            print("-" * 50)
            success, exit_code, err = stream_log_until_complete(bifrost, monitor_config)
            print("-" * 50)
            if not success:
                print(f"Training failed: {err} (exit code: {exit_code})")
        elif use_tui:
            # Route output through TUI monitor
            import threading
            import time

            from rollouts.tui.monitor import TrainingMonitor

            monitor = TrainingMonitor()
            lines_queue: list[str] = []
            streaming_done = threading.Event()
            stream_result: tuple[bool, int | None, str | None] = (False, None, None)

            def collect_lines():
                """Stream logs in background, queue lines for TUI."""
                nonlocal stream_result

                def queue_line(line: str):
                    lines_queue.append(line)

                stream_result = stream_log_until_complete(
                    bifrost, monitor_config, on_line=queue_line
                )
                streaming_done.set()

            # Start streaming in background thread
            collector = threading.Thread(target=collect_lines, daemon=True)
            collector.start()

            # Run TUI with line feeding (main thread)
            monitor._running = True
            monitor.terminal.start(on_input=lambda x: None, on_resize=monitor._on_resize)

            try:
                while monitor._running and not streaming_done.is_set():
                    # Feed queued lines to monitor
                    while lines_queue:
                        raw_line = lines_queue.pop(0)
                        monitor.feed_line(raw_line)

                    # Handle keyboard input
                    data = monitor.terminal.read_input()
                    if data:
                        monitor._handle_input(data)

                    # Render if needed
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
            success, exit_code, err = stream_log_until_complete(bifrost, monitor_config)
            print("-" * 50)
            if not success:
                print(f"Training failed: {err} (exit code: {exit_code})")

    except KeyboardInterrupt:
        print("\n\nInterrupted! Syncing logs before exit...")

    finally:
        # Always sync results/logs (even on ctrl+c or crash)
        print("\nSyncing results...")
        local_results = Path("results/rl")
        local_run_dir = local_results / run_name
        local_run_dir.mkdir(parents=True, exist_ok=True)

        # Sync all run artifacts from the single run directory
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

        if not keep_alive:
            print(f"\nTerminating instance {instance.provider}:{instance.id}...")
            instance.terminate()
        else:
            print(f"\nInstance kept alive: {instance.provider}:{instance.id}")
            print(f"Reuse with: --gpu-id {instance.provider}:{instance.id}")
