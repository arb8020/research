"""Shared utilities for RL examples.

Provides run_remote() for deploying any RL training script to a remote GPU.
"""

from __future__ import annotations

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

    Uses bifrost submit() + job_stream_until_complete() for reliable execution.

    Args:
        script_path: Absolute path to the training script (__file__)
        keep_alive: Keep GPU after completion
        gpu_id: Reuse existing GPU instance (format: "provider:instance_id")
        use_tui: Show TUI monitor for logs
        tui_debug: Print raw JSONL instead of TUI (for debugging)
    """
    from dotenv import load_dotenv

    from bifrost import GPUQuery, ProcessSpec, acquire_node, job_stream_until_complete

    load_dotenv()

    # Generate run name upfront so all logs go to the same place
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Acquire node using bifrost's tri-modal pattern
    if gpu_id:
        # Reuse existing instance
        bifrost, instance = acquire_node(node_id=gpu_id)
        print(f"Connected to existing instance: {gpu_id}")
    else:
        # Provision new instance
        print("Provisioning 2x GPU...")
        bifrost, instance = acquire_node(
            provision=GPUQuery(type="A100", count=2, min_cuda="12.8")
        )
        if instance:
            print(f"Instance: {instance.provider}:{instance.id}")

    # Compute relative path from repo root
    repo_root = Path(__file__).parent.parent.parent  # examples/rl/base_config.py -> repo root
    script_rel_path = Path(script_path).relative_to(repo_root)

    workspace = None
    remote_output_dir = None

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

        # Build environment variables
        env_vars = {"PYTHONUNBUFFERED": "1", "ROLLOUTS_RUN_NAME": run_name}
        if use_tui or tui_debug:
            env_vars["ROLLOUTS_JSON_LOGS"] = "true"

        # Submit training job using new bifrost API
        print(f"Starting training run: {run_name}")
        job = bifrost.submit(
            ProcessSpec(
                command="uv",
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

        if tui_debug:
            # Just print raw output
            print("-" * 50)
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
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

                stream_result = job_stream_until_complete(
                    bifrost, job, on_line=queue_line, timeout=7200, poll_interval=1.0
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
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
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
                print(f"Reuse with: --gpu-id {instance.provider}:{instance.id}")
