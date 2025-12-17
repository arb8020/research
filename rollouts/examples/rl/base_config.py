"""Shared utilities for RL examples.

Provides run_remote() for deploying any RL training script to a remote GPU.
"""

from __future__ import annotations

import os
from pathlib import Path


def run_remote(
    script_path: str,
    keep_alive: bool = False,
    gpu_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
) -> None:
    """Run training script on remote GPU via broker/bifrost.

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

    load_dotenv()

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

        # Run training (PYTHONUNBUFFERED=1 for real-time output)
        print("Running training...")
        env_vars = "PYTHONUNBUFFERED=1"
        if use_tui or tui_debug:
            env_vars += " ROLLOUTS_JSON_LOGS=true"
        cmd = f"cd {workspace}/rollouts && {env_vars} uv run python {script_rel_path}"
        print(f"Running: {cmd}")

        if tui_debug:
            # Just print raw JSONL for debugging
            print("-" * 50)
            for line in bifrost.exec_stream(cmd):
                print(line, flush=True)
            print("-" * 50)
        elif use_tui:
            # Route output through TUI monitor
            from rollouts.tui.monitor import TrainingMonitor

            monitor = TrainingMonitor()

            def feed_monitor():
                """Feed lines to monitor in background."""
                import threading
                import time

                lines_queue: list[str] = []
                done = threading.Event()

                def collect_lines():
                    try:
                        for line in bifrost.exec_stream(cmd):
                            lines_queue.append(line)
                    finally:
                        done.set()

                collector = threading.Thread(target=collect_lines, daemon=True)
                collector.start()

                # Run TUI with line feeding
                monitor._running = True
                monitor.terminal.start(on_input=lambda x: None, on_resize=monitor._on_resize)

                try:
                    while monitor._running and not done.is_set():
                        # Feed queued lines
                        while lines_queue:
                            raw_line = lines_queue.pop(0)
                            log_line = monitor.parse_jsonl_line(raw_line)
                            if log_line:
                                pane_name = monitor.route_log_line(log_line)
                                monitor.panes[pane_name].add_line(log_line)
                                monitor._needs_redraw = True

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

            feed_monitor()
        else:
            print("-" * 50)
            for line in bifrost.exec_stream(cmd):
                print(line, flush=True)
            print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted! Syncing logs before exit...")

    finally:
        # Always sync results/logs (even on ctrl+c)
        print("\nSyncing results...")
        local_results = Path("results/rl")
        local_results.mkdir(parents=True, exist_ok=True)

        # Sync all timestamped run directories (logs + config, NOT checkpoints)
        remote_base = "results/rl"
        files_to_sync = [
            "config.json",
            "training.jsonl",
            "sglang_server.log",
        ]

        # List remote directories to find timestamped runs
        try:
            ls_output = bifrost.exec(f"ls -1 {remote_base}")
            remote_dirs = [d.strip() for d in ls_output.split("\n") if d.strip()]
        except Exception:
            remote_dirs = []

        for remote_dir in remote_dirs:
            local_run_dir = local_results / remote_dir
            local_run_dir.mkdir(parents=True, exist_ok=True)

            for filename in files_to_sync:
                try:
                    result = bifrost.download_files(
                        remote_path=f"{remote_base}/{remote_dir}/{filename}",
                        local_path=str(local_run_dir / filename),
                        recursive=False,
                    )
                    if result and result.success:
                        print(f"  Synced: {remote_dir}/{filename}")
                except Exception:
                    pass

        if not keep_alive:
            print(f"\nTerminating instance {instance.provider}:{instance.id}...")
            instance.terminate()
        else:
            print(f"\nInstance kept alive: {instance.provider}:{instance.id}")
            print(f"Reuse with: --gpu-id {instance.provider}:{instance.id}")
