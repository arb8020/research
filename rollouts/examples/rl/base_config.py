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
    from kerbal.tmux import start_tmux_session
    from kerbal.job_monitor import LogStreamConfig, stream_log_until_complete

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

        # Run training in tmux with log file (survives SSH disconnects, captures crashes)
        print("Starting training in tmux...")
        training_log = f"{workspace}/training.log"
        env_vars = "PYTHONUNBUFFERED=1"
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
            # TODO: integrate with TUI monitor
            # For now, fall back to raw streaming
            print("-" * 50)
            success, exit_code, err = stream_log_until_complete(bifrost, monitor_config)
            print("-" * 50)
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
        local_results.mkdir(parents=True, exist_ok=True)

        # Always sync the main training log (captures early crashes)
        try:
            result = bifrost.download_files(
                remote_path=f"{workspace}/training.log",
                local_path=str(local_results / "training.log"),
                recursive=False,
            )
            if result and result.success:
                print("  Synced: training.log")
        except Exception:
            pass

        # Sync all timestamped run directories (logs + config, NOT checkpoints)
        remote_base = f"{workspace}/rollouts/results/rl"
        files_to_sync = [
            "config.json",
            "rollouts.jsonl",
            "sglang.log",
            "vllm.log",
        ]

        # List remote directories to find timestamped runs
        try:
            ls_result = bifrost.exec(f"ls -1 {remote_base} 2>/dev/null || true")
            ls_output = ls_result.stdout if hasattr(ls_result, 'stdout') else str(ls_result)
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
