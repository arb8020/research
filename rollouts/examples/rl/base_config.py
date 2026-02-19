"""Shared utilities for RL examples.

Provides run_remote() for deploying any RL training script to a remote GPU.

Two modes:
  --provision (no --tui): fire-and-forget. Submit job, save metadata, exit immediately.
  --provision --tui:      fire-and-forget + launch rollouts monitor --attach inline.
  (no --provision):       blocking. Stream logs until completion, sync results, terminate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from bifrost import BifrostClient, JobInfo
    from broker import ClientGPUInstance


async def _deploy_and_submit(
    script_path: str,
    node_id: str | None,
    gpu_count: int,
    gpu_type: str,
    use_json_logs: bool,
    exposed_ports: tuple[int, ...] = (),
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace).
    """
    import logging

    from dotenv import load_dotenv

    from bifrost import GPUQuery, ProcessSpec, acquire_node
    from pytui import Console

    load_dotenv()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Create console for coordinated spinner + logging output
    console = Console()
    console.install_logging_handler(logging.getLogger())  # Capture all log output

    # Acquire node - show which credentials profile is being used
    from broker.credentials import get_active_profile

    profile_name, _ = get_active_profile()
    profile_hint = f" [{profile_name}]" if profile_name else ""
    provision_msg = "Connecting..." if node_id else f"Provisioning {gpu_count}x {gpu_type}{profile_hint}..."
    with console.spinner(provision_msg) as spinner:
        if node_id:
            bifrost, instance = await acquire_node(node_id=node_id)
            spinner.update(f"Connected to {node_id}")
        else:
            bifrost, instance = await acquire_node(
                provision=GPUQuery(
                    type=gpu_type,
                    count=gpu_count,
                    min_cuda="12.8",
                    exposed_ports=exposed_ports,
                    enable_http_proxy=not exposed_ports,  # raw TCP for LogsServer
                    name=f"rollouts/{run_name}",
                )
            )
            node_str = f"{instance.provider}:{instance.id}" if instance else "?"
            spinner.update(f"Provisioned {node_str}")

    # Deploy code (git sync only, no bootstrap)
    repo_root = Path(__file__).parent.parent.parent
    script_rel_path = Path(script_path).relative_to(repo_root)

    with console.spinner("Deploying code..."):
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", allow_dirty=True)

    # Bootstrap steps — each gets its own spinner with ✓ on completion
    bootstrap_steps = [
        ("Installing system deps", "apt-get update && apt-get install -y tmux libnuma1 || true"),
        (
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        ),
        (
            "Syncing Python deps",
            "cd rollouts && ~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12",
        ),
        (
            "Installing ML packages",
            "~/.local/bin/uv pip install torch 'transformers>=5.0' 'huggingface_hub>=0.27' datasets accelerate sglang[all] curl_cffi peft",
        ),
    ]

    for label, cmd in bootstrap_steps:
        with console.spinner(f"{label}..."):
            bifrost.exec(cmd, working_dir=workspace)

    # Create run output directory
    remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    bifrost.exec(f"mkdir -p {remote_output_dir}")
    training_log = f"{remote_output_dir}/training.log"

    # Build environment variables
    env_vars = {"PYTHONUNBUFFERED": "1", "ROLLOUTS_RUN_NAME": run_name}
    if use_json_logs:
        env_vars["ROLLOUTS_JSON_LOGS"] = "true"

    # Submit training job
    with console.spinner(f"Starting {run_name}...") as spinner:
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
        spinner.update(f"Training started ({job.tmux_session})")

    # Clean up logging handler
    console.remove_logging_handlers()

    return bifrost, instance, job, run_name, remote_output_dir, workspace


async def _block_until_complete(
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
    # TODO: job_stream_until_complete is sync (blocking IO). Fine for training
    # runs that block for hours. Convert to async if we need concurrency here.
    from bifrost import job_stream_until_complete

    try:
        if tui_debug:
            logger.info("-" * 50)
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
            logger.info("-" * 50)
            if not success:
                logger.error("Training failed: %s (exit code: %s)", err, exit_code)
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
                logger.error("Training failed: %s (exit code: %s)", err, exit_code)
        else:
            logger.info("-" * 50)
            success, exit_code, err = job_stream_until_complete(
                bifrost, job, timeout=7200, poll_interval=1.0
            )
            logger.info("-" * 50)
            if not success:
                logger.error("Training failed: %s (exit code: %s)", err, exit_code)

    except KeyboardInterrupt:
        logger.warning("Interrupted! Syncing logs before exit...")

    finally:
        await _sync_and_cleanup(bifrost, instance, run_name, remote_output_dir, keep_alive)


async def _sync_and_cleanup(
    bifrost: BifrostClient,
    instance: ClientGPUInstance | None,
    run_name: str,
    remote_output_dir: str | None,
    keep_alive: bool,
) -> None:
    """Sync results from remote and optionally terminate instance."""
    logger.info("Syncing results...")
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
                    logger.info("Synced: %s/%s", run_name, filename)
            except Exception:
                pass

    if instance:
        if not keep_alive:
            logger.info("Terminating instance %s:%s...", instance.provider, instance.id)
            await instance.terminate()
        else:
            logger.info("Instance kept alive: %s:%s", instance.provider, instance.id)
            logger.info("Reuse with: --node-id %s:%s", instance.provider, instance.id)


async def run_remote(
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

    # Expose LogsServer port for fire-and-forget monitoring
    logs_port = 9100
    exposed_ports = (logs_port,) if fire_and_forget else ()

    bifrost, instance, job, run_name, remote_output_dir, workspace = await _deploy_and_submit(
        script_path=script_path,
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        use_json_logs=use_json_logs,
        exposed_ports=exposed_ports,
    )

    if not fire_and_forget:
        await _block_until_complete(
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

    # Start LogsServer on the remote node
    from bifrost import ProcessSpec

    logger.info("Starting LogsServer...")
    # Use relative path for --dir since cwd is workspace
    # remote_output_dir is {workspace}/rollouts/results/rl/{run_name}
    logs_dir_relative = f"rollouts/results/rl/{run_name}"
    bifrost.submit(
        ProcessSpec(
            command="python3",
            args=(
                "-m",
                "miniray.logs_server",
                "--port",
                str(logs_port),
                "--dir",
                logs_dir_relative,
            ),
            cwd=workspace,  # Root so `python3 -m miniray.logs_server` can find miniray/
        ),
        name="logs-server",
        log_file=f"{remote_output_dir}/logs_server.log",
        workspace=workspace,
    )

    # Register job in ~/.rollouts/jobs.json
    from rollouts.jobs import make_job

    make_job(
        job_id=run_name,
        provider=instance.provider,
        node_id=instance.id,
        script=script_path,
    )

    logger.info("Training submitted (fire-and-forget).")
    logger.info("  Run:       %s", run_name)
    logger.info("  Node:      %s", node_id_str)
    logger.info("  Remote:    %s", remote_output_dir)
    logger.info("Attach later:")
    logger.info("  rollouts monitor --attach %s", run_name)
    logger.info("  rollouts monitor --attach --latest")

    if use_tui:
        # Launch rollouts monitor --attach inline
        import subprocess
        import sys

        subprocess.run(
            [sys.executable, "-m", "rollouts", "monitor", "--attach", run_name],
            check=False,
        )
