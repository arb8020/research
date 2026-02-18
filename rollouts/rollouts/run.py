#!/usr/bin/env python3
"""Unified training runner.

Usage:
    # Local execution (requires local GPU)
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py

    # Modal execution (fast ~30s cold start, recommended for CI)
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --modal
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --modal --gpu H100

    # RunPod/SSH execution (slower 2-5 min cold start)
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --provision
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --node-id runpod:abc123

The config file must export:
    - config: A training config (e.g., GRPOConfig)
    - train(config, **kwargs): Function to run local training

Execution modes:
    --modal:     Run on Modal sandbox (fast cold start, ~30s)
    --provision: Provision new GPU instance via SSH (RunPod, etc.)
    --node-id:   Reuse existing SSH instance (provider:id format)
    (none):      Run locally

Options:
    --detach:    Submit and exit (don't launch TUI) [SSH only]
    --keep-alive: Keep instance running after completion [SSH only]
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker import ClientGPUInstance

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_config"] = module
    spec.loader.exec_module(module)
    return module


def _setup_run_logging(run_dir: Path) -> Callable[[str, Any], None]:
    """Create run directory and return a logging function."""
    import json
    from datetime import datetime

    run_dir.mkdir(parents=True, exist_ok=True)
    log_file = run_dir / "run.jsonl"

    def log_event(event: str, **data: Any) -> None:
        entry = {
            "ts": datetime.now().isoformat(),
            "event": event,
            **data,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    return log_event


async def _deploy_and_submit(
    script_path: str,
    node_id: str | None,
    gpu_count: int,
    gpu_type: str,
    provider: str | None = None,
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir).
    """
    from bifrost import GPUQuery, ProcessSpec, acquire_node
    from broker import AccountError, ProvisionError
    from pytui import Console

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Create local run directory immediately for logging
    local_run_dir = REPO_ROOT / "results" / "rl" / run_name
    log = _setup_run_logging(local_run_dir)
    log("run_start", config=script_path, gpu_count=gpu_count, gpu_type=gpu_type, node_id=node_id)

    logs_port = 9100

    # Create console for coordinated spinner + logging output
    console = Console()
    console.install_logging_handler(logging.getLogger())

    # Acquire node - show which credentials profile is being used
    from broker.credentials import get_active_profile

    profile_name, _ = get_active_profile()
    profile_hint = f" [{profile_name}]" if profile_name else ""

    if node_id:
        provision_msg = "Connecting..."
    elif provider:
        provision_msg = f"Provisioning {gpu_count}x {gpu_type} on {provider}{profile_hint}..."
    else:
        provision_msg = f"Provisioning {gpu_count}x {gpu_type}{profile_hint}..."
    log("provision_start", msg=provision_msg)
    try:
        with console.spinner(provision_msg) as spinner:
            if node_id:
                bifrost, instance = await acquire_node(node_id=node_id)
                spinner.update(f"Connected to {node_id}")
                log("provision_done", node_id=node_id, reused=True)
            else:
                bifrost, instance = await acquire_node(
                    provision=GPUQuery(
                        type=gpu_type,
                        count=gpu_count,
                        min_cuda="12.8",
                        exposed_ports=(logs_port,),
                        name=f"rollouts/{run_name}",
                        provider=provider,
                    )
                )
                node_str = f"{instance.provider}:{instance.id}" if instance else "?"
                spinner.update(f"Provisioned {node_str}")
                log(
                    "provision_done",
                    node_id=node_str,
                    provider=instance.provider if instance else None,
                )
    except AccountError as e:
        logger.debug("AccountError details", exc_info=True)
        print(f"\nError: {e.user_message()}", file=sys.stderr)
        sys.exit(1)
    except ProvisionError as e:
        logger.debug("ProvisionError details", exc_info=True)
        # Surface categorized one-liner based on result
        result = e.result
        if result.credential_error:
            print("\nError: Invalid API credentials. Check your API keys.", file=sys.stderr)
        elif result.no_offers_found:
            print(
                f"\nError: No {gpu_type} GPUs found. Try a different --gpu-type.",
                file=sys.stderr,
            )
        elif result.all_unavailable:
            print(
                f"\nError: No {gpu_type} GPUs available right now. Try again later or use --gpu-type to pick a different GPU.",
                file=sys.stderr,
            )
        elif result.network_error:
            print("\nError: Network error reaching GPU provider. Try again.", file=sys.stderr)
        else:
            print(f"\nError: Provisioning failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Deploy code (git sync only, no bootstrap)
    script_rel_path = Path(script_path).relative_to(REPO_ROOT)

    log("deploy_start")
    with console.spinner("Deploying code..."):
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", allow_dirty=True)
    log("deploy_done", workspace=workspace)

    # Bootstrap steps — each gets its own spinner with ✓ on completion
    bootstrap_steps = [
        ("Installing system deps", "apt-get update && apt-get install -y tmux libnuma1 || true"),
        (
            "Installing uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        ),
        (
            "Syncing Python deps",
            "~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12 --package rollouts",
        ),
        (
            "Installing ML packages",
            "~/.local/bin/uv pip install torch transformers datasets accelerate sglang[all] curl_cffi peft",
        ),
    ]

    for label, cmd in bootstrap_steps:
        log("bootstrap_step_start", label=label)
        with console.spinner(f"{label}..."):
            bifrost.exec(cmd, working_dir=workspace)
        log("bootstrap_step_done", label=label)

    # Create run output directory
    remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    bifrost.exec(f"mkdir -p {remote_output_dir}")
    training_log = f"{remote_output_dir}/training.log"

    env_vars = {
        "PYTHONUNBUFFERED": "1",
        "ROLLOUTS_RUN_NAME": run_name,
        "ROLLOUTS_OUTPUT_DIR": f"results/rl/{run_name}",
        "ROLLOUTS_JSON_LOGS": "true",
    }

    # Submit training job
    log("submit_start")
    with console.spinner(f"Starting {run_name}...") as spinner:
        job = bifrost.submit(
            ProcessSpec(
                command="/root/.local/bin/uv",
                args=("run", "python", "-m", "rollouts.run", "--config", str(script_rel_path)),
                cwd=f"{workspace}/rollouts",
                env=env_vars,
            ),
            name="rl-training",
            log_file=training_log,
            workspace=f"{workspace}/rollouts",
        )
        spinner.update(f"Training started ({job.tmux_session})")
    log("submit_done", tmux_session=job.tmux_session)

    return bifrost, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir


async def _sync_and_cleanup(
    bifrost: BifrostClient,
    instance: ClientGPUInstance | None,
    run_name: str,
    remote_output_dir: str | None,
    keep_alive: bool,
) -> None:
    """Sync results from remote and optionally terminate instance.

    Currently unused - monitor handles sync/terminate internally.
    Kept for future --detach cleanup support.
    """
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
    detach: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
    tail: bool = False,
    provider: str | None = None,
) -> None:
    """Run training script on remote GPU via bifrost."""
    (
        bifrost,
        instance,
        job,
        run_name,
        remote_output_dir,
        workspace,
        console,
        local_run_dir,
    ) = await _deploy_and_submit(
        script_path=script_path,
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        provider=provider,
    )

    assert instance is not None, "run_remote requires a provisioned instance"
    node_id_str = f"{instance.provider}:{instance.id}"

    from bifrost import ProcessSpec

    logs_port = 9100
    logs_dir_relative = f"rollouts/results/rl/{run_name}"

    # Start LogsServer - use -m since workspace is repo root with miniray/ dir
    logger.info("Starting LogsServer...")
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
            cwd=workspace,
        ),
        name="logs-server",
        log_file=f"{remote_output_dir}/logs_server.log",
        workspace=workspace,
    )

    from rollouts.jobs import make_job

    make_job(
        job_id=run_name,
        provider=instance.provider,
        node_id=instance.id,
        script=script_path,
    )

    logger.info("Training submitted: %s", run_name)
    logger.info("  Node:   %s", node_id_str)
    logger.info("  Remote: %s", remote_output_dir)

    if detach:
        logger.info("Detached. Attach later:")
        logger.info("  rollouts monitor --attach %s", run_name)
        return

    import subprocess

    if not tail:
        logger.info("Launching TUI...")
        # Clean up logging handler before launching TUI (it has its own output)
        console.remove_logging_handlers()
        # If no handlers remain, Python's logging.lastResort will still emit WARNING+
        # to stderr, corrupting the TUI. Install a NullHandler to keep the terminal clean.
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            root_logger.addHandler(logging.NullHandler())

    monitor_cmd = [sys.executable, "-m", "rollouts", "monitor", "--attach", run_name]
    if tail:
        monitor_cmd.append("--tail")
    if keep_alive:
        monitor_cmd.append("--keep-alive")
    else:
        monitor_cmd.append("--terminate")
    subprocess.run(monitor_cmd, check=False)
    # Note: monitor handles final sync and terminate internally


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Local
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py

    # Modal (recommended for CI)
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --modal

    # RunPod
    python -m rollouts.run --config examples/rl/reverse_text/grpo_01_01.py --provision
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # Remote execution
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance (provider:id)")
    parser.add_argument(
        "--modal", action="store_true", help="Run on Modal sandbox (fast cold start)"
    )
    parser.add_argument("--detach", action="store_true", help="Submit and exit (don't launch TUI)")
    parser.add_argument("--tail", action="store_true", help="Stream logs to stdout instead of TUI")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--gpu-type", type=str, default="A100", help="GPU type (default: A100)")
    parser.add_argument("--provider", type=str, help="Force specific provider (e.g., runpod, vast)")

    # Local execution
    parser.add_argument("--max-samples", type=int, help="Limit dataset size (local only)")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Config: {config_path}")

    if args.modal:
        # Modal execution (fast cold start)
        import trio

        from .modal_runner import ModalRunConfig, run_modal

        modal_config = ModalRunConfig(
            config_path=str(config_path),
            gpu_type=args.gpu_type,
            gpu_count=args.gpu_count,
        )
        results = trio.run(run_modal, modal_config)
        if not results.get("success"):
            sys.exit(1)
    elif args.provision or args.node_id:
        # Remote execution via SSH (RunPod, etc.)
        import trio

        trio.run(
            run_remote,
            str(config_path),
            args.keep_alive,
            args.node_id,
            args.detach,
            args.gpu_count,
            args.gpu_type,
            args.tail,
            args.provider,
        )
    else:
        # Local execution
        config_module = load_config_module(config_path)

        if not hasattr(config_module, "config"):
            print("Config file must export 'config'", file=sys.stderr)
            sys.exit(1)

        if not hasattr(config_module, "train"):
            print("Config file must export 'train' function for local execution", file=sys.stderr)
            sys.exit(1)

        kwargs = {}
        if args.max_samples is not None:
            kwargs["max_samples"] = args.max_samples

        results = config_module.train(config=config_module.config, **kwargs)
        print(f"Training complete. {len(results.get('metrics_history', []))} steps")


if __name__ == "__main__":
    main()
