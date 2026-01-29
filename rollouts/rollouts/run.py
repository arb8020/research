#!/usr/bin/env python3
"""Unified training runner.

Usage:
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py --provision
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py --node-id runpod:abc123
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py --provision --detach

The config file must export:
    - config: A training config (e.g., GRPOConfig)
    - train(config, **kwargs): Function to run local training

Remote execution:
    --provision: Provision a new GPU instance
    --node-id:   Reuse an existing instance (provider:id format)
    --detach:    Submit and exit (don't launch TUI)
    --keep-alive: Keep instance running after completion

Local execution (no --provision or --node-id):
    Calls train(config) directly
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
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


async def _deploy_and_submit(
    script_path: str,
    node_id: str | None,
    gpu_count: int,
    gpu_type: str,
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace).
    """
    from dotenv import load_dotenv

    from bifrost import GPUQuery, ProcessSpec, acquire_node

    load_dotenv()

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    from pytui import Spinner

    spinner = Spinner("Provisioning..." if not node_id else "Connecting...")
    spinner.start()

    logs_port = 9100

    if node_id:
        bifrost, instance = await acquire_node(node_id=node_id)
        spinner.stop(f"Connected to {node_id}")
    else:
        spinner.update(f"Provisioning {gpu_count}x {gpu_type}...")
        bifrost, instance = await acquire_node(
            provision=GPUQuery(
                type=gpu_type,
                count=gpu_count,
                min_cuda="12.8",
                exposed_ports=(logs_port,),
                name=f"rollouts/{run_name}",
            )
        )
        node_str = f"{instance.provider}:{instance.id}" if instance else "?"
        spinner.stop(f"Provisioned {node_str}")

    # Deploy code
    script_rel_path = Path(script_path).relative_to(REPO_ROOT)

    spinner = Spinner("Deploying code...")
    spinner.start()
    bootstrap = [
        "apt-get update && apt-get install -y tmux libnuma1 || true",
        "curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env",
        "cd rollouts && ~/.local/bin/uv python install 3.12 && ~/.local/bin/uv sync --python 3.12",
        "~/.local/bin/uv pip install torch transformers datasets accelerate sglang[all] curl_cffi peft",
    ]
    workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", bootstrap_cmd=bootstrap)
    spinner.stop("Code deployed")

    # Create run output directory
    remote_output_dir = f"{workspace}/rollouts/results/rl/{run_name}"
    bifrost.exec(f"mkdir -p {remote_output_dir}")
    training_log = f"{remote_output_dir}/training.log"

    env_vars = {
        "PYTHONUNBUFFERED": "1",
        "ROLLOUTS_RUN_NAME": run_name,
        "ROLLOUTS_JSON_LOGS": "true",
    }

    spinner = Spinner(f"Starting {run_name}...")
    spinner.start()
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
    spinner.stop(f"Training started ({job.tmux_session})")

    return bifrost, instance, job, run_name, remote_output_dir, workspace


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
    detach: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
) -> None:
    """Run training script on remote GPU via bifrost."""
    bifrost, instance, job, run_name, remote_output_dir, workspace = await _deploy_and_submit(
        script_path=script_path,
        node_id=node_id,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
    )

    assert instance is not None, "run_remote requires a provisioned instance"
    node_id_str = f"{instance.provider}:{instance.id}"

    from bifrost import ProcessSpec

    logs_port = 9100
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

    logger.info("Launching TUI...")
    subprocess.run(
        [sys.executable, "-m", "rollouts", "monitor", "--attach", run_name],
        check=False,
    )

    await _sync_and_cleanup(bifrost, instance, run_name, remote_output_dir, keep_alive)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py --provision
    python -m rollouts.run --config examples/rl/calculator/grpo_01_01.py --node-id runpod:abc123
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # Remote execution
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance (provider:id)")
    parser.add_argument("--detach", action="store_true", help="Submit and exit (don't launch TUI)")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--gpu-count", type=int, default=1, help="Number of GPUs (default: 1)")
    parser.add_argument("--gpu-type", type=str, default="A100", help="GPU type (default: A100)")

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

    if args.provision or args.node_id:
        # Remote execution
        import trio

        trio.run(
            run_remote,
            str(config_path),
            args.keep_alive,
            args.node_id,
            args.detach,
            args.gpu_count,
            args.gpu_type,
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
