#!/usr/bin/env python3
"""Unified training runner.

Usage:
    # Config-driven execution (NEW: reads hardware from config file)
    python -m rollouts.run --config examples/rl/kernelbench/grpo_01_01.py

    # CLI overrides (optional, override config values)
    python -m rollouts.run --config ... --gpu-type H100  # Override GPU type
    python -m rollouts.run --config ... --provider modal  # Override provider
    python -m rollouts.run --config ... --local  # Force local execution

    # Legacy CLI flags (still supported for backwards compat)
    python -m rollouts.run --config ... --modal  # Same as --provider modal
    python -m rollouts.run --config ... --provision  # Provision via config.hardware.provider

The config file should export:
    - config: A training config (e.g., GRPOConfig)
    - hardware: HardwareConfig (optional, defaults to local execution)
    - train(config, **kwargs): Function to run local training

Execution modes (determined by hardware.provider or CLI override):
    - "local":     Run on local GPU
    - "modal":     Run on Modal sandbox (fast ~30s cold start)
    - "runpod":    Provision GPU via RunPod SSH
    - "lambdalabs": Provision GPU via Lambda Labs
    - "vast":      Provision GPU via Vast.ai

Options:
    --tui:       Launch TUI after submitting (default: fire-and-forget) [SSH only]
    --tail:      Stream logs to stdout (default: fire-and-forget) [SSH only]
    --keep-alive: Keep instance running after completion [SSH only]
    --node-id:   Reuse existing SSH instance (provider:id format)

Attach to running job:
    rollouts monitor --attach <run_name>
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from collections.abc import Callable, Generator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any


@contextmanager
def _quiet_spinner(msg: str) -> Generator[None, None, None]:
    """No-op context manager for quiet mode.

    Inner operations (like bifrost.acquire_node) log their own progress,
    so we just yield without adding wrapper messages.
    """
    yield None


if TYPE_CHECKING:
    from bifrost import BifrostClient
    from broker import ClientGPUInstance

    from .training.multi_node import MultiNodeConfig

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
    allow_dirty: bool = False,
    quiet: bool = False,
) -> tuple:
    """Provision node, deploy code, submit training job.

    Returns (bifrost_client, instance, job, run_name, remote_output_dir, workspace, console, local_run_dir).
    """
    from bifrost import GPUQuery, ProcessSpec, acquire_node
    from broker import AccountError, ProvisionError
    from pytui import Console

    from .jobs import register_job, update_job_node

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"run_{timestamp}"

    # Create local run directory immediately for logging
    local_run_dir = REPO_ROOT / "results" / "rl" / run_name
    log = _setup_run_logging(local_run_dir)
    log("run_start", config=script_path, gpu_count=gpu_count, gpu_type=gpu_type, node_id=node_id)

    # Register job in local registry BEFORE provisioning
    # Use placeholder node_id if reusing, will be updated after provision
    initial_provider = "runpod"  # Default, updated after provision
    initial_node_id = "pending"
    if node_id:
        parts = node_id.split(":", 1)
        if len(parts) == 2:
            initial_provider, initial_node_id = parts
    elif provider:
        initial_provider = provider

    register_job(
        job_id=run_name,
        provider=initial_provider,
        node_id=initial_node_id,
        config_path=script_path,
        log_path=f"results/rl/{run_name}",
    )

    logs_port = 9100

    # Create console for coordinated spinner + logging output
    # In quiet mode, skip spinners and just use plain logging to stderr
    if quiet:
        console = None
        spinner = _quiet_spinner
        # Ensure logs go to stderr so agents see progress
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            stream=sys.stderr,
            force=True,
        )
        # Silence noisy HTTP loggers
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
    else:
        console = Console()
        console.install_logging_handler(logging.getLogger())
        spinner = console.spinner  # Use interactive spinner

    # Fail-fast: check for uncommitted changes BEFORE provisioning
    # This prevents wasting time/money on a pod we can't deploy to
    if not allow_dirty:
        from bifrost.git_sync import _check_uncommitted_changes, _check_untracked_files

        uncommitted = _check_uncommitted_changes() or []
        untracked = _check_untracked_files() or []
        if uncommitted or untracked:
            print(
                "\n❌ Deploy uses git bundles - only committed code is shipped to the pod.\n",
                file=sys.stderr,
            )
            total = len(uncommitted) + len(untracked)
            print(f"{total} file(s) will NOT be deployed:\n", file=sys.stderr)
            for f in uncommitted[:5]:
                print(f"   - {f} (modified)", file=sys.stderr)
            if len(uncommitted) > 5:
                print(f"   ... and {len(uncommitted) - 5} more modified", file=sys.stderr)
            for f in untracked[:5]:
                print(f"   - {f} (untracked)", file=sys.stderr)
            if len(untracked) > 5:
                print(f"   ... and {len(untracked) - 5} more untracked", file=sys.stderr)
            print("\nTo include them: git add <file> && git commit", file=sys.stderr)
            print("To proceed without them: --allow-dirty", file=sys.stderr)
            sys.exit(1)

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
        with spinner(provision_msg) as spin:
            if node_id:
                bifrost, instance = await acquire_node(node_id=node_id)
                if spin:
                    spin.update(f"Connected to {node_id}")
                log("provision_done", node_id=node_id, reused=True)
                # Update job registry with actual node info
                if instance:
                    update_job_node(run_name, instance.provider, instance.id)
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
                if spin:
                    spin.update(f"Provisioned {node_str}")
                log(
                    "provision_done",
                    node_id=node_str,
                    provider=instance.provider if instance else None,
                )
                # Update job registry with actual node info
                if instance:
                    update_job_node(run_name, instance.provider, instance.id)
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
    with spinner("Deploying code..."):
        workspace = bifrost.push("~/.bifrost/workspaces/rollouts-rl", allow_dirty=allow_dirty)
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
            # sglang 0.5.8 (latest PyPI) is incompatible with transformers>=5.x in two ways:
            #   1. transformers 4.57.1 lacks is_offline_mode (removed in huggingface_hub>=1.4)
            #   2. janus_pro.py calls AutoImageProcessor.register() in a way that broke in transformers 5.x
            # Both are fixed in sglang git main. Install from git, then pin transformers/hf_hub.
            # See: https://github.com/sgl-project/sglang/issues/4159
            "~/.local/bin/uv pip install --upgrade torch datasets accelerate curl_cffi peft"
            " 'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python'"
            " && ~/.local/bin/uv pip install --upgrade 'transformers>=5.0.0' 'huggingface_hub>=1.4.0'",
        ),
    ]

    for label, cmd in bootstrap_steps:
        log("bootstrap_step_start", label=label)
        with spinner(f"{label}..."):
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
    with spinner(f"Starting {run_name}...") as spin:
        job = bifrost.submit(
            ProcessSpec(
                command="/root/.local/bin/uv",
                args=(
                    "run",
                    "python",
                    "-m",
                    "rollouts.run",
                    "--config",
                    str(script_rel_path),
                    "--local",
                ),
                cwd=f"{workspace}/rollouts",
                env=env_vars,
            ),
            name=run_name,  # Unique per run for tmux session isolation
            log_file=training_log,
            workspace=f"{workspace}/rollouts",
        )
        if spin:
            spin.update(f"Training started ({job.tmux_session})")
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
    tui: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
    tail: bool = False,
    provider: str | None = None,
    allow_dirty: bool = False,
    quiet: bool = False,
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
        allow_dirty=allow_dirty,
        quiet=quiet,
    )

    assert instance is not None, "run_remote requires a provisioned instance"
    node_id_str = f"{instance.provider}:{instance.id}"

    from bifrost import ProcessSpec

    logs_port = 9100
    logs_dir_relative = f"rollouts/results/rl/{run_name}"

    # Kill any existing LogsServer on this port (left over from a previous run on the same pod)
    # Use multiple methods since not all may be available/work on all systems
    bifrost.exec(f"fuser -k {logs_port}/tcp 2>/dev/null || true")
    bifrost.exec(f"lsof -ti:{logs_port} | xargs -r kill -9 2>/dev/null || true")
    bifrost.exec("pkill -f 'miniray.logs_server' 2>/dev/null || true")
    bifrost.exec(
        "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^bifrost-job-logs-' | xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true"
    )

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
        name=f"logs-{run_name}",  # Unique per run for isolation
        log_file=f"{remote_output_dir}/logs_server.log",
        workspace=workspace,
    )

    logger.info("Training submitted: %s", run_name)
    logger.info("  Node:   %s", node_id_str)
    logger.info("  Local:  results/rl/%s/", run_name)

    # Start background sync daemon so logs appear locally
    # This lets agents `tail -f results/rl/<run_name>/training.log`
    import shutil
    import subprocess

    sync_session = f"sync_{run_name}"

    if shutil.which("tmux"):
        # Start sync daemon in a detached tmux session
        # cd to REPO_ROOT so logs sync to rollouts/results/rl/
        sync_cmd = [
            "tmux",
            "new",
            "-d",
            "-s",
            sync_session,
            f"cd {REPO_ROOT} && {sys.executable} -m rollouts monitor --attach {run_name} --sync-only",
        ]
        subprocess.run(sync_cmd, check=False, capture_output=True)
        local_log_path = REPO_ROOT / "results" / "rl" / run_name / "training.log"
        logger.info("Syncing to: %s", local_log_path)
    else:
        logger.info("Install tmux for automatic log sync")

    # Default: fire-and-forget (print attach instructions and exit)
    if not tui and not tail:
        if keep_alive:
            logger.info("  (instance will stay alive)")
        else:
            logger.info("  (instance will terminate when job completes)")
        return

    # --tui or --tail: launch monitor
    import subprocess

    if tui:
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
    from dataclasses import replace

    from .training.configs import HardwareConfig

    parser = argparse.ArgumentParser(
        description="Run RL training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Config-driven (reads hardware from config file)
    python -m rollouts.run --config examples/rl/kernelbench/grpo_01_01.py

    # Override provider via CLI
    python -m rollouts.run --config ... --provider modal
    python -m rollouts.run --config ... --local

    # Legacy flags (still supported)
    python -m rollouts.run --config ... --modal
    python -m rollouts.run --config ... --provision --provider runpod
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # Provider/hardware overrides
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "modal", "runpod", "lambdalabs", "vast"],
        help="Override hardware provider from config",
    )
    parser.add_argument("--local", action="store_true", help="Force local execution")
    parser.add_argument("--gpu-type", type=str, help="Override GPU type from config")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count from config")

    # Legacy flags (for backwards compat)
    parser.add_argument("--modal", action="store_true", help="[Legacy] Same as --provider modal")
    parser.add_argument(
        "--provision", action="store_true", help="[Legacy] Provision using config's provider"
    )

    # Remote execution options
    parser.add_argument("--node-id", type=str, help="Reuse existing instance (provider:id)")
    parser.add_argument(
        "--tui", action="store_true", help="Launch TUI after submitting (default: fire-and-forget)"
    )
    parser.add_argument(
        "--tail", action="store_true", help="Stream logs to stdout (default: fire-and-forget)"
    )
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow deploying with uncommitted changes (not recommended)",
    )
    parser.add_argument(
        "--spinners",
        action="store_true",
        help="Enable interactive spinners (default: plain text logging for agents)",
    )

    # Local execution
    parser.add_argument("--max-samples", type=int, help="Limit dataset size (local only)")

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config module
    config_module = load_config_module(config_path)

    if not hasattr(config_module, "config"):
        print("Config file must export 'config'", file=sys.stderr)
        sys.exit(1)

    # Get hardware config (default to local if not specified)
    hardware: HardwareConfig = getattr(config_module, "hardware", HardwareConfig(provider="local"))

    # Apply CLI overrides
    if args.local:
        hardware = replace(hardware, provider="local")
    elif args.modal:
        # Legacy --modal flag
        hardware = replace(hardware, provider="modal")
    elif args.provider:
        hardware = replace(hardware, provider=args.provider)
    elif args.provision and hardware.provider == "local":
        # Legacy --provision without provider: default to runpod
        hardware = replace(hardware, provider="runpod")

    if args.gpu_type:
        hardware = replace(hardware, gpu_type=args.gpu_type)
    if args.gpu_count:
        hardware = replace(hardware, gpu_count=args.gpu_count)

    print(f"Config: {config_path}")
    print(f"Hardware: {hardware.gpu_count}x {hardware.gpu_type} on {hardware.provider}")

    # Check for multi-node config (only if not forced to local)
    multi_node: MultiNodeConfig | None = None
    if not args.local:
        multi_node = getattr(config_module, "multi_node", None)

    # Dispatch based on provider
    if multi_node is not None:
        # Multi-node distributed training
        import trio

        from .training.multi_node import launch_multi_node_training

        print(f"Multi-node: {multi_node.num_nodes} nodes × {multi_node.gpus_per_node} GPUs")
        print(f"  Inference: {multi_node.total_inference_engines} engines")
        print(f"  Training: {multi_node.total_trainer_gpus} FSDP ranks")

        async def _run_multi_node() -> None:
            allocation = await launch_multi_node_training(
                config=multi_node,
                train_config=config_module.config,
            )
            print(f"\nCluster launched: {allocation.fsdp_world_size} FSDP ranks")
            print(f"Inference endpoints: {allocation.all_inference_endpoints}")
            print("\nMonitor with:")
            for node in allocation.nodes:
                print(f"  ssh root@{node.public_ip} tmux attach -t trainer_0")

        trio.run(_run_multi_node)

    elif hardware.provider == "modal":
        # Modal execution (fast cold start)
        import trio

        from .modal_runner import ModalRunConfig, run_modal

        modal_config = ModalRunConfig(
            config_path=str(config_path),
            gpu_type=hardware.gpu_type,
            gpu_count=hardware.gpu_count,
        )
        results = trio.run(run_modal, modal_config)
        if not results.get("success"):
            sys.exit(1)

    elif hardware.provider in ("runpod", "lambdalabs", "vast") or args.node_id:
        # Remote execution via SSH
        import trio

        trio.run(
            run_remote,
            str(config_path),
            args.keep_alive,
            args.node_id,
            args.tui,
            hardware.gpu_count,
            hardware.gpu_type,
            args.tail,
            hardware.provider if hardware.provider != "local" else None,
            args.allow_dirty,
            not args.spinners,  # quiet=True by default, --spinners to enable
        )

    else:
        # Local execution
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
