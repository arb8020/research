#!/usr/bin/env python3
"""Deploy training to remote GPU via bifrost.

This script:
1. Loads a training config
2. Deploys code to remote instance (via bifrost)
3. Runs SFT/RL training in tmux
4. Syncs results back to local

Usage:
    python deploy.py configs/01_debug_sft_rl.py --ssh root@host:port
    python deploy.py configs/01_debug_sft_rl.py --ssh root@host:port --detached
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import local config
from base_config import Config

# Import bifrost for deployment
from bifrost.client import BifrostClient
from dotenv import load_dotenv

# Import shared logging
from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Remote workspace path constant
REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/dev/integration_training"

# GPU availability thresholds (from clicker pattern)
DEFAULT_MEMORY_THRESHOLD_MB = 1000  # Consider GPU busy if > 1GB used
DEFAULT_UTIL_THRESHOLD_PCT = 5      # Consider GPU busy if > 5% utilized


def check_gpus_available(
    bifrost_client: BifrostClient,
    gpu_ids: list[int],
    memory_threshold_mb: int = DEFAULT_MEMORY_THRESHOLD_MB,
    util_threshold_pct: int = DEFAULT_UTIL_THRESHOLD_PCT,
) -> tuple[bool, str]:
    """Check if specified GPUs exist and are free (clicker pattern).

    GPU is "available" if:
    1. GPU ID exists on remote (from nvidia-smi)
    2. Memory used ≤ memory_threshold_mb (default 1GB)
    3. Utilization ≤ util_threshold_pct (default 5%)

    Tiger Style: Assert preconditions, explicit error messages.

    Args:
        bifrost_client: Connected bifrost client
        gpu_ids: List of GPU IDs to check (e.g., [0, 2, 4])
        memory_threshold_mb: Memory threshold in MB (default 1000 = 1GB)
        util_threshold_pct: Utilization threshold % (default 5%)

    Returns:
        (True, "") if all GPUs are available
        (False, error_message) if any GPU is unavailable or busy

    Example:
        >>> available, msg = check_gpus_available(client, [0, 1, 2])
        >>> if not available:
        ...     print(f"GPU check failed: {msg}")
    """
    assert len(gpu_ids) > 0, "Must specify at least one GPU to check"

    try:
        # Query all GPUs on remote
        result = bifrost_client.exec(
            "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
        )

        if result.exit_code != 0:
            return False, f"Failed to run nvidia-smi: {result.stderr}"

        # Parse nvidia-smi output
        gpu_stats = {}
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 3:
                continue

            try:
                gpu_id = int(parts[0])
                memory_mb = int(parts[1])
                util_pct = int(parts[2])
                gpu_stats[gpu_id] = {'memory_mb': memory_mb, 'util_pct': util_pct}
            except ValueError:
                continue

        # Check each requested GPU
        for gpu_id in gpu_ids:
            # Check 1: Does GPU exist?
            if gpu_id not in gpu_stats:
                return False, f"GPU {gpu_id} not found on remote (available: {sorted(gpu_stats.keys())})"

            # Check 2: Is GPU free?
            stats = gpu_stats[gpu_id]
            mem_mb = stats['memory_mb']
            util = stats['util_pct']

            if mem_mb > memory_threshold_mb or util > util_threshold_pct:
                return False, f"GPU {gpu_id} is busy ({mem_mb}MB used, {util}% util)"

        return True, ""

    except Exception as e:
        return False, f"GPU availability check failed: {e}"


def load_config_from_file(config_path: str) -> Config:
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object
    """
    import importlib.util

    assert config_path.endswith('.py'), f"Config must be .py file, got {config_path}"
    assert Path(config_path).exists(), f"Config file not found: {config_path}"

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'config'), "Config file must define 'config' variable"
    config: Config = getattr(module, 'config')
    assert isinstance(config, Config), f"Expected Config object, got {type(config)}"

    return config


def check_remote_prerequisites(bifrost_client: BifrostClient) -> list[str]:
    """Check if remote has required tools installed.

    Args:
        bifrost_client: Bifrost client instance

    Returns:
        List of missing tools (empty if all present)
    """
    missing = []

    # Check for uv (Python package installer)
    # uv installs to ~/.local/bin, need to add to PATH since installer modifies shell profiles
    # but we're running in a fresh shell each time
    result = bifrost_client.exec(
        'export PATH="$HOME/.local/bin:$PATH" && '
        'command -v uv >/dev/null 2>&1 && echo "OK" || echo "MISSING"'
    )
    if result.stdout.strip() != 'OK':
        missing.append("uv")

    # Check for tmux (terminal multiplexer)
    result = bifrost_client.exec("command -v tmux >/dev/null 2>&1 && echo 'OK' || echo 'MISSING'")
    if result.stdout.strip() != 'OK':
        missing.append("tmux")

    return missing


def deploy_code(bifrost_client: BifrostClient) -> str:
    """Deploy code to remote and bootstrap environment.

    Args:
        bifrost_client: Bifrost client instance

    Returns:
        Remote workspace path (absolute, expanded)
    """
    logger.info("📦 Deploying code and bootstrapping environment...")

    # Deploy code with bootstrap
    # Note: Need to add uv to PATH since it's installed in ~/.local/bin
    bootstrap_cmd = 'export PATH="$HOME/.local/bin:$PATH" && uv sync --extra dev-integration-training'
    workspace_path = bifrost_client.push(
        bootstrap_cmd=bootstrap_cmd
    )

    logger.info(f"✅ Code deployed to {workspace_path}")

    # Expand tilde to absolute path
    result = bifrost_client.exec(f"echo {workspace_path}")
    if result.exit_code == 0 and result.stdout:
        workspace_path = result.stdout.strip()
        logger.info(f"📍 Expanded workspace path: {workspace_path}")

    return workspace_path


def start_training(
    bifrost_client: BifrostClient,
    config: Config,
    config_path: str,
    workspace_path: str,
    remote_result_dir: str,
) -> tuple[str, str | None]:
    """Start training on remote in tmux session.

    Args:
        bifrost_client: Bifrost client instance
        config: Training configuration
        config_path: Path to config file (local)
        workspace_path: Remote workspace path (absolute)
        remote_result_dir: Remote result directory

    Returns:
        (tmux_session_name, error_message)
        error_message is None on success
    """
    import os

    # Get config filename
    config_name = Path(config_path).name

    # Build training command
    # Note: workspace_path is ~/.bifrost/workspace, we need to cd to dev/integration_training
    venv_path = f"{workspace_path}/.venv/bin/activate"
    project_dir = f"{workspace_path}/dev/integration_training"
    gpu_ranks_str = ",".join(str(r) for r in config.target.gpu_ranks)

    # Get HF_TOKEN from local environment
    hf_token = os.getenv("HF_TOKEN", "")

    # Base command with environment variables
    # Note: We DON'T set CUDA_VISIBLE_DEVICES (see train.py:144-147)
    # Instead, train.py uses explicit device placement (cuda:N means physical GPU N)
    # This avoids remapping confusion and works with non-contiguous GPU allocations
    cmd_parts = [
        f"cd {project_dir}",
        f"source {venv_path}",
    ]

    # Add HF_TOKEN if available
    if hf_token:
        cmd_parts.append(f"export HF_TOKEN='{hf_token}'")
        logger.info("🔑 Using HF_TOKEN for model download")
    else:
        logger.warning("⚠️  HF_TOKEN not set - training may hit rate limits")

    # Build training command
    # Use torchrun for FSDP (multi-GPU training)
    if config.target.train_backend == "fsdp" and len(config.target.gpu_ranks) > 1:
        # Set CUDA_VISIBLE_DEVICES to limit to requested GPUs
        gpu_list = ",".join(str(r) for r in config.target.gpu_ranks)
        cmd_parts.append(f"export CUDA_VISIBLE_DEVICES={gpu_list}")
        # Launch with torchrun for distributed training
        nproc = len(config.target.gpu_ranks)
        train_cmd = f"torchrun --nproc_per_node={nproc} train.py configs/{config_name}"
        logger.info(f"🚀 Using torchrun with {nproc} processes for FSDP training")
    else:
        train_cmd = f"python train.py configs/{config_name}"

    cmd_parts.append(train_cmd)

    full_cmd = " && ".join(cmd_parts)

    # Generate tmux session name from config
    tmux_session = f"training_{config.output.experiment_name}"

    logger.info(f"🚀 Starting training in tmux session: {tmux_session}")
    logger.info(f"   Mode: {config.output.mode}")
    logger.info(f"   Model: {config.model.name}")
    logger.info(f"   Command: {full_cmd[:150]}...")

    # Kill existing session if it exists
    bifrost_client.exec(f"tmux kill-session -t {tmux_session} 2>/dev/null || true")

    # Create result directory
    bifrost_client.exec(f"mkdir -p {remote_result_dir}")

    # Start new tmux session
    tmux_cmd = (
        f"cd {workspace_path} && "
        f"tmux new-session -d -s {tmux_session} "
        f"'{full_cmd} 2>&1 | tee {remote_result_dir}/training.log'"
    )

    result = bifrost_client.exec(tmux_cmd)
    if result.exit_code != 0:
        return tmux_session, f"Failed to start tmux session: {result.stderr}"

    logger.info(f"✅ Training started in tmux session: {tmux_session}")
    return tmux_session, None


def tail_log_until_complete(
    bifrost_client: BifrostClient,
    log_path: str,
    tmux_session: str,
    check_interval: int = 5,
) -> bool:
    """Tail training log and wait for completion.

    Args:
        bifrost_client: Bifrost client instance
        log_path: Path to training log file on remote
        tmux_session: Tmux session name
        check_interval: Seconds between checks

    Returns:
        True if training completed successfully, False otherwise
    """
    import time

    logger.info("📊 Tailing training log (Ctrl+C to detach)...")
    logger.info("=" * 60)

    last_line = 0

    try:
        while True:
            # Check if tmux session still exists
            check_result = bifrost_client.exec(f"tmux has-session -t {tmux_session} 2>&1")
            session_exists = check_result.exit_code == 0

            # Tail new lines from log
            tail_result = bifrost_client.exec(f"tail -n +{last_line + 1} {log_path} 2>/dev/null")
            if tail_result.stdout:
                lines = tail_result.stdout.splitlines()
                for line in lines:
                    print(line)
                last_line += len(lines)

            # If session ended, training is complete
            if not session_exists:
                logger.info("=" * 60)
                logger.info("✅ Training session completed")

                # Check for success indicators in log
                success_check = bifrost_client.exec(
                    f"grep -i 'Training Complete' {log_path} || "
                    f"grep -i 'complete!' {log_path}"
                )
                return success_check.exit_code == 0

            time.sleep(check_interval)

    except KeyboardInterrupt:
        logger.info("\n⚠️  Detached from log (training continues in background)")
        logger.info(f"   Attach: bifrost exec '<ssh-string>' 'tmux attach -t {tmux_session}'")
        return False


def sync_results(
    bifrost_client: BifrostClient,
    remote_result_dir: str,
    local_output_dir: Path,
) -> None:
    """Sync results from remote to local.

    Args:
        bifrost_client: Bifrost client instance
        remote_result_dir: Remote result directory
        local_output_dir: Local output directory
    """
    logger.info(f"💾 Syncing results from {remote_result_dir}...")

    # Create local directory
    local_output_dir.mkdir(parents=True, exist_ok=True)

    # Download results (recursive since remote_result_dir is a directory)
    result = bifrost_client.download_files(
        remote_path=f"{remote_result_dir}/",
        local_path=str(local_output_dir),
        recursive=True
    )

    if result and result.success:
        logger.info(f"✅ Results synced to: {local_output_dir}")
    else:
        logger.warning("⚠️  Some files may not have synced")


def main():
    """Main deployment orchestrator."""
    parser = argparse.ArgumentParser(
        description="Deploy training to remote GPU instance"
    )
    parser.add_argument(
        "config",
        help="Path to config file (e.g., configs/01_debug_sft_rl.py)"
    )
    parser.add_argument(
        "--ssh",
        required=True,
        help="SSH connection string (e.g., root@host:port)"
    )
    parser.add_argument(
        "--ssh-key",
        default="~/.ssh/id_ed25519",
        help="Path to SSH private key"
    )
    parser.add_argument(
        "--detached",
        action="store_true",
        help="Start training and exit immediately (don't wait for completion)"
    )
    args = parser.parse_args()

    # Load environment variables (for HF_TOKEN, etc.)
    load_dotenv()

    # Track deployment start time
    import time
    start_time = time.time()

    # Setup logging
    setup_logging(
        level="INFO",
        logger_levels={
            "httpx": "WARNING",
            "urllib3": "WARNING",
            "paramiko": "WARNING",
        }
    )

    # Load config
    config = load_config_from_file(args.config)

    # Create timestamped result directory name
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    result_dir_name = f"{config.output.experiment_name}_{timestamp}"

    logger.info("🎯 Integration Training Deployment")
    logger.info(f"📅 Experiment: {result_dir_name}")
    logger.info(f"🤖 Model: {config.model.name}")
    logger.info(f"🔧 Mode: {config.output.mode}")
    logger.info(f"🖥️  SSH: {args.ssh}")
    logger.info("=" * 60)

    # Local output directory
    local_output_dir = Path(config.output.save_dir) / result_dir_name

    try:
        # Connect to remote
        bifrost_client = BifrostClient(args.ssh, args.ssh_key)

        # Check remote prerequisites
        missing_tools = check_remote_prerequisites(bifrost_client)
        if missing_tools:
            logger.error("❌ Missing required tools on remote host:")
            for tool in missing_tools:
                logger.error(f"   • {tool}")
            logger.error("")
            logger.error("💡 After installing, retry the deployment")
            return 1
        logger.info("✅ All prerequisites present (uv, tmux)")

        # Check GPU availability (Tiger Style: fail fast before deploying code)
        gpu_ids = config.target.gpu_ranks
        if gpu_ids:
            # First enumerate all physical GPUs on the node
            logger.info("🔍 Enumerating GPUs on remote...")
            result = bifrost_client.exec(
                "nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits"
            )
            if result.exit_code == 0:
                all_gpu_ids = []
                free_gpu_ids = []
                busy_gpu_ids = []

                for line in result.stdout.strip().splitlines():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        try:
                            gpu_id = int(parts[0])
                            mem_mb = int(parts[1])
                            util = int(parts[2])
                            all_gpu_ids.append(gpu_id)

                            # Check if free using same thresholds
                            if mem_mb <= DEFAULT_MEMORY_THRESHOLD_MB and util <= DEFAULT_UTIL_THRESHOLD_PCT:
                                free_gpu_ids.append(gpu_id)
                            else:
                                busy_gpu_ids.append(gpu_id)
                        except ValueError:
                            pass

                logger.info(f"   Physical GPUs on node: {sorted(all_gpu_ids)}")
                logger.info(f"   Available (free) GPUs:  {sorted(free_gpu_ids)}")
                if busy_gpu_ids:
                    logger.info(f"   Busy GPUs:              {sorted(busy_gpu_ids)}")

            # Now check if requested GPUs are available
            logger.info(f"🔍 Checking availability for requested GPUs: {gpu_ids}")
            available, error_msg = check_gpus_available(bifrost_client, gpu_ids)
            if not available:
                logger.error(f"❌ GPU availability check failed: {error_msg}")
                logger.error("")
                logger.error("💡 Options:")
                logger.error("   • Update gpu_ranks in config to use different GPUs")
                logger.error("   • Wait for GPUs to become free")
                logger.error("   • Check if GPU IDs are correct for this machine")
                return 1
            logger.info(f"✅ All requested GPUs are available: {gpu_ids}")

        # Deploy code
        workspace_path = deploy_code(bifrost_client)

        # Construct remote result directory (in project subdirectory)
        project_dir = f"{workspace_path}/dev/integration_training"
        remote_result_dir = f"{project_dir}/results/{result_dir_name}"

        # Start training
        tmux_session, error = start_training(
            bifrost_client, config, args.config, workspace_path, remote_result_dir
        )
        if error:
            logger.error(f"❌ Failed to start training: {error}")
            return 1

        if args.detached:
            logger.info("\n🎯 Training started in detached mode")
            logger.info(f"   Attach: bifrost exec '{args.ssh}' 'tmux attach -t {tmux_session}'")
            logger.info(f"   Results will be in: {remote_result_dir}")
            return 0

        # Tail log and wait for completion
        log_path = f"{remote_result_dir}/training.log"
        success = tail_log_until_complete(bifrost_client, log_path, tmux_session)

        # Sync results
        logger.info("\n💾 Syncing results to local...")
        sync_results(bifrost_client, remote_result_dir, local_output_dir)

        # Calculate and display total time
        elapsed_time = time.time() - start_time
        minutes, seconds = divmod(int(elapsed_time), 60)
        hours, minutes = divmod(minutes, 60)

        if hours > 0:
            time_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            time_str = f"{minutes}m {seconds}s"
        else:
            time_str = f"{seconds}s"

        logger.info("\n🎉 Deployment complete!")
        logger.info(f"⏱️  Total time: {time_str}")
        logger.info(f"📊 Results: {local_output_dir}")

        return 0 if success else 1

    except Exception as e:
        logger.error(f"✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
