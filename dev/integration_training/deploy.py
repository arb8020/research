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

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone

# Import bifrost for deployment
from bifrost.client import BifrostClient

# Import shared logging
from shared.logging_config import setup_logging

# Import local config
from base_config import Config

logger = logging.getLogger(__name__)

# Remote workspace path constant
REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/dev/integration_training"


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

    assert hasattr(module, 'config'), f"Config file must define 'config' variable"
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
    cmd_parts = [
        f"cd {project_dir}",
        f"source {venv_path}",
        f"export CUDA_VISIBLE_DEVICES={gpu_ranks_str}",
    ]

    # Add HF_TOKEN if available
    if hf_token:
        cmd_parts.append(f"export HF_TOKEN='{hf_token}'")
        logger.info("🔑 Using HF_TOKEN for model download")
    else:
        logger.warning("⚠️  HF_TOKEN not set - training may hit rate limits")

    # Build training command
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

    logger.info(f"🎯 Integration Training Deployment")
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
