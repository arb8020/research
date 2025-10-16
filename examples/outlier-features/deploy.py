#!/usr/bin/env python3
"""Outlier Features Analysis with automated GPU deployment via broker/bifrost.

This script automatically:
1. Provisions a GPU with sufficient disk space
2. Deploys the codebase
3. Runs outlier analysis on specified model
4. Syncs results back to local
5. Cleans up GPU instance

Adapted from llm-workbench/examples/outlier_features_moe/deploy_and_analyze.py
Split into functions <70 lines following Tiger Style.

Usage:
    python deploy.py configs/02_olmoe_baseline.py
    python deploy.py configs/03_qwen_medium.py --keep-running
"""

import sys
import os
import json
import logging
import importlib.util
from pathlib import Path
from datetime import datetime
from typing import Literal, TypeAlias
from dotenv import load_dotenv

# Import broker and bifrost for deployment
from broker.client import GPUClient
from bifrost.client import BifrostClient

# Import shared logging
from shared.logging_config import setup_logging

# Import local config
from config import Config
from estimate_vram import estimate_vram_requirements

logger = logging.getLogger(__name__)

# Type aliases for provision result
ProvisionError: TypeAlias = Literal["create_failed", "ssh_timeout"]
ProvisionResult: TypeAlias = (
    tuple[Literal[True], "ClientGPUInstance", str, None] |
    tuple[Literal[False], None, None, ProvisionError]
)


def load_config_from_file(config_path: str) -> Config:
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object
    """
    assert config_path.endswith('.py'), f"Config must be .py file, got {config_path}"

    spec = importlib.util.spec_from_file_location("exp_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'config'), f"Config file must define 'config' variable"
    config: Config = getattr(module, 'config')
    assert isinstance(config, Config), f"Expected Config object, got {type(config)}"

    return config


def estimate_vram_if_needed(config: Config) -> int:
    """Estimate VRAM requirements if not specified in config.

    Args:
        config: Configuration object

    Returns:
        VRAM requirement in GB
    """
    if config.deployment.min_vram is not None:
        logger.info(f"Using configured VRAM: {config.deployment.min_vram}GB")
        return config.deployment.min_vram

    logger.info("Estimating VRAM requirements...")
    vram_estimate = estimate_vram_requirements(
        model_name=config.model.name,
        safety_factor=config.deployment.safety_factor,
        sequence_length=config.dataset.sequence_length,
        batch_size=config.analysis.batch_size
    )

    min_vram = vram_estimate['recommended_vram']
    effective_params = vram_estimate.get('effective_params_billions', 0.0)
    logger.info(f"📊 Estimated VRAM: {min_vram}GB (effective params: {effective_params:.1f}B)")

    return min_vram


def provision_gpu(config: Config, min_vram: int) -> ProvisionResult:
    """Provision GPU instance via broker.

    Args:
        config: Configuration object
        min_vram: Minimum VRAM in GB

    Returns:
        ProvisionResult tuple
    """
    gpu_desc = f"{config.deployment.gpu_count}x GPU" if config.deployment.gpu_count > 1 else "GPU"
    disk_desc = f"{config.deployment.container_disk}GB container"
    if config.deployment.volume_disk > 0:
        disk_desc += f" + {config.deployment.volume_disk}GB volume"

    logger.info(f"📡 Creating {gpu_desc} instance (min {min_vram}GB VRAM per GPU, "
                f"{config.deployment.min_cpu_ram}GB CPU RAM, max ${config.deployment.max_price}/hr, {disk_desc})...")

    # Load credentials from environment
    load_dotenv()
    credentials = {}
    if runpod_key := os.getenv("RUNPOD_API_KEY"):
        credentials["runpod"] = runpod_key
    if vast_key := os.getenv("VAST_API_KEY"):
        credentials["vast"] = vast_key

    assert credentials, "No GPU provider credentials found. Set RUNPOD_API_KEY or VAST_API_KEY in .env"

    # Load SSH key path from environment
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    gpu_client = GPUClient(credentials=credentials, ssh_key_path=ssh_key_path)

    # Build query for GPU
    query = (
        (gpu_client.vram_gb >= min_vram) &
        (gpu_client.memory_gb >= config.deployment.min_cpu_ram) &
        (gpu_client.price_per_hour <= config.deployment.max_price) &
        (gpu_client.manufacturer == 'Nvidia')
    )

    # Add GPU type filter if specified
    if config.deployment.gpu_filter:
        query = query & (gpu_client.gpu_type.contains(config.deployment.gpu_filter))

    # Create instance
    gpu_instance = gpu_client.create(
        query=query,
        name=f"outlier-analysis-{config.model.name.replace('/', '-').lower()}",
        cloud_type="secure",
        gpu_count=config.deployment.gpu_count,
        sort=lambda x: x.price_per_hour,
        reverse=False,
        container_disk_gb=config.deployment.container_disk,
        volume_disk_gb=config.deployment.volume_disk if config.deployment.volume_disk > 0 else None
    )

    if not gpu_instance:
        logger.error("Failed to create GPU instance - no matching offers available")
        return (False, None, None, "create_failed")

    logger.info(f"✅ GPU ready: {gpu_instance.id}")

    # Wait for SSH
    ssh_ready = gpu_instance.wait_until_ssh_ready(timeout=300)
    if not ssh_ready:
        logger.error(f"SSH failed to become ready on {gpu_instance.id}, terminating...")
        gpu_client.terminate_instance(gpu_instance.id, gpu_instance.provider)
        return (False, None, None, "ssh_timeout")

    ssh_connection = gpu_instance.ssh_connection_string()
    logger.info(f"✅ SSH ready: {ssh_connection}")

    return (True, gpu_instance, ssh_key_path, None)


def deploy_code(ssh_connection: str, ssh_key_path: str, config: Config) -> 'BifrostClient':
    """Deploy code to remote instance via bifrost.

    Args:
        ssh_connection: SSH connection string
        ssh_key_path: Path to SSH private key
        config: Configuration object

    Returns:
        BifrostClient instance
    """
    bifrost_client = BifrostClient(ssh_connection, ssh_key_path)

    # Deploy the codebase with outlier dependencies
    logger.info("Deploying code and installing dependencies...")
    workspace_path = bifrost_client.push(bootstrap_cmd="uv sync --extra example-outlier-features")
    logger.info(f"✅ Code deployed to {workspace_path}")

    # Configure HuggingFace cache if using volume disk
    if config.deployment.volume_disk > 0:
        hf_cache_cmd = '''
mkdir -p /workspace/hf_cache 2>/dev/null || mkdir -p ~/.cache/huggingface
export HF_HOME=/workspace/hf_cache 2>/dev/null || export HF_HOME=~/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME/transformers
'''
        bifrost_client.exec(hf_cache_cmd)
        logger.info("✅ HuggingFace cache configured for volume disk")

    return bifrost_client


def build_analysis_command(config: Config, config_path: str) -> str:
    """Build the remote analysis command string.

    Args:
        config: Configuration object
        config_path: Path to config file (for copying to remote)

    Returns:
        Command string
    """
    # Load HF token from environment
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")

    # Build environment setup
    hf_env = f"export HF_TOKEN='{hf_token}' && \\\n"
    hf_env += "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \\\n"

    if config.deployment.volume_disk > 0:
        hf_env += """export HF_HOME=/workspace/hf_cache 2>/dev/null || export HF_HOME=~/.cache/huggingface && \\
export HUGGINGFACE_HUB_CACHE=$HF_HOME && \\
export TRANSFORMERS_CACHE=$HF_HOME/transformers && \\
"""

    # Build main command
    # Note: We'll copy the config file to remote, then run with that path
    remote_config_path = f"~/.bifrost/workspace/examples/outlier-features/configs/{Path(config_path).name}"

    cmd = f"""cd ~/.bifrost/workspace/examples/outlier-features && \\
exec > outlier_analysis.log 2>&1 && \\
{hf_env}uv sync --extra example-outlier-features && \\
~/.bifrost/workspace/.venv/bin/python run_full_analysis.py {remote_config_path} \\
&& touch .analysis_complete \\
|| {{ echo "ANALYSIS FAILED with exit code $?"; touch .analysis_failed; exit 1; }}
"""

    return cmd


def start_analysis(bifrost_client: 'BifrostClient', config: Config, config_path: str):
    """Start analysis in tmux session on remote.

    Args:
        bifrost_client: Bifrost client instance
        config: Configuration object
        config_path: Path to local config file
    """
    logger.info("🔬 Starting outlier analysis...")

    # First, copy the config file to remote
    remote_config_dir = "~/.bifrost/workspace/examples/outlier-features/configs"
    bifrost_client.exec(f"mkdir -p {remote_config_dir}")

    # Upload config file
    with open(config_path, 'r') as f:
        config_content = f.read()

    remote_config_path = f"{remote_config_dir}/{Path(config_path).name}"
    bifrost_client.exec(f"cat > {remote_config_path} <<'EOF'\n{config_content}\nEOF")
    logger.info(f"✅ Config uploaded: {remote_config_path}")

    # Build and execute analysis command
    cmd = build_analysis_command(config, config_path)
    tmux_cmd = f"tmux new-session -d -s outlier-analysis '{cmd}'"
    bifrost_client.exec(tmux_cmd)

    logger.info("✅ Analysis started - will take 10-30 minutes")
    logger.info(f"📊 Monitor: bifrost exec '{bifrost_client.ssh}' "
                "'cd ~/.bifrost/workspace/examples/outlier-features && tail -20 outlier_analysis.log'")


def wait_for_analysis_completion(bifrost_client: 'BifrostClient', timeout: int = 2700) -> bool:
    """Poll for analysis completion with explicit timeout.

    Args:
        bifrost_client: Connected Bifrost client
        timeout: Max wait time in seconds (default 45 min)

    Returns:
        True if completed successfully, False if failed/timeout
    """
    assert bifrost_client is not None, "BifrostClient must not be None"
    assert timeout > 0, "Timeout must be positive"

    import time

    poll_interval = 30
    max_iterations = timeout // poll_interval
    assert max_iterations > 0, f"Timeout {timeout}s too short for poll interval {poll_interval}s"

    logger.info(f"⏳ Waiting for analysis completion (timeout: {timeout}s, polling every {poll_interval}s)...")

    for i in range(max_iterations):
        # Check for completion markers AND final results file with debug info
        check_cmd = """
cd ~/.bifrost/workspace/examples/outlier-features
echo "=== Polling Debug Info ==="
echo "Marker files:"
ls -lh .analysis_complete .analysis_failed 2>/dev/null || echo "  No marker files"
echo "Results file:"
ls -lh results/final_analysis_results.json 2>/dev/null || echo "  No results file yet"
echo "Last 3 log lines:"
tail -3 outlier_analysis.log 2>/dev/null || echo "  No log file yet"
echo "Tmux session status:"
tmux list-sessions 2>/dev/null | grep outlier-analysis || echo "  No tmux session (may have completed)"
echo "========================="

# Determine status
test -f .analysis_complete && echo 'COMPLETE' && exit 0
test -f .analysis_failed && echo 'FAILED' && exit 0
test -f results/final_analysis_results.json && echo 'COMPLETE' && exit 0
echo 'RUNNING'
"""
        result = bifrost_client.exec(check_cmd)

        # Extract status (last line of output)
        output_lines = result.stdout.strip().split('\n') if result.stdout else []
        status = output_lines[-1] if output_lines else 'UNKNOWN'

        # Log debug info every 5 polls (or first poll)
        if i == 0 or (i + 1) % 5 == 0:
            logger.debug("Poll debug output:\n" + result.stdout if result.stdout else "No output")

        if status == 'COMPLETE':
            logger.info("✅ Analysis completed successfully")
            # Show final debug info
            logger.info("Final state:\n" + result.stdout if result.stdout else "")
            return True
        elif status == 'FAILED':
            logger.error("❌ Analysis failed (marker file found)")
            logger.error("Failure state:\n" + result.stdout if result.stdout else "")
            return False

        # Show progress
        elapsed = (i + 1) * poll_interval
        logger.info(f"⏳ Analysis running... ({elapsed}s / {timeout}s)")
        time.sleep(poll_interval)

    logger.error(f"❌ Analysis timeout after {timeout}s")
    return False


def sync_results(bifrost_client: 'BifrostClient', output_dir: Path):
    """Sync analysis results from remote GPU to local directory.

    Args:
        bifrost_client: Bifrost client instance
        output_dir: Local output directory
    """
    logger.info("💾 Syncing results from remote GPU...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Display remote log
    logger.info("=" * 60)
    logger.info("Remote analysis log:")
    logger.info("=" * 60)
    try:
        result = bifrost_client.exec(
            "cat ~/.bifrost/workspace/examples/outlier-features/outlier_analysis.log"
        )
        if result.stdout and result.stdout.strip():
            print(result.stdout)
        else:
            logger.warning("⚠️  Log file not found or empty")
    except Exception as e:
        logger.warning(f"⚠️  Could not read remote log: {e}")
    logger.info("=" * 60)

    # Sync main log file
    log_path = output_dir / "outlier_analysis.log"
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier-features/outlier_analysis.log",
            local_path=str(log_path)
        )
        if result and result.success and result.files_copied > 0:
            logger.info(f"✅ Synced: outlier_analysis.log")
        else:
            logger.warning(f"⚠️  Analysis log not ready yet")
    except Exception as e:
        logger.warning(f"⚠️  Could not sync analysis log: {e}")

    # Sync final results
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier-features/results/final_analysis_results.json",
            local_path=str(output_dir / "final_analysis_results.json")
        )
        if result and result.success and result.files_copied > 0:
            logger.info(f"✅ Synced: final_analysis_results.json")
        else:
            logger.info("ℹ️  Final results not ready yet")
    except Exception as e:
        logger.info(f"ℹ️  Final results not ready yet: {e}")

    # Sync config used
    try:
        result = bifrost_client.download_files(
            remote_path="~/.bifrost/workspace/examples/outlier-features/results/config.json",
            local_path=str(output_dir / "config.json")
        )
        if result and result.success and result.files_copied > 0:
            logger.info(f"✅ Synced: config.json")
    except Exception as e:
        logger.debug(f"Config not synced: {e}")

    logger.info(f"✅ Results synced to local: {output_dir}")


def cleanup_instance(instance_id: str):
    """Terminate GPU instance.

    Args:
        instance_id: GPU instance ID
    """
    import subprocess

    logger.info(f"🧹 Terminating GPU instance {instance_id}...")

    try:
        result = subprocess.run(
            ["broker", "terminate", instance_id, "--yes"],
            text=True,
            capture_output=True
        )

        if result.returncode == 0:
            logger.info("✅ Cleanup complete")
        else:
            logger.warning(f"⚠️  Cleanup may have failed: {result.stderr}")
            logger.info(f"   Manual cleanup: broker terminate {instance_id}")

    except Exception as e:
        logger.warning(f"⚠️  Cleanup error (instance may still be running): {e}")
        logger.info(f"   Manual cleanup: broker terminate {instance_id}")


def main():
    """Main deployment orchestrator."""
    # Parse args
    import argparse
    parser = argparse.ArgumentParser(description="Automated Outlier Analysis with GPU Deployment")
    parser.add_argument("config", help="Path to config file (e.g., configs/02_olmoe_baseline.py)")
    parser.add_argument("--keep-running", action="store_true",
                        help="Keep GPU instance running after analysis")
    args = parser.parse_args()

    # Load config
    config = load_config_from_file(args.config)

    # Override keep_running from command line if specified
    if args.keep_running:
        config.deployment.keep_running = True

    # Create timestamped experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_safe_name = config.model.name.replace('/', '_').replace('-', '_')
    if config.output.experiment_name:
        experiment_name = f"{config.output.experiment_name}_{timestamp}"
    else:
        experiment_name = f"outlier_analysis_{model_safe_name}_{timestamp}"

    # Setup logging
    setup_logging(level=config.output.log_level)

    logger.info("🎯 Outlier Features Analysis - Automated Deployment")
    logger.info(f"📅 Experiment: {experiment_name}")
    logger.info(f"🤖 Model: {config.model.name}")
    logger.info("=" * 60)

    output_dir = Path(f"remote_results/{experiment_name}")

    try:
        # Step 1: Estimate VRAM
        min_vram = estimate_vram_if_needed(config)

        # Step 2: Provision GPU
        success, gpu_instance, ssh_key_path, error = provision_gpu(config, min_vram)
        if not success:
            logger.error(f"Provisioning failed: {error}")
            return 1

        # Step 3: Deploy code
        bifrost_client = deploy_code(gpu_instance.ssh_connection_string(), ssh_key_path, config)

        # Step 4: Start analysis
        start_analysis(bifrost_client, config, args.config)

        # Step 5: Wait for completion
        logger.info("\n⏳ Waiting for analysis to complete...")
        success = wait_for_analysis_completion(bifrost_client, timeout=2700)

        # Step 6: Sync results
        logger.info("\n💾 Syncing results to local...")
        if not success:
            logger.warning("⚠️  Analysis did not complete successfully, but syncing logs for debugging...")
        sync_results(bifrost_client, output_dir)

        # Step 7: Cleanup (conditional)
        if not config.deployment.keep_running:
            cleanup_instance(gpu_instance.id)
        else:
            logger.info(f"\n🎯 Keeping GPU running")
            logger.info(f"   SSH: {gpu_instance.ssh_connection_string()}")
            logger.info(f"   Terminate: broker terminate {gpu_instance.id}")

        logger.info("\n🎉 Deployment complete!")
        return 0

    except Exception as e:
        logger.error(f"✗ Deployment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
