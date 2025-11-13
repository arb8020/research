#!/usr/bin/env python3
"""Deploy evaluation to remote GPU via bifrost.

This script:
1. Loads an evaluation config
2. Deploys code to remote instance (via bifrost)
3. Runs evaluation in tmux
4. Syncs results back to local

Usage:
    python deploy.py configs/prime_backend_bench.py --ssh root@host:port
    python deploy.py configs/prime_backend_bench.py --ssh root@host:port --detached
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import bifrost for deployment
from bifrost.client import BifrostClient
from dotenv import load_dotenv

# Import kerbal for dependency management and deployment patterns
from kerbal import (
    DependencyConfig,
    setup_script_deps,
    check_gpus_available,
    start_tmux_session,
)
from kerbal.job_monitor import (
    LogStreamConfig,
    stream_log_until_complete,
)

# Import shared logging
from shared.logging_config import setup_logging

logger = logging.getLogger(__name__)

# Remote workspace path constant
REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/dev/integration_evaluation"

# GPU availability thresholds
DEFAULT_MEMORY_THRESHOLD_MB = 1000  # Consider GPU busy if > 1GB used
DEFAULT_UTIL_THRESHOLD_PCT = 5      # Consider GPU busy if > 5% utilized


def load_config_from_file(config_path: str):
    """Load config from Python file.

    Args:
        config_path: Path to config .py file

    Returns:
        Config object with 'config' attribute
    """
    import importlib.util

    assert config_path.endswith('.py'), f"Config must be .py file, got {config_path}"
    assert Path(config_path).exists(), f"Config file not found: {config_path}"

    spec = importlib.util.spec_from_file_location("eval_config", config_path)
    assert spec is not None, f"Failed to load spec from {config_path}"
    assert spec.loader is not None, f"Spec loader is None for {config_path}"

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, 'config'), "Config file must define 'config' variable"
    config = getattr(module, 'config')

    return config


def check_remote_prerequisites(bifrost_client: BifrostClient) -> list[str]:
    """Check if remote has required tools installed.

    Args:
        bifrost_client: Bifrost client instance

    Returns:
        List of missing tools (empty if all present)
    """
    missing = []

    # Check for tmux (terminal multiplexer)
    result = bifrost_client.exec("command -v tmux >/dev/null 2>&1 && echo 'OK' || echo 'MISSING'")
    if result.stdout.strip() != 'OK':
        missing.append("tmux")

    return missing


def deploy_code(bifrost_client: BifrostClient) -> str:
    """Deploy code to remote and bootstrap environment using kerbal.

    Args:
        bifrost_client: Bifrost client instance

    Returns:
        Remote workspace path (absolute, expanded)
    """
    logger.info("📦 Deploying code and bootstrapping environment...")

    # Deploy code with explicit workspace path for isolation
    workspace_path = bifrost_client.push(workspace_path="~/.bifrost/workspaces/integration_evaluation")

    logger.info(f"✅ Code deployed to {workspace_path}")

    # Expand tilde to absolute path
    result = bifrost_client.exec(f"echo {workspace_path}")
    if result.exit_code == 0 and result.stdout:
        workspace_path = result.stdout.strip()
        logger.info(f"📍 Expanded workspace path: {workspace_path}")

    # Check what actually got deployed
    logger.info("🔍 Checking deployed directory structure...")
    result = bifrost_client.exec(f"ls -la {workspace_path}")
    if result.exit_code == 0:
        logger.info(f"Workspace contents:\n{result.stdout}")

    # The project_workspace needs to point to where local.py is
    # Bifrost pushes the entire monorepo, so we need to find integration-evaluation
    project_workspace = f"{workspace_path}/dev/integration-evaluation"

    # Verify the directory exists
    result = bifrost_client.exec(f"test -d {project_workspace} && echo 'EXISTS' || echo 'MISSING'")
    if result.stdout.strip() != 'EXISTS':
        logger.error(f"❌ Project directory not found: {project_workspace}")
        logger.info("Looking for integration-evaluation...")
        result = bifrost_client.exec(f"find {workspace_path} -name 'integration-evaluation' -type d 2>/dev/null | head -5")
        logger.info(f"Found:\n{result.stdout}")
        raise RuntimeError(f"Project directory does not exist: {project_workspace}")

    logger.info(f"✅ Project directory exists: {project_workspace}")

    # Define dependencies for dev/integration_evaluation
    deps = DependencyConfig(
        project_name="integration-evaluation",
        dependencies=[
            "rollouts @ git+https://github.com/arb8020/research.git#subdirectory=rollouts",
            "shared @ git+https://github.com/arb8020/research.git#subdirectory=shared",
            "torch>=2.4.0",
            "trio>=0.27.0",
            "trio-asyncio>=0.15.0",
            "python-dotenv>=1.0.0",
            "triton>=3.0.0",  # For GPU kernel compilation
            "rich>=13.0.0",
            # Install backendbench BEFORE Prime CLI's backend-bench installation
            # This satisfies the URL dependency so Prime CLI doesn't need to resolve it
            "backendbench @ git+https://github.com/meta-pytorch/BackendBench.git",
        ],
        extras={
            "dev": ["pytest", "black"],
        },
        python_version=">=3.10",
    )

    # Setup dependencies using kerbal
    logger.info("📦 Installing dependencies via kerbal...")
    logger.info(f"   Dependencies to install: {len(deps.dependencies)} packages")
    for dep in deps.dependencies:
        logger.info(f"   - {dep}")
    setup_script_deps(bifrost_client, project_workspace, deps, install_extras=None)

    # Debug: Check the generated pyproject.toml
    logger.info("🔍 Checking generated pyproject.toml...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"cat pyproject.toml 2>&1"
    )
    logger.info(f"Generated pyproject.toml:\n{result.stdout}")

    # Debug: Check what's actually in the venv
    logger.info("🔍 Debugging venv state...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"source .venv/bin/activate && "
        f"echo '=== Python location ===' && "
        f"which python && "
        f"echo '=== Python version ===' && "
        f"python --version && "
        f"echo '=== pip list (backendbench-related) ===' && "
        f"pip list | grep -i backend && "
        f"echo '=== Site packages location ===' && "
        f"python -c 'import site; print(site.getsitepackages())' && "
        f"echo '=== List site-packages contents ===' && "
        f"ls -la .venv/lib/*/site-packages/ | grep -i backend 2>&1"
    )
    logger.info(f"Debug output:\n{result.stdout}")

    # Verify backendbench was installed successfully
    # Note: The package might be named BackendBench (capital B) in the module
    logger.info("🔍 Verifying backendbench installation...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"source .venv/bin/activate && "
        f"python -c 'try: import BackendBench as backendbench; print(backendbench.__file__)\nexcept: import backendbench; print(backendbench.__file__)' 2>&1"
    )
    if result.exit_code != 0:
        logger.warning("⚠️  backendbench not found - likely uv sync skipped it (already resolved)")
        logger.info("📦 Installing backendbench manually via uv pip...")

        # Install manually using uv pip
        install_result = bifrost_client.exec(
            f"cd {project_workspace} && "
            f"source .venv/bin/activate && "
            f"export PATH=$HOME/.local/bin:$PATH && "
            f"uv pip install 'backendbench @ git+https://github.com/meta-pytorch/BackendBench.git' 2>&1"
        )

        if install_result.exit_code != 0:
            logger.error(f"❌ Failed to install backendbench (exit code {install_result.exit_code})")
            logger.error(f"Output: {install_result.stdout}")
            raise RuntimeError("Failed to install backendbench")

        logger.info("✅ backendbench installed manually")
        logger.info(f"Install output: {install_result.stdout}")

        # Check what was actually installed
        logger.info("🔍 Checking site-packages after manual install...")
        check = bifrost_client.exec(
            f"cd {project_workspace} && "
            f"source .venv/bin/activate && "
            f"pip list | grep -i backend && "
            f"ls -la .venv/lib/python*/site-packages/ | grep -i backend 2>&1"
        )
        logger.info(f"Post-install check:\n{check.stdout}")

        # Verify it works now
        verify = bifrost_client.exec(
            f"cd {project_workspace} && "
            f"source .venv/bin/activate && "
            f"python -c 'import backendbench; print(backendbench.__file__)' 2>&1"
        )
        if verify.exit_code != 0:
            logger.error("❌ backendbench still not importable!")
            logger.error(f"Output: {verify.stdout}")

            # Try to understand why
            logger.info("🔍 Checking if package files exist...")
            debug = bifrost_client.exec(
                f"cd {project_workspace} && "
                f"find .venv/lib/python*/site-packages -name '*backend*' -o -name '*.dist-info' | grep -i backend 2>&1"
            )
            logger.info(f"Package files found:\n{debug.stdout}")

            raise RuntimeError("backendbench installed but not importable")

        logger.info(f"✅ backendbench verified at: {verify.stdout.strip()}")
    else:
        logger.info(f"✅ backendbench installed at: {result.stdout.strip()}")

    # Install Prime packages directly from Prime's package index
    # Install verifiers first, then try backend-bench with --no-deps to skip URL dependency check
    logger.info("📦 Installing verifiers from Prime's index...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"source .venv/bin/activate && "
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"uv pip install verifiers --extra-index-url https://hub.primeintellect.ai/siro/simple/ 2>&1"
    )
    if result.exit_code != 0:
        logger.error(f"❌ Failed to install verifiers (exit code {result.exit_code})")
        logger.error(f"Output: {result.stdout}")
        raise RuntimeError("Failed to install verifiers")

    logger.info("✅ verifiers installed")

    # Try installing backend-bench with --no-deps to skip dependency resolution
    logger.info("📦 Installing backend-bench (with --no-deps to skip URL dependency check)...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"source .venv/bin/activate && "
        f"export PATH=$HOME/.local/bin:$PATH && "
        f"uv pip install backend-bench --no-deps --extra-index-url https://hub.primeintellect.ai/siro/simple/ 2>&1"
    )
    if result.exit_code != 0:
        logger.error(f"❌ Failed to install Prime packages (exit code {result.exit_code})")
        logger.error(f"stdout: {result.stdout}")
        logger.error(f"stderr: {result.stderr}")
        logger.error("These packages are required for backend-bench evaluation")
        raise RuntimeError("Failed to install Prime packages (verifiers + backend-bench)")
    else:
        logger.info("✅ Prime packages installed (verifiers + backend-bench)")
        logger.info(f"Output: {result.stdout}")

    # Verify all required packages are importable
    logger.info("🔍 Verifying all required packages...")
    result = bifrost_client.exec(
        f"cd {project_workspace} && "
        f"source .venv/bin/activate && "
        f"python -c 'import verifiers, backend_bench, backendbench; "
        f"print(f\"verifiers: {{verifiers.__file__}}\"); "
        f"print(f\"backend_bench: {{backend_bench.__file__}}\"); "
        f"print(f\"backendbench: {{backendbench.__file__}}\")' 2>&1"
    )
    if result.exit_code != 0:
        logger.error("❌ Package verification failed!")
        logger.error(f"Output: {result.stdout}")
        raise RuntimeError("Required packages not importable after installation")
    else:
        logger.info("✅ All packages verified:")
        for line in result.stdout.strip().split('\n'):
            logger.info(f"   {line}")

    return workspace_path


def start_evaluation(
    bifrost_client: BifrostClient,
    config,
    config_path: str,
    workspace_path: str,
    remote_result_dir: str,
) -> tuple[str, str | None]:
    """Start evaluation on remote in tmux session.

    Args:
        bifrost_client: Bifrost client instance
        config: Evaluation configuration
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

    # Build evaluation command
    # NOTE: Directory name uses hyphen, not underscore!
    project_dir = f"{workspace_path}/dev/integration-evaluation"
    venv_path = f"{project_dir}/.venv/bin/activate"

    # Get API key from local environment
    api_key_var = config.api_key_env_var
    api_key = os.getenv(api_key_var, "")

    # Build environment variables
    env_vars = {}

    # Add API key if available
    if api_key:
        env_vars[api_key_var] = api_key
        logger.info(f"🔑 Using {api_key_var} for model API")
    else:
        logger.warning(f"⚠️  {api_key_var} not set - evaluation may fail")

    # Set CUDA_VISIBLE_DEVICES to requested GPUs
    if hasattr(config, 'gpu_ranks') and config.gpu_ranks:
        gpu_list = ",".join(str(r) for r in config.gpu_ranks)
        env_vars["CUDA_VISIBLE_DEVICES"] = gpu_list
        logger.info(f"🎮 Using GPUs: {gpu_list}")

    # Build evaluation command
    eval_cmd = f"python local.py configs/{config_name}"

    # Activate venv and run evaluation command
    full_cmd = f"source {venv_path} && {eval_cmd}"

    # Generate tmux session name from config
    experiment_name = getattr(config, 'experiment_name', 'eval')
    tmux_session = f"eval_{experiment_name}"
    eval_log = f"{remote_result_dir}/evaluation.log"

    logger.info(f"🚀 Starting evaluation in tmux session: {tmux_session}")
    logger.info(f"   Environment: {config.env_name}")
    logger.info(f"   Model: {config.model_name}")
    logger.info(f"   Samples: {config.num_samples}")
    logger.info(f"   Backend-bench GPU: {config.backend_bench_gpu}")
    logger.info(f"   Command: {full_cmd[:150]}...")

    # Kill existing session if it exists
    bifrost_client.exec(f"tmux kill-session -t {tmux_session} 2>/dev/null || true")

    # Create result directory
    bifrost_client.exec(f"mkdir -p {remote_result_dir}")

    # Start evaluation using kerbal
    session, err = start_tmux_session(
        bifrost_client,
        session_name=tmux_session,
        command=full_cmd,
        workspace=project_dir,
        log_file=eval_log,
        env_vars=env_vars,
    )

    if err:
        return session, f"Failed to start tmux session: {err}"

    logger.info(f"✅ Evaluation started in tmux session: {session}")
    return session, None


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
        description="Deploy evaluation to remote GPU instance"
    )
    parser.add_argument(
        "config",
        help="Path to config file (e.g., configs/prime_backend_bench.py)"
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
        help="Start evaluation and exit immediately (don't wait for completion)"
    )
    args = parser.parse_args()

    # Load environment variables (for API keys, etc.)
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
    experiment_name = getattr(config, 'experiment_name', 'eval')
    result_dir_name = f"{experiment_name}_{timestamp}"

    logger.info("🎯 Integration Evaluation Deployment")
    logger.info(f"📅 Experiment: {result_dir_name}")
    logger.info(f"🎮 Environment: {config.env_name}")
    logger.info(f"🤖 Model: {config.model_name}")
    logger.info(f"🖥️  SSH: {args.ssh}")
    logger.info("=" * 60)

    # Local output directory
    local_output_dir = Path(config.output_dir) / result_dir_name

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
        logger.info("✅ All prerequisites present (tmux)")

        # Check GPU availability
        gpu_ids = getattr(config, 'gpu_ranks', [0])
        if gpu_ids and config.backend_bench_gpu == "local":
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

        # Construct remote result directory
        # NOTE: Directory name uses hyphen, not underscore!
        project_dir = f"{workspace_path}/dev/integration-evaluation"
        remote_result_dir = f"{project_dir}/eval_results/{result_dir_name}"

        # Start evaluation
        tmux_session, error = start_evaluation(
            bifrost_client, config, args.config, workspace_path, remote_result_dir
        )
        if error:
            logger.error(f"❌ Failed to start evaluation: {error}")
            return 1

        if args.detached:
            logger.info("\n🎯 Evaluation started in detached mode")
            logger.info(f"   Attach: bifrost exec '{args.ssh}' 'tmux attach -t {tmux_session}'")
            logger.info(f"   Results will be in: {remote_result_dir}")
            return 0

        # Monitor evaluation with real-time log streaming
        log_path = f"{remote_result_dir}/evaluation.log"
        logger.info("📊 Monitoring evaluation with real-time log streaming...")
        logger.info(f"💡 If evaluation fails early, check: tmux attach -t {tmux_session}")

        monitor_config = LogStreamConfig(
            session_name=tmux_session,
            log_file=log_path,
            timeout_sec=86400,  # 24 hours max for evaluation
        )

        success, exit_code, err = stream_log_until_complete(
            bifrost_client,
            monitor_config,
        )

        if not success:
            logger.error(f"❌ Evaluation failed: {err}")

            # Try to get more error details from tmux pane
            logger.info("🔍 Checking tmux pane for error details...")

            # First check if session still exists
            check_result = bifrost_client.exec(f"tmux has-session -t {tmux_session} 2>&1 && echo 'EXISTS' || echo 'GONE'")
            if 'GONE' in check_result.stdout:
                logger.warning(f"⚠️  Tmux session '{tmux_session}' already exited")
                logger.info("💡 The script failed before writing to the log. Common causes:")
                logger.info("   • Import error (missing dependency)")
                logger.info("   • Python version mismatch")
                logger.info("   • Config file not found")
                logger.info("   • Syntax error in code")
            else:
                # Session exists, try to capture it
                result = bifrost_client.exec(f"tmux capture-pane -t {tmux_session} -p")
                if result.exit_code == 0 and result.stdout.strip():
                    logger.error("Tmux pane output:")
                    logger.error(result.stdout[-2000:])  # Last 2000 chars
                else:
                    logger.warning("⚠️  Tmux pane is empty or capture failed")

            # Continue to sync results anyway
            success = False
        else:
            logger.info("✅ Evaluation completed successfully")

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
