#!/usr/bin/env python3
"""Deploy SGLang inference server to remote GPU.

Simplified from qwen3_next/deploy.py but follows same patterns.
Designed for both evaluation (clicker) and RL training use cases.

Tiger Style: Explicit parameters, tuple returns, comprehensive logging.

Usage:
    from rollouts.deploy import deploy_sglang_server, ServerConfig

    config = ServerConfig(
        model="Qwen/Qwen2.5-VL-7B-Instruct",
        gpu_type="H100",
        port=30000,
    )

    server_info, error = await deploy_sglang_server(config)
"""

import asyncio
import logging
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)

# GPU availability thresholds
DEFAULT_MEMORY_THRESHOLD_MB = 1000  # Consider GPU busy if > 1GB used
DEFAULT_UTIL_THRESHOLD_PCT = 5      # Consider GPU busy if > 5% utilized


@dataclass(frozen=True)
class ServerConfig:
    """Configuration for deploying SGLang server.

    Tiger Style: All parameters explicit with sensible defaults.
    Simplified from qwen3_next but still comprehensive.
    """
    # Model configuration
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    dtype: str = "auto"  # "auto", "float16", "bfloat16"
    max_model_len: Optional[int] = None  # Context length (None = model default)
    quantization: Optional[str] = None  # None, "awq", "gptq", "fp8"
    trust_remote_code: bool = True

    # Engine configuration
    attention_backend: Optional[str] = None  # "triton" for Blackwell, "flashinfer" for others

    # GPU configuration
    gpu_type: str = "H100"  # For memory estimation
    gpu_ranks: list[int] = field(default_factory=lambda: [0])  # GPU indices

    # Server configuration
    port: int = 30000
    host: str = "0.0.0.0"

    # Performance tuning
    tensor_parallel_size: int = 1  # Must match len(gpu_ranks)
    gpu_memory_utilization: float = 0.9
    enable_prefix_caching: bool = False  # RadixAttention

    # Deployment configuration
    ssh_connection: Optional[str] = None  # "user@host:port" for remote, None for local
    ssh_key: str = "~/.ssh/id_ed25519"
    tmux_session_name: str = "sglang-server"

    # Environment
    hf_cache_dir: str = "/home/ubuntu/.cache/huggingface"
    use_hf_transfer: bool = True
    allow_long_max_model_len: bool = True
    flash_attn_version: Optional[str] = None

    def __post_init__(self):
        """Validate configuration."""
        assert len(self.model) > 0, "model cannot be empty"
        assert self.port > 1024 and self.port < 65536, f"Invalid port: {self.port}"
        assert len(self.gpu_ranks) > 0, "gpu_ranks cannot be empty"

        # Validate TP matches GPU count
        if self.tensor_parallel_size != len(self.gpu_ranks):
            raise ValueError(
                f"tensor_parallel_size ({self.tensor_parallel_size}) "
                f"must match number of GPUs ({len(self.gpu_ranks)})"
            )


@dataclass(frozen=True)
class ServerInfo:
    """Information about deployed server.

    Tiger Style: Explicit return type for deployment results.
    """
    url: str  # Server URL (e.g., "http://localhost:30000")
    model: str  # Model name
    port: int  # Server port
    ssh_connection: Optional[str]  # SSH connection string if remote
    tmux_session: str  # Tmux session name
    gpu_ranks: list[int]  # GPU indices used

    def get_api_base(self) -> str:
        """Get OpenAI-compatible API base URL."""
        return f"{self.url}/v1"


def get_venv_path() -> str:
    """Get venv path for SGLang.

    Tiger Style: Explicit venv path for isolation.
    """
    return ".venv-sglang"


def generate_bootstrap_command(config: ServerConfig) -> str:
    """Generate bootstrap command to install SGLang on remote server.

    Tiger Style: Minimal bootstrap, let SGLang manage its dependencies.
    Creates venv for isolation (matches qwen3_next pattern).

    Args:
        config: Server configuration

    Returns:
        Bootstrap command string
    """
    assert config is not None
    assert isinstance(config, ServerConfig)

    venv_name = get_venv_path()
    assert venv_name is not None
    assert len(venv_name) > 0

    steps = [
        # Ensure uv is in PATH
        'export PATH="$HOME/.local/bin:$PATH"',

        # Create venv if doesn't exist (skip if exists to avoid reinstalling)
        f'[ -d {venv_name} ] && echo "Using existing venv: {venv_name}" || uv venv {venv_name} --python 3.10',

        # Activate venv
        f'source {venv_name}/bin/activate',

        # Install SGLang from git (latest main branch)
        # SGLang manages its own torch/flashinfer dependencies
        # --index-strategy unsafe-best-match: search all indexes for best CUDA-compatible torch
        'uv pip install "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python" '
        '--index-strategy unsafe-best-match',

        # Show installed version for verification
        'echo "=== Installed SGLang version ===" && uv pip show sglang | grep Version || true',
    ]

    assert len(steps) > 0
    result = ' && '.join(steps)
    assert len(result) > 0
    return result


def check_local_prerequisites() -> Tuple[bool, Optional[str]]:
    """Check if required tools are installed locally.

    Tiger Style: Preflight validation, fail-fast.

    Returns:
        (True, None) if all prerequisites met
        (False, error_message) if missing tools
    """
    missing = []

    # Check for nvidia-smi (GPU driver)
    try:
        result = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            timeout=5,
        )
        assert result is not None
        if result.returncode != 0:
            missing.append("nvidia-smi (NVIDIA drivers not installed or GPU not available)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("nvidia-smi (NVIDIA drivers not installed)")

    # Check for tmux
    try:
        result = subprocess.run(
            ["tmux", "-V"],
            capture_output=True,
            timeout=5,
        )
        assert result is not None
        if result.returncode != 0:
            missing.append("tmux (install with: brew install tmux or apt install tmux)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("tmux (install with: brew install tmux or apt install tmux)")

    # Check for python
    try:
        result = subprocess.run(
            ["python", "--version"],
            capture_output=True,
            timeout=5,
        )
        assert result is not None
        if result.returncode != 0:
            missing.append("python")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        missing.append("python")

    assert isinstance(missing, list)
    if missing:
        error_msg = "Missing prerequisites:\n" + "\n".join(f"  - {m}" for m in missing)
        assert error_msg is not None
        assert len(error_msg) > 0
        return False, error_msg

    return True, None


def check_gpus_available_local(
    gpu_ids: list[int],
    memory_threshold_mb: int = DEFAULT_MEMORY_THRESHOLD_MB,
    util_threshold_pct: int = DEFAULT_UTIL_THRESHOLD_PCT,
) -> Tuple[bool, Optional[str]]:
    """Check if specified GPUs are free on local machine.

    Tiger Style: Assert preconditions, explicit thresholds.

    Args:
        gpu_ids: List of GPU IDs to check
        memory_threshold_mb: Memory threshold in MB
        util_threshold_pct: Utilization threshold %

    Returns:
        (True, None) if all GPUs are free
        (False, error_message) if any GPU is busy
    """
    assert gpu_ids is not None
    assert isinstance(gpu_ids, list)
    assert len(gpu_ids) > 0, "Must specify at least one GPU to check"
    assert all(isinstance(gpu_id, int) for gpu_id in gpu_ids)
    assert all(gpu_id >= 0 for gpu_id in gpu_ids)
    assert memory_threshold_mb > 0
    assert util_threshold_pct >= 0
    assert util_threshold_pct <= 100

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode != 0:
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

        # Check if requested GPUs are free
        for gpu_id in gpu_ids:
            if gpu_id not in gpu_stats:
                available = list(gpu_stats.keys())
                return False, f"GPU {gpu_id} not found (available GPUs: {available})"

            stats = gpu_stats[gpu_id]
            mem_mb = stats['memory_mb']
            util = stats['util_pct']

            if mem_mb > memory_threshold_mb or util > util_threshold_pct:
                return False, f"GPU {gpu_id} is busy ({mem_mb}MB used, {util}% util)"

        return True, None

    except Exception as e:
        return False, f"GPU availability check failed: {e}"


def check_port_available_local(port: int) -> Tuple[bool, Optional[str]]:
    """Check if port is available on local machine.

    Args:
        port: Port number to check

    Returns:
        (True, None) if port is free
        (False, error_message) if port is in use
    """
    try:
        result = subprocess.run(
            ["lsof", f"-i:{port}"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        # lsof returns 0 if port is in use, 1 if free
        if result.returncode == 0:
            return False, f"Port {port} is already in use"

        return True, None

    except FileNotFoundError:
        # lsof not available, skip check
        logger.warning("lsof not available, skipping port check")
        return True, None
    except Exception as e:
        logger.warning(f"Port check failed: {e}")
        return True, None


def check_remote_prerequisites(ssh_connection: str, ssh_key: str) -> Tuple[bool, Optional[str]]:
    """Check if required tools are installed on remote.

    Tiger Style: Assert preconditions before proceeding.

    Args:
        ssh_connection: SSH connection string (user@host:port)
        ssh_key: Path to SSH private key

    Returns:
        (True, None) if all prerequisites met
        (False, error_message) if missing tools
    """
    try:
        from bifrost.client import BifrostClient
    except ImportError:
        return False, "bifrost not installed. Install with: pip install rollouts[deploy]"

    bifrost_client = BifrostClient(ssh_connection, ssh_key)
    missing = []

    # Check for uv
    result = bifrost_client.exec(
        'export PATH="$HOME/.local/bin:$PATH" && '
        'command -v uv >/dev/null 2>&1 && echo "OK" || echo "MISSING"'
    )
    if result.stdout.strip() != 'OK':
        missing.append("uv (install: curl -LsSf https://astral.sh/uv/install.sh | sh)")

    # Check for tmux
    result = bifrost_client.exec('command -v tmux >/dev/null 2>&1 && echo "OK" || echo "MISSING"')
    if result.stdout.strip() != 'OK':
        missing.append("tmux (install: apt install tmux or yum install tmux)")

    # Check for nvidia-smi
    result = bifrost_client.exec('command -v nvidia-smi >/dev/null 2>&1 && echo "OK" || echo "MISSING"')
    if result.stdout.strip() != 'OK':
        missing.append("nvidia-smi (NVIDIA drivers not installed)")

    if missing:
        error_msg = "Missing prerequisites on remote:\n" + "\n".join(f"  - {m}" for m in missing)
        return False, error_msg

    return True, None


def check_gpus_available_remote(
    ssh_connection: str,
    ssh_key: str,
    gpu_ids: list[int],
    memory_threshold_mb: int = DEFAULT_MEMORY_THRESHOLD_MB,
    util_threshold_pct: int = DEFAULT_UTIL_THRESHOLD_PCT,
) -> Tuple[bool, Optional[str]]:
    """Check if specified GPUs are free on remote machine.

    Args:
        ssh_connection: SSH connection string
        ssh_key: Path to SSH private key
        gpu_ids: List of GPU IDs to check
        memory_threshold_mb: Memory threshold in MB
        util_threshold_pct: Utilization threshold %

    Returns:
        (True, None) if all GPUs are free
        (False, error_message) if any GPU is busy
    """
    try:
        from bifrost.client import BifrostClient
    except ImportError:
        return False, "bifrost not installed"

    bifrost_client = BifrostClient(ssh_connection, ssh_key)

    try:
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

        # Check if requested GPUs are free
        for gpu_id in gpu_ids:
            if gpu_id not in gpu_stats:
                return False, f"GPU {gpu_id} not found on remote (available: {list(gpu_stats.keys())})"

            stats = gpu_stats[gpu_id]
            mem_mb = stats['memory_mb']
            util = stats['util_pct']

            if mem_mb > memory_threshold_mb or util > util_threshold_pct:
                return False, f"GPU {gpu_id} is busy ({mem_mb}MB used, {util}% util)"

        return True, None

    except Exception as e:
        return False, f"GPU availability check failed: {e}"


def check_port_available_remote(
    ssh_connection: str,
    ssh_key: str,
    port: int
) -> Tuple[bool, Optional[str]]:
    """Check if port is available on remote machine.

    Args:
        ssh_connection: SSH connection string
        ssh_key: Path to SSH private key
        port: Port number to check

    Returns:
        (True, None) if port is free
        (False, error_message) if port is in use
    """
    try:
        from bifrost.client import BifrostClient
    except ImportError:
        return False, "bifrost not installed"

    bifrost_client = BifrostClient(ssh_connection, ssh_key)

    try:
        result = bifrost_client.exec(f"lsof -i :{port} || echo 'PORT_FREE'")

        if "PORT_FREE" not in result.stdout:
            return False, f"Port {port} is already in use on remote"

        return True, None

    except Exception as e:
        logger.warning(f"Remote port check failed: {e}")
        return True, None


def build_sglang_command(config: ServerConfig) -> str:
    """Build SGLang launch command.

    Tiger Style: Pure function, explicit parameters.
    Based on qwen3_next/engines/sglang.py but simplified.

    Args:
        config: Server configuration

    Returns:
        Command string to launch SGLang server
    """
    cmd_parts = [
        "python -m sglang.launch_server",
        f"--model {config.model}",
        f"--host {config.host}",
        f"--port {config.port}",
    ]

    # Tensor parallelism
    if config.tensor_parallel_size > 1:
        cmd_parts.append(f"--tp {config.tensor_parallel_size}")

    # Model parameters
    if config.max_model_len:
        cmd_parts.append(f"--context-length {config.max_model_len}")

    if config.dtype != "auto":
        cmd_parts.append(f"--dtype {config.dtype}")

    if config.quantization:
        cmd_parts.append(f"--quantization {config.quantization}")

    if config.trust_remote_code:
        cmd_parts.append("--trust-remote-code")

    # Performance tuning
    cmd_parts.append(f"--mem-fraction-static {config.gpu_memory_utilization}")

    if config.enable_prefix_caching:
        cmd_parts.append("--enable-radix-cache")

    # Engine configuration
    if config.attention_backend:
        cmd_parts.append(f"--attention-backend {config.attention_backend}")

    return " \\\n    ".join(cmd_parts)


async def deploy_sglang_server(
    config: ServerConfig,
    wait_for_ready: bool = True,
    timeout_seconds: int = 300,
) -> Tuple[Optional[ServerInfo], Optional[str]]:
    """Deploy SGLang server to local or remote GPU.

    Tiger Style: Tuple return for explicit error handling.

    Args:
        config: Server configuration
        wait_for_ready: Wait for health check before returning
        timeout_seconds: Max wait time for server startup

    Returns:
        (ServerInfo, None) on success
        (None, error_message) on failure
    """
    logger.info("🚀 Deploying SGLang server...")
    logger.info(f"   Model: {config.model}")
    logger.info(f"   GPUs: {config.gpu_ranks} (TP={config.tensor_parallel_size})")
    logger.info(f"   Port: {config.port}")

    if config.ssh_connection:
        # Remote deployment via bifrost
        return await _deploy_remote(config, wait_for_ready, timeout_seconds)
    else:
        # Local deployment
        return await _deploy_local(config, wait_for_ready, timeout_seconds)


async def _deploy_local(
    config: ServerConfig,
    wait_for_ready: bool,
    timeout_seconds: int,
) -> Tuple[Optional[ServerInfo], Optional[str]]:
    """Deploy server locally using subprocess + tmux.

    Tiger Style: Helper function with tuple return.
    """
    import subprocess

    logger.info("")
    logger.info("🔍 Preflight Checks")
    logger.info("-" * 60)

    # Check 1: Prerequisites
    logger.info("Checking prerequisites...")
    prereq_ok, prereq_error = check_local_prerequisites()
    if not prereq_ok:
        logger.error(f"❌ {prereq_error}")
        return None, prereq_error
    logger.info("✅ Prerequisites OK")

    # Check 2: GPU availability
    logger.info(f"Checking GPU availability: {config.gpu_ranks}...")
    gpu_ok, gpu_error = check_gpus_available_local(config.gpu_ranks)
    if not gpu_ok:
        logger.error(f"❌ {gpu_error}")
        return None, gpu_error
    logger.info(f"✅ GPUs available: {config.gpu_ranks}")

    # Check 3: Port availability
    logger.info(f"Checking port {config.port}...")
    port_ok, port_error = check_port_available_local(config.port)
    if not port_ok:
        logger.error(f"❌ {port_error}")
        return None, port_error
    logger.info(f"✅ Port {config.port} available")

    logger.info("")
    logger.info("✅ All preflight checks passed!")
    logger.info("")

    # Build command
    sglang_cmd = build_sglang_command(config)

    # Build environment variables
    env_vars = []
    cuda_visible_devices = ",".join(str(r) for r in config.gpu_ranks)
    env_vars.append(f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    env_vars.append(f"export HF_HOME={config.hf_cache_dir}")

    if config.use_hf_transfer:
        env_vars.append("export HF_HUB_ENABLE_HF_TRANSFER=1")

    env_str = " && ".join(env_vars)

    # Check if tmux session exists
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", config.tmux_session_name],
            capture_output=True,
            timeout=5,
        )
        if result.returncode == 0:
            error_msg = (
                f"Tmux session '{config.tmux_session_name}' already exists. "
                f"Kill it first: tmux kill-session -t {config.tmux_session_name}"
            )
            logger.error(f"❌ {error_msg}")
            return None, error_msg
    except Exception as e:
        logger.warning(f"Could not check tmux session: {e}")

    # Start server in tmux
    tmux_cmd = f"""
tmux new-session -d -s {config.tmux_session_name} "
{env_str} && \\
{sglang_cmd} 2>&1 | tee sglang_server.log
"
"""

    logger.info("📦 Starting SGLang server in tmux...")
    try:
        result = subprocess.run(
            tmux_cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            error_msg = f"Failed to start tmux: {result.stderr}"
            logger.error(f"❌ {error_msg}")
            return None, error_msg
    except Exception as e:
        error_msg = f"Failed to start tmux: {e}"
        logger.error(f"❌ {error_msg}")
        return None, error_msg

    logger.info(f"✅ Server started in tmux session: {config.tmux_session_name}")

    # Create ServerInfo
    server_info = ServerInfo(
        url=f"http://localhost:{config.port}",
        model=config.model,
        port=config.port,
        ssh_connection=None,
        tmux_session=config.tmux_session_name,
        gpu_ranks=list(config.gpu_ranks),
    )

    if wait_for_ready:
        # Wait for health check
        logger.info("⏳ Waiting for server to be ready...")
        ready, error = await _wait_for_health(server_info, timeout_seconds)
        if not ready:
            return None, error
        logger.info("✅ Server is ready!")

    return server_info, None


async def _deploy_remote(
    config: ServerConfig,
    wait_for_ready: bool,
    timeout_seconds: int,
) -> Tuple[Optional[ServerInfo], Optional[str]]:
    """Deploy server to remote GPU via bifrost.

    Tiger Style: Helper function with tuple return.
    Based on qwen3_next/deploy.py pattern.
    """
    try:
        from bifrost.client import BifrostClient
    except ImportError:
        error_msg = "bifrost not installed. Install with: pip install rollouts[deploy]"
        logger.error(f"❌ {error_msg}")
        return None, error_msg

    assert config.ssh_connection is not None

    # Connect to remote
    logger.info(f"🔗 Connecting to {config.ssh_connection}...")
    bifrost_client = BifrostClient(config.ssh_connection, config.ssh_key)

    logger.info("")
    logger.info("🔍 Preflight Checks (Remote)")
    logger.info("-" * 60)

    # Check 1: Prerequisites
    logger.info("Checking remote prerequisites...")
    prereq_ok, prereq_error = check_remote_prerequisites(config.ssh_connection, config.ssh_key)
    if not prereq_ok:
        logger.error(f"❌ {prereq_error}")
        return None, prereq_error
    logger.info("✅ Prerequisites OK")

    # Check 2: GPU availability
    logger.info(f"Checking remote GPU availability: {config.gpu_ranks}...")
    gpu_ok, gpu_error = check_gpus_available_remote(
        config.ssh_connection, config.ssh_key, config.gpu_ranks
    )
    if not gpu_ok:
        logger.error(f"❌ {gpu_error}")
        return None, gpu_error
    logger.info(f"✅ GPUs available: {config.gpu_ranks}")

    # Check 3: Port availability
    logger.info(f"Checking remote port {config.port}...")
    port_ok, port_error = check_port_available_remote(
        config.ssh_connection, config.ssh_key, config.port
    )
    if not port_ok:
        logger.error(f"❌ {port_error}")
        return None, port_error
    logger.info(f"✅ Port {config.port} available")

    logger.info("")
    logger.info("✅ All preflight checks passed!")
    logger.info("")

    # Deploy code (without bootstrap first)
    logger.info("📦 Deploying code...")
    try:
        workspace_path = bifrost_client.push()
        logger.info(f"✅ Code deployed to {workspace_path}")
    except Exception as e:
        logger.warning(f"Code deployment skipped: {e}")
        workspace_path = "~"

    # Expand tilde to absolute path
    result = bifrost_client.exec(f"echo {workspace_path}")
    if result.exit_code == 0 and result.stdout:
        workspace_path = result.stdout.strip()
        logger.info(f"📍 Expanded workspace path: {workspace_path}")

    # Run bootstrap to install SGLang with streaming output
    bootstrap_cmd = generate_bootstrap_command(config)
    logger.info("🔧 Running bootstrap to install SGLang...")
    logger.info(f"   Bootstrap command: {bootstrap_cmd[:100]}...")
    logger.info("=" * 60)

    # Execute bootstrap with streaming output so we can see progress
    # Tiger Style: Explicit control flow - capture actual exit code
    full_cmd = f"cd {workspace_path} && {bootstrap_cmd}; EXIT=$?; echo '::EXIT_CODE::'$EXIT; exit $EXIT"

    exit_code = None
    output_lines = []

    try:
        for line in bifrost_client.exec_stream(full_cmd):
            # Capture exit code from marker line
            if line.startswith("::EXIT_CODE::"):
                exit_code_str = line.replace("::EXIT_CODE::", "").strip()
                if exit_code_str.isdigit():
                    exit_code = int(exit_code_str)
                    logger.info(f"📊 Bootstrap exit code: {exit_code}")
            else:
                # Print output in real-time
                print(line, end='', flush=True)
                output_lines.append(line)
    except Exception as e:
        logger.error(f"❌ Bootstrap streaming failed: {e}")
        return None, f"Bootstrap execution failed: {e}"

    logger.info("=" * 60)

    # Check bootstrap succeeded
    if exit_code is None:
        error_msg = "Could not determine bootstrap exit code"
        logger.error(f"❌ {error_msg}")
        logger.error("   Last 10 lines of output:")
        for line in output_lines[-10:]:
            logger.error(f"   {line.rstrip()}")
        return None, error_msg

    if exit_code != 0:
        error_msg = f"Bootstrap failed with exit code {exit_code}"
        logger.error(f"❌ {error_msg}")
        logger.error("=" * 60)
        logger.error("📋 Error context (last 30 lines):")
        logger.error("=" * 60)
        for line in output_lines[-30:]:
            if 'error:' in line.lower() or 'failed' in line.lower() or '×' in line:
                logger.error(f">>> {line.rstrip()}")
            else:
                logger.error(f"    {line.rstrip()}")
        logger.error("=" * 60)
        return None, error_msg

    logger.info(f"✅ Bootstrap completed successfully")

    # Verify bootstrap postconditions (paired assertion - like qwen3_next)
    venv_name = get_venv_path()
    venv_activate_path = f"{workspace_path}/{venv_name}/bin/activate"
    verify_venv_cmd = f"test -f {venv_activate_path} && echo 'EXISTS' || echo 'MISSING'"
    verify_result = bifrost_client.exec(verify_venv_cmd)
    venv_exists = verify_result.stdout.strip() == 'EXISTS' if verify_result.stdout else False

    if not venv_exists:
        error_msg = f"Bootstrap reported success but venv not found: {venv_name}"
        logger.error(f"❌ {error_msg}")
        logger.error("   Possible causes:")
        logger.error("   - uv failed silently during venv creation")
        logger.error("   - Python version installation failed")
        logger.error("   - Disk space or permission issues")
        logger.error("")
        logger.error("   To debug, SSH to the remote and run:")
        logger.error(f"   ls -la {workspace_path}/{venv_name}/")
        return None, error_msg

    logger.info(f"✅ Verified venv exists: {venv_name}")

    # Verify SGLang is actually installed in the venv (critical postcondition!)
    logger.info("🔍 Verifying SGLang installation...")
    sglang_check_cmd = f"cd {workspace_path} && source {venv_name}/bin/activate && python -c 'import sglang; print(sglang.__version__)'"
    sglang_result = bifrost_client.exec(sglang_check_cmd)

    if sglang_result.exit_code != 0:
        error_msg = "Bootstrap succeeded but SGLang not importable in venv"
        logger.error(f"❌ {error_msg}")
        logger.error(f"   Import check failed with exit code: {sglang_result.exit_code}")
        logger.error("   Output:")
        logger.error(f"   {sglang_result.stdout}")
        logger.error(f"   {sglang_result.stderr}")
        logger.error("")
        logger.error("   This means the bootstrap completed but SGLang wasn't actually installed.")
        logger.error("   Check the bootstrap output above for errors.")
        return None, error_msg

    sglang_version = sglang_result.stdout.strip()
    logger.info(f"✅ SGLang {sglang_version} verified installed")
    logger.info(f"✅ Installation complete at {workspace_path}")

    # Build command
    sglang_cmd = build_sglang_command(config)

    # Build environment and venv activation
    venv_name = get_venv_path()
    setup_commands = []

    # Activate venv
    setup_commands.append(f"source {venv_name}/bin/activate")

    # Set environment variables
    cuda_visible_devices = ",".join(str(r) for r in config.gpu_ranks)
    setup_commands.append(f"export CUDA_VISIBLE_DEVICES={cuda_visible_devices}")
    setup_commands.append(f"export HF_HOME={config.hf_cache_dir}")

    if config.use_hf_transfer:
        setup_commands.append("export HF_HUB_ENABLE_HF_TRANSFER=1")

    setup_str = " && ".join(setup_commands)

    # Start server in tmux
    tmux_cmd = f"""
cd {workspace_path} && \\
tmux new-session -d -s {config.tmux_session_name} "
{setup_str} && \\
{sglang_cmd} 2>&1 | tee sglang_server.log
"
"""

    logger.info("🚀 Starting SGLang server on remote...")
    result = bifrost_client.exec(tmux_cmd)
    if result.exit_code != 0:
        error_msg = f"Failed to start server: {result.stderr}"
        logger.error(f"❌ {error_msg}")
        return None, error_msg

    logger.info(f"✅ Server started in tmux session: {config.tmux_session_name}")

    # Parse host from SSH connection (user@host:port or user@host)
    ssh_parts = config.ssh_connection.split("@")
    if len(ssh_parts) == 2:
        host_port = ssh_parts[1]
        host = host_port.split(":")[0]
    else:
        host = "localhost"

    # Create ServerInfo
    server_info = ServerInfo(
        url=f"http://{host}:{config.port}",
        model=config.model,
        port=config.port,
        ssh_connection=config.ssh_connection,
        tmux_session=config.tmux_session_name,
        gpu_ranks=list(config.gpu_ranks),
    )

    if wait_for_ready:
        # Wait for health check (via bifrost exec)
        logger.info("⏳ Waiting for server to be ready...")
        ready, error = await _wait_for_health_remote(server_info, bifrost_client, timeout_seconds)
        if not ready:
            return None, error
        logger.info("✅ Server is ready!")

    return server_info, None


async def _wait_for_health(
    server_info: ServerInfo,
    timeout_seconds: int,
) -> Tuple[bool, Optional[str]]:
    """Wait for local server health check.

    Tiger Style: Tuple return for success/failure.
    Checks tmux session first to detect crashes early.
    """
    import aiohttp

    health_url = f"{server_info.url}/health"
    poll_interval = 5
    max_iterations = timeout_seconds // poll_interval

    async with aiohttp.ClientSession() as session:
        for i in range(max_iterations):
            # First check if tmux session is still alive
            try:
                result = subprocess.run(
                    ["tmux", "has-session", "-t", server_info.tmux_session],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode != 0:
                    error_msg = "Server crashed during startup (tmux session exited)"
                    logger.error(f"❌ {error_msg}")
                    # Fetch last 50 lines from log file
                    try:
                        log_result = subprocess.run(
                            ["tail", "-50", "sglang_server.log"],
                            capture_output=True,
                            timeout=5,
                        )
                        if log_result.returncode == 0 and log_result.stdout:
                            logger.error("📋 Last 50 lines of server log:")
                            logger.error("=" * 60)
                            logger.error(log_result.stdout.decode('utf-8', errors='ignore'))
                            logger.error("=" * 60)
                    except Exception:
                        pass
                    return False, error_msg
            except Exception:
                pass  # tmux check failed, continue to health check

            # Then check health endpoint
            try:
                async with session.get(health_url, timeout=5) as resp:
                    if resp.status == 200:
                        logger.info(f"✅ Server is ready (took ~{(i+1)*poll_interval}s)")
                        return True, None
            except Exception:
                pass

            if i < max_iterations - 1:
                await asyncio.sleep(poll_interval)

    error_msg = f"Server failed to become ready within {timeout_seconds}s"
    logger.error(f"❌ {error_msg}")
    return False, error_msg


async def _wait_for_health_remote(
    server_info: ServerInfo,
    bifrost_client: Any,
    timeout_seconds: int,
) -> Tuple[bool, Optional[str]]:
    """Wait for remote server health check via bifrost.

    Tiger Style: Tuple return for success/failure.
    Checks tmux session first to detect crashes early.
    """
    poll_interval = 5
    max_iterations = timeout_seconds // poll_interval

    for i in range(max_iterations):
        # First check if tmux session is still alive
        tmux_check = bifrost_client.exec(
            f"tmux has-session -t {server_info.tmux_session} 2>/dev/null"
        )
        if tmux_check.exit_code != 0:
            error_msg = "Server crashed during startup (tmux session exited)"
            logger.error(f"❌ {error_msg}")
            # Fetch last 50 lines from log file
            log_result = bifrost_client.exec(
                "tail -50 sglang_server.log 2>/dev/null || echo 'Could not read log file'"
            )
            if log_result.stdout and "Could not read log file" not in log_result.stdout:
                logger.error("📋 Last 50 lines of server log:")
                logger.error("=" * 60)
                logger.error(log_result.stdout)
                logger.error("=" * 60)
            return False, error_msg

        # Then check health endpoint via curl on remote
        result = bifrost_client.exec(
            f"curl -s http://localhost:{server_info.port}/health || echo 'NOT_READY'"
        )

        if result.exit_code == 0 and "NOT_READY" not in result.stdout:
            logger.info(f"✅ Server is ready (took ~{(i+1)*poll_interval}s)")
            return True, None

        if i < max_iterations - 1:
            await asyncio.sleep(poll_interval)

    error_msg = f"Server failed to become ready within {timeout_seconds}s"
    logger.error(f"❌ {error_msg}")
    return False, error_msg


# Convenience function for creating Endpoint from ServerInfo
def server_info_to_endpoint(server_info: ServerInfo) -> "Endpoint":
    """Convert ServerInfo to rollouts Endpoint.

    Tiger Style: Explicit conversion function.
    """
    from rollouts.rollout import Endpoint

    return Endpoint(
        model=server_info.model,
        api_base=server_info.get_api_base(),
        api_key="EMPTY",  # SGLang doesn't require API key
        temperature=0.0,
        max_tokens=2048,
        max_retries=3,
        backoff_base_seconds=1.0,
    )
