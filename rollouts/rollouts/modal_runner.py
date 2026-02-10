"""Modal-based training runner.

Run training workloads on Modal sandboxes with GPU access.
Uses Modal's native async APIs via trio_asyncio bridge.

Usage:
    # From config file
    python -m rollouts.modal_runner --config examples/rl/reverse_text/grpo_01_01.py

    # With specific GPU
    python -m rollouts.modal_runner --config examples/rl/reverse_text/grpo_01_01.py --gpu H100

Design:
    Unlike run.py which uses bifrost (SSH-based), this uses Modal sandboxes directly.
    Modal sandboxes have ~10-30s cold start vs RunPod's 2-5 min.

    For GRPO training, everything runs in a single sandbox:
    - SGLang inference server (launched by grpo_train)
    - Training backend (PyTorch)
    - Agent loop (rollout generation)

    Weight sync is local disk since trainer + inference are colocated.
    When we scale to separate sandboxes, we'll use ModalVolumeWeightSync.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    pass

from ._logging import setup_logging

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# Default Modal app name
MODAL_APP_NAME = "rollouts-training"


@dataclass(frozen=True)
class ModalDeps:
    """Dependencies for Modal sandbox.

    Explicit specification of what goes into the container.
    """

    python_version: str = "3.12"
    system_packages: tuple[str, ...] = ("bash", "curl", "git", "build-essential", "libnuma1")
    pip_packages: tuple[str, ...] = (
        "torch>=2.4",
        "transformers>=4.50",
        "datasets",
        "accelerate",
        "safetensors",
        "sglang[all]",
        "curl_cffi",
        "peft",
    )
    pip_index_url: str = "https://download.pytorch.org/whl/cu124"
    pip_extra_index_url: str = "https://pypi.org/simple"
    bootstrap_commands: tuple[str, ...] = ()


@dataclass
class ModalRunConfig:
    """Configuration for a Modal training run."""

    config_path: str
    gpu_type: str = "A100"
    gpu_count: int = 1
    deps: ModalDeps = field(default_factory=ModalDeps)
    timeout_hours: int = 4

    # Git sync config
    git_repo: str = "https://github.com/your-org/research.git"
    git_branch: str = "main"

    # Use local code sync instead of git clone
    use_local_sync: bool = True


def _build_modal_image(modal: Any, deps: ModalDeps, gpu_type: str) -> Any:
    """Build Modal image from deps specification.

    Handles GPU-specific CUDA requirements (B200 needs cu128 nightly).
    """
    # GPU-specific torch index
    if gpu_type in ("B200", "GB200"):
        pip_index = "https://download.pytorch.org/whl/nightly/cu128"
    else:
        pip_index = deps.pip_index_url

    image = modal.Image.debian_slim(python_version=deps.python_version)

    if deps.system_packages:
        image = image.apt_install(*deps.system_packages)

    if deps.pip_packages:
        image = image.pip_install(
            *deps.pip_packages,
            index_url=pip_index,
            extra_index_url=deps.pip_extra_index_url,
        )

    for cmd in deps.bootstrap_commands:
        image = image.run_commands(cmd)

    # Set up HuggingFace cache
    image = image.env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })

    return image


async def _create_sandbox(
    config: ModalRunConfig,
) -> tuple[Any, str]:
    """Create Modal sandbox using native async API.

    Returns (sandbox, sandbox_id).
    """
    import modal
    import trio_asyncio

    logger.info(f"Looking up app: {MODAL_APP_NAME}")
    app = await trio_asyncio.aio_as_trio(
        modal.App.lookup.aio(MODAL_APP_NAME, create_if_missing=True)
    )

    logger.info("Building image...")
    image = _build_modal_image(modal, config.deps, config.gpu_type)
    logger.info("Image built")

    # GPU spec
    gpu_count = config.gpu_count
    gpu_type = config.gpu_type
    if gpu_count > 1:
        gpu_spec = f"{gpu_type}:{gpu_count}"
    else:
        gpu_spec = gpu_type

    # Unique name
    ts = int(datetime.now(timezone.utc).timestamp())
    sandbox_name = f"rollouts-{config.gpu_type.lower()}-{ts}"

    timeout_seconds = config.timeout_hours * 3600

    logger.info(f"Creating sandbox: {sandbox_name} (gpu={gpu_spec})...")
    sandbox = await trio_asyncio.aio_as_trio(
        modal.Sandbox.create.aio(
            app=app,
            image=image,
            gpu=gpu_spec,
            timeout=timeout_seconds,
            name=sandbox_name,
        )
    )

    assert sandbox is not None, "Sandbox.create() returned None"
    assert sandbox.object_id, "Sandbox missing object_id"

    logger.info(f"Sandbox created: {sandbox.object_id}")

    return sandbox, sandbox.object_id


def _exec_sync(sandbox: Any, command: str, timeout: int = 300) -> tuple[str, str, int]:
    """Execute command in sandbox. Blocking.

    Returns (stdout, stderr, exit_code).
    """
    proc = sandbox.exec("bash", "-c", command, timeout=timeout)

    stdout_lines = []
    for line in proc.stdout:
        stdout_lines.append(line)
        logger.info(f"[sandbox] {line.rstrip()}")

    stderr_lines = []
    for line in proc.stderr:
        stderr_lines.append(line)
        logger.warning(f"[sandbox stderr] {line.rstrip()}")

    proc.wait()

    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


async def _sync_code_to_sandbox(sandbox: Any, local_root: Path) -> str:
    """Sync local code to sandbox via sandbox.open() file API.

    Uses git bundle + sandbox.open() for efficient file transfer.

    Returns workspace path in sandbox (the rollouts subdir within the cloned repo).
    """
    # Git root is ~/research (parent), we clone to /workspace/research
    # The rollouts code is at /workspace/research/rollouts
    clone_dir = "/workspace/research"
    workspace = "/workspace/research/rollouts"

    def _sync() -> None:
        # Create git bundle of current HEAD (fast, includes all needed objects)
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        try:
            # Get current branch/commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(local_root),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            logger.info(f"Bundling commit {commit[:8]}...")

            # Create bundle
            subprocess.run(
                ["git", "bundle", "create", bundle_path, "HEAD"],
                cwd=str(local_root),
                check=True,
                capture_output=True,
            )

            bundle_size = os.path.getsize(bundle_path)
            logger.info(f"Bundle size: {bundle_size / 1024 / 1024:.1f} MB")

            # Read bundle data
            with open(bundle_path, "rb") as f:
                bundle_data = f.read()

            # Create workspace directory
            _exec_sync(sandbox, "mkdir -p /workspace", timeout=30)

            # Use sandbox.open() for proper file transfer (Alpha API)
            logger.info("Uploading bundle via sandbox.open()...")
            remote_file = sandbox.open("/tmp/repo.bundle", "wb")
            remote_file.write(bundle_data)
            remote_file.close()
            logger.info(f"Uploaded {len(bundle_data) / 1024 / 1024:.1f} MB")

            logger.info("Extracting bundle...")
            # Clone from bundle - clones parent repo (research) to /workspace/research
            _exec_sync(
                sandbox,
                "cd /workspace && git clone /tmp/repo.bundle research && "
                "cd research && git checkout HEAD",
                timeout=120,
            )

            logger.info(f"Code synced to {workspace}")

        finally:
            os.unlink(bundle_path)

    await trio.to_thread.run_sync(_sync)
    return workspace


async def _run_training_in_sandbox(
    sandbox: Any,
    workspace: str,
    config_path: str,
    run_name: str,
) -> dict[str, Any]:
    """Run training script inside Modal sandbox.

    Returns metrics from training.
    """
    # Get relative config path (may already be relative)
    config_p = Path(config_path)
    if config_p.is_absolute():
        config_rel = config_p.relative_to(REPO_ROOT)
    else:
        config_rel = config_p

    # Install base dependencies (not using editable install to avoid PyPI deps)
    logger.info("Installing dependencies...")

    def _install() -> None:
        # Install just the rollouts deps (skip bifrost/broker from PyPI)
        _exec_sync(
            sandbox,
            f"cd {workspace} && pip install openai anthropic dacite aiohttp trio httpx "
            f"'transformers>=4.50' datasets peft accelerate --quiet",
            timeout=300,
        )

    await trio.to_thread.run_sync(_install)
    logger.info("Dependencies installed")

    # Run training with PYTHONPATH set to include our code
    logger.info(f"Starting training: {config_rel}")

    env_vars = (
        f"PYTHONUNBUFFERED=1 "
        f"PYTHONPATH={workspace} "
        f"ROLLOUTS_RUN_NAME={run_name} "
        f"ROLLOUTS_OUTPUT_DIR=results/rl/{run_name} "
    )

    cmd = f"cd {workspace} && {env_vars} python {config_rel}"

    def _train() -> tuple[str, str, int]:
        return _exec_sync(sandbox, cmd, timeout=14400)  # 4 hour timeout

    stdout, stderr, exit_code = await trio.to_thread.run_sync(_train)

    if exit_code != 0:
        logger.error(f"Training failed with exit code {exit_code}")
        logger.error(f"stderr: {stderr}")
        return {"success": False, "exit_code": exit_code, "stderr": stderr}

    logger.info("Training completed successfully")
    return {"success": True, "exit_code": 0}


async def run_modal(config: ModalRunConfig) -> dict[str, Any]:
    """Run training on Modal.

    Full lifecycle:
    1. Create sandbox with deps
    2. Sync code
    3. Run training
    4. Terminate sandbox

    Returns training results.
    """
    import trio_asyncio

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = f"modal_{timestamp}"

    logger.info("=" * 60)
    logger.info(f"Modal Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {config.config_path}")
    logger.info(f"GPU: {config.gpu_count}x {config.gpu_type}")

    async with trio_asyncio.open_loop():
        # Create sandbox
        logger.info("Creating Modal sandbox...")
        sandbox, sandbox_id = await _create_sandbox(config)

        try:
            # Test GPU access
            logger.info("Verifying GPU access...")

            def _check_gpu() -> None:
                stdout, _, exit_code = _exec_sync(sandbox, "nvidia-smi", timeout=30)
                assert exit_code == 0, "nvidia-smi failed"

            await trio.to_thread.run_sync(_check_gpu)
            logger.info("GPU access verified")

            # Sync code
            logger.info("Syncing code to sandbox...")
            if config.use_local_sync:
                workspace = await _sync_code_to_sandbox(sandbox, REPO_ROOT)
            else:
                # Git clone
                workspace = "/workspace/rollouts"

                def _git_clone() -> None:
                    _exec_sync(
                        sandbox,
                        f"git clone --depth 1 --branch {config.git_branch} "
                        f"{config.git_repo} {workspace}",
                        timeout=300,
                    )

                await trio.to_thread.run_sync(_git_clone)

            logger.info(f"Code synced to {workspace}")

            # Run training
            logger.info("Starting training...")
            results = await _run_training_in_sandbox(
                sandbox, workspace, config.config_path, run_name
            )

            return results

        finally:
            # Terminate sandbox
            logger.info(f"Terminating sandbox: {sandbox_id}")

            def _terminate() -> None:
                sandbox.terminate()

            await trio.to_thread.run_sync(_terminate)
            logger.info("Sandbox terminated")


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_config"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run training on Modal")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., examples/rl/reverse_text/grpo_01_01.py)",
    )
    parser.add_argument(
        "--gpu",
        default="A100",
        help="GPU type (default: A100)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=4,
        help="Sandbox timeout in hours (default: 4)",
    )

    args = parser.parse_args()

    # Setup logging with JSONL file output for debugging
    # Creates results/modal_runs/{timestamp}/run.jsonl
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = REPO_ROOT / "results" / "modal_runs" / f"modal_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.jsonl"

    setup_logging(
        level="INFO",
        use_color=True,
        log_file=str(log_file),
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING", "modal": "WARNING"},
    )
    logger.info(f"Log file: {log_file}")

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    assert config_path.exists(), f"Config not found: {config_path}"

    # Build run config
    run_config = ModalRunConfig(
        config_path=str(config_path),
        gpu_type=args.gpu,
        gpu_count=args.gpu_count,
        timeout_hours=args.timeout_hours,
    )

    # Run
    results = trio.run(run_modal, run_config)

    if results.get("success"):
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"Training failed: {results}")
        sys.exit(1)


if __name__ == "__main__":
    main()
