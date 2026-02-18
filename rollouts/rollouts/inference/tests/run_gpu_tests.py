#!/usr/bin/env python3
"""Run inference engine GPU tests via Modal sandbox.

Usage:
    python3 rollouts/inference/tests/run_gpu_tests.py           # Reuse existing sandbox or create new
    python3 rollouts/inference/tests/run_gpu_tests.py --fresh   # Force new sandbox
    python3 rollouts/inference/tests/run_gpu_tests.py --terminate  # Terminate existing sandbox

This syncs local code to a Modal sandbox with GPU and runs:
1. test_equivalence.py - numerical equivalence vs HuggingFace
2. test_mini_sglang_parity.py - radix cache, KV cache correctness

Keep-alive mode: Sandbox persists between runs for fast iteration (~5s vs ~70s).
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


# Find repo root by looking for .git
def find_repo_root() -> Path:
    current = Path(__file__).resolve()
    while current != current.parent:
        if (current / ".git").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not find repo root (no .git directory)")


REPO_ROOT = find_repo_root()
ROLLOUTS_ROOT = REPO_ROOT / "rollouts"
SANDBOX_STATE_FILE = Path.home() / ".rollouts" / "gpu_test_sandbox.json"


def get_saved_sandbox_id() -> str | None:
    """Get saved sandbox ID if it exists."""
    if not SANDBOX_STATE_FILE.exists():
        return None
    try:
        data = json.loads(SANDBOX_STATE_FILE.read_text())
        return data.get("sandbox_id")
    except (json.JSONDecodeError, KeyError):
        return None


def save_sandbox_id(sandbox_id: str) -> None:
    """Save sandbox ID for reuse."""
    SANDBOX_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SANDBOX_STATE_FILE.write_text(json.dumps({"sandbox_id": sandbox_id}))
    logger.info(f"Saved sandbox ID to {SANDBOX_STATE_FILE}")


def clear_sandbox_id() -> None:
    """Clear saved sandbox ID."""
    if SANDBOX_STATE_FILE.exists():
        SANDBOX_STATE_FILE.unlink()


def check_sandbox_alive(sandbox_id: str) -> bool:
    """Check if a sandbox is still alive."""
    try:
        import modal

        sandbox = modal.Sandbox.from_id(sandbox_id)
        # Try a simple command to verify it's responsive
        proc = sandbox.exec("echo", "alive", timeout=10)
        proc.wait()
        return proc.returncode == 0
    except Exception as e:
        logger.debug(f"Sandbox {sandbox_id} not alive: {e}")
        return False


def create_sandbox() -> str:
    """Create a new Modal sandbox and return its ID."""
    sandbox_script = """
import asyncio
import modal

async def main():
    app = modal.App.lookup("inference-engine-tests", create_if_missing=True)

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04",
            add_python="3.11",
        )
        .apt_install("git", "build-essential")
        .pip_install(
            "torch",
            index_url="https://download.pytorch.org/whl/cu124",
            extra_index_url="https://pypi.org/simple",
        )
        .pip_install(
            "transformers>=4.50",
            "accelerate",
            "safetensors",
            "numpy",
            "uv",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
        })
        .run_commands("mkdir -p /workspace")
    )

    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        gpu="A100",
        timeout=1800,  # 30 min
    )

    print(sandbox.object_id)

asyncio.run(main())
"""

    logger.info("Creating new Modal sandbox...")
    result = subprocess.run(
        [sys.executable, "-c", sandbox_script],
        capture_output=True,
        text=True,
        env={**os.environ},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create sandbox: {result.stderr}")

    sandbox_id = result.stdout.strip().split("\n")[-1]
    logger.info(f"Created sandbox: {sandbox_id}")
    return sandbox_id


def setup_sandbox(sandbox) -> bool:
    """Initial setup: clone repo and install deps. Only needed for new sandboxes."""
    import tempfile

    logger.info("Setting up sandbox (first-time setup)...")

    # Sync code via git bundle
    logger.info("Syncing code via git bundle...")
    with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
        bundle_path = f.name

    try:
        subprocess.run(
            ["git", "bundle", "create", bundle_path, "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
        )

        with open(bundle_path, "rb") as f:
            bundle_data = f.read()

        logger.info(f"Bundle size: {len(bundle_data) / 1024 / 1024:.1f} MB")

        remote_file = sandbox.open("/tmp/repo.bundle", "wb")
        remote_file.write(bundle_data)
        remote_file.close()

        proc = sandbox.exec(
            "bash",
            "-c",
            "rm -rf /workspace/research && cd /workspace && git clone /tmp/repo.bundle research && cd research && git checkout HEAD",
            timeout=120,
        )
        proc.wait()
        if proc.returncode != 0:
            logger.error(f"Git clone failed: {proc.stderr.read()}")
            return False

        logger.info("Code synced to /workspace/research")

    finally:
        os.unlink(bundle_path)

    # Install dependencies
    logger.info("Installing dependencies...")
    proc = sandbox.exec(
        "bash",
        "-c",
        "cd /workspace/research/rollouts && "
        "uv pip install --system -e '.[training]' && "
        "uv pip install --system git+https://github.com/sgl-project/mini-sglang.git 2>&1",
        timeout=300,
    )
    for line in proc.stdout:
        if line.strip():
            # Only log key lines to reduce noise
            if any(x in line for x in ["Successfully", "error", "Error", "WARNING"]):
                logger.info(f"[sandbox] {line.rstrip()}")
    proc.wait()

    if proc.returncode != 0:
        logger.error("pip install failed")
        return False

    logger.info("Setup complete")
    return True


def sync_modified_files(sandbox) -> int:
    """Sync only modified/untracked files. Returns count of files synced."""
    import tarfile
    import tempfile
    import time

    t_start = time.time()
    files_to_sync = []

    # Find files to sync
    t_find_start = time.time()
    inference_dir = ROLLOUTS_ROOT / "rollouts" / "inference"
    for subdir in ["", "attention", "models", "layers", "tests"]:
        src_dir = inference_dir / subdir if subdir else inference_dir
        if not src_dir.exists():
            continue
        for py_file in src_dir.glob("*.py"):
            rel_path = py_file.relative_to(ROLLOUTS_ROOT)

            # Check if untracked or modified
            untracked = (
                subprocess.run(
                    ["git", "ls-files", "--error-unmatch", str(rel_path)],
                    cwd=str(ROLLOUTS_ROOT),
                    capture_output=True,
                ).returncode
                != 0
            )

            modified = (
                subprocess.run(
                    ["git", "diff", "--quiet", str(rel_path)],
                    cwd=str(ROLLOUTS_ROOT),
                    capture_output=True,
                ).returncode
                != 0
            )

            if untracked or modified:
                files_to_sync.append((py_file, rel_path))

    t_find_end = time.time()
    logger.info(
        f"[timing] Found {len(files_to_sync)} files to sync in {t_find_end - t_find_start:.2f}s"
    )

    if not files_to_sync:
        return 0

    # Create tar archive with all files
    t_tar_start = time.time()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
        tar_path = f.name

    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for py_file, rel_path in files_to_sync:
                tar.add(py_file, arcname=str(rel_path))

        tar_size = os.path.getsize(tar_path)
        t_tar_end = time.time()
        logger.info(
            f"[timing] Created tar ({tar_size / 1024:.1f}KB) in {t_tar_end - t_tar_start:.2f}s"
        )

        # Upload tar as single file
        t_upload_start = time.time()
        with open(tar_path, "rb") as f:
            tar_data = f.read()
        tar_b64 = base64.b64encode(tar_data).decode()

        # Write and extract in one command
        proc = sandbox.exec(
            "bash",
            "-c",
            f"echo '{tar_b64}' | base64 -d > /tmp/sync.tar.gz && "
            f"cd /workspace/research/rollouts && tar xzf /tmp/sync.tar.gz && "
            f"rm /tmp/sync.tar.gz",
            timeout=60,
        )
        proc.wait()

        t_upload_end = time.time()
        logger.info(f"[timing] Uploaded and extracted in {t_upload_end - t_upload_start:.2f}s")
        logger.info(f"[timing] Total sync: {t_upload_end - t_start:.2f}s")

    finally:
        os.unlink(tar_path)

    return len(files_to_sync)


def run_tests(sandbox) -> bool:
    """Run the test suite in the sandbox."""
    import time

    t_start = time.time()

    test_script = """
import sys
import subprocess

print("=" * 60)
print("InferenceEngineV2 GPU Tests")
print("=" * 60)

import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CUDA version: {torch.version.cuda}")
print()

cwd = "/workspace/research/rollouts"

# Test functional model first
print("Testing functional Llama model...")
print("-" * 40)
r0 = subprocess.run(
    [sys.executable, "-m", "rollouts.inference.models.llama_functional"],
    cwd=cwd,
)
print()

# Run test_mini_sglang_parity
print("Running test_mini_sglang_parity.py...")
print("-" * 40)
r1 = subprocess.run(
    [sys.executable, "-m", "rollouts.inference.tests.test_mini_sglang_parity"],
    cwd=cwd,
)
print()

# Run test_equivalence
print("Running test_equivalence.py...")
print("-" * 40)
r2 = subprocess.run(
    [sys.executable, "-m", "rollouts.inference.tests.test_equivalence"],
    cwd=cwd,
)

print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"llama_functional: {'PASS' if r0.returncode == 0 else 'FAIL'}")
print(f"test_mini_sglang_parity: {'PASS' if r1.returncode == 0 else 'FAIL'}")
print(f"test_equivalence: {'PASS' if r2.returncode == 0 else 'FAIL'}")

all_passed = r0.returncode == 0 and r1.returncode == 0 and r2.returncode == 0
sys.exit(0 if all_passed else 1)
"""

    # Write test script
    test_script_b64 = base64.b64encode(test_script.encode()).decode()
    proc = sandbox.exec(
        "bash",
        "-c",
        f"echo '{test_script_b64}' | base64 -d > /workspace/research/rollouts/run_tests.py",
        timeout=30,
    )
    proc.wait()

    t_script_written = time.time()
    logger.info(f"[timing] Test script written in {t_script_written - t_start:.2f}s")

    # Run tests
    logger.info("Running tests...")
    logger.info("=" * 60)
    proc = sandbox.exec(
        "bash",
        "-c",
        "cd /workspace/research/rollouts && PYTHONUNBUFFERED=1 python run_tests.py",
        timeout=600,
    )

    for line in proc.stdout:
        logger.info(f"[sandbox] {line.rstrip()}")
    for line in proc.stderr:
        logger.warning(f"[sandbox stderr] {line.rstrip()}")

    proc.wait()
    success = proc.returncode == 0

    t_tests_done = time.time()
    logger.info("=" * 60)
    logger.info(f"[timing] Tests execution: {t_tests_done - t_script_written:.2f}s")
    logger.info(f"Tests {'PASSED' if success else 'FAILED'}")
    return success


def run_tests_in_modal(fresh: bool = False) -> bool:
    """Run GPU tests in Modal sandbox with keep-alive support."""
    import modal

    sandbox_id = None
    sandbox = None
    needs_setup = False

    # Try to reuse existing sandbox
    if not fresh:
        sandbox_id = get_saved_sandbox_id()
        if sandbox_id:
            logger.info(f"Found saved sandbox: {sandbox_id}")
            if check_sandbox_alive(sandbox_id):
                logger.info("Sandbox is alive, reusing...")
                sandbox = modal.Sandbox.from_id(sandbox_id)
            else:
                logger.info("Sandbox is dead, creating new one...")
                clear_sandbox_id()
                sandbox_id = None

    # Create new sandbox if needed
    if sandbox is None:
        sandbox_id = create_sandbox()
        sandbox = modal.Sandbox.from_id(sandbox_id)
        save_sandbox_id(sandbox_id)
        needs_setup = True

    # Setup sandbox if new
    if needs_setup:
        if not setup_sandbox(sandbox):
            logger.error("Sandbox setup failed")
            return False

    # Sync modified files
    count = sync_modified_files(sandbox)
    logger.info(f"Synced {count} modified files")

    # Run tests
    return run_tests(sandbox)


def terminate_sandbox() -> bool:
    """Terminate the saved sandbox."""
    sandbox_id = get_saved_sandbox_id()
    if not sandbox_id:
        logger.info("No saved sandbox to terminate")
        return True

    try:
        import modal

        sandbox = modal.Sandbox.from_id(sandbox_id)
        sandbox.terminate()
        logger.info(f"Terminated sandbox: {sandbox_id}")
    except Exception as e:
        logger.warning(f"Failed to terminate sandbox: {e}")

    clear_sandbox_id()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Run GPU tests in Modal sandbox")
    parser.add_argument("--fresh", action="store_true", help="Force create new sandbox")
    parser.add_argument(
        "--terminate", action="store_true", help="Terminate existing sandbox and exit"
    )
    args = parser.parse_args()

    from datetime import datetime, timezone

    from rollouts._logging import setup_logging

    # Setup logging with JSONL file output
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = ROLLOUTS_ROOT / "results" / "gpu_tests" / f"run_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.jsonl"

    setup_logging(
        level="INFO",
        use_color=True,
        log_file=str(log_file),
        logger_levels={"httpx": "WARNING", "modal": "WARNING"},
    )
    logger.info(f"Log file: {log_file}")

    try:
        import modal
    except ImportError:
        logger.error("Modal not installed. Install with: pip install modal")
        logger.error("Then authenticate with: modal setup")
        return 1

    if args.terminate:
        terminate_sandbox()
        return 0

    success = run_tests_in_modal(fresh=args.fresh)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
