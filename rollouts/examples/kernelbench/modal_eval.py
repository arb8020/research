"""Modal runner for KernelBench evaluation.

Runs the eval inside a Modal sandbox with GPU access.
Uses the same pattern as rollouts/modal_runner.py.

Usage:
    python modal_eval.py --config configs/api_smoke.py --limit 2
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trio

# Get repo root
REPO_ROOT = Path(__file__).parent.parent.parent

# Get API key from rollouts credentials
sys.path.insert(0, str(REPO_ROOT))
from rollouts.credentials import get_api_key

logger = logging.getLogger(__name__)


@dataclass
class ModalEvalConfig:
    """Config for Modal eval run."""

    config_path: str
    gpu_type: str = "A10G"
    limit: int | None = None
    skip: int | None = None
    max_turns: int | None = None
    model: str | None = None
    levels: list[int] | None = None
    backend: str | None = None
    keep_alive: bool = False
    sandbox_id: str | None = None  # Reuse existing sandbox


def _build_image(modal_module: Any) -> Any:
    """Build Modal image with dependencies."""
    image = (
        modal_module.Image.debian_slim(python_version="3.12")
        .apt_install("git", "build-essential", "curl", "ninja-build")
        .pip_install(
            "torch>=2.4",
            "triton",
            "anthropic",
            "openai",
            "httpx",
            "trio",
            "datasets",
            "huggingface_hub",
            "dacite",
            "markdownify",  # For coding environment
            "tenacity",
            "tomli",
            index_url="https://download.pytorch.org/whl/cu124",
            extra_index_url="https://pypi.org/simple",
        )
        .env({
            "HF_HOME": "/root/.cache/huggingface",
        })
    )
    return image


def _exec_sync(sandbox: Any, command: str, timeout: int = 300) -> tuple[str, str, int]:
    """Execute command in sandbox. Returns (stdout, stderr, exit_code)."""
    proc = sandbox.exec("bash", "-c", command, timeout=timeout)

    stdout_lines = []
    for line in proc.stdout:
        stdout_lines.append(line)
        print(f"[sandbox] {line.rstrip()}")

    stderr_lines = []
    for line in proc.stderr:
        stderr_lines.append(line)
        print(f"[sandbox stderr] {line.rstrip()}", file=sys.stderr)

    proc.wait()
    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


async def _sync_code_to_sandbox(sandbox: Any, local_root: Path) -> str:
    """Sync local code to sandbox via git bundle."""
    workspace = "/workspace/research/rollouts"

    def _sync() -> None:
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        try:
            # Get current commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(local_root),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            print(f"Bundling commit {commit[:8]}...")

            # Create bundle
            subprocess.run(
                ["git", "bundle", "create", bundle_path, "HEAD"],
                cwd=str(local_root),
                check=True,
                capture_output=True,
            )

            bundle_size = os.path.getsize(bundle_path)
            print(f"Bundle size: {bundle_size / 1024 / 1024:.1f} MB")

            # Read bundle
            with open(bundle_path, "rb") as f:
                bundle_data = f.read()

            # Create workspace
            _exec_sync(sandbox, "mkdir -p /workspace", timeout=30)

            # Upload via sandbox.open()
            print("Uploading bundle...")
            remote_file = sandbox.open("/tmp/repo.bundle", "wb")
            remote_file.write(bundle_data)
            remote_file.close()

            # Clone from bundle
            print("Extracting bundle...")
            _exec_sync(
                sandbox,
                "cd /workspace && git clone /tmp/repo.bundle research && "
                "cd research && git checkout HEAD",
                timeout=120,
            )

            print(f"Code synced to {workspace}")

        finally:
            os.unlink(bundle_path)

    await trio.to_thread.run_sync(_sync)
    return workspace


async def run_modal_eval(config: ModalEvalConfig) -> dict:
    """Run evaluation in Modal sandbox."""
    import modal

    # Get API key
    api_key = get_api_key("anthropic") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("No Anthropic API key found")

    app = modal.App.lookup("kernelbench-eval", create_if_missing=True)

    # Reuse existing sandbox or create new one
    if config.sandbox_id:
        print(f"Reusing sandbox: {config.sandbox_id}")
        sandbox = await trio.to_thread.run_sync(
            lambda: modal.Sandbox.from_id(config.sandbox_id)
        )
    else:
        print(f"Creating Modal sandbox with {config.gpu_type} GPU...")
        image = _build_image(modal)
        sandbox = await trio.to_thread.run_sync(
            lambda: modal.Sandbox.create(
                app=app,
                image=image,
                gpu=config.gpu_type,
                timeout=3600,
            )
        )
        print(f"Sandbox created: {sandbox.object_id}")
        if config.keep_alive:
            print(f"  (use --sandbox-id {sandbox.object_id} to reuse)")

    try:
        # Sync code
        workspace = await _sync_code_to_sandbox(sandbox, REPO_ROOT.parent)

        # Set environment
        _exec_sync(sandbox, f"export ANTHROPIC_API_KEY='{api_key}'", timeout=10)

        # Build eval command
        # Run directly instead of as module
        cmd_parts = [
            f"cd {workspace} &&",
            f"PYTHONPATH={workspace}:{workspace}/rollouts",
            f"ANTHROPIC_API_KEY='{api_key}'",
            "python examples/kernelbench/run_eval.py",
            config.config_path,
        ]

        if config.limit:
            cmd_parts.extend(["--limit", str(config.limit)])
        if config.skip:
            cmd_parts.extend(["--skip", str(config.skip)])
        if config.max_turns:
            cmd_parts.extend(["--max-turns", str(config.max_turns)])
        if config.model:
            cmd_parts.extend(["--model", config.model])
        if config.levels:
            cmd_parts.extend(["--levels"] + [str(l) for l in config.levels])
        if config.backend:
            cmd_parts.extend(["--backend", config.backend])

        cmd = " ".join(cmd_parts)
        print(f"Running: {cmd}")

        # Run eval
        stdout, stderr, exit_code = _exec_sync(sandbox, cmd, timeout=3600)

        return {
            "success": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
        }

    finally:
        if config.keep_alive:
            print(f"Keeping sandbox alive: {sandbox.object_id}")
            print(f"  Reuse with: --sandbox-id {sandbox.object_id}")
        else:
            print("Terminating sandbox...")
            await trio.to_thread.run_sync(sandbox.terminate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KernelBench eval on Modal")
    parser.add_argument("--config", required=True, help="Config file path")
    parser.add_argument("--gpu-type", default="A10G", help="GPU type")
    parser.add_argument("--limit", type=int, help="Max samples")
    parser.add_argument("--skip", type=int, help="Skip first N problems")
    parser.add_argument("--max-turns", type=int, help="Max turns")
    parser.add_argument("--model", help="Model override")
    parser.add_argument("--levels", type=int, nargs="+", help="Levels")
    parser.add_argument("--backend", help="Backend (CUDA/HIP)")
    parser.add_argument("--keep-alive", action="store_true", help="Keep sandbox alive after eval")
    parser.add_argument("--sandbox-id", help="Reuse existing sandbox by ID")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    config = ModalEvalConfig(
        config_path=args.config,
        gpu_type=args.gpu_type,
        limit=args.limit,
        skip=args.skip,
        max_turns=args.max_turns,
        model=args.model,
        levels=args.levels,
        backend=args.backend,
        keep_alive=args.keep_alive,
        sandbox_id=args.sandbox_id,
    )

    result = trio.run(run_modal_eval, config)
    print(f"\nResult: {result}")

    if not result.get("success"):
        sys.exit(1)


if __name__ == "__main__":
    main()
