"""Modal sandbox interface for running experiments with GPU access.

Provides a simple interface for Claude Code to:
1. Create sandboxes with appropriate images
2. Run commands and get results
3. Log everything as wide events for later analysis

Usage:
    from src.sandbox import Sandbox

    async with Sandbox.create(gpu="A10G") as sb:
        result = await sb.run("python -c 'import torch; print(torch.cuda.is_available())'")
        print(result.stdout)
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SANDBOX_LOG = RESULTS_DIR / "sandbox_events.jsonl"
SNAPSHOT_CACHE = PROJECT_ROOT / ".snapshot_cache.json"


def _load_snapshot_cache() -> dict[str, str]:
    """Load snapshot ID cache from disk."""
    if SNAPSHOT_CACHE.exists():
        return json.loads(SNAPSHOT_CACHE.read_text())
    return {}


def _save_snapshot_cache(cache: dict[str, str]) -> None:
    """Save snapshot ID cache to disk."""
    SNAPSHOT_CACHE.write_text(json.dumps(cache, indent=2))


@dataclass
class CommandResult:
    """Result from running a command in sandbox."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: int
    success: bool = field(init=False)

    def __post_init__(self):
        self.success = self.exit_code == 0

    def __str__(self) -> str:
        if self.success:
            return self.stdout
        return f"[exit {self.exit_code}]\n{self.stdout}\n{self.stderr}"


def _log_event(event: dict[str, Any]) -> None:
    """Append a wide event to the sandbox log."""
    SANDBOX_LOG.parent.mkdir(parents=True, exist_ok=True)
    event["logged_at"] = datetime.now().isoformat()
    with open(SANDBOX_LOG, "a") as f:
        f.write(json.dumps(event) + "\n")


def _create_sandbox_sync(gpu_type: str, image_tag: str) -> tuple[str, Any]:
    """Create Modal sandbox. Returns (sandbox_id, modal_module)."""
    import modal

    app = modal.App.lookup("dormant-llm-puzzle", create_if_missing=True)

    # Build image based on tag
    if image_tag == "transformers":
        # For running HuggingFace models
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .apt_install("git")
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
                "bitsandbytes",
            )
            .env({
                "HF_HOME": "/root/.cache/huggingface",
                "TRANSFORMERS_CACHE": "/root/.cache/huggingface",
            })
        )
    elif image_tag == "minimal":
        # Just torch
        image = (
            modal.Image.debian_slim(python_version="3.11")
            .pip_install(
                "torch",
                index_url="https://download.pytorch.org/whl/cu124",
                extra_index_url="https://pypi.org/simple",
            )
            .pip_install("numpy")
        )
    else:
        raise ValueError(f"Unknown image_tag: {image_tag}")

    ts = int(time.time())
    sandbox_name = f"dormant-{gpu_type.lower()}-{ts}"

    sandbox = modal.Sandbox.create(
        app=app,
        image=image,
        gpu=gpu_type,
        timeout=60 * 60,  # 1 hour
        name=sandbox_name,
    )

    return sandbox.object_id, modal


class Sandbox:
    """Modal sandbox wrapper with logging."""

    def __init__(self, sandbox_id: str, gpu_type: str, image_tag: str):
        self.sandbox_id = sandbox_id
        self.gpu_type = gpu_type
        self.image_tag = image_tag
        self._sandbox = None
        self._command_count = 0

    @classmethod
    async def create(
        cls,
        gpu: str = "A10G",
        image: str = "transformers",
        snapshot_key: str | None = None,
    ) -> Sandbox:
        """Create a new sandbox.

        Args:
            gpu: GPU type (T4, A10G, A100, H100, etc.)
            image: Image tag (transformers, minimal)
            snapshot_key: If provided, try to restore from cached snapshot first
        """
        import trio

        # Check if we have a cached snapshot for this key
        if snapshot_key:
            cache = _load_snapshot_cache()
            snapshot_id = cache.get(snapshot_key)
            if snapshot_id:
                logger.info(f"Restoring from snapshot: {snapshot_key} -> {snapshot_id}")
                try:
                    sb = await cls.from_snapshot(snapshot_id, gpu)
                    logger.info(f"Restored sandbox: {sb.sandbox_id}")
                    return sb
                except Exception as e:
                    logger.warning(f"Failed to restore snapshot {snapshot_id}: {e}")
                    # Fall through to create fresh

        logger.info(f"Creating sandbox: gpu={gpu}, image={image}")

        sandbox_id, _ = await trio.to_thread.run_sync(
            lambda: _create_sandbox_sync(gpu, image)
        )

        sb = cls(sandbox_id, gpu, image)

        # Log creation event
        _log_event({
            "event": "sandbox_created",
            "sandbox_id": sandbox_id,
            "gpu_type": gpu,
            "image_tag": image,
        })

        logger.info(f"Sandbox created: {sandbox_id}")
        return sb

    @classmethod
    async def from_snapshot(cls, image_id: str, gpu: str = "A10G") -> "Sandbox":
        """Create sandbox from a filesystem snapshot (stored as Image)."""
        import trio
        import modal

        def _create_from_snapshot():
            app = modal.App.lookup("dormant-llm-puzzle", create_if_missing=True)
            # snapshot_filesystem() returns an Image, so we restore from Image ID
            image = modal.Image.from_id(image_id)
            sandbox = modal.Sandbox.create(
                app=app,
                image=image,
                gpu=gpu,
                timeout=60 * 60,
            )
            return sandbox.object_id

        sandbox_id = await trio.to_thread.run_sync(_create_from_snapshot)
        sb = cls(sandbox_id, gpu, "from_snapshot")

        _log_event({
            "event": "sandbox_restored",
            "sandbox_id": sandbox_id,
            "image_id": image_id,
            "gpu_type": gpu,
        })

        return sb

    @classmethod
    def from_id(cls, sandbox_id: str) -> Sandbox:
        """Reconnect to existing sandbox."""
        # We don't know gpu/image, use unknown
        return cls(sandbox_id, "unknown", "unknown")

    def _get_sandbox(self):
        """Get modal.Sandbox object."""
        if self._sandbox is None:
            import modal
            self._sandbox = modal.Sandbox.from_id(self.sandbox_id)
        return self._sandbox

    async def run(
        self,
        command: str,
        timeout: int = 300,
        tag: str | None = None,
    ) -> CommandResult:
        """Run a command in the sandbox.

        Args:
            command: Bash command to run
            timeout: Timeout in seconds
            tag: Optional tag for logging (e.g., "setup", "inference")
        """
        import trio

        self._command_count += 1
        cmd_id = f"{self.sandbox_id[:8]}-{self._command_count:03d}"

        start = time.time()

        def _run_sync() -> tuple[str, str, int]:
            sandbox = self._get_sandbox()
            proc = sandbox.exec("bash", "-c", command, timeout=timeout)
            proc.wait()
            return proc.stdout.read(), proc.stderr.read(), proc.returncode

        stdout, stderr, exit_code = await trio.to_thread.run_sync(_run_sync)
        duration_ms = int((time.time() - start) * 1000)

        result = CommandResult(
            command=command,
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            duration_ms=duration_ms,
        )

        # Log as wide event
        _log_event({
            "event": "command_executed",
            "cmd_id": cmd_id,
            "sandbox_id": self.sandbox_id,
            "gpu_type": self.gpu_type,
            "command": command[:500],  # Truncate long commands
            "tag": tag,
            "exit_code": exit_code,
            "duration_ms": duration_ms,
            "success": result.success,
            "stdout_len": len(stdout),
            "stderr_len": len(stderr),
            # Include output preview for quick debugging
            "stdout_preview": stdout[:500] if stdout else None,
            "stderr_preview": stderr[:500] if stderr else None,
        })

        return result

    async def write_file(self, path: str, content: str) -> CommandResult:
        """Write a file to the sandbox."""
        import base64

        content_b64 = base64.b64encode(content.encode()).decode()

        # Ensure parent directory exists
        parent = str(Path(path).parent)
        cmd = f"mkdir -p {parent} && echo '{content_b64}' | base64 -d > {path}"

        return await self.run(cmd, tag="write_file")

    async def read_file(self, path: str) -> str:
        """Read a file from the sandbox."""
        result = await self.run(f"cat {path}", tag="read_file")
        if not result.success:
            raise FileNotFoundError(f"Could not read {path}: {result.stderr}")
        return result.stdout

    async def snapshot(self, key: str) -> str:
        """Create filesystem snapshot and cache it under the given key.

        snapshot_filesystem() returns an Image, so we store the Image ID.
        Returns the image ID.
        """
        import trio

        def _snapshot_sync() -> str:
            sandbox = self._get_sandbox()
            image = sandbox.snapshot_filesystem()
            return image.object_id

        image_id = await trio.to_thread.run_sync(_snapshot_sync)

        # Cache the image ID
        cache = _load_snapshot_cache()
        cache[key] = image_id
        _save_snapshot_cache(cache)

        _log_event({
            "event": "snapshot_created",
            "sandbox_id": self.sandbox_id,
            "image_id": image_id,
            "snapshot_key": key,
        })

        logger.info(f"Snapshot created: {key} -> {image_id}")
        return image_id

    async def terminate(self) -> None:
        """Terminate the sandbox."""
        import trio

        def _terminate_sync():
            sandbox = self._get_sandbox()
            sandbox.terminate()

        await trio.to_thread.run_sync(_terminate_sync)

        _log_event({
            "event": "sandbox_terminated",
            "sandbox_id": self.sandbox_id,
            "total_commands": self._command_count,
        })

        logger.info(f"Sandbox terminated: {self.sandbox_id}")

    async def __aenter__(self) -> Sandbox:
        return self

    async def __aexit__(self, *args) -> None:
        await self.terminate()


# ── Sync wrapper for simple scripts ──────────────────────────────────────────


def run_sync(coro):
    """Run async code from sync context."""
    import trio
    return trio.run(coro)


# ── CLI for quick testing ────────────────────────────────────────────────────


async def _cli_main():
    import argparse

    parser = argparse.ArgumentParser(description="Modal sandbox CLI")
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # create
    create_p = subparsers.add_parser("create", help="Create sandbox")
    create_p.add_argument("--gpu", default="A10G")
    create_p.add_argument("--image", default="transformers")

    # run
    run_p = subparsers.add_parser("run", help="Run command in sandbox")
    run_p.add_argument("sandbox_id")
    run_p.add_argument("command")
    run_p.add_argument("--timeout", type=int, default=300)

    # terminate
    term_p = subparsers.add_parser("terminate", help="Terminate sandbox")
    term_p.add_argument("sandbox_id")

    args = parser.parse_args()

    if args.cmd == "create":
        sb = await Sandbox.create(gpu=args.gpu, image=args.image)
        print(sb.sandbox_id)

    elif args.cmd == "run":
        sb = Sandbox.from_id(args.sandbox_id)
        result = await sb.run(args.command, timeout=args.timeout)
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, file=sys.stderr, end="")
        sys.exit(result.exit_code)

    elif args.cmd == "terminate":
        sb = Sandbox.from_id(args.sandbox_id)
        await sb.terminate()
        print("Terminated")


if __name__ == "__main__":
    import trio
    trio.run(_cli_main)
