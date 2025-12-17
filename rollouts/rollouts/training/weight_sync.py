"""Inference engine lifecycle management.

Abstracts SGLang/vLLM server lifecycle: launch, health check, log tailing, weight sync.

Key design principles (from SLIME + code style guides):
- Casey Muratori: Fine-grained immediate mode + coarse-grained convenience
- Tiger Style: Assert preconditions, no hidden state
- Sean Goedecke: Stateless coordination, boring patterns

Architecture:
- Protocol: InferenceEngine with full lifecycle (launch, health, logs, weight sync)
- Adapters: SGLangEngine, VLLMEngine implement protocol
- Fine-grained functions for each operation
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import httpx
import trio

# ══════════════════════════════════════════════════════════════
# Fine-grained immediate mode (Casey Muratori style)
# ══════════════════════════════════════════════════════════════


async def update_sglang_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Update SGLang server weights from checkpoint on disk.

    Calls SGLang's /update_weights_from_disk HTTP endpoint.

    Args:
        base_url: SGLang server URL (e.g. "http://localhost:30000")
        checkpoint_path: Path to checkpoint (local path or HF model ID)

    Returns:
        Response dict with keys:
            - success: bool
            - message: str

    Raises:
        httpx.HTTPError: If HTTP request fails
        AssertionError: If preconditions violated
        trio.TooSlowError: If request takes >5 minutes (use trio.fail_after for custom timeout)

    Example:
        >>> with trio.fail_after(300):  # 5 minute timeout
        ...     response = await update_sglang_weights_from_disk(
        ...         "http://localhost:30000",
        ...         "/checkpoints/step_1000",
        ...     )
        >>> assert response["success"]
    """
    # Tiger Style: assert preconditions
    assert base_url, "base_url cannot be empty"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Simple HTTP POST - no abstraction, no state
    # Note: No timeout parameter - caller should use trio.fail_after
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/update_weights_from_disk",
            json={"model_path": checkpoint_path},
        )
        response.raise_for_status()
        result = response.json()

    # Tiger Style: assert postconditions
    assert "success" in result, "Response must have 'success' field"

    return result


async def update_vllm_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Update vLLM server weights from checkpoint on disk.

    Calls vLLM's collective_rpc endpoint with reload_weights method.

    Args:
        base_url: vLLM server URL (e.g. "http://localhost:30001")
        checkpoint_path: Path to checkpoint (local path or HF model ID)

    Returns:
        Response dict from vLLM RPC

    Raises:
        httpx.HTTPError: If HTTP request fails
        AssertionError: If preconditions violated
        trio.TooSlowError: If request takes >5 minutes (use trio.fail_after for custom timeout)

    Example:
        >>> with trio.fail_after(300):  # 5 minute timeout
        ...     response = await update_vllm_weights_from_disk(
        ...         "http://localhost:30001",
        ...         "/checkpoints/step_1000",
        ...     )
    """
    # Tiger Style: assert preconditions
    assert base_url, "base_url cannot be empty"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Call vLLM's reload_weights RPC
    # Note: No timeout parameter - caller should use trio.fail_after
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/collective_rpc",
            json={
                "method": "reload_weights",
                "params": {"model_path": checkpoint_path},
            },
        )
        response.raise_for_status()
        return response.json()


# ══════════════════════════════════════════════════════════════
# Minimal Protocol (Tiger Style: just type hints, not inheritance)
# ══════════════════════════════════════════════════════════════


class InferenceEngine(Protocol):
    """Protocol for inference engines with full lifecycle management.

    Tiger Style: This is JUST a type annotation (Protocol), not a base class.
    No inheritance! Just duck typing.

    Lifecycle:
    1. launch() -> subprocess.Popen  # Start the server
    2. start_log_tailer() -> Thread  # Tail logs as JSONL to stdout
    3. wait_until_ready() -> None    # Block until health check passes
    4. update_weights_from_checkpoint() -> dict  # Sync weights
    5. shutdown() -> None            # Terminate the server
    """

    @property
    def name(self) -> str:
        """Engine name for logging (e.g., 'sglang', 'vllm')."""
        ...

    @property
    def log_path(self) -> Path:
        """Path to server log file."""
        ...

    @property
    def health_url(self) -> str:
        """URL for health check endpoint."""
        ...

    @property
    def api_base(self) -> str:
        """Base URL for OpenAI-compatible API (e.g., 'http://localhost:30000/v1')."""
        ...

    def build_launch_cmd(self) -> str:
        """Build the shell command to launch the server."""
        ...

    def launch(self) -> subprocess.Popen:
        """Launch the inference server as a subprocess.

        Returns:
            subprocess.Popen handle for the server process
        """
        ...

    def start_log_tailer(self) -> threading.Thread:
        """Start a daemon thread that tails logs and emits JSONL to stdout.

        Returns:
            The started daemon thread
        """
        ...

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        """Wait until the server is ready (health check passes).

        Args:
            max_wait: Max seconds to wait before raising RuntimeError

        Raises:
            RuntimeError: If server doesn't become ready within max_wait
        """
        ...

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update model weights from checkpoint on disk.

        Args:
            checkpoint_path: Path to checkpoint directory or HF model ID

        Returns:
            Response dict from inference engine
        """
        ...

    def shutdown(self, proc: subprocess.Popen) -> None:
        """Shutdown the inference server.

        Args:
            proc: The Popen handle returned by launch()
        """
        ...


# ══════════════════════════════════════════════════════════════
# Adapters (Casey Muratori: redundancy - multiple ways to do same thing)
# ══════════════════════════════════════════════════════════════


@dataclass
class SGLangEngine:
    """SGLang inference engine with full lifecycle management.

    Implements InferenceEngine protocol for SGLang servers.

    Example:
        >>> engine = SGLangEngine(
        ...     model_name="Qwen/Qwen3-0.6B",
        ...     port=30000,
        ...     gpu_ids=(0,),
        ...     output_dir=Path("results/rl/run_001"),
        ... )
        >>> proc = engine.launch()
        >>> engine.start_log_tailer()  # Emits JSONL to stdout for TUI
        >>> await engine.wait_until_ready()
        >>> # ... use engine ...
        >>> await engine.update_weights_from_checkpoint("/ckpt/step_100")
        >>> engine.shutdown(proc)
    """

    model_name: str
    port: int
    gpu_ids: tuple[int, ...]
    output_dir: Path
    dtype: str = "bfloat16"
    mem_fraction: float = 0.7
    timeout: float = 300.0
    _log_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "sglang.log"

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def log_path(self) -> Path:
        return self._log_file

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/health"

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def base_url(self) -> str:
        """Base URL without /v1 suffix (for weight sync API)."""
        return f"http://localhost:{self.port}"

    def build_launch_cmd(self) -> str:
        """Build SGLang launch command."""
        gpu_str = ",".join(str(g) for g in self.gpu_ids)
        return (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"python -m sglang.launch_server "
            f"--model-path {self.model_name} "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--mem-fraction-static {self.mem_fraction} "
            f">> {self._log_file} 2>&1"
        )

    def launch(self) -> subprocess.Popen:
        """Launch SGLang server as subprocess."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_launch_cmd()
        return subprocess.Popen(cmd, shell=True, start_new_session=True)

    def start_log_tailer(self) -> threading.Thread:
        """Start daemon thread that tails SGLang logs as JSONL."""

        def tail_log() -> None:
            try:
                # Wait for log file to exist
                for _ in range(30):
                    if self._log_file.exists():
                        break
                    time.sleep(0.1)

                with open(self._log_file) as f:
                    while True:
                        line = f.readline()
                        if line:
                            line = line.strip()
                            if line:
                                print(
                                    json.dumps({"logger": "sglang", "message": line}),
                                    flush=True,
                                )
                        else:
                            time.sleep(0.1)
            except Exception:
                pass  # File closed or thread killed

        thread = threading.Thread(target=tail_log, daemon=True)
        thread.start()
        return thread

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        """Wait until SGLang health check passes."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _attempt in range(int(max_wait)):
                try:
                    resp = await client.get(self.health_url)
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await trio.sleep(1.0)

        msg = f"SGLang failed to start after {max_wait}s. Check {self._log_file}"
        raise RuntimeError(msg)

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update SGLang server weights from checkpoint."""
        assert checkpoint_path, "checkpoint_path cannot be empty"

        with trio.fail_after(self.timeout):
            return await update_sglang_weights_from_disk(
                self.base_url,
                checkpoint_path,
            )

    def shutdown(self, proc: subprocess.Popen) -> None:
        """Terminate the SGLang server process."""
        proc.terminate()


@dataclass
class VLLMEngine:
    """vLLM inference engine with full lifecycle management.

    Implements InferenceEngine protocol for vLLM servers.

    Example:
        >>> engine = VLLMEngine(
        ...     model_name="Qwen/Qwen3-0.6B",
        ...     port=30001,
        ...     gpu_ids=(0,),
        ...     output_dir=Path("results/rl/run_001"),
        ... )
        >>> proc = engine.launch()
        >>> engine.start_log_tailer()  # Emits JSONL to stdout for TUI
        >>> await engine.wait_until_ready()
        >>> # ... use engine ...
        >>> await engine.update_weights_from_checkpoint("/ckpt/step_100")
        >>> engine.shutdown(proc)
    """

    model_name: str
    port: int
    gpu_ids: tuple[int, ...]
    output_dir: Path
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.7
    timeout: float = 300.0
    _log_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "vllm.log"

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def log_path(self) -> Path:
        return self._log_file

    @property
    def health_url(self) -> str:
        return f"http://localhost:{self.port}/health"

    @property
    def api_base(self) -> str:
        return f"http://localhost:{self.port}/v1"

    @property
    def base_url(self) -> str:
        """Base URL without /v1 suffix (for weight sync API)."""
        return f"http://localhost:{self.port}"

    def build_launch_cmd(self) -> str:
        """Build vLLM launch command."""
        gpu_str = ",".join(str(g) for g in self.gpu_ids)
        return (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name} "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f">> {self._log_file} 2>&1"
        )

    def launch(self) -> subprocess.Popen:
        """Launch vLLM server as subprocess."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_launch_cmd()
        return subprocess.Popen(cmd, shell=True, start_new_session=True)

    def start_log_tailer(self) -> threading.Thread:
        """Start daemon thread that tails vLLM logs as JSONL."""

        def tail_log() -> None:
            try:
                # Wait for log file to exist
                for _ in range(30):
                    if self._log_file.exists():
                        break
                    time.sleep(0.1)

                with open(self._log_file) as f:
                    while True:
                        line = f.readline()
                        if line:
                            line = line.strip()
                            if line:
                                print(
                                    json.dumps({"logger": "vllm", "message": line}),
                                    flush=True,
                                )
                        else:
                            time.sleep(0.1)
            except Exception:
                pass  # File closed or thread killed

        thread = threading.Thread(target=tail_log, daemon=True)
        thread.start()
        return thread

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        """Wait until vLLM health check passes."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _attempt in range(int(max_wait)):
                try:
                    resp = await client.get(self.health_url)
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await trio.sleep(1.0)

        msg = f"vLLM failed to start after {max_wait}s. Check {self._log_file}"
        raise RuntimeError(msg)

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update vLLM server weights from checkpoint."""
        assert checkpoint_path, "checkpoint_path cannot be empty"

        with trio.fail_after(self.timeout):
            return await update_vllm_weights_from_disk(
                self.base_url,
                checkpoint_path,
            )

    def shutdown(self, proc: subprocess.Popen) -> None:
        """Terminate the vLLM server process."""
        proc.terminate()


# ══════════════════════════════════════════════════════════════
# Stateless orchestration (Sean Goedecke: boring coordination)
# ══════════════════════════════════════════════════════════════


async def sync_weights_to_engines(
    engines: list[InferenceEngine],
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    """Sync checkpoint to multiple inference engines in parallel.

    Pure function - no state! No retention!
    Sean Goedecke: This is stateless coordination (that's good!).

    Uses trio for structured concurrency (not asyncio).

    Args:
        engines: List of inference engines (SGLang or vLLM)
        checkpoint_path: Path to checkpoint directory

    Returns:
        List of responses from each engine (in same order as engines)

    Raises:
        AssertionError: If preconditions violated

    Example:
        >>> engines = [
        ...     SGLangEngine("http://localhost:30000"),
        ...     VLLMEngine("http://localhost:30001"),
        ... ]
        >>> responses = await sync_weights_to_engines(engines, "/ckpt/step_100")
        >>> assert len(responses) == 2
        >>> assert all(r.get("success") or "method" in r for r in responses)
    """
    # Tiger Style: assert preconditions
    assert len(engines) > 0, "Must provide at least one engine"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Parallel sync with trio structured concurrency
    results = []

    async with trio.open_nursery() as nursery:

        async def sync_one(engine: InferenceEngine) -> None:
            """Sync to single engine and append result."""
            response = await engine.update_weights_from_checkpoint(checkpoint_path)
            results.append(response)

        # Start all syncs in parallel
        for engine in engines:
            nursery.start_soon(sync_one, engine)

    # Tiger Style: assert postconditions
    assert len(results) == len(engines), f"Expected {len(engines)} results, got {len(results)}"

    return results
