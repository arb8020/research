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

import logging
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


def get_fast_sync_dir() -> Path:
    """Get a fast directory for weight sync (RAM disk if available).

    Returns /dev/shm on Linux (tmpfs, ~10GB RAM disk) for fast I/O.
    Falls back to /tmp if /dev/shm not available.

    This speeds up disk-based weight sync from ~2-4s to ~0.5-1s.
    """
    shm_path = Path("/dev/shm")
    if shm_path.exists() and shm_path.is_dir():
        sync_dir = shm_path / "rollouts_weight_sync"
        sync_dir.mkdir(exist_ok=True)
        return sync_dir
    else:
        # Fallback to /tmp (may be slow if not tmpfs)
        sync_dir = Path("/tmp/rollouts_weight_sync")
        sync_dir.mkdir(exist_ok=True)
        return sync_dir


# ══════════════════════════════════════════════════════════════
# NCCL Weight Sync (pure functions for multi-node)
# ══════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class NCCLSyncGroup:
    """Immutable config for NCCL weight sync group.

    This is just data - the actual process group is created by
    init_nccl_weight_sync() and passed around.
    """

    master_addr: str
    master_port: int
    trainer_rank: int  # Usually 0
    world_size: int  # trainer + all inference GPUs
    group_name: str = "weight_sync"


def init_nccl_weight_sync(
    config: NCCLSyncGroup,
    inference_endpoints: list[str],
) -> Any:  # Returns torch.distributed ProcessGroup
    """Initialize NCCL process group for weight sync.

    Pure function: takes config, returns process group.
    Must be called once at startup.

    Args:
        config: NCCL group configuration
        inference_endpoints: List of SGLang server URLs

    Returns:
        NCCL process group for broadcasting weights

    Example:
        >>> config = NCCLSyncGroup(
        ...     master_addr="10.0.0.1",
        ...     master_port=29500,
        ...     trainer_rank=0,
        ...     world_size=5,  # 1 trainer + 4 inference GPUs
        ... )
        >>> pg = init_nccl_weight_sync(config, ["http://10.0.0.2:30000", ...])
        >>> # Later: sync_weights_nccl(model, pg, endpoints)
    """

    import requests
    import torch.distributed as dist

    # 1. Tell each SGLang server to join the NCCL group
    for i, endpoint in enumerate(inference_endpoints):
        response = requests.post(
            f"{endpoint}/init_weights_update_group",
            json={
                "master_address": config.master_addr,
                "master_port": config.master_port,
                "rank": i + 1,  # Inference ranks start at 1
                "world_size": config.world_size,
                "group_name": config.group_name,
                "backend": "nccl",
            },
            timeout=60,
        )
        response.raise_for_status()

    # 2. Trainer joins as rank 0
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{config.master_addr}:{config.master_port}",
            rank=config.trainer_rank,
            world_size=config.world_size,
        )

    # 3. Create named group for weight sync
    pg = dist.new_group(
        ranks=list(range(config.world_size)),
        backend="nccl",
    )

    return pg


def sync_weights_nccl(
    model: Any,  # nn.Module
    process_group: Any,  # ProcessGroup
    inference_endpoints: list[str],
    weight_version: int,
) -> None:
    """Broadcast model weights to inference engines via NCCL.

    Pure function: no hidden state, explicit inputs.

    Args:
        model: PyTorch model to sync
        process_group: NCCL process group from init_nccl_weight_sync()
        inference_endpoints: SGLang server URLs
        weight_version: Version number for this weight update

    Example:
        >>> sync_weights_nccl(model, pg, endpoints, step)
    """
    import requests
    import torch.distributed as dist

    state_dict = model.state_dict()

    # 1. Send metadata to inference engines (names, shapes, dtypes)
    param_info = [
        {"name": name, "shape": list(p.shape), "dtype": str(p.dtype)}
        for name, p in state_dict.items()
    ]

    for endpoint in inference_endpoints:
        requests.post(
            f"{endpoint}/update_weights_from_distributed",
            json={
                "names": [p["name"] for p in param_info],
                "shapes": [p["shape"] for p in param_info],
                "dtypes": [p["dtype"] for p in param_info],
                "group_name": "weight_sync",
                "weight_version": str(weight_version),
            },
            timeout=300,
        )

    # 2. Broadcast each tensor via NCCL (GPU-to-GPU, no serialization)
    for name, param in state_dict.items():
        param_data = param.data.contiguous().cuda()
        dist.broadcast(param_data, src=0, group=process_group)

    # 3. Wait for completion
    dist.barrier(group=process_group)


def cleanup_nccl_weight_sync(
    process_group: Any,
    inference_endpoints: list[str],
) -> None:
    """Cleanup NCCL weight sync group.

    Call at shutdown to properly cleanup distributed resources.
    """
    import requests
    import torch.distributed as dist

    # Tell SGLang servers to leave the group
    for endpoint in inference_endpoints:
        try:
            requests.post(
                f"{endpoint}/destroy_weights_update_group",
                json={"group_name": "weight_sync"},
                timeout=10,
            )
        except Exception:
            pass  # Best effort cleanup

    # Destroy local process group
    if process_group is not None:
        dist.destroy_process_group(process_group)


# ══════════════════════════════════════════════════════════════
# Minimal Protocol (Tiger Style: just type hints, not inheritance)
# ══════════════════════════════════════════════════════════════


class InferenceEngine(Protocol):
    """Protocol for inference engines with full lifecycle management.

    Tiger Style: This is JUST a type annotation (Protocol), not a base class.
    No inheritance! Just duck typing.

    Lifecycle:
    1. launch() -> str               # Start server in tmux, return session name
    2. start_log_tailer() -> Thread  # Tail logs via Python logging
    3. wait_until_ready() -> None    # Block until health check passes
    4. update_weights_from_checkpoint() -> dict  # Sync weights
    5. shutdown() -> None            # Kill tmux session
    """

    @property
    def name(self) -> str:
        """Engine name for logging (e.g., 'sglang', 'vllm')."""
        ...

    @property
    def session_name(self) -> str:
        """Tmux session name for this engine."""
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

    def launch(self) -> str:
        """Launch the inference server in a tmux session.

        Returns:
            The tmux session name (for shutdown)
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

    def shutdown(self) -> None:
        """Shutdown the inference server (kill tmux session)."""
        ...


# ══════════════════════════════════════════════════════════════
# Adapters (Casey Muratori: redundancy - multiple ways to do same thing)
# ══════════════════════════════════════════════════════════════


@dataclass
class SGLangEngine:
    """SGLang inference engine with full lifecycle management.

    Implements InferenceEngine protocol for SGLang servers.
    Launches server in tmux for reliability (survives parent process crashes).

    Example:
        >>> engine = SGLangEngine(
        ...     model_name="Qwen/Qwen3-0.6B",
        ...     port=30000,
        ...     cuda_device_ids=(0,),
        ...     output_dir=Path("results/rl/run_001"),
        ... )
        >>> engine.launch()
        >>> engine.start_log_tailer()  # Routes logs via Python logging
        >>> await engine.wait_until_ready()
        >>> # ... use engine ...
        >>> await engine.update_weights_from_checkpoint("/ckpt/step_100")
        >>> engine.shutdown()
    """

    model_name: str
    port: int
    cuda_device_ids: tuple[int, ...]
    output_dir: Path
    dtype: str = "bfloat16"
    mem_fraction: float = 0.7
    timeout: float = 300.0
    _log_file: Path = field(init=False)
    _session_name: str = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "sglang.log"
        self._session_name = f"sglang-{self.port}"

    @property
    def name(self) -> str:
        return "sglang"

    @property
    def session_name(self) -> str:
        return self._session_name

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
        """Build SGLang launch command (without redirection - tmux handles that)."""
        gpu_str = ",".join(str(g) for g in self.cuda_device_ids)
        return (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"python -m sglang.launch_server "
            f"--model-path {self.model_name} "
            f"--host 0.0.0.0 "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--mem-fraction-static {self.mem_fraction} "
            f"--trust-remote-code"
        )

    def launch(self) -> str:
        """Launch SGLang server in tmux session.

        Uses tmux for reliability - survives parent process crashes.
        Logs are piped to a file for tailing.

        Returns:
            The tmux session name
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Kill existing session if present
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
        )

        # Kill any orphaned processes using our GPUs
        for gpu_id in self.cuda_device_ids:
            subprocess.run(
                f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
                shell=True,
                capture_output=True,
            )

        # Build command with log piping
        cmd = self.build_launch_cmd()
        full_cmd = f"{cmd} 2>&1 | tee {self._log_file}"

        # Launch in tmux
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self._session_name, full_cmd],
            check=True,
        )

        return self._session_name

    def start_log_tailer(self) -> threading.Thread:
        """Start daemon thread that tails SGLang logs via Python logging.

        Uses a dedicated 'sglang' logger so logs go through the same
        formatting as training logs (JSONL when TUI is active).
        """
        sglang_logger = logging.getLogger("sglang")

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
                                sglang_logger.info(line)
                        else:
                            time.sleep(0.1)
            except Exception:
                pass  # File closed or thread killed

        thread = threading.Thread(target=tail_log, daemon=True)
        thread.start()
        return thread

    def _is_session_alive(self) -> bool:
        """Check if tmux session is still running."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", self._session_name],
            capture_output=True,
        )
        return result.returncode == 0

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        """Wait until SGLang health check passes."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _attempt in range(int(max_wait)):
                # Check if tmux session crashed
                if not self._is_session_alive():
                    msg = f"SGLang server crashed during startup! Check {self._log_file}"
                    raise RuntimeError(msg)

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

    def shutdown(self) -> None:
        """Kill the tmux session running SGLang."""
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
        )


@dataclass
class VLLMEngine:
    """vLLM inference engine with full lifecycle management.

    Implements InferenceEngine protocol for vLLM servers.
    Launches server in tmux for reliability (survives parent process crashes).

    Example:
        >>> engine = VLLMEngine(
        ...     model_name="Qwen/Qwen3-0.6B",
        ...     port=30001,
        ...     cuda_device_ids=(0,),
        ...     output_dir=Path("results/rl/run_001"),
        ... )
        >>> engine.launch()
        >>> engine.start_log_tailer()  # Routes logs via Python logging
        >>> await engine.wait_until_ready()
        >>> # ... use engine ...
        >>> await engine.update_weights_from_checkpoint("/ckpt/step_100")
        >>> engine.shutdown()
    """

    model_name: str
    port: int
    cuda_device_ids: tuple[int, ...]
    output_dir: Path
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.7
    timeout: float = 300.0
    _log_file: Path = field(init=False)
    _session_name: str = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "vllm.log"
        self._session_name = f"vllm-{self.port}"

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def session_name(self) -> str:
        return self._session_name

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
        """Build vLLM launch command (without redirection - tmux handles that)."""
        gpu_str = ",".join(str(g) for g in self.cuda_device_ids)
        return (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"python -m vllm.entrypoints.openai.api_server "
            f"--model {self.model_name} "
            f"--host 0.0.0.0 "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--gpu-memory-utilization {self.gpu_memory_utilization} "
            f"--trust-remote-code"
        )

    def launch(self) -> str:
        """Launch vLLM server in tmux session.

        Uses tmux for reliability - survives parent process crashes.
        Logs are piped to a file for tailing.

        Returns:
            The tmux session name
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Kill existing session if present
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
        )

        # Kill any orphaned processes using our GPUs
        for gpu_id in self.cuda_device_ids:
            subprocess.run(
                f"nvidia-smi --id={gpu_id} --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9",
                shell=True,
                capture_output=True,
            )

        # Build command with log piping
        cmd = self.build_launch_cmd()
        full_cmd = f"{cmd} 2>&1 | tee {self._log_file}"

        # Launch in tmux
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self._session_name, full_cmd],
            check=True,
        )

        return self._session_name

    def start_log_tailer(self) -> threading.Thread:
        """Start daemon thread that tails vLLM logs via Python logging.

        Uses a dedicated 'vllm' logger so logs go through the same
        formatting as training logs (JSONL when TUI is active).
        """
        vllm_logger = logging.getLogger("vllm")

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
                                vllm_logger.info(line)
                        else:
                            time.sleep(0.1)
            except Exception:
                pass  # File closed or thread killed

        thread = threading.Thread(target=tail_log, daemon=True)
        thread.start()
        return thread

    def _is_session_alive(self) -> bool:
        """Check if tmux session is still running."""
        result = subprocess.run(
            ["tmux", "has-session", "-t", self._session_name],
            capture_output=True,
        )
        return result.returncode == 0

    async def wait_until_ready(self, max_wait: float = 120.0) -> None:
        """Wait until vLLM health check passes."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _attempt in range(int(max_wait)):
                # Check if tmux session crashed
                if not self._is_session_alive():
                    msg = f"vLLM server crashed during startup! Check {self._log_file}"
                    raise RuntimeError(msg)

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

    def shutdown(self) -> None:
        """Kill the tmux session running vLLM."""
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
        )


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
