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
                "rank_offset": i + 1,  # Inference ranks start at 1
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
        {"name": name, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", "")}
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
    # NOTE: NCCL weight sync is done via HTTP API (/init_weights_update_group),
    # not via CLI flags. This field is kept for compatibility but not used.
    rl_on_policy_target: str | None = None
    _log_file: Path = field(init=False)
    _session_name: str = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "sglang.log"
        # Use output_dir name (run_id) for session isolation across runs
        run_id = self.output_dir.name
        self._session_name = f"sglang-{run_id}"

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
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"HF_HUB_DOWNLOAD_TIMEOUT=300 "  # 5 min timeout for model downloads
            # NCCL environment for cross-process weight sync:
            # - NCCL_SHM_DISABLE=1: Use sockets instead of shared memory (avoids IPC issues)
            # - NCCL_CUMEM_ENABLE=0: Consistent with SGLang defaults (see miles/ray/actor_group.py)
            f"NCCL_SHM_DISABLE=1 "
            f"NCCL_CUMEM_ENABLE=0 "
            f"python -m rollouts.training.sglang_launcher "
            f"--model-path {self.model_name} "
            f"--host 0.0.0.0 "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--mem-fraction-static {self.mem_fraction} "
            f"--trust-remote-code"
        )
        # NOTE: NCCL weight sync uses HTTP API (/init_weights_update_group),
        # not SGLang CLI flags. The --rl-on-policy-target flag only supports 'fsdp'.
        return cmd

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

        # Kill any old sglang sessions (from previous runs on same pod)
        subprocess.run(
            "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^sglang-' | xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true",
            shell=True,
            capture_output=True,
        )

        # Kill any process bound to our port (stale server from previous run)
        subprocess.run(
            f"fuser -k {self.port}/tcp 2>/dev/null || true",
            shell=True,
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
        # Use output_dir name (run_id) for session isolation across runs
        run_id = self.output_dir.name
        self._session_name = f"vllm-{run_id}"

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
            f"HF_HUB_DOWNLOAD_TIMEOUT=300 "  # 5 min timeout for model downloads
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

        # Kill any old vllm sessions (from previous runs on same pod)
        subprocess.run(
            "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^vllm-' | xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true",
            shell=True,
            capture_output=True,
        )

        # Kill any process bound to our port (stale server from previous run)
        subprocess.run(
            f"fuser -k {self.port}/tcp 2>/dev/null || true",
            shell=True,
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
# ENGINE V2 (rollouts native inference)
# ══════════════════════════════════════════════════════════════


@dataclass
class EngineV2Engine:
    """Rollouts native inference engine with full lifecycle management.

    Implements InferenceEngine protocol for engine_v2.
    Launches server in tmux for reliability (survives parent process crashes).

    This is the rollouts-native inference engine, alternative to SGLang/vLLM.
    Supports same weight sync APIs for RL training.

    Example:
        >>> engine = EngineV2Engine(
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
    max_batch_size: int = 32
    max_seq_len: int = 4096
    attention_backend: str = "auto"
    _log_file: Path = field(init=False)
    _session_name: str = field(init=False)

    def __post_init__(self) -> None:
        self._log_file = self.output_dir / "engine_v2.log"
        run_id = self.output_dir.name
        self._session_name = f"engine_v2-{run_id}"

    @property
    def name(self) -> str:
        return "engine_v2"

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
        """Build engine_v2 launch command."""
        gpu_str = ",".join(str(g) for g in self.cuda_device_ids)
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"HF_HUB_DOWNLOAD_TIMEOUT=300 "
            # NCCL environment for cross-process weight sync
            f"NCCL_SHM_DISABLE=1 "
            f"NCCL_CUMEM_ENABLE=0 "
            f"python -m rollouts.inference.server "
            f"--model {self.model_name} "
            f"--port {self.port} "
            f"--dtype {self.dtype} "
            f"--max-batch-size {self.max_batch_size} "
            f"--max-seq-len {self.max_seq_len} "
            f"--attention-backend {self.attention_backend}"
        )
        return cmd

    def launch(self) -> str:
        """Launch engine_v2 server in tmux session.

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

        # Kill any old engine_v2 sessions
        subprocess.run(
            "tmux list-sessions -F '#{session_name}' 2>/dev/null | grep '^engine_v2-' | xargs -r -I{} tmux kill-session -t {} 2>/dev/null || true",
            shell=True,
            capture_output=True,
        )

        # Kill any process bound to our port
        subprocess.run(
            f"fuser -k {self.port}/tcp 2>/dev/null || true",
            shell=True,
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
        """Start daemon thread that tails engine_v2 logs via Python logging."""
        engine_logger = logging.getLogger("engine_v2")

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
                                engine_logger.info(line)
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
        """Wait until engine_v2 health check passes."""
        async with httpx.AsyncClient(timeout=5.0) as client:
            for _attempt in range(int(max_wait)):
                # Check if tmux session crashed
                if not self._is_session_alive():
                    msg = f"engine_v2 server crashed during startup! Check {self._log_file}"
                    raise RuntimeError(msg)

                try:
                    resp = await client.get(self.health_url)
                    if resp.status_code == 200:
                        return
                except Exception:
                    pass
                await trio.sleep(1.0)

        msg = f"engine_v2 failed to start after {max_wait}s. Check {self._log_file}"
        raise RuntimeError(msg)

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update engine_v2 server weights from checkpoint."""
        assert checkpoint_path, "checkpoint_path cannot be empty"

        # Use same endpoint as SGLang for compatibility
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                f"{self.base_url}/update_weights_from_disk",
                json={"model_path": checkpoint_path},
            )
            response.raise_for_status()
            return response.json()

    def shutdown(self) -> None:
        """Kill the tmux session running engine_v2."""
        subprocess.run(
            ["tmux", "kill-session", "-t", self._session_name],
            capture_output=True,
        )


# ══════════════════════════════════════════════════════════════
# Stateless orchestration (Sean Goedecke: boring coordination)
# ══════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════
# True PipelineRL: Non-blocking weight sync (inference never stops)
# ══════════════════════════════════════════════════════════════


@dataclass
class PipelineWeightSyncManager:
    """True PipelineRL-style weight sync: inference never stops.

    Unlike stop-and-sync (Miles/verl), this broadcasts weights while
    inference continues. Samples are tagged with weight_version.

    Architecture:
        Training loop:
            for step in steps:
                batch = get_batch()  # May have samples from v-1, v-2, etc.
                train(batch)
                sync_manager.broadcast_weights_async(model)  # Non-blocking!
                # Training continues immediately, doesn't wait for sync

        Inference side:
            - Receives NCCL broadcast in background
            - Updates weights parameter-by-parameter
            - New requests use new weights, in-flight requests use old weights
            - Returns weight_version with each response

    Warning:
        This is "slightly sketchy" (PipelineRL's words) - during a sync,
        some layers may have new weights while others have old weights.
        PipelineRL accepts this for the throughput benefit.

    Example:
        >>> manager = PipelineWeightSyncManager(
        ...     inference_endpoints=["http://localhost:30000"],
        ...     max_lag=2,
        ... )
        >>> await manager.init_nccl_group()
        >>>
        >>> for step in range(num_steps):
        ...     batch = await get_batch_with_staleness_filter(manager.current_version, max_lag=2)
        ...     train(batch)
        ...     manager.broadcast_weights_async(model)  # Non-blocking!
    """

    inference_endpoints: list[str]
    max_lag: int = 2
    nccl_master_port: int = 29500

    # Internal state
    _process_group: Any = field(default=None, init=False, repr=False)
    _current_version: int = field(default=0, init=False)
    _pending_sync: trio.Event | None = field(default=None, init=False, repr=False)
    _sync_nursery: trio.Nursery | None = field(default=None, init=False, repr=False)
    _sync_in_progress: bool = field(default=False, init=False)

    @property
    def sync_in_progress(self) -> bool:
        """True if a weight sync is currently in progress (SGLang is blocked)."""
        return self._sync_in_progress

    @property
    def current_version(self) -> int:
        """Current weight version (increments after each successful sync)."""
        return self._current_version

    async def init_nccl_group(self) -> None:
        """Initialize NCCL process group between trainer and inference engines.

        Must be called once at startup before any broadcasts.

        NCCL requires all ranks to join concurrently - SGLang's /init_weights_update_group
        blocks on dist.init_process_group, so we must run HTTP requests AND trainer join
        in parallel to avoid deadlock.
        """
        import os
        import socket

        from ..inference.weight_sync import create_stateless_process_group

        logger = logging.getLogger(__name__)

        # Find an available port (avoids conflicts with stale processes)
        def find_free_port(start_port: int, max_attempts: int = 100) -> int:
            for port in range(start_port, start_port + max_attempts):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        s.bind(("", port))
                        return port
                except OSError:
                    continue
            raise RuntimeError(
                f"No free port found in range {start_port}-{start_port + max_attempts}"
            )

        master_port = find_free_port(self.nccl_master_port)
        if master_port != self.nccl_master_port:
            logger.info(f"Port {self.nccl_master_port} in use, using {master_port}")

        # Get master address
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        world_size = 1 + len(self.inference_endpoints)  # trainer + inference

        logger.info(f"Initializing PipelineRL NCCL group (world_size={world_size})")

        # NCCL init requires all ranks to join concurrently
        # Run HTTP requests AND trainer join in parallel
        async def register_inference_endpoint(endpoint: str, rank: int) -> None:
            async with httpx.AsyncClient(timeout=300.0) as client:
                await client.post(
                    f"{endpoint}/init_weights_update_group",
                    json={
                        "master_address": master_addr,
                        "master_port": master_port,
                        "rank_offset": rank,
                        "world_size": world_size,
                        "group_name": "pipeline_weight_sync",
                        "backend": "nccl",
                    },
                )

        async def trainer_join() -> None:
            def _join() -> None:
                # Set NCCL env vars
                os.environ.setdefault("NCCL_SHM_DISABLE", "1")
                os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")

                self._process_group = create_stateless_process_group(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank=0,
                    world_size=world_size,
                    group_name="pipeline_weight_sync",
                    backend="nccl",
                    timeout_seconds=300.0,
                )

            await trio.to_thread.run_sync(_join)

        # Launch all NCCL participants concurrently
        async with trio.open_nursery() as nursery:
            for i, endpoint in enumerate(self.inference_endpoints):
                nursery.start_soon(register_inference_endpoint, endpoint, i + 1)
            nursery.start_soon(trainer_join)

        logger.info("PipelineRL NCCL group initialized")

    async def broadcast_weights_async(
        self,
        model: Any,  # nn.Module
        nursery: trio.Nursery,
    ) -> None:
        """Broadcast weights to inference engines in background (non-blocking).

        This is the key PipelineRL primitive: training continues immediately
        while weight sync happens in background.

        Args:
            model: PyTorch model to sync
            nursery: Trio nursery to spawn background sync task

        Note:
            The sync may complete after the next training step starts.
            Samples generated during sync may use old or new weights.
        """

        async def _do_sync() -> None:
            await self._sync_weights_nccl(model)
            self._current_version += 1

        # Spawn sync task in background - training continues immediately
        nursery.start_soon(_do_sync)

    async def _sync_weights_nccl(self, model: Any) -> None:
        """Internal: perform NCCL weight sync.

        IMPORTANT: While this runs, SGLang is blocked receiving weights and
        cannot process inference requests. Callers should pause sampling.
        """
        import torch.distributed as dist

        logger = logging.getLogger(__name__)

        self._sync_in_progress = True
        try:
            state_dict = model.state_dict()
            new_version = self._current_version + 1

            # Build parameter info
            param_info = [
                {"name": name, "shape": list(p.shape), "dtype": str(p.dtype).replace("torch.", "")}
                for name, p in state_dict.items()
            ]

            logger.debug(f"Starting weight sync to v{new_version}")

            # IMPORTANT: HTTP POST and NCCL broadcast must run concurrently!
            # The HTTP endpoint blocks waiting for NCCL weights, so we can't do them sequentially.

            async def send_http_requests() -> None:
                """Tell inference engines to prepare for NCCL receive."""
                async with httpx.AsyncClient(timeout=300.0) as client:
                    for endpoint in self.inference_endpoints:
                        await client.post(
                            f"{endpoint}/update_weights_from_distributed",
                            json={
                                "names": [p["name"] for p in param_info],
                                "shapes": [p["shape"] for p in param_info],
                                "dtypes": [p["dtype"] for p in param_info],
                                "group_name": "pipeline_weight_sync",
                                "weight_version": str(new_version),
                            },
                        )

            async def do_nccl_broadcast() -> None:
                """Broadcast each tensor via NCCL (runs in thread to not block event loop)."""

                def _broadcast() -> None:
                    for _name, param in state_dict.items():
                        param_data = param.data.contiguous()
                        if param_data.device.type != "cuda":
                            param_data = param_data.cuda()
                        dist.broadcast(param_data, src=0, group=self._process_group)
                    # NOTE: No barrier here! SGLang doesn't call barrier after receiving.
                    # The HTTP response serves as implicit synchronization.

                await trio.to_thread.run_sync(_broadcast)

            # Run HTTP requests and NCCL broadcast concurrently
            async with trio.open_nursery() as sync_nursery:
                sync_nursery.start_soon(send_http_requests)
                sync_nursery.start_soon(do_nccl_broadcast)

            logger.debug(f"Weight sync to v{new_version} complete")

            # Free CUDA memory after weight sync (QED-Nano pattern)
            # The broadcast creates temporary GPU tensors that should be freed
            import torch

            torch.cuda.empty_cache()
        finally:
            self._sync_in_progress = False

    async def get_inference_weight_version(self, endpoint: str) -> int:
        """Query current weight version from an inference engine.

        Useful for debugging / monitoring staleness.
        """
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{endpoint}/get_weight_version")
            response.raise_for_status()
            return int(response.json().get("weight_version", 0))

    async def cleanup(self) -> None:
        """Cleanup NCCL process group."""
        import torch.distributed as dist

        if self._process_group is not None:
            # Tell inference engines to leave
            async with httpx.AsyncClient(timeout=10.0) as client:
                for endpoint in self.inference_endpoints:
                    try:
                        await client.post(
                            f"{endpoint}/destroy_weights_update_group",
                            json={"group_name": "pipeline_weight_sync"},
                        )
                    except Exception:
                        pass  # Best effort

            dist.destroy_process_group(self._process_group)
            self._process_group = None


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
