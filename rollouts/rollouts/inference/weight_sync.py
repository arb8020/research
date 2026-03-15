"""NCCL weight sync for engine_v2.

Implements PipelineRL-style weight sync: training broadcasts weights via NCCL,
inference engines receive and load them.

Two roles:
- Sender (training side): Creates NCCL group, broadcasts state_dict
- Receiver (inference side): Joins NCCL group, receives and loads weights

Usage (sender - training process):
    sender = WeightSyncSender(
        master_addr="10.0.0.1",
        master_port=29500,
        inference_world_size=4,  # Number of inference GPUs
    )
    sender.init_group()

    # In training loop:
    sender.broadcast_weights(model.state_dict())

    sender.cleanup()

Usage (receiver - inference engine):
    receiver = WeightSyncReceiver(
        master_addr="10.0.0.1",
        master_port=29500,
        rank=1,  # This inference GPU's rank (1-indexed, 0 is trainer)
        world_size=5,  # trainer + inference GPUs
    )
    receiver.init_group()

    # When trainer broadcasts:
    state_dict = receiver.receive_weights(param_info)
    engine.reload_weights(state_dict)

    receiver.cleanup()
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from ..training.weight_sync_protocol import WeightUpdatePayload, WeightWireTensor

logger = logging.getLogger(__name__)
_ARGUS_DIAG_EVENT_SENTINEL = "__ARGUS_DIAG__"


def _emit_argus_diag(event: str, **data: object) -> None:
    """Best-effort structured diagnostics for remote sandbox runs."""
    try:
        sys.stderr.write(
            f"{_ARGUS_DIAG_EVENT_SENTINEL}{json.dumps({'event': event, **data}, sort_keys=True)}\n"
        )
        sys.stderr.flush()
    except Exception:
        return


def _tensor_sync_metadata(name: str, tensor: Tensor) -> dict[str, object]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
        "is_contiguous": bool(tensor.is_contiguous()),
        "stride": list(tensor.stride()),
    }


def _normalize_cuda_device(device: torch.device) -> torch.device:
    """Resolve ambiguous CUDA devices to an explicit local index."""
    if device.type != "cuda":
        return device
    if device.index is not None:
        return device
    return torch.device("cuda", torch.cuda.current_device())


# ═══════════════════════════════════════════════════════════════════════════════
# STATELESS PROCESS GROUP (following vLLM pattern)
# ═══════════════════════════════════════════════════════════════════════════════


def create_stateless_process_group(
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    group_name: str = "weight_sync",
    backend: str = "nccl",
    timeout_seconds: float = 300.0,
) -> dist.ProcessGroup:
    """Create a process group without touching global torch.distributed state.

    This follows vLLM's StatelessProcessGroup pattern - allows creating NCCL
    groups for weight sync without interfering with training's distributed setup.

    Args:
        master_addr: IP address of the master (trainer rank 0)
        master_port: Port for rendezvous
        rank: This process's rank in the weight sync group
        world_size: Total processes (trainer + inference GPUs)
        group_name: Name for the process group
        backend: "nccl" for GPU-to-GPU, "gloo" for CPU
        timeout_seconds: Timeout for initialization

    Returns:
        Process group for weight sync operations
    """
    from datetime import timedelta

    from torch.distributed.distributed_c10d import (
        Backend,
        PrefixStore,
        _new_process_group_helper,
        _world,
        rendezvous,
    )

    timeout = timedelta(seconds=timeout_seconds)
    init_method = f"tcp://{master_addr}:{master_port}"
    logger.info(
        "weight_sync_pg_create_start group=%s backend=%s rank=%s world_size=%s init_method=%s timeout_s=%.1f",
        group_name,
        backend,
        rank,
        world_size,
        init_method,
        timeout_seconds,
    )
    _emit_argus_diag(
        "weight_sync_pg_create_start",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
        init_method=init_method,
        timeout_s=timeout_seconds,
    )

    # Rendezvous to get store
    logger.info(
        "weight_sync_pg_rendezvous_start group=%s rank=%s world_size=%s",
        group_name,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_rendezvous_start",
        group=group_name,
        rank=rank,
        world_size=world_size,
    )
    rendezvous_iterator = rendezvous(init_method, rank, world_size, timeout=timeout)
    store, rank, world_size = next(rendezvous_iterator)
    store.set_timeout(timeout)
    logger.info(
        "weight_sync_pg_rendezvous_ok group=%s rank=%s world_size=%s",
        group_name,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_rendezvous_ok",
        group=group_name,
        rank=rank,
        world_size=world_size,
    )

    # Use PrefixStore to namespace this group
    store = PrefixStore(group_name, store)

    # Create process group without touching the *default* global process group.
    # NOTE: PyTorch has renamed the kwarg from `pg_options` → `backend_options`
    # in some versions; detect by signature instead of version-string compares.
    import inspect

    pg_helper_sig = inspect.signature(_new_process_group_helper)
    if "backend_options" in pg_helper_sig.parameters:
        pg_kwargs = {"backend_options": None}
    elif "pg_options" in pg_helper_sig.parameters:
        pg_kwargs = {"pg_options": None}
    else:
        raise RuntimeError(
            "Unsupported torch.distributed version: _new_process_group_helper has neither "
            "'backend_options' nor 'pg_options' parameter."
        )

    logger.info(
        "weight_sync_pg_helper_start group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_helper_start",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    pg, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        store,
        group_name=group_name,
        timeout=timeout,
        **pg_kwargs,
    )
    logger.info(
        "weight_sync_pg_helper_ok group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_helper_ok",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    # Register in world for cleanup
    _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
    logger.info(
        "weight_sync_pg_create_ok group=%s backend=%s rank=%s world_size=%s",
        group_name,
        backend,
        rank,
        world_size,
    )
    _emit_argus_diag(
        "weight_sync_pg_create_ok",
        group=group_name,
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    return pg


# ═══════════════════════════════════════════════════════════════════════════════
# SENDER (training side)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WeightSyncSender:
    """Sends weights from training to inference via NCCL broadcast.

    The sender is rank 0 in the weight sync group. It broadcasts each tensor
    to all inference GPUs.
    """

    master_addr: str
    master_port: int
    inference_world_size: int  # Number of inference GPUs (not including trainer)
    group_name: str = "weight_sync"
    timeout_seconds: float = 300.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    _process_group: Any = field(default=None, init=False, repr=False)
    _weight_version: int = field(default=0, init=False)

    @property
    def world_size(self) -> int:
        return self.inference_world_size + 1  # +1 for trainer

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def init_group(self) -> None:
        """Initialize NCCL process group. Call once at startup."""
        self.device = _normalize_cuda_device(self.device)
        logger.info(
            "weight_sync_sender_init_start world_size=%s master=%s:%s device=%s group=%s",
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
        )
        _emit_argus_diag(
            "weight_sync_sender_init_start",
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._process_group = create_stateless_process_group(
            master_addr=self.master_addr,
            master_port=self.master_port,
            rank=0,  # Trainer is always rank 0
            world_size=self.world_size,
            group_name=self.group_name,
            timeout_seconds=self.timeout_seconds,
        )
        logger.info(
            "weight_sync_sender_init_ok world_size=%s master=%s:%s device=%s group=%s",
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
        )
        _emit_argus_diag(
            "weight_sync_sender_init_ok",
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
        )

    def broadcast_payload(
        self,
        payload: WeightUpdatePayload,
        async_op: bool = False,
    ) -> list[Any] | None:
        """Broadcast explicit wire payload to all inference GPUs.

        Args:
            payload: Concrete payload to broadcast
            async_op: If True, return handles for async wait

        Returns:
            List of async handles if async_op=True, else None
        """
        assert self._process_group is not None, "Call init_group() first"

        handles = []
        total_tensors = len(payload.tensors)
        total_bytes = sum(
            int(item.tensor.numel() * item.tensor.element_size()) for item in payload.tensors
        )
        first_items = list(payload.tensors[:3])
        logger.info(
            "weight_sync_sender_broadcast_start world_size=%s payload_kind=%s total_tensors=%s total_bytes=%s first_tensors=%s",
            self.world_size,
            payload.payload_kind,
            total_tensors,
            total_bytes,
            [_tensor_sync_metadata(item.wire_name, item.tensor) for item in first_items],
        )

        for index, item in enumerate(payload.tensors):
            name = item.wire_name
            param = item.tensor
            # Ensure contiguous and on GPU
            data = param.data.contiguous()
            if data.device != self.device:
                data = data.to(self.device)
            tensor_meta = _tensor_sync_metadata(name, data)
            broadcast_meta = {
                **tensor_meta,
                "load_name": item.load_name,
                "payload_kind": item.payload_kind,
                "param_device": str(param.device),
                "current_cuda_device": (
                    torch.cuda.current_device() if torch.cuda.is_available() else None
                ),
                "data_ptr": int(data.data_ptr()),
                "storage_offset": int(data.storage_offset()),
                "element_size": int(data.element_size()),
                "nbytes": int(data.numel() * data.element_size()),
                "group_rank": dist.get_rank(self._process_group),
                "group_world_size": dist.get_world_size(self._process_group),
            }
            logger.info(
                "weight_sync_sender_broadcast_tensor index=%s total_tensors=%s async_op=%s tensor=%s",
                index,
                total_tensors,
                async_op,
                broadcast_meta,
            )
            try:
                handle = dist.broadcast(data, src=0, group=self._process_group, async_op=async_op)
            except Exception as exc:
                raise RuntimeError(
                    "Weight sync sender broadcast failed "
                    f"index={index} total_tensors={total_tensors} tensor={broadcast_meta}"
                ) from exc
            if async_op:
                handles.append(handle)

        self._weight_version += 1

        if async_op:
            return handles
        return None

    def broadcast_weights(
        self,
        state_dict: dict[str, Tensor],
        async_op: bool = False,
    ) -> list[Any] | None:
        """Backward-compatible wrapper for raw tensor dictionaries."""
        payload = WeightUpdatePayload(
            tensors=tuple(
                WeightWireTensor(
                    wire_name=name,
                    load_name=name,
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype).replace("torch.", ""),
                    tensor=tensor,
                    payload_kind="inference_load_tensor",
                )
                for name, tensor in state_dict.items()
            ),
            payload_kind="inference_load_tensor",
        )
        return self.broadcast_payload(payload, async_op=async_op)

    def cleanup(self) -> None:
        """Cleanup process group."""
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None
            logger.info("Weight sync sender cleaned up")


# ═══════════════════════════════════════════════════════════════════════════════
# RECEIVER (inference side)
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ParamInfo:
    """Metadata for a parameter to receive."""

    wire_name: str
    load_name: str
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class WeightSyncReceiver:
    """Receives weights from training via NCCL broadcast.

    The receiver joins the weight sync group and receives tensors broadcast
    by the sender (trainer).
    """

    master_addr: str
    master_port: int
    rank: int  # This GPU's rank in weight sync group (1-indexed)
    world_size: int  # Total: trainer + inference GPUs
    group_name: str = "weight_sync"
    timeout_seconds: float = 300.0
    device: torch.device = field(default_factory=lambda: torch.device("cuda"))

    _process_group: Any = field(default=None, init=False, repr=False)
    _communicator: Any = field(default=None, init=False, repr=False)
    _weight_version: int = field(default=0, init=False)

    @property
    def weight_version(self) -> int:
        return self._weight_version

    def init_group(self) -> None:
        """Initialize NCCL process group. Call once at startup."""
        assert self.rank > 0, "Rank 0 is reserved for trainer (sender)"
        self.device = _normalize_cuda_device(self.device)
        logger.info(
            "weight_sync_receiver_init_start rank=%s world_size=%s master=%s:%s device=%s group=%s",
            self.rank,
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
        )
        _emit_argus_diag(
            "weight_sync_receiver_init_start",
            rank=self.rank,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
        )
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self._process_group = create_stateless_process_group(
            master_addr=self.master_addr,
            master_port=self.master_port,
            rank=self.rank,
            world_size=self.world_size,
            group_name=self.group_name,
            timeout_seconds=self.timeout_seconds,
        )
        try:
            from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
            from vllm.distributed.utils import StatelessProcessGroup

            stateless_pg = StatelessProcessGroup.create(
                host=self.master_addr,
                port=self.master_port,
                rank=self.rank,
                world_size=self.world_size,
                store_timeout=int(self.timeout_seconds),
            )
            self._communicator = PyNcclCommunicator(stateless_pg, device=self.device)
        except Exception as exc:
            logger.warning(
                "weight_sync_receiver_communicator_init_failed rank=%s world_size=%s device=%s error=%r; "
                "falling back to torch.distributed broadcast",
                self.rank,
                self.world_size,
                self.device,
                exc,
            )
            self._communicator = None
        logger.info(
            "weight_sync_receiver_init_ok rank=%s world_size=%s master=%s:%s device=%s group=%s",
            self.rank,
            self.world_size,
            self.master_addr,
            self.master_port,
            self.device,
            self.group_name,
        )
        _emit_argus_diag(
            "weight_sync_receiver_init_ok",
            rank=self.rank,
            world_size=self.world_size,
            master_addr=self.master_addr,
            master_port=self.master_port,
            device=str(self.device),
            group=self.group_name,
        )

    def receive_weights(
        self,
        param_info: list[ParamInfo],
    ) -> dict[str, Tensor]:
        """Receive weights from trainer broadcast.

        Args:
            param_info: List of (name, shape, dtype) for expected params

        Returns:
            State dict with received weights
        """
        assert self._process_group is not None, "Call init_group() first"

        state_dict = {}
        total_tensors = len(param_info)
        logger.info(
            "weight_sync_receiver_receive_start rank=%s world_size=%s total_tensors=%s first_tensors=%s",
            self.rank,
            self.world_size,
            total_tensors,
            [
                {
                    "wire_name": info.wire_name,
                    "load_name": info.load_name,
                    "shape": list(info.shape),
                    "dtype": str(info.dtype).replace("torch.", ""),
                }
                for info in param_info[:3]
            ],
        )
        for index, info in enumerate(param_info):
            # Allocate buffer
            buffer = torch.empty(info.shape, dtype=info.dtype, device=self.device)

            # Receive broadcast from rank 0
            tensor_meta = _tensor_sync_metadata(info.wire_name, buffer)
            receive_meta = {
                **tensor_meta,
                "load_name": info.load_name,
                "current_cuda_device": (
                    torch.cuda.current_device() if torch.cuda.is_available() else None
                ),
                "data_ptr": int(buffer.data_ptr()),
                "storage_offset": int(buffer.storage_offset()),
                "element_size": int(buffer.element_size()),
                "nbytes": int(buffer.numel() * buffer.element_size()),
                "group_rank": dist.get_rank(self._process_group),
                "group_world_size": dist.get_world_size(self._process_group),
            }
            logger.info(
                "weight_sync_receiver_receive_tensor index=%s total_tensors=%s tensor=%s",
                index,
                total_tensors,
                receive_meta,
            )
            try:
                if self._communicator is not None:
                    self._communicator.broadcast(buffer, src=0, stream=torch.cuda.current_stream())
                else:
                    dist.broadcast(buffer, src=0, group=self._process_group)
            except Exception as exc:
                raise RuntimeError(
                    "Weight sync receiver broadcast failed "
                    f"index={index} total_tensors={total_tensors} tensor={receive_meta}"
                ) from exc

            state_dict[info.load_name] = buffer

        self._weight_version += 1
        logger.info(f"Received weights v{self._weight_version}")
        return state_dict

    def receive_weights_into(
        self,
        state_dict: dict[str, Tensor],
    ) -> None:
        """Receive weights directly into existing state_dict tensors (in-place).

        This is more efficient than receive_weights() as it avoids allocation.

        Args:
            state_dict: Existing state dict to receive into
        """
        assert self._process_group is not None, "Call init_group() first"

        for _name, param in state_dict.items():
            # Receive broadcast from rank 0 directly into existing tensor
            if self._communicator is not None:
                self._communicator.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
            else:
                dist.broadcast(param.data, src=0, group=self._process_group)

        self._weight_version += 1
        logger.info(f"Received weights v{self._weight_version} (in-place)")

    def cleanup(self) -> None:
        """Cleanup process group."""
        if self._communicator is not None:
            del self._communicator
            self._communicator = None
        if self._process_group is not None:
            dist.destroy_process_group(self._process_group)
            self._process_group = None
            logger.info("Weight sync receiver cleaned up")


# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════


def add_weight_sync_to_engine(
    engine: Any,
    master_addr: str,
    master_port: int,
    rank: int,
    world_size: int,
    group_name: str = "weight_sync",
) -> WeightSyncReceiver:
    """Add NCCL weight sync capability to an inference engine.

    This creates a WeightSyncReceiver and attaches it to the engine.
    When weights are broadcast, call engine.receive_and_reload_weights().

    Args:
        engine: InferenceEngineV2 instance
        master_addr: IP of trainer
        master_port: Port for NCCL rendezvous
        rank: This engine's rank (1-indexed)
        world_size: Total processes

    Returns:
        WeightSyncReceiver attached to engine
    """
    receiver = WeightSyncReceiver(
        master_addr=master_addr,
        master_port=master_port,
        rank=rank,
        world_size=world_size,
        group_name=group_name,
        device=engine.device,
    )
    receiver.init_group()

    # Store receiver on engine for later use
    engine._weight_sync_receiver = receiver

    return receiver


def receive_and_reload_weights(engine: Any) -> None:
    """Receive broadcasted weights and reload into engine.

    Call this after sender.broadcast_weights().
    """
    receiver = getattr(engine, "_weight_sync_receiver", None)
    assert receiver is not None, "Call add_weight_sync_to_engine() first"

    # Get current model's state dict structure
    if engine._use_functional_model:
        template = engine._functional_weights
    else:
        template = engine.model.state_dict()

    # Receive in-place
    receiver.receive_weights_into(template)

    # Reload (handles cache invalidation)
    engine.reload_weights(template)
