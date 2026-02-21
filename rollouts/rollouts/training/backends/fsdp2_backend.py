"""FSDP2 training backend adapter.

Wraps PyTorch's FSDP2 (Fully Sharded Data Parallel) for distributed training.

FSDP2 (torch.distributed.fsdp2) provides:
- Per-parameter sharding with DTensor
- Mixed precision training
- Activation checkpointing
- CPU offloading

This is the recommended backend for models up to ~30B on 8xH100.

Based on SLIME's fsdp_utils implementation.
Reference: https://github.com/THUDM/slime

Usage:
    backend = FSDP2TrainingBackend(
        model=fsdp_wrapped_model,
        optimizer=optimizer,
        config=FSDP2Config(),
        checkpoint_dir=Path("/checkpoints"),
    )
    metrics = await backend.forward_backward(batch).result()
    step_metrics = await backend.optim_step().result()
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...training.types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class FSDP2Config:
    """Configuration for FSDP2 training backend.

    Attributes:
        sharding_strategy: How to shard model
            - "FULL_SHARD": Shard params, grads, optimizer states (most memory efficient)
            - "SHARD_GRAD_OP": Shard grads and optimizer states only
            - "NO_SHARD": No sharding (DDP mode)
        mixed_precision: Enable mixed precision training
        compute_dtype: Dtype for forward/backward ("bfloat16", "float16")
        reduce_dtype: Dtype for gradient reduction ("float32", "bfloat16")
        cpu_offload: Offload params/grads to CPU
        use_orig_params: Use original parameter containers (required for some optimizers)
        clip_grad: Gradient clipping norm (0 = disabled)
        gradient_checkpointing: Enable activation checkpointing
        forward_prefetch: Prefetch next FSDP unit during forward
        backward_prefetch: Prefetch next FSDP unit during backward
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    compute_dtype: str = "bfloat16"
    reduce_dtype: str = "float32"
    cpu_offload: bool = False
    use_orig_params: bool = True
    clip_grad: float = 1.0
    gradient_checkpointing: bool = False
    forward_prefetch: bool = True
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE or BACKWARD_POST


@dataclass
class FSDP2TrainingBackend:
    """FSDP2 training backend for distributed training.

    Implements TrainingBackend protocol using PyTorch FSDP2.

    FSDP2 automatically shards model parameters across GPUs,
    gathering them only when needed for computation.

    Attributes:
        model: FSDP-wrapped PyTorch model
        optimizer: Optimizer instance
        loss_fn: Loss function (model_output, batch) -> loss
        config: FSDP2Config with sharding settings
        checkpoint_dir: Directory for checkpoints
        scheduler: Optional learning rate scheduler

    Example:
        >>> from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        >>> model = FSDP(model, **fsdp_config)
        >>> optimizer = torch.optim.AdamW(model.parameters())
        >>>
        >>> backend = FSDP2TrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=my_loss_fn,
        ...     config=FSDP2Config(),
        ...     checkpoint_dir=Path("/checkpoints"),
        ... )
    """

    model: Any  # FSDP wrapped nn.Module
    optimizer: Any  # torch.optim.Optimizer
    loss_fn: Callable[[Any, dict[str, Any]], Any]
    config: FSDP2Config = field(default_factory=FSDP2Config)
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    scheduler: Any | None = None

    # Internal state
    _step: int = field(default=0, init=False)
    _grad_scaler: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize gradient scaler for mixed precision if needed."""
        import torch

        if self.config.mixed_precision and self.config.compute_dtype == "float16":
            # FP16 requires gradient scaling
            self._grad_scaler = torch.amp.GradScaler("cuda")
        else:
            # BF16 doesn't need gradient scaling
            self._grad_scaler = None

    def forward_backward(self, batch: dict[str, Any]) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients using FSDP.

        Args:
            batch: {
                "input_ids": Tensor [batch, seq_len],
                "labels": Tensor [batch, seq_len],
                "attention_mask": Tensor [batch, seq_len],
                "advantages": Optional Tensor [batch] for RL,
            }

        Returns:
            Future resolving to {"loss": float, ...}
        """
        import torch

        # Determine compute dtype
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        compute_dtype = dtype_map.get(self.config.compute_dtype, torch.bfloat16)

        # Zero gradients
        self.optimizer.zero_grad()

        # Forward pass with autocast
        with torch.amp.autocast("cuda", dtype=compute_dtype, enabled=self.config.mixed_precision):
            # Get model outputs
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch.get("labels"),
            )

            # Compute loss
            loss = self.loss_fn(outputs, batch)

            # Apply advantage weighting for RL
            advantages = batch.get("advantages")
            if advantages is not None:
                loss = loss * advantages.mean()

        # Backward pass
        if self._grad_scaler is not None:
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

        metrics = {
            "loss": float(loss.detach()),
        }

        return ImmediateTrainFuture(metrics)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients with gradient clipping.

        Returns:
            Future resolving to {"lr": float, "step": int, "grad_norm": float, ...}
        """
        import torch.nn.utils as nn_utils

        # Unscale gradients for clipping
        if self._grad_scaler is not None:
            self._grad_scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = 0.0
        if self.config.clip_grad > 0:
            grad_norm = nn_utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad)
            grad_norm = float(grad_norm)

        # Optimizer step
        if self._grad_scaler is not None:
            self._grad_scaler.step(self.optimizer)
            self._grad_scaler.update()
        else:
            self.optimizer.step()

        # Scheduler step
        lr = 0.0
        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
        else:
            lr = self.optimizer.param_groups[0].get("lr", 0.0)

        self._step += 1

        metrics = {
            "step": self._step,
            "lr": float(lr),
            "grad_norm": grad_norm,
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing to inference.

        Uses FSDP's full_state_dict to gather sharded params.

        Returns:
            Future resolving to state_dict
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            # Use full state dict (gathers all shards)
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
            ):
                state_dict = self.model.state_dict()

            return ImmediateTrainFuture(state_dict)
        except ImportError:
            # Fallback for non-FSDP models
            return ImmediateTrainFuture(self.model.state_dict())

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights.

        Args:
            weights: state_dict to load
        """
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
            ):
                self.model.load_state_dict(weights)
        except ImportError:
            self.model.load_state_dict(weights)

        return ImmediateTrainFuture(None)

    def save_checkpoint(self, step: int) -> TrainFuture[Path]:
        """Save checkpoint using FSDP's distributed checkpointing.

        Args:
            step: Current training step

        Returns:
            Future resolving to checkpoint path
        """
        import torch
        import torch.distributed as dist

        ckpt_path = self.checkpoint_dir / f"step_{step}"

        try:
            from torch.distributed.checkpoint import save
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            # Use sharded state dict for efficient distributed save
            with FSDP.state_dict_type(
                self.model,
                StateDictType.SHARDED_STATE_DICT,
            ):
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": step,
                }
                if self.scheduler is not None:
                    state_dict["scheduler"] = self.scheduler.state_dict()

            ckpt_path.mkdir(parents=True, exist_ok=True)
            save(state_dict, checkpoint_id=str(ckpt_path))

        except ImportError:
            # Fallback to simple torch.save (rank 0 only)
            if not dist.is_initialized() or dist.get_rank() == 0:
                ckpt_path.mkdir(parents=True, exist_ok=True)
                state_dict = self.get_weights().result()
                torch.save(
                    {
                        "model": state_dict,
                        "optimizer": self.optimizer.state_dict(),
                        "step": step,
                    },
                    ckpt_path / "checkpoint.pt",
                )

        return ImmediateTrainFuture(ckpt_path)

    @staticmethod
    def wrap_model(
        model: nn.Module,
        config: FSDP2Config,
    ) -> Any:
        """Wrap a model with FSDP.

        Utility function to create FSDP-wrapped model from config.

        Args:
            model: PyTorch model to wrap
            config: FSDP2Config with wrapping settings

        Returns:
            FSDP-wrapped model
        """
        import torch
        from torch.distributed.fsdp import (
            BackwardPrefetch,
            CPUOffload,
            MixedPrecision,
            ShardingStrategy,
        )
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
        )

        # Map config to FSDP enums
        sharding_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = sharding_map.get(config.sharding_strategy, ShardingStrategy.FULL_SHARD)

        backward_prefetch_map = {
            "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
            "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
        }
        backward_prefetch = backward_prefetch_map.get(
            config.backward_prefetch, BackwardPrefetch.BACKWARD_PRE
        )

        # Mixed precision config
        mixed_precision = None
        if config.mixed_precision:
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            compute_dtype = dtype_map.get(config.compute_dtype, torch.bfloat16)
            reduce_dtype = dtype_map.get(config.reduce_dtype, torch.float32)

            mixed_precision = MixedPrecision(
                param_dtype=compute_dtype,
                reduce_dtype=reduce_dtype,
                buffer_dtype=compute_dtype,
            )

        # CPU offload
        cpu_offload = CPUOffload(offload_params=config.cpu_offload) if config.cpu_offload else None

        # Wrap model
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            forward_prefetch=config.forward_prefetch,
            use_orig_params=config.use_orig_params,
        )

        return fsdp_model
