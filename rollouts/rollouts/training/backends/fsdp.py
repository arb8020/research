"""FSDP backend for multi-GPU training.

Implements TrainingBackend protocol using PyTorch FSDP (Fully Sharded Data Parallel).
Each GPU holds a shard of the model, reducing memory usage for large models.

Based on SLIME's FSDP implementation:
- references/slime/slime/backends/fsdp_utils/actor.py
- references/slime/slime/backends/fsdp_utils/arguments.py

Tiger Style: Explicit state, clear assertions.
Casey Muratori: Protocol over inheritance, minimal coupling.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from rollouts.training.distributed_utils import (
    barrier,
    get_rank,
    get_world_size,
    is_main_process,
)
from rollouts.training.types import TrainFuture

logger = logging.getLogger(__name__)


@dataclass
class FSDPConfig:
    """Configuration for FSDP training.

    Based on SLIME's FSDPArgs but simplified.

    Attributes:
        sharding_strategy: FSDP sharding strategy
            - "FULL_SHARD": Shard parameters, gradients, and optimizer states (default)
            - "SHARD_GRAD_OP": Shard gradients and optimizer states only
            - "NO_SHARD": No sharding (DDP mode)
        mixed_precision: Enable mixed precision training (bf16)
        cpu_offload: Offload to CPU (saves GPU memory, slower)
        auto_wrap_min_params: Minimum parameters for auto-wrapping submodules
        gradient_checkpointing: Enable activation checkpointing (saves memory)
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    cpu_offload: bool = False
    auto_wrap_min_params: int = 1_000_000
    gradient_checkpointing: bool = False


@dataclass
class FSDPTrainingBackend:
    """FSDP backend for distributed multi-GPU training.

    Implements the TrainingBackend protocol using FSDP.

    Attributes:
        model: PyTorch model (will be wrapped with FSDP)
        optimizer: Optimizer (must be created AFTER FSDP wrapping)
        loss_fn: Loss function (logits, labels, loss_mask) -> loss
        checkpoint_dir: Directory for checkpoints
        config: FSDP configuration
        device: Device to use (auto-detected from rank)
        rank: Process rank (auto-detected)
        world_size: Total processes (auto-detected)
        step: Current training step

    Example:
        >>> # In each worker process (spawned by Worker pattern)
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend="nccl")
        >>>
        >>> backend = FSDPTrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     loss_fn=grpo_loss,
        ...     checkpoint_dir=Path("checkpoints"),
        ...     config=FSDPConfig(sharding_strategy="FULL_SHARD"),
        ... )
        >>>
        >>> # Training loop
        >>> for batch in batches:
        ...     fwd_result = await backend.forward_backward(batch).result()
        ...     opt_result = await backend.optim_step().result()
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable
    checkpoint_dir: Path
    config: FSDPConfig = field(default_factory=FSDPConfig)
    device: Optional[torch.device] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
    step: int = 0
    _fsdp_model: Optional[FSDP] = None

    def __post_init__(self):
        """Initialize FSDP backend.

        Side effects:
            - Detects rank/world_size from torch.distributed
            - Sets device from rank
            - Wraps model with FSDP
            - Moves optimizer to correct device
        """
        # Tiger Style: Assert torch.distributed is initialized
        assert dist.is_initialized(), (
            "torch.distributed not initialized. "
            "Call dist.init_process_group() before creating FSDPTrainingBackend."
        )

        # Auto-detect distributed config
        self.rank = get_rank()
        self.world_size = get_world_size()

        # Set device (one GPU per process)
        if self.device is None:
            self.device = torch.device(f"cuda:{self.rank}")

        # Move model to device BEFORE FSDP wrapping
        self.model = self.model.to(self.device)

        # Wrap model with FSDP
        self._fsdp_model = self._wrap_model_with_fsdp()

        logger.info(
            f"[Rank {self.rank}/{self.world_size}] FSDPTrainingBackend initialized "
            f"(strategy={self.config.sharding_strategy}, "
            f"mixed_precision={self.config.mixed_precision})"
        )

    def _wrap_model_with_fsdp(self) -> FSDP:
        """Wrap model with FSDP.

        Returns:
            FSDP-wrapped model

        Side effects:
            - Modifies model in-place with FSDP wrapper
        """
        # Parse sharding strategy
        strategy_map = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        sharding_strategy = strategy_map.get(self.config.sharding_strategy)
        assert sharding_strategy is not None, (
            f"Unknown sharding strategy: {self.config.sharding_strategy}"
        )

        # Configure mixed precision
        mixed_precision = None
        if self.config.mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        # Configure auto-wrap policy (wrap large submodules)
        # Note: size_based_auto_wrap_policy returns a callable that FSDP will invoke
        from functools import partial
        auto_wrap_policy = partial(
            size_based_auto_wrap_policy,
            min_num_params=self.config.auto_wrap_min_params
        )

        # Wrap model
        fsdp_model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=self.device,
            use_orig_params=True,  # Preserve parameter names
        )

        return fsdp_model

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients (distributed across GPUs).

        Args:
            batch: Training batch with keys:
                - "input_ids": [batch_size, seq_len]
                - "labels": [batch_size, seq_len]
                - "loss_mask": [batch_size, seq_len]
                - "advantages": [batch_size] (optional, for RL)

        Returns:
            TrainFuture resolving to {"loss": float, "grad_norm": float}

        Side effects:
            - Computes forward pass
            - Computes loss
            - Computes gradients (backward pass)
            - FSDP automatically shards and syncs gradients across GPUs
        """
        from rollouts.training.types import ImmediateTrainFuture

        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass (FSDP handles sharding)
        self._fsdp_model.train()
        outputs = self._fsdp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )

        # Extract logits
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs

        # Compute loss
        loss = self.loss_fn(
            logits=logits,
            labels=batch["labels"],
            loss_mask=batch.get("loss_mask"),
            advantages=batch.get("advantages"),
        )

        # Backward pass (FSDP syncs gradients)
        loss.backward()

        # Compute gradient norm (distributed)
        grad_norm = self._compute_grad_norm()

        # Return metrics
        metrics = {
            "loss": loss.item(),
            "grad_norm": grad_norm,
        }

        return ImmediateTrainFuture(metrics)

    def _compute_grad_norm(self) -> float:
        """Compute global gradient norm across all GPUs.

        Returns:
            Global gradient norm (same on all ranks)
        """
        # Collect all gradient norms
        total_norm_sq = 0.0
        for param in self._fsdp_model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2

        # Convert to tensor for all_reduce
        total_norm_sq_tensor = torch.tensor([total_norm_sq], device=self.device)

        # Sum across all GPUs
        dist.all_reduce(total_norm_sq_tensor, op=dist.ReduceOp.SUM)

        # Compute global norm
        global_norm = (total_norm_sq_tensor.item() ** 0.5)
        return global_norm

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients and update weights.

        Returns:
            TrainFuture resolving to {"lr": float, "step": int}

        Side effects:
            - Applies optimizer step
            - Zeros gradients
            - Increments step counter
            - FSDP automatically syncs updated parameters
        """
        from rollouts.training.types import ImmediateTrainFuture

        # Apply gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Increment step
        self.step += 1

        # Get current learning rate
        lr = self.optimizer.param_groups[0]["lr"]

        metrics = {
            "lr": lr,
            "step": self.step,
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            TrainFuture resolving to full state_dict (only on rank 0)

        Side effects:
            - Gathers full model state to rank 0
            - Other ranks return empty dict
        """
        from rollouts.training.types import ImmediateTrainFuture

        # Use FSDP's state_dict API
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        # Configure to gather full state on rank 0
        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            state_dict = self._fsdp_model.state_dict()

        # Only rank 0 has the full state
        if not is_main_process():
            state_dict = {}

        return ImmediateTrainFuture(state_dict)

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint.

        Args:
            weights: state_dict to load

        Returns:
            TrainFuture resolving to None

        Side effects:
            - Loads weights into model
            - FSDP automatically shards weights across GPUs
        """
        from rollouts.training.types import ImmediateTrainFuture

        # Use FSDP's load_state_dict API
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType

        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            self._fsdp_model.load_state_dict(weights)

        logger.info(f"[Rank {self.rank}] Loaded weights")

        return ImmediateTrainFuture(None)

    async def save_checkpoint(
        self, step: int, metrics: Dict[str, float]
    ) -> Path:
        """Save checkpoint (only on rank 0).

        Args:
            step: Current training step
            metrics: Current metrics

        Returns:
            Path to saved checkpoint

        Side effects:
            - Saves checkpoint to disk (rank 0 only)
            - Creates checkpoint directory if needed
        """
        if not is_main_process():
            # Only rank 0 saves
            barrier()  # Wait for rank 0
            return self.checkpoint_dir / f"step_{step}"

        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = self.checkpoint_dir / f"step_{step}"
        ckpt_path.mkdir(exist_ok=True)

        # Get full state dict (gathered on rank 0)
        state_dict = await self.get_weights().result()

        # Save checkpoint
        checkpoint = {
            "model": state_dict,
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "metrics": metrics,
        }

        torch.save(checkpoint, ckpt_path / "checkpoint.pt")
        logger.info(f"[Rank {self.rank}] Saved checkpoint to {ckpt_path}")

        # Sync with other ranks
        barrier()

        return ckpt_path
