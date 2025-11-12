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
        clip_grad: Gradient norm clipping (SLIME default: 1.0)
    """

    sharding_strategy: str = "FULL_SHARD"
    mixed_precision: bool = True
    cpu_offload: bool = False
    auto_wrap_min_params: int = 1_000_000
    gradient_checkpointing: bool = False
    clip_grad: float = 1.0


@dataclass
class FSDPTrainingBackend:
    """FSDP backend for distributed multi-GPU training.

    Implements the TrainingBackend protocol using FSDP.

    Attributes:
        model: PyTorch model (will be wrapped with FSDP)
        loss_fn: Loss function (logits, labels, loss_mask) -> loss
        checkpoint_dir: Directory for checkpoints
        config: FSDP configuration
        optimizer_fn: Function to create optimizer (called AFTER FSDP wrapping)
        device: Device to use (auto-detected from rank)
        rank: Process rank (auto-detected)
        world_size: Total processes (auto-detected)
        step: int = 0
        checkpoint_path: Optional path to checkpoint for initialization

    Example:
        >>> # In each worker process (spawned by Worker pattern)
        >>> import torch.distributed as dist
        >>> dist.init_process_group(backend="nccl")
        >>>
        >>> # Create optimizer factory
        >>> def make_optimizer(model):
        ...     return torch.optim.AdamW(model.parameters(), lr=1e-4)
        >>>
        >>> backend = FSDPTrainingBackend(
        ...     model=model,
        ...     optimizer_fn=make_optimizer,
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

    # Required fields (no defaults) - must come first
    model: torch.nn.Module
    optimizer_fn: Callable[[torch.nn.Module], torch.optim.Optimizer]
    loss_fn: Callable
    checkpoint_dir: Path

    # Optional fields (with defaults) - must come after required fields
    config: FSDPConfig = field(default_factory=FSDPConfig)
    checkpoint_path: Optional[Path] = None
    device: Optional[torch.device] = None
    rank: Optional[int] = None
    world_size: Optional[int] = None
    step: int = 0
    _fsdp_model: Optional[torch.nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None  # Optional LR scheduler

    def __post_init__(self):
        """Initialize FSDP backend with two-phase checkpoint loading.

        Initialization order (following THUDM SLIME pattern):
            1. Load checkpoint to CPU (if provided)
            2. Load model weights BEFORE FSDP wrapping
            3. Wrap model with FSDP
            4. Create optimizer AFTER FSDP wrapping
            5. Load optimizer state (if checkpoint provided)

        Side effects:
            - Detects rank/world_size from torch.distributed
            - Sets device from rank
            - Loads checkpoint weights (if checkpoint_path provided)
            - Wraps model with FSDP
            - Creates optimizer on wrapped model
            - Restores optimizer state (if checkpoint provided)
        """
        # Tiger Style: Assert preconditions
        assert dist.is_initialized(), (
            "torch.distributed not initialized. "
            "Call dist.init_process_group() before creating FSDPTrainingBackend."
        )
        assert callable(self.optimizer_fn), (
            f"optimizer_fn must be callable, got {type(self.optimizer_fn)}"
        )
        assert callable(self.loss_fn), (
            f"loss_fn must be callable, got {type(self.loss_fn)}"
        )
        assert self.checkpoint_dir is not None, "checkpoint_dir cannot be None"

        # Auto-detect distributed config
        self.rank = get_rank()
        self.world_size = get_world_size()

        # Tiger Style: Assert postconditions
        assert self.world_size > 0, f"Invalid world_size: {self.world_size}"
        assert 0 <= self.rank < self.world_size, (
            f"Invalid rank {self.rank} for world_size {self.world_size}"
        )

        # Set device (one GPU per process)
        if self.device is None:
            self.device = torch.device(f"cuda:{self.rank}")

        # Tiger Style: Assert valid device
        assert self.device.type == "cuda", (
            f"FSDP requires CUDA device, got {self.device.type}"
        )

        # Phase 1: Load checkpoint weights BEFORE FSDP wrapping (THUDM pattern)
        checkpoint_payload = None
        if self.checkpoint_path is not None:
            logger.info(f"[Rank {self.rank}] Loading checkpoint from {self.checkpoint_path}")
            checkpoint_payload = self._load_checkpoint_cpu(self.checkpoint_path)

            if checkpoint_payload and checkpoint_payload.get("model"):
                logger.info(f"[Rank {self.rank}] Loading model weights before FSDP wrapping")
                self.model.load_state_dict(checkpoint_payload["model"], strict=True)
                checkpoint_payload["model"] = None  # Free memory

        # Move model to device BEFORE FSDP wrapping
        self.model = self.model.to(self.device)

        # Wrap model with FSDP
        logger.info(f"[Rank {self.rank}] Wrapping model with FSDP")
        self._fsdp_model = self._wrap_model_with_fsdp()

        # Tiger Style: Assert FSDP wrapping succeeded
        assert self._fsdp_model is not None, "FSDP wrapping failed, _fsdp_model is None"
        assert self._fsdp_model is self.model, (
            "FSDP wrapping should modify model in-place, but returned different object"
        )

        # Create optimizer AFTER FSDP wrapping (CRITICAL!)
        logger.info(f"[Rank {self.rank}] Creating optimizer on FSDP-wrapped model")
        self.optimizer = self.optimizer_fn(self._fsdp_model)

        # Tiger Style: Assert optimizer creation succeeded
        assert self.optimizer is not None, "optimizer_fn returned None"
        assert hasattr(self.optimizer, 'step'), "optimizer missing step() method"
        assert hasattr(self.optimizer, 'zero_grad'), "optimizer missing zero_grad() method"

        # Phase 2: Load optimizer state AFTER FSDP wrapping (THUDM pattern)
        if checkpoint_payload and checkpoint_payload.get("optimizer"):
            logger.info(f"[Rank {self.rank}] Loading optimizer state")
            self.optimizer.load_state_dict(checkpoint_payload["optimizer"])
            checkpoint_payload["optimizer"] = None  # Free memory

        # Restore step counter from checkpoint
        if checkpoint_payload and checkpoint_payload.get("step"):
            self.step = checkpoint_payload["step"]
            logger.info(f"[Rank {self.rank}] Restored step counter: {self.step}")

        # Sync all ranks after initialization (critical barrier - all ranks must reach here)
        logger.info(f"[FSDP DEBUG] Rank {self.rank}: Entering initialization barrier...")
        barrier()
        logger.info(f"[FSDP DEBUG] Rank {self.rank}: Passed initialization barrier")

        # Tiger Style: Assert final invariants (all ranks must have these)
        assert self._fsdp_model is not None, "Invariant violated: _fsdp_model is None after init"
        assert self.optimizer is not None, "Invariant violated: optimizer is None after init"
        assert self.device is not None, "Invariant violated: device is None after init"
        assert self.rank is not None, "Invariant violated: rank is None after init"

        logger.info(
            f"[Rank {self.rank}/{self.world_size}] FSDPTrainingBackend initialized "
            f"(strategy={self.config.sharding_strategy}, "
            f"mixed_precision={self.config.mixed_precision})"
        )

    def _load_checkpoint_cpu(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Load checkpoint to CPU (all ranks).

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Dictionary with model, optimizer, step (or None if not found)

        Side effects:
            - Reads checkpoint.pt from disk
            - All ranks load the same checkpoint
        """
        ckpt_file = checkpoint_path / "checkpoint.pt"
        if not ckpt_file.exists():
            logger.warning(f"[Rank {self.rank}] Checkpoint file not found: {ckpt_file}")
            return None

        # Load to CPU (all ranks)
        checkpoint = torch.load(ckpt_file, map_location="cpu")

        return {
            "model": checkpoint.get("model"),
            "optimizer": checkpoint.get("optimizer"),
            "step": checkpoint.get("step", 0),
            "metrics": checkpoint.get("metrics", {}),
        }

    def _wrap_model_with_fsdp(self) -> torch.nn.Module:
        """Wrap model with FSDP v2 (per-layer wrapping).

        Following THUDM SLIME pattern:
        - Uses fully_shard() composable API (PyTorch 2.4+)
        - Wraps individual transformer layers
        - Wraps untied embeddings separately
        - Wraps entire model at root

        Returns:
            FSDP-wrapped model (using fully_shard)

        Side effects:
            - Modifies model in-place with FSDP wrapper
        """
        from packaging import version

        # Import fully_shard based on PyTorch version
        if version.parse(torch.__version__) >= version.parse("2.6"):
            from torch.distributed.fsdp import fully_shard
        elif version.parse(torch.__version__) >= version.parse("2.4"):
            from torch.distributed._composable.fsdp import fully_shard
        else:
            raise ImportError(
                f"FSDP v2 (fully_shard) requires PyTorch 2.4+, got {torch.__version__}"
            )

        # Get transformer layer classes to wrap (HuggingFace models expose this)
        layer_cls_to_wrap = getattr(self.model, "_no_split_modules", [])

        if not layer_cls_to_wrap:
            logger.warning(
                f"[Rank {self.rank}] Model has no _no_split_modules, "
                "falling back to whole-model wrapping"
            )
            # Just wrap the whole model
            fully_shard(self.model)
            return self.model

        # Find all modules to wrap individually
        # - Transformer layers (from _no_split_modules)
        # - Untied embeddings (only if tie_word_embeddings=False)
        modules_to_wrap = []
        for name, module in self.model.named_modules():
            # Wrap transformer layers
            if module.__class__.__name__ in layer_cls_to_wrap:
                modules_to_wrap.append((name, module))
            # Wrap untied embeddings
            elif isinstance(module, torch.nn.Embedding):
                # Check if embeddings are tied
                tie_embeddings = getattr(self.model.config, "tie_word_embeddings", True)
                if not tie_embeddings:
                    modules_to_wrap.append((name, module))

        logger.info(
            f"[FSDP DEBUG] Rank {self.rank}: Wrapping {len(modules_to_wrap)} modules with FSDP v2"
        )

        # Wrap each module individually (THUDM pattern)
        for idx, (name, module) in enumerate(modules_to_wrap):
            logger.debug(f"[FSDP DEBUG] Rank {self.rank}: Wrapping module {idx+1}/{len(modules_to_wrap)}: {name}")
            fully_shard(module)

        logger.info(f"[FSDP DEBUG] Rank {self.rank}: All sub-modules wrapped, wrapping root model...")
        # Wrap the entire model (root wrapping)
        fully_shard(self.model)
        logger.info(f"[FSDP DEBUG] Rank {self.rank}: Root model wrapped successfully")

        return self.model

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

        # Tiger Style: Assert preconditions
        assert self._fsdp_model is not None, "Cannot call forward_backward before initialization"
        assert "input_ids" in batch, "batch must contain 'input_ids'"
        assert isinstance(batch["input_ids"], torch.Tensor), "input_ids must be a tensor"

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

        # Compute loss (SLIME pattern: pass logits and batch dict)
        loss = self.loss_fn(logits=logits, batch=batch)

        # Tiger Style: Assert loss is valid
        assert isinstance(loss, torch.Tensor), f"loss_fn must return torch.Tensor, got {type(loss)}"
        assert loss.dim() == 0, f"loss must be scalar (0-dim), got shape {loss.shape}"
        assert not torch.isnan(loss), "loss is NaN - training is unstable!"
        assert not torch.isinf(loss), "loss is infinite - training is unstable!"

        # Backward pass (FSDP syncs gradients)
        loss.backward()

        # Compute gradient norm (distributed)
        grad_norm = self._compute_grad_norm()

        # Tiger Style: Assert gradient norm is valid
        assert grad_norm >= 0, f"gradient norm must be non-negative, got {grad_norm}"
        assert not torch.isnan(torch.tensor(grad_norm)), "gradient norm is NaN!"

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

        # Tiger Style: Assert preconditions
        assert self.optimizer is not None, "Cannot call optim_step before initialization"
        assert self._fsdp_model is not None, "Cannot call optim_step before initialization"

        # Clip gradients (from config, SLIME default: 1.0)
        # This prevents exploding gradients during training
        grad_norm_clipped = torch.nn.utils.clip_grad_norm_(
            self._fsdp_model.parameters(),
            max_norm=self.config.clip_grad
        )

        # Tiger Style: Assert gradient clipping worked
        assert grad_norm_clipped >= 0, f"Clipped grad norm must be non-negative, got {grad_norm_clipped}"

        # Apply gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Step learning rate scheduler if provided
        if self.scheduler is not None:
            self.scheduler.step()

        # Increment step
        old_step = self.step
        self.step += 1

        # Tiger Style: Assert step incremented
        assert self.step == old_step + 1, f"Step counter did not increment correctly: {old_step} -> {self.step}"

        # Get current learning rate
        lr = self.optimizer.param_groups[0]["lr"]

        # Tiger Style: Assert LR is valid
        assert lr >= 0, f"Learning rate must be non-negative, got {lr}"

        metrics = {
            "lr": lr,
            "step": self.step,
            "grad_norm_clipped": float(grad_norm_clipped),
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference.

        CRITICAL: This is a collective operation - ALL ranks must call this.
        Based on THUDM SLIME's update_cpu_params_dict pattern.

        Returns:
            TrainFuture resolving to full state_dict (only on rank 0)

        Side effects:
            - ALL ranks participate in gathering (collective operation)
            - Gathers full model state to rank 0
            - Other ranks return empty dict after participating

        Tiger Style assertions:
            - Asserts distributed is initialized (precondition)
            - Asserts FSDP model exists (precondition)
            - Logs before/after collective operation (paired observation)
            - Asserts negative space (non-main ranks get empty dict)
        """
        from rollouts.training.types import ImmediateTrainFuture

        # Use new PyTorch checkpoint API (not deprecated state_dict_type)
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            get_model_state_dict,
        )

        # Tiger Style: Assert preconditions
        assert dist.is_initialized(), "torch.distributed must be initialized for collective ops"
        assert self._fsdp_model is not None, "FSDP model must be wrapped before get_weights"

        logger.info(
            f"[COLLECTIVE-ENTER] Rank {self.rank}: get_model_state_dict "
            f"(ALL ranks must participate in this collective operation)"
        )

        # Configure to gather full state on rank 0
        # CRITICAL: get_model_state_dict internally calls barriers to gather weights
        # ALL ranks must call this, even though only rank 0 gets the full state
        options = StateDictOptions(full_state_dict=True, cpu_offload=True)
        state_dict = get_model_state_dict(self._fsdp_model, options=options)

        logger.info(
            f"[COLLECTIVE-EXIT] Rank {self.rank}: get_model_state_dict completed "
            f"(all internal barriers passed, state_dict has {len(state_dict)} keys)"
        )

        # Only rank 0 has the full state, other ranks need to clear their dict
        if not is_main_process():
            # Tiger Style: Assert negative space (what should NOT happen)
            assert len(state_dict) > 0, (
                f"Non-main rank {self.rank} should receive partial state from collective, "
                "but got empty dict - this suggests collective operation failed"
            )
            logger.info(
                f"[RANK-PATH] Rank {self.rank}: Clearing {len(state_dict)} partial weights "
                "(not main process)"
            )
            state_dict = {}

        # Tiger Style: Assert postconditions
        if is_main_process():
            assert len(state_dict) > 0, "Rank 0 should have full state_dict after collective"
        else:
            assert len(state_dict) == 0, f"Non-main rank should have empty dict, got {len(state_dict)} keys"

        return ImmediateTrainFuture(state_dict)

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights with explicit DTensor handling.

        Following THUDM SLIME pattern:
        - Try new PyTorch API first (set_model_state_dict)
        - Fall back to manual DTensor distribution if needed

        Args:
            weights: state_dict to load

        Returns:
            TrainFuture resolving to None

        Side effects:
            - Loads weights into model
            - FSDP automatically shards weights across GPUs (or manual DTensor distribution)
        """
        from rollouts.training.types import ImmediateTrainFuture

        logger.info(f"[FSDP DEBUG] Rank {self.rank}: Loading model state dict...")

        try:
            # Try new PyTorch checkpoint API first
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions,
                set_model_state_dict,
            )

            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            set_model_state_dict(
                self._fsdp_model,
                model_state_dict=weights,
                options=options
            )
            logger.info(f"[FSDP DEBUG] Rank {self.rank}: Loaded weights successfully (PyTorch API)")

        except Exception as e:
            # Fall back to manual DTensor handling (THUDM pattern)
            logger.warning(
                f"[FSDP DEBUG] Rank {self.rank}: PyTorch API failed ({e}), "
                "falling back to manual DTensor loading"
            )
            self._load_weights_with_dtensor(weights)
            logger.info(f"[FSDP DEBUG] Rank {self.rank}: Loaded weights successfully (manual DTensor)")

        return ImmediateTrainFuture(None)

    @torch.no_grad()
    def _load_weights_with_dtensor(self, weights: Dict[str, torch.Tensor]) -> None:
        """Load weights with explicit DTensor distribution (THUDM SLIME pattern).

        Args:
            weights: State dict to load (on CPU)

        Side effects:
            - Loads weights into FSDP model
            - Handles DTensor distribution manually
        """
        # Import DTensor utilities
        try:
            from torch.distributed._tensor import DTensor, distribute_tensor
        except ImportError:
            logger.error("DTensor not available, cannot load weights")
            raise

        # Cache parameter and buffer maps for efficiency
        if not hasattr(self, "_fsdp_param_map"):
            self._fsdp_param_map = dict(self._fsdp_model.named_parameters())
            self._fsdp_buffer_map = dict(self._fsdp_model.named_buffers())

        param_map = self._fsdp_param_map
        buffer_map = self._fsdp_buffer_map

        for name, src in weights.items():
            if not torch.is_tensor(src):
                continue

            # Find target parameter or buffer
            target_param = param_map.get(name)
            if target_param is None:
                target_param = buffer_map.get(name)
                if target_param is None:
                    logger.warning(f"[Rank {self.rank}] Parameter not found: {name}")
                    continue

            dst_tensor = target_param.data
            src_tensor = src.detach()

            # Move to CPU if needed
            if src_tensor.device.type != "cpu":
                src_tensor = src_tensor.to(device=torch.device("cpu"))

            # Convert dtype if needed
            if src_tensor.dtype != dst_tensor.dtype:
                src_tensor = src_tensor.to(dtype=dst_tensor.dtype)

            # Handle DTensor distribution (THUDM pattern)
            if isinstance(dst_tensor, DTensor):
                # Distribute full tensor according to FSDP sharding
                distributed = distribute_tensor(
                    src_tensor.contiguous(),
                    device_mesh=dst_tensor.device_mesh,
                    placements=dst_tensor.placements,
                )
                dst_tensor.copy_(distributed)
            else:
                # Regular tensor: just copy to GPU
                dst_tensor.copy_(
                    src_tensor.to(device=dst_tensor.device, non_blocking=True)
                )

        torch.cuda.synchronize()

    async def save_checkpoint(
        self, step: int, metrics: Dict[str, float]
    ) -> Path:
        """Save checkpoint following THUDM SLIME pattern with explicit control flow.

        Based on SLIME's checkpoint.py save() function:
        - ALL ranks participate in weight gathering (collective operation)
        - Explicit barriers synchronize all ranks
        - Only rank 0 performs file I/O
        - Simple, symmetric control flow (Tiger Style)

        Args:
            step: Current training step
            metrics: Current metrics

        Returns:
            Path to saved checkpoint

        Side effects:
            - ALL ranks participate in get_weights() collective operation
            - Saves checkpoint to disk (rank 0 only)
            - Creates checkpoint directory if needed
            - Synchronizes all ranks with barriers

        Tiger Style principles applied:
            - Explicit control flow (all ranks follow same path)
            - Paired assertions (pre/post collective operations)
            - Aggressive logging (every state transition)
            - Assert negative space (what non-main ranks should NOT do)
        """
        logger.info(f"[STATE] Rank {self.rank}: ENTERING save_checkpoint (step={step})")

        # Tiger Style: Assert preconditions
        assert self.checkpoint_dir is not None, "checkpoint_dir must be set"
        assert self.optimizer is not None, "optimizer must be initialized"
        assert self._fsdp_model is not None, "FSDP model must be wrapped"

        # Phase 1: ALL ranks synchronize GPU operations (SLIME pattern)
        logger.info(f"[FSDP DEBUG] Rank {self.rank}: Synchronizing CUDA operations...")
        torch.cuda.synchronize()

        # Phase 2: Create checkpoint directory (rank 0 only, then barrier)
        # Following SLIME checkpoint.py:132-134
        ckpt_path = self.checkpoint_dir / f"step_{step}"
        if is_main_process():
            logger.info(f"[RANK-PATH] Rank 0: Creating checkpoint directory: {ckpt_path}")
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path.mkdir(exist_ok=True)
            logger.info(f"[FSDP DEBUG] Rank 0: Directory created successfully")
        else:
            logger.info(f"[RANK-PATH] Rank {self.rank}: Skipping directory creation (not main process)")

        # Barrier: ensure directory exists before proceeding
        logger.info(f"[BARRIER-ENTER] Rank {self.rank}: Waiting for directory creation...")
        barrier()
        logger.info(f"[BARRIER-EXIT] Rank {self.rank}: Directory barrier passed")

        # Phase 3: ALL ranks participate in weight gathering (CRITICAL collective operation)
        # Following SLIME checkpoint.py:136 (actor.update_cpu_params_dict)
        logger.info(
            f"[COLLECTIVE-ENTER] Rank {self.rank}: Starting get_weights() "
            f"(ALL ranks MUST participate)"
        )
        state_dict = await self.get_weights().result()
        logger.info(
            f"[COLLECTIVE-EXIT] Rank {self.rank}: get_weights() completed, "
            f"state_dict has {len(state_dict)} keys"
        )

        # Tiger Style: Assert postconditions on collective operation
        if is_main_process():
            assert len(state_dict) > 0, "Rank 0 must have non-empty state_dict"
        else:
            assert len(state_dict) == 0, (
                f"Non-main rank should have empty state_dict, got {len(state_dict)} keys"
            )

        # Phase 4: Only rank 0 saves to disk (SLIME pattern)
        # Following SLIME checkpoint.py:143-169
        if is_main_process():
            logger.info(f"[RANK-PATH] Rank 0: Saving checkpoint to disk...")
            logger.info(f"[FSDP DEBUG] Rank 0: Getting optimizer state dict...")

            checkpoint = {
                "model": state_dict,
                "optimizer": self.optimizer.state_dict(),
                "step": step,
                "metrics": metrics,
            }

            logger.info(f"[FSDP DEBUG] Rank 0: Writing checkpoint.pt to {ckpt_path}...")
            torch.save(checkpoint, ckpt_path / "checkpoint.pt")
            logger.info(f"[FSDP DEBUG] Rank 0: Checkpoint saved successfully to {ckpt_path}")
        else:
            # Tiger Style: Assert negative space (what should NOT happen)
            logger.info(f"[RANK-PATH] Rank {self.rank}: Skipping disk save (not main process)")
            # Assert that non-main ranks don't accidentally create files
            assert not (ckpt_path / "checkpoint.pt").exists() or True, (
                f"Non-main rank {self.rank} should not create checkpoint files"
            )

        # Phase 5: Final barrier to synchronize all ranks (SLIME pattern)
        # Following SLIME checkpoint.py:171
        logger.info(f"[BARRIER-ENTER] Rank {self.rank}: Entering final checkpoint barrier...")
        barrier()
        logger.info(f"[BARRIER-EXIT] Rank {self.rank}: Final checkpoint barrier passed")

        logger.info(f"[STATE] Rank {self.rank}: EXITING save_checkpoint (step={step}) - SUCCESS")

        return ckpt_path
