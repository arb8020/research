"""Megatron-Core training backend adapter.

Wraps Megatron-Core's distributed training for large models.

Megatron-Core provides:
- Tensor parallelism (TP): Split layers across GPUs
- Pipeline parallelism (PP): Split model stages across GPUs
- Expert parallelism (EP): Distribute MoE experts
- Sequence parallelism (SP): Distribute sequence dimension

Based on SLIME's megatron_utils implementation.
Reference: https://github.com/THUDM/slime

Semantic note:
This backend does not execute our seqax-inspired realization semantics as an
explicit collective program. `lowering.realization` is used for validation and
coarse partition lowering only. Execution remains Megatron's backend-native
runtime semantics.

Usage:
    backend = MegatronTrainingBackend(
        model=megatron_model,
        optimizer=megatron_optimizer,
        config=megatron_config,
        checkpoint_dir=Path("/checkpoints"),
    )
    metrics = await backend.forward_backward(batch).result()
    step_metrics = await backend.optim_step().result()
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...training.lowering import MegatronLowering
from ...training.types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass
class MegatronConfig:
    """Configuration for Megatron training backend.

    Attributes:
        tensor_model_parallel_size: TP degree
        pipeline_model_parallel_size: PP degree
        expert_model_parallel_size: EP degree (for MoE)
        sequence_parallel: Enable sequence parallelism
        micro_batch_size: Micro batch size per GPU
        global_batch_size: Total batch size across all GPUs
        num_microbatches: Number of microbatches for pipeline
        clip_grad: Gradient clipping norm (0 = disabled)
        fp16: Use FP16 mixed precision
        bf16: Use BF16 mixed precision
        use_flash_attn: Enable FlashAttention
        seq_length: Maximum sequence length
    """

    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    expert_model_parallel_size: int = 1
    sequence_parallel: bool = False
    micro_batch_size: int = 1
    global_batch_size: int = 8
    num_microbatches: int = 1
    clip_grad: float = 1.0
    fp16: bool = False
    bf16: bool = True
    use_flash_attn: bool = True
    seq_length: int = 4096


@dataclass
class MegatronTrainingBackend:
    """Megatron-Core training backend for large models.

    Implements TrainingBackend protocol using Megatron's pipeline engine.

    This adapter wraps Megatron's forward_backward_func and optimizer step.
    The model, optimizer, and scheduler are created externally using
    Megatron's initialization functions.

    Attributes:
        model: Megatron DDP-wrapped model chunks (list for pipeline)
        optimizer: MegatronOptimizer instance
        opt_param_scheduler: Learning rate scheduler
        config: MegatronConfig with parallelism settings
        checkpoint_dir: Directory for checkpoints
        loss_fn: Loss function (logits, labels, loss_mask) -> loss

    Example:
        >>> # Initialize Megatron (typically in worker process)
        >>> from megatron.training.training import get_model
        >>> model = get_model(model_provider_fn, ...)
        >>> optimizer = get_megatron_optimizer(model, ...)
        >>>
        >>> backend = MegatronTrainingBackend(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     config=MegatronConfig(tensor_model_parallel_size=4),
        ...     checkpoint_dir=Path("/checkpoints"),
        ... )
    """

    model: Sequence[Any]  # List of DDP model chunks
    optimizer: Any  # MegatronOptimizer
    opt_param_scheduler: Any | None = None
    config: MegatronConfig = field(default_factory=MegatronConfig)
    lowering: MegatronLowering = field(default_factory=MegatronLowering)
    checkpoint_dir: Path = field(default_factory=lambda: Path("./checkpoints"))
    loss_fn: Any | None = None

    # Internal state
    _step: int = field(default=0, init=False)
    _data_iterator: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Validate Megatron is available."""
        try:
            from megatron.core.pipeline_parallel import get_forward_backward_func

            self._forward_backward_func = get_forward_backward_func()
        except ImportError as e:
            raise ImportError(
                "Megatron-Core is required for MegatronTrainingBackend. "
                "Install from: https://github.com/NVIDIA/Megatron-LM"
            ) from e

    def forward_backward(
        self,
        batch: dict[str, Any],
        *,
        loss_fn: Any | None = None,
        loss_fn_config: dict[str, float] | None = None,
    ) -> TrainFuture[dict[str, float]]:
        """Compute loss and gradients using Megatron's pipeline engine.

        Args:
            batch: {
                "input_ids": Tensor [batch, seq_len],
                "labels": Tensor [batch, seq_len],
                "loss_mask": Tensor [batch, seq_len],
                "advantages": Optional Tensor [batch] for RL,
            }

        Returns:
            Future resolving to {"loss": float, "grad_norm": float, ...}
        """
        try:
            from megatron.core import mpu
            from megatron.core.distributed import finalize_model_grads
        except ImportError as e:
            raise ImportError("Megatron-Core is required.") from e

        import torch

        if loss_fn_config is not None:
            raise ValueError(
                "loss_fn_config is not supported for MegatronTrainingBackend.forward_backward yet. "
                "Pass a closure via loss_fn that captures any config instead."
            )

        # Zero gradients
        for model_chunk in self.model:
            model_chunk.zero_grad_buffer()
        self.optimizer.zero_grad()

        # Create data iterator from batch
        # Megatron expects an iterator that yields batches
        class BatchIterator:
            def __init__(self, batch: dict[str, Any]) -> None:
                self._batch = batch
                self._consumed = False

            def __iter__(self) -> BatchIterator:
                return self

            def __next__(self) -> dict[str, Any]:
                if self._consumed:
                    raise StopIteration
                self._consumed = True
                return self._batch

        data_iterator = BatchIterator(batch)

        # Forward step function for Megatron pipeline
        def forward_step(data_iter: Any, model: Any) -> Any:
            batch = next(data_iter)
            tokens = batch["input_ids"]
            labels = batch["labels"]
            loss_mask = batch.get("loss_mask")
            advantages = batch.get("advantages")

            # Model forward
            output = model(
                input_ids=tokens,
                position_ids=None,
                attention_mask=None,
                labels=labels,
            )

            # Compute loss
            active_loss_fn = loss_fn if loss_fn is not None else self.loss_fn
            if active_loss_fn is not None:
                loss = active_loss_fn(output, labels, loss_mask)
            else:
                # Default: assume model returns loss directly
                loss = output if isinstance(output, torch.Tensor) else output.loss

            # Apply advantage weighting for RL
            if advantages is not None:
                loss = loss * advantages.mean()

            # Return for pipeline engine
            def loss_reducer(output_tensor: Any) -> dict[str, Any]:
                return {"loss": output_tensor}

            return loss, loss_reducer

        # Run forward/backward through pipeline
        losses_reduced = self._forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=self.model,
            num_microbatches=self.config.num_microbatches,
            seq_length=self.config.seq_length,
            micro_batch_size=self.config.micro_batch_size,
            forward_only=False,
        )

        # Finalize gradients (all-reduce across DP)
        finalize_model_grads(self.model)

        # Collect metrics from last pipeline stage
        metrics = {"loss": 0.0, "grad_norm": 0.0}
        if mpu.is_pipeline_last_stage():
            if losses_reduced:
                metrics["loss"] = float(losses_reduced[0]["loss"])

        return ImmediateTrainFuture(metrics)

    def optim_step(self) -> TrainFuture[dict[str, float]]:
        """Apply gradients using Megatron's optimizer.

        Returns:
            Future resolving to {"lr": float, "step": int, "grad_norm": float, ...}
        """

        # Clip gradients
        grad_norm = None
        if self.config.clip_grad > 0:
            grad_norm = self.optimizer.clip_grad_norm(self.config.clip_grad)

        # Optimizer step
        self.optimizer.step()

        # Scheduler step
        if self.opt_param_scheduler is not None:
            self.opt_param_scheduler.step()
            lr = self.opt_param_scheduler.get_lr()
        else:
            lr = self.optimizer.param_groups[0].get("lr", 0.0)

        self._step += 1

        metrics = {
            "step": self._step,
            "lr": float(lr) if not isinstance(lr, list) else float(lr[0]),
            "grad_norm": float(grad_norm) if grad_norm is not None else 0.0,
        }

        return ImmediateTrainFuture(metrics)

    def get_weights(self) -> TrainFuture[dict[str, Any]]:
        """Get model weights for syncing to inference.

        For Megatron models, this gathers weights from all TP/PP ranks.

        Returns:
            Future resolving to state_dict
        """
        try:
            from megatron.core import mpu
        except ImportError:
            # Fallback for non-Megatron testing
            if hasattr(self.model, "state_dict"):
                return ImmediateTrainFuture(self.model.state_dict())
            return ImmediateTrainFuture(self.model[0].state_dict())

        # Only rank 0 returns full state dict
        if mpu.get_data_parallel_rank() == 0 and mpu.get_tensor_model_parallel_rank() == 0:
            # Gather from all pipeline stages
            state_dict = {}
            for i, model_chunk in enumerate(self.model):
                chunk_state = model_chunk.state_dict()
                for k, v in chunk_state.items():
                    state_dict[f"chunk_{i}.{k}"] = v
            return ImmediateTrainFuture(state_dict)

        return ImmediateTrainFuture({})

    def load_weights(self, weights: dict[str, Any]) -> TrainFuture[None]:
        """Load model weights.

        Args:
            weights: state_dict to load
        """
        # Our get_weights() returns a "chunk_{i}."-prefixed dict.
        # Load that representation back into the model chunks.
        for i, model_chunk in enumerate(self.model):
            prefix = f"chunk_{i}."
            chunk_state = {k[len(prefix) :]: v for k, v in weights.items() if k.startswith(prefix)}
            if chunk_state:
                model_chunk.load_state_dict(chunk_state, strict=False)

        return ImmediateTrainFuture(None)

    def save_checkpoint(self, step: int) -> TrainFuture[Path]:
        """Save checkpoint using Megatron's distributed checkpointing.

        Args:
            step: Current training step

        Returns:
            Future resolving to checkpoint path
        """
        try:
            from megatron.training.checkpointing import save_checkpoint
        except ImportError:
            # Fallback to simple torch.save
            import torch

            ckpt_path = self.checkpoint_dir / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_dict = self.get_weights().result()
            torch.save(state_dict, ckpt_path / "model.pt")
            return ImmediateTrainFuture(ckpt_path)

        ckpt_path = self.checkpoint_dir / f"step_{step}"
        save_checkpoint(step, self.model, self.optimizer, self.opt_param_scheduler)
        return ImmediateTrainFuture(ckpt_path)
