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

import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from ...training.lowering import MegatronLowering
from ...training.types import ImmediateTrainFuture, TrainFuture

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _extract_megatron_logits(output_tensor: Any) -> Any:
    """Normalize Megatron forward output into a logits tensor."""
    if isinstance(output_tensor, tuple):
        assert output_tensor, "Megatron output tuple must be non-empty"
        return output_tensor[0]
    if hasattr(output_tensor, "logits"):
        return output_tensor.logits
    return output_tensor


def _normalize_megatron_checkpoint_args(
    args: Any,
    checkpoint_dir: Path,
    *,
    save_optimizer_state: bool,
) -> None:
    """Fill checkpoint config defaults expected by newer Megatron saves.

    Our worker constructs a minimal Megatron args namespace for model/runtime
    initialization. Recent Megatron checkpoint code assumes the checkpoint
    config product type has already been populated as well. Normalize those
    fields here instead of depending on parser/version-specific side effects.
    """

    desired_ckpt_format = "torch" if save_optimizer_state else "torch_dist"

    checkpoint_defaults = {
        "async_ckpt_cpu_priority": 10,
        "async_ckpt_io_priority": 3,
        "async_save": False,
        "ckpt_assume_constant_structure": False,
        # Megatron's torch_dist optimizer checkpoint path is currently dishonest
        # for our mixed-precision runtime: it cannot map optimizer params back to
        # the model sharded-state tensors. Use the legacy torch checkpoint format
        # when optimizer state is requested, and keep torch_dist for model-only
        # checkpoints where the distributed format is working.
        "ckpt_format": desired_ckpt_format,
        "ckpt_fully_parallel_save": True,
        "ckpt_fully_parallel_save_process_group": "dp",
        "dist_ckpt_optim_fully_reshardable": False,
        "dist_ckpt_save_pre_mcore_014": False,
        "dist_ckpt_workers": 1,
        "distrib_optim_fully_reshardable_mem_efficient": False,
        "log_progress": False,
        "non_persistent_ckpt_type": None,
        "non_persistent_global_ckpt_dir": None,
        "non_persistent_local_ckpt_algo": "fully_parallel",
        "save_interval": 1,
        "save_retain_interval": None,
        "use_persistent_ckpt_worker": False,
    }
    for name, default in checkpoint_defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, default)

    # The worker args namespace often already carries a default torch_dist
    # checkpoint format from Megatron initialization. For checkpoint save
    # semantics we need the format to follow the requested product type, not
    # whatever ambient default happened to be set earlier.
    args.ckpt_format = desired_ckpt_format
    args.no_save_optim = not save_optimizer_state
    if not hasattr(args, "no_save_rng"):
        args.no_save_rng = not getattr(args, "save_rng", True)

    # This backend owns the save root; keep Megatron's tracker and naming
    # rooted under the configured checkpoint directory.
    args.save = str(checkpoint_dir)

    try:
        from megatron.training.utils import update_use_dist_ckpt
    except ImportError:
        args.use_dist_ckpt = args.ckpt_format != "torch"
    else:
        update_use_dist_ckpt(args)


def _naive_per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def _naive_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)


def _tp_per_token_logprobs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    process_group: dist.ProcessGroup,
) -> torch.Tensor:
    from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy

    flat_logits = logits.reshape(-1, logits.size(-1)).contiguous()
    flat_labels = labels.reshape(-1).contiguous()
    losses = fused_vocab_parallel_cross_entropy(
        flat_logits.unsqueeze(1),
        flat_labels.unsqueeze(1),
        process_group,
    )
    return (-losses.squeeze(1)).reshape_as(labels)


class _VocabParallelEntropy(torch.autograd.Function):
    """Entropy over tensor-parallel vocab shards.

    TODO: If we later model sharded forward products explicitly, we can move
    the shared GRPO formulas back above this backend layer. Until then, keep
    the TP-specific realization glue local to Megatron.
    """

    @staticmethod
    def forward(
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        process_group: dist.ProcessGroup,
    ) -> torch.Tensor:
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

        normalized_logits = vocab_parallel_logits - logits_max
        exp_logits = normalized_logits.exp_()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(sum_exp_logits, group=process_group)

        softmax_logits = exp_logits.div_(sum_exp_logits)
        sum_softmax_times_logits = (softmax_logits * vocab_parallel_logits).sum(
            dim=-1, keepdim=True
        )
        dist.all_reduce(sum_softmax_times_logits, group=process_group)

        entropy = logits_max + sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        vocab_parallel_logits.sub_(sum_softmax_times_logits)
        softmax_logits.mul_(vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        vocab_parallel_logits.add_(sum_softmax_times_logits)
        softmax_logits.mul_(-1)
        return softmax_logits, None


def _tp_entropy(logits: torch.Tensor, process_group: dist.ProcessGroup) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.size(-1)).contiguous()
    entropy = _VocabParallelEntropy.apply(flat_logits, process_group)
    return entropy.reshape(logits.shape[:-1])


def _sequence_mean(values: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return (values * loss_mask).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)


def _masked_mean(values: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    return (values * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)


def _megatron_per_token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_world_size() <= 1:
        return _naive_per_token_logprobs(logits, labels)
    return _tp_per_token_logprobs(logits, labels, mpu.get_tensor_model_parallel_group())


def _megatron_entropy(logits: torch.Tensor) -> torch.Tensor:
    from megatron.core import mpu

    if mpu.get_tensor_model_parallel_world_size() <= 1:
        return _naive_entropy(logits)
    return _tp_entropy(logits, mpu.get_tensor_model_parallel_group())


def megatron_grpo_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]

    token_logprobs = _megatron_per_token_logprobs(logits, labels)
    seq_logprobs = _sequence_mean(token_logprobs, loss_mask)
    pg_loss = -(seq_logprobs * advantages).mean()

    with torch.no_grad():
        entropy = _masked_mean(_megatron_entropy(logits), loss_mask).item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "avg_logprob": seq_logprobs.mean().item(),
        "avg_advantage": advantages.mean().item(),
    }
    return pg_loss, metrics


def megatron_grpo_loss_clipped(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    clip_range: float = 0.2,
    entropy_coef: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]
    old_logprobs = batch["old_logprobs"]

    token_logprobs = _megatron_per_token_logprobs(logits, labels)
    seq_logprobs = _sequence_mean(token_logprobs, loss_mask)

    log_ratio = seq_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    pg_loss1 = -ratio * advantages
    pg_loss2 = -torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

    entropy = _masked_mean(_megatron_entropy(logits), loss_mask)
    loss = pg_loss - entropy_coef * entropy

    with torch.no_grad():
        clipped = (ratio < 1.0 - clip_range) | (ratio > 1.0 + clip_range)
        clipfrac = clipped.float().mean().item()
        approx_kl = ((ratio - 1) - log_ratio).mean().item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy.item(),
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
        "avg_ratio": ratio.mean().item(),
        "avg_logprob": seq_logprobs.mean().item(),
        "avg_advantage": advantages.mean().item(),
    }
    return loss, metrics


def megatron_grpo_loss_masked(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
    ratio_low: float = 0.1,
    ratio_high: float = 10.0,
    kl_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    advantages = batch["advantages"]
    old_logprobs = batch["old_logprobs"]

    token_logprobs = _megatron_per_token_logprobs(logits, labels)
    seq_logprobs = _sequence_mean(token_logprobs, loss_mask)

    log_ratio = seq_logprobs - old_logprobs
    ratio = torch.exp(log_ratio)
    is_masked_low = ratio < ratio_low
    is_masked_high = ratio > ratio_high
    keep_mask = ~(is_masked_low | is_masked_high)
    coeff = ratio * (advantages - kl_coef * log_ratio)

    if keep_mask.sum() > 0:
        pg_loss = -(coeff.detach() * seq_logprobs)[keep_mask].sum() / keep_mask.sum()
    else:
        # Megatron mutates the returned loss tensor in-place during reduction,
        # so return a graph-derived zero instead of a fresh leaf tensor.
        pg_loss = seq_logprobs.sum() * 0.0

    with torch.no_grad():
        entropy = _masked_mean(_megatron_entropy(logits), loss_mask).item()
        mismatch_kl = (torch.exp(log_ratio) - log_ratio - 1).mean().item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "masked_frac": (~keep_mask).float().mean().item(),
        "masked_low_frac": is_masked_low.float().mean().item(),
        "masked_high_frac": is_masked_high.float().mean().item(),
        "mismatch_kl": mismatch_kl,
        "avg_ratio": ratio.mean().item(),
        "avg_logprob": seq_logprobs.mean().item(),
        "avg_advantage": advantages.mean().item(),
        "avg_coeff": coeff.mean().item(),
    }
    return pg_loss, metrics


def megatron_opd_loss(
    logits: torch.Tensor,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    labels = batch["labels"]
    loss_mask = batch["loss_mask"]
    teacher_logprobs = batch["teacher_logprobs"]

    student_logprobs = _megatron_per_token_logprobs(logits, labels)
    if "advantages" in batch:
        advantages = batch["advantages"]
    else:
        advantages = (teacher_logprobs - student_logprobs) * loss_mask

    masked_pg = student_logprobs * advantages.detach() * loss_mask
    num_tokens = loss_mask.sum().clamp(min=1.0)
    pg_loss = -masked_pg.sum() / num_tokens

    with torch.no_grad():
        entropy = _masked_mean(_megatron_entropy(logits), loss_mask).item()
        avg_student_lp = (student_logprobs * loss_mask).sum().item() / num_tokens.item()
        avg_teacher_lp = (teacher_logprobs * loss_mask).sum().item() / num_tokens.item()
        avg_advantage = (advantages * loss_mask).sum().item() / num_tokens.item()
        kl_div = (
            (student_logprobs - teacher_logprobs) * loss_mask
        ).sum().item() / num_tokens.item()

    metrics = {
        "pg_loss": pg_loss.item(),
        "entropy": entropy,
        "avg_student_logprob": avg_student_lp,
        "avg_teacher_logprob": avg_teacher_lp,
        "avg_advantage": avg_advantage,
        "kl_div": kl_div,
        "num_tokens": num_tokens.item(),
    }
    return pg_loss, metrics


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
        save_optimizer_state: Include optimizer/scheduler state in checkpoints
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
    save_optimizer_state: bool = True


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
        loss_fn: Optional backend-native loss function for dict batches.

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
            active_loss_fn = loss_fn if loss_fn is not None else self.loss_fn

            # Model forward
            output = model(
                input_ids=tokens,
                position_ids=None,
                attention_mask=None,
                labels=None if active_loss_fn is not None else labels,
            )

            # Return for pipeline engine
            def loss_reducer(output_tensor: Any) -> dict[str, Any]:
                extra_metrics: dict[str, float] = {}
                if active_loss_fn is not None:
                    logits = _extract_megatron_logits(output_tensor)
                    loss_result = active_loss_fn(logits, batch)
                    if isinstance(loss_result, tuple):
                        loss, extra_metrics = loss_result
                    else:
                        loss = loss_result
                else:
                    # Default: assume model returns loss directly
                    loss = (
                        output_tensor
                        if isinstance(output_tensor, torch.Tensor)
                        else output_tensor.loss
                    )
                    if isinstance(loss, torch.Tensor) and loss.ndim > 0:
                        if loss_mask is not None:
                            masked = loss.float() * loss_mask.float()
                            loss = masked.sum() / torch.clamp_min(loss_mask.sum(), 1.0)
                        else:
                            loss = loss.float().mean()

                detached_loss = loss.detach()
                metric_tensors = {
                    "loss": detached_loss.float(),
                    **{
                        key: torch.as_tensor(
                            value, device=detached_loss.device, dtype=torch.float32
                        )
                        for key, value in extra_metrics.items()
                    },
                }
                return (
                    loss,
                    torch.tensor(1, device=detached_loss.device),
                    {
                        "keys": list(metric_tensors.keys()),
                        "values": torch.stack([
                            torch.tensor(1.0, device=detached_loss.device),
                            *metric_tensors.values(),
                        ]),
                    },
                )

            return output, loss_reducer

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
                first_loss = losses_reduced[0]
                if isinstance(first_loss, dict) and "keys" in first_loss and "values" in first_loss:
                    keys = list(first_loss["keys"])
                    values = first_loss["values"]
                    count = float(values[0])
                    for metric_index, key in enumerate(keys, start=1):
                        metrics[key] = float(values[metric_index]) / max(count, 1.0)

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
            self.opt_param_scheduler.step(increment=self.config.global_batch_size)
            primary_group = self.optimizer.param_groups[0]
            lr = self.opt_param_scheduler.get_lr(primary_group)
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
        """Get a checkpoint-like Megatron local state_dict view.

        This is not the honest inference-export boundary. It only returns
        chunk-local state from rank 0 / TP-rank 0 and is suitable for simple
        checkpoint-style save/load, not Megatron->inference hot weight sync.
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
            from megatron.training.global_vars import get_args
        except ImportError:
            # Fallback to simple torch.save
            import torch

            ckpt_path = self.checkpoint_dir / f"step_{step}"
            ckpt_path.mkdir(parents=True, exist_ok=True)
            state_dict = self.get_weights().result()
            torch.save(state_dict, ckpt_path / "model.pt")
            return ImmediateTrainFuture(ckpt_path)

        args = get_args()
        _normalize_megatron_checkpoint_args(
            args,
            self.checkpoint_dir,
            save_optimizer_state=self.config.save_optimizer_state,
        )
        logger.info(
            "Megatron save_checkpoint step=%s format=%s save_optimizer_state=%s use_dist_ckpt=%s",
            step,
            getattr(args, "ckpt_format", None),
            self.config.save_optimizer_state,
            getattr(args, "use_dist_ckpt", None),
        )
        signature = inspect.signature(save_checkpoint)
        checkpoint_kwargs: dict[str, Any] = {}

        # We do not currently track Megatron's cumulative FLOP counter in this
        # backend, so preserve the same explicit zero sentinel used by slime.
        if "num_floating_point_operations_so_far" in signature.parameters:
            checkpoint_kwargs["num_floating_point_operations_so_far"] = 0
        if "checkpointing_context" in signature.parameters:
            checkpoint_kwargs["checkpointing_context"] = None
        if "train_data_iterator" in signature.parameters:
            checkpoint_kwargs["train_data_iterator"] = None
        if "preprocess_common_state_dict_fn" in signature.parameters:
            checkpoint_kwargs["preprocess_common_state_dict_fn"] = None

        save_checkpoint(
            step,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            **checkpoint_kwargs,
        )
        return ImmediateTrainFuture(self.checkpoint_dir)
