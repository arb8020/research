"""Pure function implementation of SFT training loop.

No classes, no hidden state - just explicit orchestration of
stateful dependencies (backend, data).

Design: Casey Muratori (no retention), Tiger Style (explicit state).
"""

import logging
from typing import Any

from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.metrics import MetricsLogger
from rollouts.training.types import Sample, SFTTrainingConfig

logger = logging.getLogger(__name__)


async def run_sft_training(
    backend: PyTorchTrainingBackend,
    samples: list[Sample],
    config: SFTTrainingConfig,
    metrics_logger: MetricsLogger | None = None,
) -> list[dict[str, float]]:
    """Run SFT training (pure function, no hidden state).

    Args:
        backend: Training backend (has its own state)
        samples: Training samples (immutable)
        config: Training configuration (immutable)
        metrics_logger: Optional metrics logger (Casey: explicit parameter)

    Returns:
        List of metrics dicts (one per step)

    Example:
        >>> from rollouts.training.metrics import JSONLLogger
        >>>
        >>> backend = PyTorchTrainingBackend(model, optimizer, loss_fn)
        >>> samples = load_sft_samples("dataset.jsonl")
        >>> config = SFTTrainingConfig(num_steps=1000, batch_size=4)
        >>> logger = JSONLLogger(Path("logs/exp_001"))
        >>>
        >>> metrics = await run_sft_training(backend, samples, config, logger)
        >>> print(f"Final loss: {metrics[-1]['loss']:.4f}")

    Casey Muratori: No retention, explicit inputs/outputs.
    Sean Goedecke: Boring coordination, no magic.
    """
    # Tiger Style: Assert preconditions
    assert len(samples) > 0, "samples cannot be empty"
    assert config.num_steps > 0, "num_steps must be > 0"
    assert config.batch_size > 0, "batch_size must be > 0"

    metrics_history = []

    logger.info("Starting SFT training...")
    logger.info(f"  Samples: {len(samples)}")
    logger.info(f"  Steps: {config.num_steps}")
    logger.info(f"  Batch size: {config.batch_size}")

    for step in range(config.num_steps):
        # Get batch (pure function)
        batch = collate_batch(samples, config.batch_size, step)

        # Train (backend has state, but we don't!)
        fwd_metrics = await backend.forward_backward(batch).result()
        opt_metrics = await backend.optim_step().result()

        # Combine metrics (pure)
        step_metrics = {
            **fwd_metrics,
            **opt_metrics,
            "step": step,
        }
        metrics_history.append(step_metrics)

        # ═══════════════════════════════════════════════════════
        # ERROR LOGGING: Events (sporadic)
        # ═══════════════════════════════════════════════════════
        if step % config.log_every == 0:
            logger.info(
                f"Step {step}: "
                f"loss={fwd_metrics['loss']:.4f}, "
                f"grad_norm={fwd_metrics['grad_norm']:.4f}, "
                f"lr={opt_metrics['lr']:.4e}"
            )

        # ═══════════════════════════════════════════════════════
        # METRICS LOGGING: Timeseries (regular)
        # ═══════════════════════════════════════════════════════
        if metrics_logger and step % config.log_every == 0:
            metrics_logger.log(step_metrics, step=step)

        # Checkpoint (side effect, but explicit)
        if step % config.checkpoint_every == 0 and step > 0:
            ckpt_path = await backend.save_checkpoint(step, step_metrics)
            logger.info(f"  Saved checkpoint to {ckpt_path}")

    logger.info("Training complete!")

    # Finish metrics logging
    if metrics_logger:
        metrics_logger.finish()

    return metrics_history


def collate_batch(
    samples: list[Sample],
    batch_size: int,
    step: int,
) -> dict[str, Any]:
    """Pure function: Collate samples into training batch.

    Args:
        samples: All training samples
        batch_size: Batch size
        step: Current training step (for cycling through data)

    Returns:
        Batch dict with {input_ids, labels, loss_mask}

    Tiger Style: Explicit parameters, no hidden state.
    """

    # Cycle through dataset (simple modulo indexing)
    start_idx = (step * batch_size) % len(samples)
    end_idx = start_idx + batch_size

    # Handle wrap-around
    if end_idx <= len(samples):
        batch_samples = samples[start_idx:end_idx]
    else:
        # Wrap around to beginning
        batch_samples = samples[start_idx:] + samples[: end_idx - len(samples)]

    # Collate (pure function)
    return prepare_sft_batch(batch_samples)


def prepare_sft_batch(samples: list[Sample]) -> dict[str, Any]:
    """Pure function: Convert samples to training batch using sequence packing.

    Uses SLIME-style packing: concatenates sequences instead of padding.
    More efficient than padding (no wasted computation on pad tokens).

    Args:
        samples: List of Sample objects

    Returns:
        Batch dict with:
        - tokens: Concatenated token sequence [total_tokens]
        - labels: Same as tokens (for causal LM)
        - loss_mask: Concatenated loss mask [total_tokens]
        - cu_seqlens: Cumulative sequence lengths [batch_size + 1]
        - position_ids: Position IDs for each token [total_tokens]

    Example:
        >>> samples = [Sample(tokens=[1,2,3]), Sample(tokens=[4,5])]
        >>> batch = prepare_sft_batch(samples)
        >>> batch["tokens"]  # tensor([1,2,3,4,5])
        >>> batch["cu_seqlens"]  # tensor([0, 3, 5])
    """
    import torch

    # Packing (SLIME pattern): concatenate sequences instead of padding
    # This is more efficient - no wasted computation on padding tokens
    flat_tokens = []
    flat_masks = []
    flat_position_ids = []
    cu_seqlens = [0]  # Cumulative sequence lengths

    for s in samples:
        # Convert to lists if needed
        tokens = s.tokens if isinstance(s.tokens, list) else s.tokens.tolist()
        mask = s.loss_mask if isinstance(s.loss_mask, list) else s.loss_mask.tolist()

        # Concatenate this sequence
        flat_tokens.extend(tokens)
        flat_masks.extend(mask)
        # Position IDs reset for each sequence
        flat_position_ids.extend(range(len(tokens)))

        # Track sequence boundary
        cu_seqlens.append(cu_seqlens[-1] + len(tokens))

    # Convert to tensors
    # Add batch dimension [1, total_tokens] for compatibility with HF models
    # Note: For Flash Attention / advanced packing, this would stay 1D with cu_seqlens
    return {
        "input_ids": torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0),  # [1, total_tokens]
        "labels": torch.tensor(flat_tokens, dtype=torch.long).unsqueeze(0),  # [1, total_tokens]
        "loss_mask": torch.tensor(flat_masks, dtype=torch.float).unsqueeze(0),  # [1, total_tokens]
        "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.long),  # [batch_size + 1]
        "position_ids": torch.tensor(flat_position_ids, dtype=torch.long).unsqueeze(
            0
        ),  # [1, total_tokens]
    }
