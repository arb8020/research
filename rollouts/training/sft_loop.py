"""Pure function implementation of SFT training loop.

No classes, no hidden state - just explicit orchestration of
stateful dependencies (backend, data).

Design: Casey Muratori (no retention), Tiger Style (explicit state).
"""

import trio
from typing import List, Dict
from pathlib import Path

from rollouts.training.backends import PyTorchTrainingBackend
from training.types import Sample, TrainingConfig


async def run_sft_training(
    backend: PyTorchTrainingBackend,
    samples: List[Sample],
    config: TrainingConfig,
) -> List[Dict[str, float]]:
    """Run SFT training (pure function, no hidden state).

    Args:
        backend: Training backend (has its own state)
        samples: Training samples (immutable)
        config: Training configuration (immutable)

    Returns:
        List of metrics dicts (one per step)

    Example:
        >>> backend = PyTorchTrainingBackend(model, optimizer, loss_fn)
        >>> samples = load_sft_samples("dataset.jsonl")
        >>> config = TrainingConfig(num_steps=1000, batch_size=4)
        >>>
        >>> metrics = await run_sft_training(backend, samples, config)
        >>> print(f"Final loss: {metrics[-1]['loss']:.4f}")

    Casey Muratori: No retention, explicit inputs/outputs.
    Sean Goedecke: Boring coordination, no magic.
    """
    # Tiger Style: Assert preconditions
    assert len(samples) > 0, "samples cannot be empty"
    assert config.num_steps > 0, "num_steps must be > 0"
    assert config.batch_size > 0, "batch_size must be > 0"

    metrics_history = []

    print(f"Starting SFT training...")
    print(f"  Samples: {len(samples)}")
    print(f"  Steps: {config.num_steps}")
    print(f"  Batch size: {config.batch_size}")

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

        # Log (side effect, but explicit)
        if step % config.log_every == 0:
            print(
                f"Step {step}: "
                f"loss={fwd_metrics['loss']:.4f}, "
                f"grad_norm={fwd_metrics['grad_norm']:.4f}, "
                f"lr={opt_metrics['lr']:.4e}"
            )

        # Checkpoint (side effect, but explicit)
        if step % config.checkpoint_every == 0 and step > 0:
            ckpt_path = await backend.save_checkpoint(step, step_metrics)
            print(f"  Saved checkpoint to {ckpt_path}")

    print(f"\nTraining complete!")
    return metrics_history


def collate_batch(
    samples: List[Sample],
    batch_size: int,
    step: int,
) -> Dict[str, any]:
    """Pure function: Collate samples into training batch.

    Args:
        samples: All training samples
        batch_size: Batch size
        step: Current training step (for cycling through data)

    Returns:
        Batch dict with {input_ids, labels, loss_mask}

    Tiger Style: Explicit parameters, no hidden state.
    """
    import torch

    # Cycle through dataset (simple modulo indexing)
    start_idx = (step * batch_size) % len(samples)
    end_idx = start_idx + batch_size

    # Handle wrap-around
    if end_idx <= len(samples):
        batch_samples = samples[start_idx:end_idx]
    else:
        # Wrap around to beginning
        batch_samples = samples[start_idx:] + samples[:end_idx - len(samples)]

    # Collate (pure function)
    return prepare_sft_batch(batch_samples)


def prepare_sft_batch(samples: List[Sample]) -> Dict[str, any]:
    """Pure function: Convert samples to training batch.

    Args:
        samples: List of Sample objects

    Returns:
        Batch dict with {input_ids, labels, loss_mask}
    """
    import torch

    # Stack tensors
    # Note: Samples should already have torch tensors in tokens/loss_mask
    # If they're lists, convert them
    input_ids_list = []
    labels_list = []
    loss_mask_list = []

    for s in samples:
        # Convert to tensors if needed
        if isinstance(s.tokens, list):
            input_ids_list.append(torch.tensor(s.tokens, dtype=torch.long))
            labels_list.append(torch.tensor(s.tokens, dtype=torch.long))  # Same as input for causal LM
            loss_mask_list.append(torch.tensor(s.loss_mask, dtype=torch.float))
        else:
            input_ids_list.append(s.tokens)
            labels_list.append(s.tokens)
            loss_mask_list.append(s.loss_mask)

    # Stack into batch
    input_ids = torch.stack(input_ids_list)
    labels = torch.stack(labels_list)
    loss_mask = torch.stack(loss_mask_list)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }
