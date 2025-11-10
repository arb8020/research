#!/usr/bin/env python3
"""Example: Run SFT training with functional loop.

⚠️ NOTE: This example requires a GPU and torch installed.
For remote GPU deployment, use deploy.py pattern (like ~/wafer_stuff/clicker/).

Demonstrates:
- Loading/creating SFT samples
- Running functional SFT training loop
- No classes, just pure functions + stateful dependencies
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import trio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.backends import PyTorchTrainingBackend
from training.sft_loop import run_sft_training
from training.types import Sample, TrainingConfig


# ────────────────────── Simple Model ──────────────────────


class TinyGPT(nn.Module):
    """Tiny GPT for testing (minimal viable model)."""

    def __init__(self, vocab_size=100, dim=64, n_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=4, batch_first=True),
            num_layers=n_layers,
        )
        self.lm_head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        """Forward pass.

        Args:
            input_ids: [batch, seq_len]

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        x = self.embedding(input_ids)  # [batch, seq_len, dim]
        x = self.transformer(x)  # [batch, seq_len, dim]
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]
        return logits


# ────────────────────── Loss Function ──────────────────────


def compute_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    loss_mask: torch.Tensor,  # [batch, seq_len]
) -> torch.Tensor:
    """Compute masked cross-entropy loss (Tinker: token-level control).

    Args:
        logits: Model predictions
        labels: Target labels
        loss_mask: Token-level loss weights (0.0 = ignore, 1.0 = train)

    Returns:
        Scalar loss
    """
    # Flatten for cross_entropy
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.reshape(-1, vocab_size)
    labels_flat = labels.reshape(-1)

    # Compute loss per token
    loss_per_token = F.cross_entropy(
        logits_flat,
        labels_flat,
        reduction='none',
    )

    # Apply loss mask (Tinker: token-level weights)
    loss_per_token = loss_per_token.reshape(batch_size, seq_len)
    masked_loss = loss_per_token * loss_mask

    # Average over non-masked tokens
    return masked_loss.sum() / loss_mask.sum().clamp(min=1.0)


# ────────────────────── Synthetic Data ──────────────────────


def generate_synthetic_samples(num_samples=100, vocab_size=100, seq_len=32) -> list[Sample]:
    """Generate synthetic SFT samples for testing.

    Returns:
        List of Sample objects
    """
    samples = []

    for i in range(num_samples):
        # Random tokens
        tokens = torch.randint(0, vocab_size, (seq_len,)).tolist()

        # Mask first 25% of tokens (simulating prompt)
        loss_mask = [0.0] * (seq_len // 4) + [1.0] * (seq_len - seq_len // 4)

        sample = Sample(
            prompt=f"Prompt {i}",
            response=f"Response {i}",
            tokens=tokens,
            loss_mask=loss_mask,
            reward=0.0,  # SFT doesn't use rewards
        )
        samples.append(sample)

    return samples


# ────────────────────── Main ──────────────────────


async def main():
    print("=" * 70)
    print("SFT Training Example (Functional Loop)")
    print("=" * 70)

    # 1. Generate synthetic samples
    print("\n1. Generating synthetic samples...")
    samples = generate_synthetic_samples(num_samples=100, vocab_size=100, seq_len=32)
    print(f"   ✓ Generated {len(samples)} samples")

    # 2. Create model and backend
    print("\n2. Creating model and training backend...")
    model = TinyGPT(vocab_size=100, dim=64, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=compute_loss,
        checkpoint_dir=Path("/tmp/sft_checkpoints"),
    )
    print(f"   ✓ Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ Checkpoint dir: {backend.checkpoint_dir}")

    # 3. Configure training
    print("\n3. Configuring training...")
    config = TrainingConfig(
        num_steps=50,  # Short run for demo
        batch_size=4,
        log_every=10,
        checkpoint_every=25,
    )
    print(f"   ✓ Num steps: {config.num_steps}")
    print(f"   ✓ Batch size: {config.batch_size}")

    # 4. Run SFT training (pure function!)
    print("\n4. Running SFT training...")
    print("-" * 70)

    metrics = await run_sft_training(backend, samples, config)

    print("-" * 70)

    # 5. Print results
    print("\n5. Training Results:")
    print(f"   ✓ Total steps: {len(metrics)}")
    print(f"   ✓ Initial loss: {metrics[0]['loss']:.4f}")
    print(f"   ✓ Final loss: {metrics[-1]['loss']:.4f}")
    print(f"   ✓ Loss reduction: {metrics[0]['loss'] - metrics[-1]['loss']:.4f}")

    print("\n" + "=" * 70)
    print("✅ SFT training complete!")
    print("=" * 70)

    # Print some metrics history
    print("\nMetrics history (first 5 steps):")
    for i in range(min(5, len(metrics))):
        m = metrics[i]
        print(f"  Step {m['step']}: loss={m['loss']:.4f}, grad_norm={m['grad_norm']:.4f}")


if __name__ == "__main__":
    trio.run(main)
