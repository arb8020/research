#!/usr/bin/env python3
"""Example: Run RL training with functional loop.

⚠️ NOTE: This example requires a GPU and torch installed.
For remote GPU deployment, use deploy.py pattern (like ~/wafer_stuff/clicker/).

Demonstrates:
- Setting up DataBuffer, AsyncRolloutManager, inference engines
- Running functional RL training loop
- Environment-based reward computation
- Weight sync to inference engines (D5)
- SLIME-style Generation → Training → Weight Sync loop
"""

import sys
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import trio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.backends import PyTorchTrainingBackend
from training.rl_loop import run_rl_training
from training.rl_losses import grpo_loss
from training.data_buffer import DataBuffer, load_prompts_from_list
from training.async_rollout_manager import AsyncRolloutManager
from training.types import Sample, RolloutConfig, RLTrainingConfig

# Setup basic logging for examples
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


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
    advantages: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """Compute GRPO loss for RL training.

    Args:
        logits: Model predictions
        labels: Target labels
        loss_mask: Token-level loss weights
        advantages: Advantage estimates (rewards - baseline)

    Returns:
        Scalar loss
    """
    return grpo_loss(logits, labels, loss_mask, advantages)


# ────────────────────── Mock Rollout Function ──────────────────────


def mock_rl_rollout(prompts, **kwargs):
    """Mock rollout function for testing RL loop.

    In real RL, this would call your agent/model to generate responses.
    For this demo, we generate synthetic samples with mock rewards.

    Args:
        prompts: List of prompts

    Returns:
        List of Sample objects with rewards
    """
    samples = []
    for i, prompt in enumerate(prompts):
        # Generate synthetic tokens
        seq_len = 32
        tokens = torch.randint(0, 100, (seq_len,)).tolist()

        # Mask first 25% (prompt)
        loss_mask = [0.0] * (seq_len // 4) + [1.0] * (seq_len - seq_len // 4)

        # Mock reward: some samples are "correct"
        is_correct = (i % 3 == 0)  # Every 3rd sample is correct

        sample = Sample(
            prompt=prompt,
            response=f"Response {i}",
            tokens=tokens,
            loss_mask=loss_mask,
            reward=0.0,  # Will be computed by reward function
            metadata={"correct": is_correct},  # For reward computation
        )
        samples.append(sample)

    return samples


# ────────────────────── Main ──────────────────────


async def main():
    print("=" * 70)
    print("RL Training Example (Functional Loop + SLIME Pattern)")
    print("=" * 70)

    # 1. Create data buffer
    print("\n1. Creating data buffer...")
    prompts = load_prompts_from_list([
        "Solve 2+2",
        "Calculate 5*7",
        "What is 10-3?",
        "Compute 8/2",
        "What is 6*6?",
        "Calculate 15-7",
    ])
    data_buffer = DataBuffer(prompts=prompts, seed=42)
    print(f"   ✓ Created DataBuffer with {len(prompts)} prompts")

    # 2. Create rollout manager
    print("\n2. Creating rollout manager...")
    rollout_config = RolloutConfig(
        batch_size=4,
        generate_fn=mock_rl_rollout,
        n_samples_per_prompt=1,
    )
    rollout_manager = AsyncRolloutManager(
        data_buffer=data_buffer,
        config=rollout_config,
    )
    print(f"   ✓ Created AsyncRolloutManager")
    print(f"     Batch size: {rollout_config.batch_size}")

    # 3. Create model and backend
    print("\n3. Creating model and training backend...")
    model = TinyGPT(vocab_size=100, dim=64, n_layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=compute_loss,  # GRPO loss
        checkpoint_dir=Path("/tmp/rl_checkpoints"),
    )
    print(f"   ✓ Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   ✓ Checkpoint dir: {backend.checkpoint_dir}")

    # 4. Mock inference engines (for weight sync demonstration)
    print("\n4. Setting up inference engines...")
    # In real RL, these would be SGLangEngine or VLLMEngine instances
    # For this demo, we just use an empty list (no actual syncing)
    inference_engines = []
    print(f"   ✓ Inference engines: {len(inference_engines)} (mock)")
    print(f"     (In real RL, use SGLangEngine or VLLMEngine)")

    # 5. Configure RL training
    print("\n5. Configuring RL training...")
    config = RLTrainingConfig(
        num_steps=25,  # Short run for demo
        sync_every=10,  # Sync weights every 10 steps
        baseline=0.33,  # Baseline for advantages (1/3 of samples are correct)
        log_every=5,
        checkpoint_every=15,
    )
    print(f"   ✓ Num steps: {config.num_steps}")
    print(f"   ✓ Weight sync every: {config.sync_every} steps")
    print(f"   ✓ Baseline: {config.baseline}")

    # 6. Run RL training (pure function!)
    print("\n6. Running RL training...")
    print("   SLIME Loop: Generation → Training → Weight Sync")
    print("-" * 70)

    metrics = await run_rl_training(
        backend=backend,
        data_buffer=data_buffer,
        rollout_manager=rollout_manager,
        inference_engines=inference_engines,
        config=config,
    )

    print("-" * 70)

    # 7. Print results
    print("\n7. Training Results:")
    print(f"   ✓ Total steps: {len(metrics)}")
    print(f"   ✓ Initial mean reward: {metrics[0]['mean_reward']:.2f}")
    print(f"   ✓ Final mean reward: {metrics[-1]['mean_reward']:.2f}")
    print(f"   ✓ Initial loss: {metrics[0]['loss']:.4f}")
    print(f"   ✓ Final loss: {metrics[-1]['loss']:.4f}")

    print("\n" + "=" * 70)
    print("✅ RL training complete!")
    print("=" * 70)

    # Print some metrics history
    print("\nMetrics history (first 5 steps):")
    for i in range(min(5, len(metrics))):
        m = metrics[i]
        print(
            f"  Step {m['step']}: "
            f"reward={m['mean_reward']:.2f}, "
            f"loss={m['loss']:.4f}, "
            f"grad_norm={m['grad_norm']:.4f}"
        )


if __name__ == "__main__":
    trio.run(main)
