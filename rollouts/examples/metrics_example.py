"""Example: Using metrics logging in training loops.

Shows how to use MetricsLogger with different backends.
"""

import sys
import trio
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.metrics import (
    JSONLLogger,
    WandbLogger,
    CompositeLogger,
    compute_stats_from_jsonl,
)


# ══════════════════════════════════════════════════════════════
# Example 1: Simple JSONL logging (most common)
# ══════════════════════════════════════════════════════════════

def example_jsonl_logging():
    """Example: Basic JSONL logging (no dependencies)."""
    print("=" * 60)
    print("Example 1: JSONL Logging")
    print("=" * 60)

    # Create logger (Casey: explicit dependency injection)
    logger = JSONLLogger(Path("/tmp/metrics_example/exp_001"))

    # Simulate training
    print("\nSimulating training...")
    for step in range(10):
        # Fake training step
        loss = 1.0 / (step + 1)
        reward = step * 0.1

        # Log metrics (simple!)
        logger.log(
            metrics={
                "loss": loss,
                "reward": reward,
                "lr": 1e-4,
            },
            step=step,
        )

        if step % 5 == 0:
            print(f"  Step {step}: loss={loss:.4f}, reward={reward:.2f}")

    # Finish
    logger.finish()

    print(f"\n✓ Metrics saved to: {logger.metrics_file}")
    print("\nFirst 3 lines:")
    with open(logger.metrics_file) as f:
        for i, line in enumerate(f):
            if i < 3:
                print(f"  {line.strip()}")

    # Compute stats (Casey: no retention, compute from file)
    stats = compute_stats_from_jsonl(logger.metrics_file, "loss")
    print(f"\nLoss statistics:")
    print(f"  Mean: {stats['mean']:.4f}")
    print(f"  Min:  {stats['min']:.4f}")
    print(f"  Max:  {stats['max']:.4f}")
    print(f"  Std:  {stats['std']:.4f}")


# ══════════════════════════════════════════════════════════════
# Example 2: Composite logging (JSONL + W&B)
# ══════════════════════════════════════════════════════════════

def example_composite_logging():
    """Example: Log to both JSONL and W&B simultaneously."""
    print("\n" + "=" * 60)
    print("Example 2: Composite Logging (JSONL + W&B)")
    print("=" * 60)

    # Create composite logger (Casey: redundancy)
    logger = CompositeLogger([
        JSONLLogger(Path("/tmp/metrics_example/exp_002")),
        # WandbLogger(project="metrics-example", name="exp-002"),  # Uncomment if you have W&B
    ])

    print("\nSimulating training...")
    for step in range(5):
        loss = 0.5 - step * 0.05

        logger.log({"loss": loss}, step=step)

        if step % 2 == 0:
            print(f"  Step {step}: loss={loss:.4f}")

    logger.finish()
    print("\n✓ Logged to both JSONL and W&B (if enabled)")


# ══════════════════════════════════════════════════════════════
# Example 3: In actual training loop (realistic)
# ══════════════════════════════════════════════════════════════

async def run_fake_training(
    num_steps: int,
    metrics_logger: JSONLLogger | None = None,  # Casey: optional, explicit
):
    """Fake training loop showing real usage pattern.

    This is how you'd use it in rollouts/training/sft_loop.py
    """
    print("\nRunning fake training loop...")

    for step in range(num_steps):
        # Simulate async training step
        await trio.sleep(0.01)

        # Fake metrics
        loss = 1.0 / (step + 1)
        grad_norm = 0.5 + (step % 3) * 0.1
        lr = 1e-4

        # Log metrics (if logger provided)
        if metrics_logger:
            metrics_logger.log(
                {
                    "loss": loss,
                    "grad_norm": grad_norm,
                    "lr": lr,
                },
                step=step,
            )

        # Occasional checkpoint
        if step % 10 == 0:
            print(f"  [Checkpoint] Step {step}: loss={loss:.4f}")

    if metrics_logger:
        metrics_logger.finish()

    print(f"\n✓ Training complete ({num_steps} steps)")


async def example_training_loop():
    """Example: Using metrics logger in async training loop."""
    print("\n" + "=" * 60)
    print("Example 3: Real Training Loop Usage")
    print("=" * 60)

    # Create logger
    logger = JSONLLogger(Path("/tmp/metrics_example/exp_003"))

    # Run training (Casey: explicit parameter passing)
    await run_fake_training(num_steps=20, metrics_logger=logger)

    print(f"\n✓ Metrics saved to: {logger.metrics_file}")


# ══════════════════════════════════════════════════════════════
# Example 4: No logging (logger=None)
# ══════════════════════════════════════════════════════════════

async def example_no_logging():
    """Example: Training without metrics logging (testing, debugging)."""
    print("\n" + "=" * 60)
    print("Example 4: No Logging (logger=None)")
    print("=" * 60)

    # Run training without logger (Casey: explicit None)
    await run_fake_training(num_steps=5, metrics_logger=None)

    print("\n✓ No metrics logged (useful for testing)")


# ══════════════════════════════════════════════════════════════
# Run all examples
# ══════════════════════════════════════════════════════════════

async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("METRICS LOGGING EXAMPLES")
    print("=" * 60)

    # Sync examples
    example_jsonl_logging()
    example_composite_logging()

    # Async examples
    await example_training_loop()
    await example_no_logging()

    print("\n" + "=" * 60)
    print("KEY POINTS")
    print("=" * 60)
    print("""
1. MetricsLogger is a Protocol (swap implementations easily)
2. Logger is passed as parameter (Casey: dependency injection)
3. Logger can be None (optional logging)
4. JSONL is default (no dependencies, easy to analyze)
5. Composite logger for multiple backends (JSONL + W&B)
6. Statistics computed from file (Casey: no retention in memory)

Compare to error logging:
- Error logs: Use shared/logging_config.py (Python logging module)
- Metrics: Use this (rollouts/training/metrics.py)

They are SEPARATE systems for SEPARATE purposes!
    """)


if __name__ == "__main__":
    trio.run(main)
