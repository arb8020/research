"""Example: Using metrics logging with training loops.

Shows three usage patterns:
1. No metrics logging (testing/debugging)
2. JSONL only (default, no dependencies)
3. JSONL + W&B (production, redundancy)
"""

import trio
from pathlib import Path

# Your training code
from rollouts.training.sft_loop import run_sft_training
from rollouts.training.rl_loop import run_rl_training

# Metrics logging
from rollouts.training.metrics import (
    JSONLLogger,
    WandbLogger,
    CompositeLogger,
)


# ══════════════════════════════════════════════════════════════
# Example 1: No metrics logging (testing)
# ══════════════════════════════════════════════════════════════

async def example_no_metrics():
    """Example: Training without metrics (useful for testing)."""
    print("=" * 60)
    print("Example 1: No Metrics Logging")
    print("=" * 60)

    # Setup (your actual setup code here)
    backend = ...  # PyTorchTrainingBackend(model, optimizer, loss_fn)
    samples = ...  # load_sft_samples("dataset.jsonl")
    config = ...   # SFTTrainingConfig(num_steps=1000, batch_size=4)

    # Run training WITHOUT metrics logger (Casey: explicit None)
    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=config,
        metrics_logger=None,  # <-- No metrics logging
    )

    print(f"\n✓ Training complete. Final loss: {metrics[-1]['loss']:.4f}")
    print("  (No metrics file created)")


# ══════════════════════════════════════════════════════════════
# Example 2: JSONL logging only (default)
# ══════════════════════════════════════════════════════════════

async def example_jsonl_only():
    """Example: Log to JSONL only (no dependencies, local files)."""
    print("\n" + "=" * 60)
    print("Example 2: JSONL Logging Only")
    print("=" * 60)

    # Setup
    backend = ...  # PyTorchTrainingBackend(...)
    samples = ...  # load_sft_samples(...)
    config = ...   # SFTTrainingConfig(...)

    # Create JSONL logger (Casey: explicit dependency injection)
    metrics_logger = JSONLLogger(Path("logs/sft_exp_001"))

    # Run training WITH metrics logger
    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=config,
        metrics_logger=metrics_logger,  # <-- JSONL logging
    )

    print(f"\n✓ Training complete. Final loss: {metrics[-1]['loss']:.4f}")
    print(f"✓ Metrics saved to: {metrics_logger.metrics_file}")

    # Analyze metrics after training (Casey: no retention, compute from file)
    from rollouts.training.metrics import compute_stats_from_jsonl

    loss_stats = compute_stats_from_jsonl(metrics_logger.metrics_file, "loss")
    print(f"\nLoss statistics:")
    print(f"  Mean: {loss_stats['mean']:.4f}")
    print(f"  Min:  {loss_stats['min']:.4f}")
    print(f"  Max:  {loss_stats['max']:.4f}")
    print(f"  Std:  {loss_stats['std']:.4f}")


# ══════════════════════════════════════════════════════════════
# Example 3: JSONL + W&B (production, redundancy)
# ══════════════════════════════════════════════════════════════

async def example_jsonl_and_wandb():
    """Example: Log to both JSONL and W&B (recommended for production)."""
    print("\n" + "=" * 60)
    print("Example 3: JSONL + W&B (Production)")
    print("=" * 60)

    # Setup
    backend = ...  # PyTorchTrainingBackend(...)
    samples = ...  # load_sft_samples(...)
    config = ...   # SFTTrainingConfig(...)

    # Create composite logger (Casey: redundancy)
    # JSONL = always save locally (backup, analysis)
    # W&B = dashboards, team visibility
    metrics_logger = CompositeLogger([
        JSONLLogger(Path("logs/sft_exp_002")),
        WandbLogger(project="my-rl-project", name="sft-exp-002"),
    ])

    # Run training (SAME CODE!)
    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=config,
        metrics_logger=metrics_logger,  # <-- Both JSONL and W&B
    )

    print(f"\n✓ Training complete. Final loss: {metrics[-1]['loss']:.4f}")
    print("✓ Metrics logged to:")
    print("  - JSONL: logs/sft_exp_002/metrics.jsonl")
    print("  - W&B: https://wandb.ai/my-rl-project/runs/...")


# ══════════════════════════════════════════════════════════════
# Example 4: RL Training with metrics
# ══════════════════════════════════════════════════════════════

async def example_rl_training():
    """Example: RL training with metrics logging."""
    print("\n" + "=" * 60)
    print("Example 4: RL Training with Metrics")
    print("=" * 60)

    # Setup
    backend = ...           # PyTorchTrainingBackend(...)
    data_buffer = ...       # DataBuffer(prompts=[...])
    rollout_manager = ...   # AsyncRolloutManager(...)
    engines = ...           # [SGLangEngine(...)]
    config = ...            # RLTrainingConfig(num_steps=1000, sync_every=10)

    # Create metrics logger
    metrics_logger = CompositeLogger([
        JSONLLogger(Path("logs/rl_exp_001")),
        # WandbLogger(project="my-rl-project", name="rl-exp-001"),  # Optional
    ])

    # Run RL training
    metrics = await run_rl_training(
        backend=backend,
        data_buffer=data_buffer,
        rollout_manager=rollout_manager,
        inference_engines=engines,
        config=config,
        metrics_logger=metrics_logger,  # <-- Metrics logging
    )

    print(f"\n✓ RL training complete.")
    print(f"  Final reward: {metrics[-1]['mean_reward']:.2f}")
    print(f"  Final loss: {metrics[-1]['loss']:.4f}")
    print(f"✓ Metrics saved to: logs/rl_exp_001/metrics.jsonl")


# ══════════════════════════════════════════════════════════════
# What gets logged where
# ══════════════════════════════════════════════════════════════

def show_logging_outputs():
    """Show what gets logged to console vs metrics file."""
    print("\n" + "=" * 60)
    print("WHAT GETS LOGGED WHERE")
    print("=" * 60)

    print("""
CONSOLE OUTPUT (from error logging):
────────────────────────────────────────────────────────────
[INFO] Starting SFT training...
[INFO]   Samples: 1000
[INFO]   Steps: 100
[INFO]   Batch size: 4
[INFO] Step 0: loss=1.0000, grad_norm=0.5000, lr=1.0e-04
[INFO] Step 10: loss=0.9000, grad_norm=0.4500, lr=1.0e-04
[INFO]   Saved checkpoint to logs/ckpt_step_20.pt
[INFO] Step 20: loss=0.8000, grad_norm=0.4000, lr=1.0e-04
...
[INFO] Training complete!

METRICS FILE (logs/sft_exp_001/metrics.jsonl):
────────────────────────────────────────────────────────────
{"step": 0, "timestamp": 1704816000.0, "loss": 1.0, "grad_norm": 0.5, "lr": 0.0001}
{"step": 1, "timestamp": 1704816000.1, "loss": 0.99, "grad_norm": 0.49, "lr": 0.0001}
{"step": 2, "timestamp": 1704816000.2, "loss": 0.98, "grad_norm": 0.48, "lr": 0.0001}
...
{"step": 100, "timestamp": 1704816010.0, "loss": 0.5, "grad_norm": 0.25, "lr": 0.0001}

KEY DIFFERENCES:
────────────────────────────────────────────────────────────
Error Logging (Console):
  - Human-readable messages
  - Sporadic events (checkpoints, errors, warnings)
  - Low volume (~10 lines for 100 steps)

Metrics Logging (JSONL file):
  - Structured numeric data
  - Every step or every N steps
  - High volume (100+ entries for 100 steps)
  - Easy to analyze with pandas, plot with matplotlib
    """)


# ══════════════════════════════════════════════════════════════
# Key Points
# ══════════════════════════════════════════════════════════════

def show_key_points():
    """Show key design points."""
    print("=" * 60)
    print("KEY POINTS")
    print("=" * 60)

    print("""
1. TWO SEPARATE SYSTEMS:
   - Error logs: Python logging (shared/logging_config.py)
   - Metrics: Custom protocol (rollouts/training/metrics.py)

2. OPTIONAL METRICS LOGGING:
   - Pass metrics_logger=None for no logging (testing)
   - Pass JSONLLogger for local files (default)
   - Pass CompositeLogger for multiple backends (production)

3. EXPLICIT DEPENDENCY INJECTION:
   - Logger is passed as parameter (Casey: no global singleton)
   - Training loop has no hidden state
   - Easy to swap implementations (Ray-ready!)

4. PROTOCOL-BASED DESIGN:
   - Any class with log() and finish() works
   - No inheritance required
   - Easy to add custom loggers (MLflow, TensorBoard, etc.)

5. CLEAN SEPARATION IN CODE:

   # Error logging: events
   if step % config.log_every == 0:
       logger.info(f"Step {step}: loss={loss:.4f}")  # Python logging

   # Metrics logging: timeseries
   if metrics_logger and step % config.log_every == 0:
       metrics_logger.log(step_metrics, step=step)  # Custom system

6. CASEY PRINCIPLES:
   - No retention (compute stats from file later)
   - Explicit parameters (no global state)
   - Pure function orchestration
   - Redundancy (JSONL + W&B simultaneously)

7. TIGER PRINCIPLES:
   - Bounded (max_lines in JSONLLogger)
   - Assertions on inputs
   - Explicit state
   - Simple implementation (~100 LOC)
    """)


if __name__ == "__main__":
    show_logging_outputs()
    show_key_points()

    print("\n" + "=" * 60)
    print("To run the examples, uncomment the code below and fill in")
    print("the setup sections with your actual backend/data/config.")
    print("=" * 60)

    # Uncomment to run:
    # trio.run(example_no_metrics)
    # trio.run(example_jsonl_only)
    # trio.run(example_jsonl_and_wandb)
    # trio.run(example_rl_training)
