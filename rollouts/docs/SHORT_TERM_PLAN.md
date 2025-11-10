# Short-Term Plan: Refactor to Functional & Implement SFT/RL Loops

**Date**: 2025-11-09
**Goal**: Refactor stateful classes to pure functions, then implement functional SFT/RL training loops

---

## Overview

We're 90% of the way to a complete nanochat/SLIME-style post-training pipeline. All core components (D1-D6v1) are complete. What's missing is ~250 lines of orchestration code, but **before** we write that, we need to refactor away unnecessary stateful classes to prevent propagating bad patterns.

**Critical insight**: Training loops have NO legitimate mutable state of their own. They should be pure functions that orchestrate stateful dependencies (DataBuffer, TrainingBackend, AsyncRolloutManager).

---

## Phase 1: Refactor RolloutManager → Pure Functions

**Priority**: 🔴 **CRITICAL** - Do before building SFT/RL loops
**Effort**: ~2-3 hours
**Why**: Prevents propagating stateful pattern into training loops

### Tasks:

#### 1.1 Create `training/rollout_generation.py` (~30 min)

Extract pure functions from `rollout_manager.py`:

```python
# training/rollout_generation.py
"""Pure functions for rollout generation.

Replaces the stateful RolloutManager class with pure functions.
Following Casey Muratori: minimize state, maximize pure functions.
"""

from typing import Iterator
from training.data_buffer import DataBuffer
from rollouts.training.types import RolloutBatch, RolloutConfig


def generate_rollout_batches(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    **rollout_kwargs,
) -> Iterator[RolloutBatch]:
    """Generate rollout batches indefinitely (generator).

    Args:
        data_buffer: DataBuffer for prompt iteration (has state)
        config: RolloutConfig with batch_size and generate_fn
        **rollout_kwargs: Additional kwargs for generate_fn

    Yields:
        RolloutBatch objects ready for training

    Example:
        >>> data_buffer = DataBuffer(prompts=[...])
        >>> config = RolloutConfig(batch_size=4, generate_fn=my_fn)
        >>> batches = generate_rollout_batches(data_buffer, config)
        >>> for batch in batches:
        ...     train_on_batch(batch)
    """
    step = 0
    while True:
        # Get prompts (data_buffer manages its own state)
        prompts = data_buffer.get_prompts(config.batch_size)

        # Generate samples
        samples = config.generate_fn(prompts, **rollout_kwargs)

        # Apply transforms (pure function)
        samples = apply_sample_transforms(samples, config)

        # Convert to batch (pure function)
        batch = convert_to_batch(
            samples,
            epoch_id=data_buffer.epoch_id,
            step_id=step,
        )

        yield batch
        step += 1


def apply_sample_transforms(samples, config):
    """Pure function: Apply filters/transforms to samples.

    Moved from RolloutManager (already pure!).
    """
    # Move implementation from rollout_manager.py
    # This is already a pure function, just relocate it
    ...


def convert_to_batch(samples, epoch_id, step_id):
    """Pure function: Convert samples to RolloutBatch.

    Moved from RolloutManager (already pure!).
    """
    # Move implementation from rollout_manager.py
    # This is already a pure function, just relocate it
    ...
```

**Action**: Move existing pure functions from `rollout_manager.py` to new file.

---

#### 1.2 Update `AsyncRolloutManager` (~20 min)

Update imports to use pure functions from `rollout_generation.py`:

```python
# training/async_rollout_manager.py

from training.rollout_generation import (
    convert_to_batch,
    apply_sample_transforms,
)

class AsyncRolloutManager:
    # ... existing code ...

    async def generate_batch(self):
        # Use imported pure functions instead of RolloutManager methods
        samples = apply_sample_transforms(samples, self.config)
        batch = convert_to_batch(samples, epoch_id, step_id)
        # ...
```

**Action**: Replace any RolloutManager dependencies with pure function imports.

---

#### 1.3 Deprecate `RolloutManager` (~10 min)

Add deprecation warning to `rollout_manager.py`:

```python
# training/rollout_manager.py

import warnings

class RolloutManager:
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "RolloutManager is deprecated and will be removed in a future version. "
            "Use generate_rollout_batches() from training.rollout_generation instead. "
            "For async rollout generation, use AsyncRolloutManager.",
            DeprecationWarning,
            stacklevel=2,
        )
        # ... existing implementation ...
```

**Action**: Add deprecation warning to RolloutManager.__init__

---

#### 1.4 Update Examples (~30 min)

Update examples to use generator pattern:

**`examples/test_rollout_manager.py` → `examples/test_rollout_generation.py`**

```python
# OLD (class-based):
manager = RolloutManager(data_buffer, config)
for batch in manager:
    train_on_batch(batch)

# NEW (generator-based):
from training.rollout_generation import generate_rollout_batches

batches = generate_rollout_batches(data_buffer, config)
for batch in batches:
    train_on_batch(batch)
```

**`examples/sft_pipeline_example.py`**
- Update to use `generate_rollout_batches()` if it uses RolloutManager

**Action**: Update all examples to use pure function generator pattern.

---

#### 1.5 Update Public API (~10 min)

Remove RolloutManager from public API:

```python
# training/__init__.py

# REMOVE:
# from training.rollout_manager import RolloutManager

# ADD:
from training.rollout_generation import (
    generate_rollout_batches,
    apply_sample_transforms,
    convert_to_batch,
)
```

**Action**: Export pure functions instead of RolloutManager class.

---

#### 1.6 Commit (~10 min)

```bash
git add training/rollout_generation.py
git add training/async_rollout_manager.py
git add training/rollout_manager.py
git add training/__init__.py
git add examples/test_rollout_generation.py
git add examples/sft_pipeline_example.py

git commit -m "Refactor: Extract RolloutManager pure functions, deprecate class

- Create training/rollout_generation.py with pure functions
- Extract generate_rollout_batches() generator function
- Extract apply_sample_transforms() and convert_to_batch() helpers
- Update AsyncRolloutManager to import pure functions
- Deprecate RolloutManager class (add warning)
- Update examples to use generator pattern
- Update public API to export pure functions

Follows functional-first design: classes only for legitimate state.
RolloutManager had no real state (just loop counter), now a generator.
"
```

---

### Phase 1 Summary

**Before**: Stateful class wrapper around pure functions
**After**: Pure generator function + helper functions

**Benefits**:
- ✅ No unnecessary class
- ✅ Clear data flow (generator pattern)
- ✅ Easy to test (pure functions)
- ✅ No hidden state (step counter is local variable)
- ✅ Foundation for functional SFT/RL loops

---

## Phase 2: Implement Functional SFT Training Loop

**Priority**: 🔴 **HIGH** - Immediate value, validates all components work
**Effort**: ~4-6 hours
**Why**: Fastest path to complete training pipeline

### Tasks:

#### 2.1 Create `training/sft_loop.py` (~2 hours)

Pure function for SFT training orchestration:

```python
# training/sft_loop.py
"""Pure function implementation of SFT training loop.

No classes, no hidden state - just explicit orchestration of
stateful dependencies (backend, data).

Design: Casey Muratori (no retention), Tiger Style (explicit state).
"""

import trio
from typing import List, Dict
from pathlib import Path

from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.types import Sample, TrainingConfig


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

    print(f"Training complete!")
    return metrics_history


def collate_batch(
    samples: List[Sample],
    batch_size: int,
    step: int,
) -> Dict[str, torch.Tensor]:
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


def prepare_sft_batch(samples: List[Sample]) -> Dict[str, torch.Tensor]:
    """Pure function: Convert samples to training batch.

    Args:
        samples: List of Sample objects

    Returns:
        Batch dict with {input_ids, labels, loss_mask}
    """
    import torch

    # Stack tensors
    input_ids = torch.stack([s.input_ids for s in samples])
    labels = torch.stack([s.labels for s in samples])
    loss_mask = torch.stack([s.loss_mask for s in samples])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
    }
```

**Action**: Implement pure SFT training loop function.

---

#### 2.2 Create `examples/run_sft.py` (~1 hour)

End-to-end example demonstrating SFT training:

```python
# examples/run_sft.py
"""Example: Run SFT training with functional loop.

Demonstrates:
- Loading SFT dataset
- Tokenizing conversations
- Running functional SFT training loop
- No classes, just pure functions + stateful dependencies
"""

import sys
import torch
import trio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.sft_loop import run_sft_training
from rollouts.training.sample_prep import tokenize_conversation, compute_loss_mask
from rollouts.training.types import Sample, TrainingConfig


async def main():
    # 1. Load dataset (manual for now, HF datasets later)
    conversations = [
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
        ],
        [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
        ],
        # ... more conversations ...
    ]

    # 2. Tokenize and prepare samples
    tokenizer = ...  # Load your tokenizer
    samples = []

    for conv in conversations:
        tokens, user_spans = tokenize_conversation(conv, tokenizer)
        loss_mask = compute_loss_mask(tokens, user_spans)

        samples.append(Sample(
            input_ids=torch.tensor(tokens),
            labels=torch.tensor(tokens),  # Same for causal LM
            loss_mask=torch.tensor(loss_mask),
        ))

    # 3. Create model and backend
    model = ...  # Your model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=compute_loss,  # Your loss function
        checkpoint_dir=Path("/tmp/sft_checkpoints"),
    )

    # 4. Configure training
    config = TrainingConfig(
        num_steps=1000,
        batch_size=4,
        log_every=100,
        checkpoint_every=500,
    )

    # 5. Run training (pure function!)
    metrics = await run_sft_training(backend, samples, config)

    # 6. Print results
    print(f"\nTraining complete!")
    print(f"  Total steps: {len(metrics)}")
    print(f"  Final loss: {metrics[-1]['loss']:.4f}")


if __name__ == "__main__":
    trio.run(main)
```

**Action**: Create complete example script for SFT training.

---

#### 2.3 Test SFT Loop (~1-2 hours)

Test with `test_pytorch_backend.py` model:

```bash
# Run SFT example
python examples/run_sft.py

# Expected output:
# Starting SFT training...
#   Samples: 100
#   Steps: 1000
#   Batch size: 4
# Step 0: loss=3.2451, grad_norm=2.1234, lr=1.0e-04
# Step 100: loss=2.1234, grad_norm=1.5678, lr=1.0e-04
# ...
# Training complete!
#   Total steps: 1000
#   Final loss: 0.8765
```

**Action**: Verify SFT loop works end-to-end.

---

#### 2.4 Commit (~10 min)

```bash
git add training/sft_loop.py
git add examples/run_sft.py

git commit -m "Implement functional SFT training loop

- Add run_sft_training() pure function (no class!)
- Add collate_batch() and prepare_sft_batch() helpers
- Add examples/run_sft.py demonstrating full SFT pipeline
- Validates D6v1 PyTorchTrainingBackend works end-to-end

Design: Pure function orchestration (Casey Muratori pattern).
No hidden state, explicit dependencies, boring coordination.
"
```

---

### Phase 2 Summary

**Deliverable**: Complete SFT training pipeline
**Time**: ~4-6 hours
**Lines of code**: ~200 lines

**Benefits**:
- ✅ First complete training pipeline
- ✅ Validates all D1-D6 components work together
- ✅ Pure function design (easy to test/modify)
- ✅ Foundation for RL loop

---

## Phase 3: Implement Functional RL Training Loop

**Priority**: 🟡 **MEDIUM** - After SFT works
**Effort**: ~1-2 days
**Why**: Completes SLIME-style RL training

### Tasks:

#### 3.1 Create `training/rl_loop.py` (~3-4 hours)

Pure function for RL training orchestration:

```python
# training/rl_loop.py
"""Pure function implementation of RL training loop.

SLIME-inspired: Generation → Training → Weight Sync loop.
No classes, no hidden state - just explicit orchestration.
"""

import trio
from typing import List, Dict
from pathlib import Path

from rollouts.training.backends import PyTorchTrainingBackend
from training.data_buffer import DataBuffer
from training.async_rollout_manager import AsyncRolloutManager
from training.weight_sync import sync_weights_to_engines
from rollouts.training.types import RLTrainingConfig, InferenceEngine


async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: List[InferenceEngine],
    config: RLTrainingConfig,
) -> List[Dict[str, float]]:
    """Run RL training (pure function, no hidden state).

    Args:
        backend: Training backend (stateful)
        data_buffer: Data buffer (stateful)
        rollout_manager: Rollout manager (stateful)
        inference_engines: Inference engines for weight sync
        config: RL training configuration (immutable)

    Returns:
        List of metrics dicts (one per step)

    Example:
        >>> backend = PyTorchTrainingBackend(...)
        >>> data_buffer = DataBuffer(prompts=[...])
        >>> rollout_manager = AsyncRolloutManager(data_buffer, rollout_config)
        >>> engines = [SGLangEngine(...)]
        >>> config = RLTrainingConfig(num_steps=1000, sync_every=10)
        >>>
        >>> metrics = await run_rl_training(
        ...     backend, data_buffer, rollout_manager, engines, config
        ... )

    SLIME-inspired: Generation → Training → Weight Sync loop.
    Casey Muratori: No retention, explicit flow.
    """
    # Tiger Style: Assert preconditions
    assert config.num_steps > 0, "num_steps must be > 0"
    assert config.sync_every > 0, "sync_every must be > 0"

    metrics_history = []

    print(f"Starting RL training...")
    print(f"  Steps: {config.num_steps}")
    print(f"  Weight sync every: {config.sync_every} steps")

    async with rollout_manager:  # Context manager for cleanup
        for step in range(config.num_steps):
            # SLIME Step 1: Generate rollouts
            batch = await rollout_manager.generate_batch()

            # SLIME Step 2: Compute rewards (pure function)
            rewards = [compute_reward(s) for s in batch.samples]

            # SLIME Step 3: Prepare RL batch (pure function)
            rl_batch = prepare_grpo_batch(batch, rewards, config)

            # SLIME Step 4: Train
            fwd_metrics = await backend.forward_backward(rl_batch).result()
            opt_metrics = await backend.optim_step().result()

            # Combine metrics
            step_metrics = {
                **fwd_metrics,
                **opt_metrics,
                "mean_reward": sum(rewards) / len(rewards),
                "max_reward": max(rewards),
                "min_reward": min(rewards),
                "step": step,
            }
            metrics_history.append(step_metrics)

            # SLIME Step 5: Sync weights to inference engines (D5)
            if step % config.sync_every == 0 and step > 0:
                ckpt_path = await backend.save_checkpoint(step, step_metrics)
                await sync_weights_to_engines(inference_engines, str(ckpt_path))
                print(f"  Synced weights to {len(inference_engines)} engines")

            # Log
            if step % config.log_every == 0:
                print(
                    f"Step {step}: "
                    f"reward={step_metrics['mean_reward']:.2f}, "
                    f"loss={fwd_metrics['loss']:.4f}, "
                    f"grad_norm={fwd_metrics['grad_norm']:.4f}"
                )

            # Checkpoint
            if step % config.checkpoint_every == 0 and step > 0:
                ckpt_path = await backend.save_checkpoint(step, step_metrics)
                print(f"  Saved checkpoint to {ckpt_path}")

    print(f"RL training complete!")
    return metrics_history


def compute_reward(sample: Sample) -> float:
    """Pure function: Compute reward from sample.

    Uses environment grading (for now, reward model later).

    Args:
        sample: Sample with metadata containing grading result

    Returns:
        Reward (1.0 if correct, 0.0 otherwise)
    """
    # Simple environment-based grading
    if sample.metadata.get("correct", False):
        return 1.0
    return 0.0


def prepare_grpo_batch(
    batch: RolloutBatch,
    rewards: List[float],
    config: RLTrainingConfig,
) -> Dict[str, torch.Tensor]:
    """Pure function: Prepare GRPO training batch.

    Args:
        batch: Rollout batch
        rewards: Rewards for each sample
        config: RL config with baseline

    Returns:
        RL training batch with advantages
    """
    import torch

    # Compute advantages (pure function)
    advantages = compute_advantages(rewards, config.baseline)

    # Convert to tensors
    advantage_tensor = torch.tensor(advantages, dtype=torch.float32)

    # Prepare batch (similar to SFT, but with advantages)
    return {
        "input_ids": torch.stack([s.input_ids for s in batch.samples]),
        "labels": torch.stack([s.labels for s in batch.samples]),
        "loss_mask": torch.stack([s.loss_mask for s in batch.samples]),
        "advantages": advantage_tensor,
    }


def compute_advantages(
    rewards: List[float],
    baseline: float = 0.0,
) -> List[float]:
    """Pure function: Compute advantages from rewards.

    Args:
        rewards: List of rewards
        baseline: Baseline for advantage computation

    Returns:
        List of advantages (rewards - baseline)
    """
    return [r - baseline for r in rewards]
```

**Action**: Implement pure RL training loop function.

---

#### 3.2 Implement GRPO Loss (~2 hours)

Add GRPO loss to D6v1 backend or as separate function:

```python
# training/rl_losses.py
"""RL loss functions (pure).

GRPO (Group Relative Policy Optimization) - simplest RL loss.
"""

import torch
import torch.nn.functional as F


def grpo_loss(
    logits: torch.Tensor,  # [batch, seq_len, vocab_size]
    labels: torch.Tensor,  # [batch, seq_len]
    loss_mask: torch.Tensor,  # [batch, seq_len]
    advantages: torch.Tensor,  # [batch]
) -> torch.Tensor:
    """Compute GRPO loss (simplified policy gradient).

    Args:
        logits: Model predictions
        labels: Target labels
        loss_mask: Token-level loss weights
        advantages: Advantage estimates per sample

    Returns:
        Scalar loss
    """
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)

    # Get log probs for target tokens
    batch_size, seq_len, vocab_size = logits.shape
    target_log_probs = log_probs.gather(
        dim=-1,
        index=labels.unsqueeze(-1)
    ).squeeze(-1)

    # Apply loss mask and average over sequence
    masked_log_probs = target_log_probs * loss_mask
    seq_log_probs = masked_log_probs.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)

    # GRPO: policy gradient weighted by advantages
    loss = -(seq_log_probs * advantages).mean()

    return loss
```

**Action**: Implement GRPO loss function.

---

#### 3.3 Create `examples/run_rl.py` (~2 hours)

End-to-end example for RL training:

```python
# examples/run_rl.py
"""Example: Run RL training with functional loop.

Demonstrates:
- Setting up DataBuffer, AsyncRolloutManager, inference engines
- Running functional RL training loop
- Environment-based reward computation
- Weight sync to inference engines (D5)
"""

import sys
import trio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training.rl_loop import run_rl_training
from training.data_buffer import DataBuffer
from training.async_rollout_manager import AsyncRolloutManager
from training.weight_sync import SGLangEngine
from rollouts.training.types import RolloutConfig, RLTrainingConfig


async def main():
    # 1. Create data buffer
    prompts = [
        "Solve 2+2",
        "Calculate 5*7",
        "What is 10-3?",
        # ... more math prompts
    ]
    data_buffer = DataBuffer(prompts=prompts)

    # 2. Create rollout manager
    rollout_config = RolloutConfig(
        batch_size=4,
        generate_fn=my_agent.run,  # Your agent
    )
    rollout_manager = AsyncRolloutManager(data_buffer, rollout_config)

    # 3. Create inference engines
    engines = [
        SGLangEngine(base_url="http://localhost:30000"),
    ]

    # 4. Create training backend
    model = ...  # Your model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=grpo_loss,  # GRPO loss
        checkpoint_dir=Path("/tmp/rl_checkpoints"),
    )

    # 5. Configure RL training
    config = RLTrainingConfig(
        num_steps=1000,
        sync_every=10,  # Sync weights every 10 steps
        baseline=0.5,  # Baseline for advantages
        log_every=10,
        checkpoint_every=100,
    )

    # 6. Run RL training (pure function!)
    metrics = await run_rl_training(
        backend=backend,
        data_buffer=data_buffer,
        rollout_manager=rollout_manager,
        inference_engines=engines,
        config=config,
    )

    # 7. Print results
    print(f"\nRL training complete!")
    print(f"  Total steps: {len(metrics)}")
    print(f"  Final mean reward: {metrics[-1]['mean_reward']:.2f}")
    print(f"  Final loss: {metrics[-1]['loss']:.4f}")


if __name__ == "__main__":
    trio.run(main)
```

**Action**: Create complete example script for RL training.

---

#### 3.4 Test RL Loop (~2-3 hours)

Test with environment grading:

```bash
# Run RL example
python examples/run_rl.py

# Expected output:
# Starting RL training...
#   Steps: 1000
#   Weight sync every: 10 steps
# Step 0: reward=0.45, loss=2.1234, grad_norm=1.5678
# Step 10: reward=0.52, loss=1.8765, grad_norm=1.2345
#   Synced weights to 1 engines
# ...
# RL training complete!
#   Total steps: 1000
#   Final mean reward: 0.78
#   Final loss: 0.5432
```

**Action**: Verify RL loop works end-to-end with weight sync.

---

#### 3.5 Commit (~10 min)

```bash
git add training/rl_loop.py
git add training/rl_losses.py
git add examples/run_rl.py

git commit -m "Implement functional RL training loop

- Add run_rl_training() pure function (SLIME-style)
- Add compute_reward(), prepare_grpo_batch(), compute_advantages() helpers
- Add grpo_loss() implementation (simplified policy gradient)
- Add examples/run_rl.py demonstrating full RL pipeline
- Integrates D3 (DataBuffer) + D4 (AsyncRolloutManager) + D5 (weight sync) + D6v1

Design: Pure function orchestration with explicit SLIME loop.
Generation → Training → Weight Sync, no hidden state.
"
```

---

### Phase 3 Summary

**Deliverable**: Complete RL training pipeline
**Time**: ~1-2 days
**Lines of code**: ~350 lines

**Benefits**:
- ✅ First complete SLIME-style RL pipeline
- ✅ Validates all D1-D6 components work together
- ✅ Pure function design (easy to test/modify)
- ✅ Environment-based rewards (no reward model needed yet)

---

## Timeline Summary

| Phase | Task | Time | Priority |
|-------|------|------|----------|
| **Phase 1** | Refactor RolloutManager | ~2-3 hours | 🔴 CRITICAL |
| **Phase 2** | Implement SFT Loop | ~4-6 hours | 🔴 HIGH |
| **Phase 3** | Implement RL Loop | ~1-2 days | 🟡 MEDIUM |
| **TOTAL** | | **~2-3 days** | |

---

## Success Criteria

### Phase 1 Complete:
- ✅ RolloutManager deprecated with warning
- ✅ Pure functions extracted to `rollout_generation.py`
- ✅ AsyncRolloutManager uses pure functions
- ✅ Examples updated to use generator pattern
- ✅ Committed and tested

### Phase 2 Complete:
- ✅ `run_sft_training()` pure function works
- ✅ Example `run_sft.py` runs end-to-end
- ✅ Can train any model on any SFT dataset
- ✅ Validates D6v1 PyTorchTrainingBackend
- ✅ Committed and tested

### Phase 3 Complete:
- ✅ `run_rl_training()` pure function works
- ✅ GRPO loss implemented and tested
- ✅ Example `run_rl.py` runs end-to-end with weight sync
- ✅ Can do SLIME-style RL training
- ✅ Committed and tested

---

## Design Philosophy Reinforced

**Classes are for STATE MANAGEMENT, not code organization.**

### Legitimate Stateful Classes (KEEP):
- ✅ `DataBuffer` - Iteration state (`epoch_id`, `sample_offset`)
- ✅ `AsyncRolloutManager` - Async coordination state (`partial_samples`, `_abort_requested`)
- ✅ `PyTorchTrainingBackend` - Training state (`model.parameters()`, `optimizer.state`, `weight_version`)

### Pure Functions (EVERYTHING ELSE):
- ✅ Training loops: `run_sft_training()`, `run_rl_training()`
- ✅ Data prep: `collate_batch()`, `prepare_sft_batch()`, `prepare_grpo_batch()`
- ✅ Loss computation: `grpo_loss()`, `compute_advantages()`
- ✅ Reward computation: `compute_reward()`
- ✅ Rollout generation: `generate_rollout_batches()`

**Rule**: If you can implement it as a pure function or generator, DO THAT FIRST.

Only reach for classes when you have real mutable state to manage.

---

## Next Actions

**Start Phase 1 immediately** (after user approval):

1. Create `training/rollout_generation.py`
2. Extract pure functions from `rollout_manager.py`
3. Update `AsyncRolloutManager` imports
4. Add deprecation warning to `RolloutManager`
5. Update examples
6. Commit

**Expected time to first commit**: ~2-3 hours

---

## Questions Before Starting

None - plan is clear and actionable. Ready to execute Phase 1.

**Ready to start?** 🚀
