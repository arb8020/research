# Functional vs OOP Design Philosophy

**Date**: 2025-11-09

## Design Principle: Minimize Stateful Classes

Following Casey Muratori's philosophy: "State is the root of all evil."

**Rule of thumb**: Use classes ONLY when you have legitimate mutable state to manage. Otherwise, use pure functions.

## Current State Analysis

### ✅ Correctly Stateful (Classes with Legitimate Mutable State)

| Component | State | Why Class? | Justification |
|-----------|-------|------------|---------------|
| `DataBuffer` | `epoch_id`, `sample_offset` | Tracks position in dataset | ✅ Legitimate: Must remember where we are in epoch |
| `AsyncRolloutManager` | `partial_samples`, `_abort_requested` | Caches incomplete rollouts | ✅ Legitimate: Handles async abort/resume |
| `PyTorchTrainingBackend` | `model`, `optimizer`, `weight_version`, `current_step` | Training state | ✅ Legitimate: PyTorch is inherently stateful |

**Pattern**: These are "stateful resources" - they manage mutable state that MUST persist across calls.

### ✅ Correctly Stateless (Pure Data / Pure Functions)

| Component | Type | Why Not Class? |
|-----------|------|----------------|
| `Sample` | `@dataclass(frozen=True)` | Pure data, immutable | ✅ Correct: Just a value object |
| `RolloutBatch` | `@dataclass` | Collection of samples | ✅ Correct: Just a container |
| `RolloutConfig` | `@dataclass` | Configuration | ✅ Correct: Just settings |
| `compute_loss_mask()` | Pure function | Deterministic computation | ✅ Correct: No state needed |
| `tokenize_conversation()` | Pure function | Deterministic computation | ✅ Correct: No state needed |
| `sync_weights_to_engines()` | Pure function | Stateless orchestration | ✅ Correct: No retention |

**Pattern**: These are pure values or pure functions - deterministic, no hidden state.

### ⚠️ Questionable (Might Be Over-Engineered)

| Component | Current | Issue | Better Design? |
|-----------|---------|-------|----------------|
| `RolloutManager` (old) | Class | Only wraps pure functions | ✅ Already deprecated, replaced with pure functions |

## The Training Loop Question

### What We Wrote in Gap Analysis:

```python
# Gap analysis suggested:
class SFTTrainer:
    def train(self):
        for step in range(num_steps):
            batch = self.get_batch()
            metrics = await self.backend.forward_backward(batch).result()
            # ...
```

### The Critical Question:

**Does `SFTTrainer` have legitimate mutable state?**

Let's analyze:

```python
# Potential state in SFTTrainer:
- current_step: int           # Is this state or just a loop variable?
- metrics_history: list       # Is this state or just accumulated results?
- checkpoint_schedule: ...    # Is this state or just config?
- backend: TrainingBackend    # Is this state or just a dependency?
- data_buffer: DataBuffer     # Is this state or just a dependency?
```

**Analysis:**
- `current_step`: Just a loop counter (not real state!)
- `metrics_history`: Just accumulated return values (not state!)
- `backend`, `data_buffer`: Dependencies that have their own state

**Conclusion: `SFTTrainer` has NO legitimate mutable state of its own!**

## Functional Alternative: Training Loop as Pure Function

### Option 1: Pure Function (Recommended)

```python
async def run_sft_training(
    backend: PyTorchTrainingBackend,
    samples: list[Sample],
    config: TrainingConfig,
) -> list[dict[str, float]]:
    """Run SFT training (pure function, no hidden state).

    Args:
        backend: Training backend (has its own state)
        samples: Training samples (immutable)
        config: Training configuration (immutable)

    Returns:
        List of metrics dicts (one per step)

    Casey Muratori: No retention, explicit inputs/outputs.
    Sean Goedecke: Boring coordination, no magic.
    """
    # Tiger Style: Assert preconditions
    assert len(samples) > 0, "samples cannot be empty"
    assert config.num_steps > 0, "num_steps must be > 0"

    metrics_history = []

    for step in range(config.num_steps):
        # Get batch (pure function)
        batch = collate_batch(samples, config.batch_size, step)

        # Train (backend has state, but we don't!)
        fwd_metrics = await backend.forward_backward(batch).result()
        opt_metrics = await backend.optim_step().result()

        # Combine metrics (pure)
        step_metrics = {**fwd_metrics, **opt_metrics, "step": step}
        metrics_history.append(step_metrics)

        # Log (side effect, but explicit)
        if step % config.log_every == 0:
            print(f"Step {step}: loss={fwd_metrics['loss']:.4f}")

        # Checkpoint (side effect, but explicit)
        if step % config.checkpoint_every == 0:
            await backend.save_checkpoint(step, step_metrics)

    return metrics_history


# Usage (clear, explicit):
metrics = await run_sft_training(
    backend=backend,
    samples=my_samples,
    config=TrainingConfig(num_steps=1000, batch_size=4),
)
```

**Benefits:**
- ✅ No hidden state
- ✅ Easy to test (pure function)
- ✅ Easy to understand (just read top to bottom)
- ✅ Easy to modify (no class inheritance nonsense)
- ✅ Explicit dependencies (backend, samples, config)

### Option 2: RL Training Loop (Also Pure Function)

```python
async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: list[InferenceEngine],
    config: RLTrainingConfig,
) -> list[dict[str, float]]:
    """Run RL training (pure function, no hidden state).

    Args:
        backend: Training backend (stateful)
        data_buffer: Data buffer (stateful)
        rollout_manager: Rollout manager (stateful)
        inference_engines: Inference engines for weight sync
        config: RL training configuration (immutable)

    Returns:
        List of metrics dicts (one per step)

    SLIME-inspired: Generation → Training → Weight Sync loop.
    Casey Muratori: No retention, explicit flow.
    """
    metrics_history = []

    for step in range(config.num_steps):
        # Generate rollouts (rollout_manager has state)
        batch = await rollout_manager.generate_batch()

        # Compute rewards (pure function)
        rewards = [compute_reward(s) for s in batch.samples]

        # Prepare RL batch (pure function)
        rl_batch = prepare_grpo_batch(batch, rewards, config)

        # Train (backend has state)
        fwd_metrics = await backend.forward_backward(rl_batch).result()
        opt_metrics = await backend.optim_step().result()

        # Combine metrics
        step_metrics = {
            **fwd_metrics,
            **opt_metrics,
            "mean_reward": sum(rewards) / len(rewards),
            "step": step,
        }
        metrics_history.append(step_metrics)

        # Sync weights to inference engines (D5, stateless)
        if step % config.sync_every == 0:
            ckpt_path = await backend.save_checkpoint(step, step_metrics)
            await sync_weights_to_engines(inference_engines, str(ckpt_path))

        # Log
        if step % config.log_every == 0:
            print(f"Step {step}: reward={step_metrics['mean_reward']:.2f}")

    return metrics_history
```

**Benefits:**
- ✅ Clear SLIME-style loop (generate → train → sync)
- ✅ All state is in dependencies (buffer, manager, backend)
- ✅ No hidden state in the orchestration layer
- ✅ Easy to add custom logic (just modify the function)

## Design Rules (Going Forward)

### Rule 1: State Inventory

Before creating a class, ask:

1. **What mutable state does this manage?**
2. **Is this state that MUST persist across calls?**
3. **Or is it just accumulated return values / config?**

If it's #3, use a pure function.

### Rule 2: Stateful Components (Allowed Classes)

These are the ONLY legitimate stateful components in a training system:

| Component | Mutable State | Justification |
|-----------|---------------|---------------|
| **DataBuffer** | `epoch_id`, `sample_offset` | Iterates through dataset |
| **TrainingBackend** | `model.parameters()`, `optimizer.state`, `weight_version`, `current_step` | Training state (PyTorch/JAX inherently stateful) |
| **RolloutManager** | `partial_samples`, `_abort_requested` | Async rollout state |
| **Metrics Logger** (future) | `metrics_history`, `file_handle` | Accumulates/persists metrics |

**That's it. Everything else should be pure functions or pure data.**

### Rule 3: Orchestration = Pure Functions

Training loops, pipelines, workflows → **Pure functions**.

Why?
- Easier to test (no setup/teardown)
- Easier to understand (no hidden state)
- Easier to compose (just function calls)
- Easier to parallelize (no shared state)

### Rule 4: Configuration = Immutable Data

```python
# Good:
@dataclass(frozen=True)
class TrainingConfig:
    num_steps: int
    batch_size: int
    log_every: int = 100

# Bad:
class TrainingConfig:
    def __init__(self):
        self.num_steps = None  # Mutable!

    def set_num_steps(self, n):
        self.num_steps = n  # Why is config mutable?!
```

## Refactored Design for Training Loops

### SFT Training

```python
# training/sft_loop.py (new file, pure functions)

async def run_sft_training(
    backend: PyTorchTrainingBackend,
    samples: list[Sample],
    config: TrainingConfig,
) -> list[dict[str, float]]:
    """Pure function for SFT training loop."""
    # ... (implementation above)


def collate_batch(
    samples: list[Sample],
    batch_size: int,
    step: int,
) -> dict[str, torch.Tensor]:
    """Pure function for batching (no state!)."""
    start_idx = (step * batch_size) % len(samples)
    end_idx = start_idx + batch_size
    batch_samples = samples[start_idx:end_idx]

    # ... collate logic
    return batch


def prepare_sft_batch(samples: list[Sample]) -> dict[str, torch.Tensor]:
    """Pure function to convert samples to training batch."""
    # ... (pure function, no state)
```

### RL Training

```python
# training/rl_loop.py (new file, pure functions)

async def run_rl_training(
    backend: PyTorchTrainingBackend,
    data_buffer: DataBuffer,
    rollout_manager: AsyncRolloutManager,
    inference_engines: list[InferenceEngine],
    config: RLTrainingConfig,
) -> list[dict[str, float]]:
    """Pure function for RL training loop."""
    # ... (implementation above)


def compute_reward(sample: Sample) -> float:
    """Pure function to compute reward from sample."""
    # Use environment grading
    if sample.metadata.get("correct"):
        return 1.0
    return 0.0


def prepare_grpo_batch(
    batch: RolloutBatch,
    rewards: list[float],
    config: RLTrainingConfig,
) -> dict[str, torch.Tensor]:
    """Pure function to prepare GRPO training batch."""
    # Compute advantages (pure!)
    advantages = compute_advantages(rewards, config.baseline)

    # ... rest is pure computation
    return rl_batch


def compute_advantages(
    rewards: list[float],
    baseline: float = 0.0,
) -> list[float]:
    """Pure function: advantages = rewards - baseline."""
    return [r - baseline for r in rewards]
```

## Summary: Functional-First Design

### Stateful Classes (Minimal Set):
1. `DataBuffer` - iteration state
2. `TrainingBackend` - model/optimizer state
3. `AsyncRolloutManager` - rollout cache state
4. `MetricsLogger` (future) - accumulated metrics

### Pure Functions (Everything Else):
1. Training loops (`run_sft_training`, `run_rl_training`)
2. Data preparation (`collate_batch`, `prepare_grpo_batch`)
3. Loss computation (`compute_loss_mask`, `compute_advantages`)
4. Reward computation (`compute_reward`)
5. Weight sync orchestration (`sync_weights_to_engines`)

### Design Principles Applied:

- **Casey Muratori**: Minimize state, maximize pure functions
- **Tiger Style**: Explicit inputs/outputs, no hidden state
- **Sean Goedecke**: Boring coordination (just function calls)
- **Tinker**: Minimal surface area (few classes, many functions)

### Benefits:

1. **Testability**: Pure functions are trivial to test
2. **Composability**: Easy to combine/reuse functions
3. **Debuggability**: No hidden state to track
4. **Maintainability**: Clear data flow, no surprises
5. **Performance**: Easy to parallelize pure functions

## Recommendation

**Implement training loops as pure functions**, not classes:

```python
# training/sft_loop.py
async def run_sft_training(...) -> list[dict]:
    """SFT training loop (pure function)"""

# training/rl_loop.py
async def run_rl_training(...) -> list[dict]:
    """RL training loop (pure function)"""

# examples/run_sft.py
async def main():
    backend = PyTorchTrainingBackend(...)
    samples = load_samples(...)
    config = TrainingConfig(...)

    metrics = await run_sft_training(backend, samples, config)
    print(f"Training complete! Final loss: {metrics[-1]['loss']}")
```

**No `SFTTrainer` class. No `RLTrainer` class. Just pure functions.**

This is cleaner, simpler, and more in line with your existing design philosophy.
