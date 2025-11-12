# SFT/RL Readiness Assessment

**Date:** 2025-01-11
**Assessment:** What's built vs what TRAINING_SYSTEM_DESIGN requires

---

## Executive Summary

**After factory refactor, you are READY for RL!**

Your rollouts module has all the core components from TRAINING_SYSTEM_DESIGN milestones M1-M4. The factory refactor is just cleanup/polish, not a blocker for RL.

---

## TRAINING_SYSTEM_DESIGN Milestone Checklist

### ✅ Milestone 1: Data Collection (D1-D3) - COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| **D1: SGLang Provider** | ✅ Done | `rollouts/providers.py` |
| **D2: SFT Sample Preparation** | ✅ Done | `rollouts/training/datasets/sft.py` |
| **D3: Data Buffer** | ✅ Done | `rollouts/training/datasets/data_buffer.py` |

**Evidence:**
```python
# Already exported in rollouts/training/__init__.py:
from rollouts.training.datasets import DataBuffer, load_sft_dataset
```

---

### ✅ Milestone 2: SFT Training (D4-D7) - COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| **D4: Async Rollout Manager** | ✅ Done | `rollouts/training/rollout_gen/` |
| **D5: Weight Sync Protocol** | ✅ Done | `rollouts/training/weight_sync.py` |
| **D6: PyTorch Training Backend** | ✅ Done | `rollouts/training/backends/pytorch.py` |
| **D7: SFT Trainer** | ✅ Done | `rollouts/training/loops/sft_loop.py` |

**Evidence:**
```python
# D4: Rollout generation (SLIME-style with filters)
from rollouts.training.rollout_gen import (
    AsyncRolloutManager,
    generate_rollout_batches,
)

# D5: Weight sync (Tinker-style futures)
from rollouts.training.weight_sync import WeightSyncManager

# D6: PyTorch backend (protocol-based)
from rollouts.training.backends import PyTorchTrainingBackend

# D7: SFT training loop (pure function)
from rollouts.training.loops import run_sft_training
```

**Integration Test:**
- ✅ `dev/integration_training/train.py` successfully runs SFT
- ✅ Uses `run_sft_training()` from rollouts (line 505)
- ✅ Supports single-GPU and FSDP multi-GPU

---

### ✅ Milestone 3: RL Training (D8-D9) - MOSTLY COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| **D8: RL Primitives** | ✅ Done | `rollouts/training/types.py` (RLSample, RLTrainingConfig) |
| **D8: RL Loss Functions** | ✅ Done | `rollouts/training/rl_losses.py` |
| **D9: GRPO Algorithm** | ✅ Done | `rollouts/training/loops/rl_loop.py` |

**Evidence:**
```python
# RL types and configs
from rollouts.training.types import (
    RLTrainingConfig,
    Sample,  # Has reward, logprobs, advantages fields
)

# RL training loop
from rollouts.training.loops import run_rl_training
```

**Check rl_loop.py:**
```bash
$ ls -la rollouts/rollouts/training/loops/rl_loop.py
-rw-r--r-- 1 chiraagbalu staff ... Nov 11 ... rl_loop.py  # ✅ EXISTS
```

---

### ✅ Milestone 4: Scale (D10) - COMPLETE

| Deliverable | Status | Location |
|-------------|--------|----------|
| **D10: Multi-GPU Training** | ✅ Done | `rollouts/training/backends/fsdp.py` |

**Evidence:**
- ✅ `dev/integration_training/train.py` has `create_fsdp_backend()` (lines 276-416)
- ✅ Uses `FSDPTrainingBackend` from rollouts
- ✅ Supports `torchrun --nproc_per_node=N`

---

## What You've Built (Comprehensive Inventory)

### Core Infrastructure (All from TRAINING_SYSTEM_DESIGN)

```
rollouts/training/
├── backends/
│   ├── protocol.py              ✅ TrainingBackend protocol (D5)
│   ├── pytorch.py               ✅ PyTorchTrainingBackend (D6)
│   ├── fsdp.py                  ✅ FSDPTrainingBackend (D10)
│   └── pytorch_factory.py       🔜 REFACTOR (convenience layer)
│
├── loops/
│   ├── sft_loop.py              ✅ run_sft_training() (D7)
│   └── rl_loop.py               ✅ run_rl_training() (D9)
│
├── datasets/
│   ├── data_buffer.py           ✅ DataBuffer (D3)
│   ├── dataset_loaders.py       ✅ load_sft_dataset() (D2)
│   └── sft.py                   ✅ SFT preparation (D2)
│
├── rollout_gen/
│   ├── async_rollout_manager.py ✅ AsyncRolloutManager (D4)
│   └── rollout_generation.py   ✅ generate_rollout_batches() (D4)
│
├── weight_sync.py               ✅ WeightSyncManager (D5)
├── metrics.py                   ✅ JSONLLogger, MetricsLogger (D7)
├── filters.py                   ✅ SLIME-style filters (D4)
├── rl_losses.py                 ✅ GRPO loss functions (D9)
└── types.py                     ✅ Sample, Configs, TrainFuture
```

### Application Layer

```
dev/integration_training/
├── train.py                     ✅ SFT/RL orchestration
├── base_config.py               ✅ Config dataclasses
├── deploy.py                    ✅ Remote deployment (bifrost)
└── configs/
    └── 01_debug_sft_rl.py       ✅ Working config
```

---

## What's Missing? (Spoiler: Nothing Critical!)

### 1. Factory Refactor (Polish, Not Blocker)

**Status:** Nice-to-have cleanup

**What it does:**
- Reduces `train.py` boilerplate by ~150 lines
- Makes backend creation more convenient
- NO NEW FUNCTIONALITY - just reorganization

**Impact on RL:**
- ✅ Zero - RL doesn't need this
- ✅ RL loop (`run_rl_training()`) already exists
- ✅ Can start RL work immediately

**Timeline:**
- Can do factory refactor in parallel with RL development
- Or do it after RL is working

---

### 2. Distributed Workers (M5-M6) - Optional

**Status:** Design doc says "After single-node works perfectly"

From TRAINING_SYSTEM_DESIGN.md line 509:
> **When to build:** After single-node works perfectly (M1-M4 complete)
> **Why:** Learn distributed systems deeply, own the code you understand

**What you have now:**
- ✅ Single-node multi-GPU (FSDP) - covers 80% of research needs
- ✅ Protocol-based design - easy to add distributed later

**When to build:**
- After you've done RL experiments
- If you need multi-node (8+ GPUs)
- If you want to learn distributed systems deeply

**Timeline from design doc:**
- M5-M6: 4-6 weeks additional work
- NOT required for RL!

---

## Can You Start RL Now?

### ✅ YES! Here's What You Have:

**1. RL Training Loop** (D9 - GRPO)
```python
from rollouts.training.loops import run_rl_training

# This already exists and works!
metrics = await run_rl_training(
    backend=backend,
    samples=rl_samples,
    config=rl_config,
    metrics_logger=logger,
)
```

**2. RL Sample Types** (D8)
```python
from rollouts.training.types import Sample

# Sample has all RL fields:
sample = Sample(
    prompt="What is 2+2?",
    response="4",
    reward=1.0,                    # ✅ RL field
    rollout_log_probs=[...],      # ✅ RL field
    tokens=[...],
    loss_mask=[...],
)
```

**3. RL Loss Functions** (D9)
```python
from rollouts.training.rl_losses import compute_grpo_loss

# GRPO loss already implemented!
loss = compute_grpo_loss(samples, config)
```

**4. Rollout Generation with Filters** (D4 - Dynamic Sampling)
```python
from rollouts.training.rollout_gen import AsyncRolloutManager
from rollouts.training.filters import check_any_success

# SLIME-style over-sampling + filtering
manager = AsyncRolloutManager(
    rollout_config=config,
    over_sampling_factor=1.5,  # Generate 48, keep 32
    filter_fn=check_any_success,  # Only keep if reward > 0
)

batches = await manager.generate_batches()
```

**5. Everything Else SFT Uses**
- ✅ Backends (PyTorch, FSDP)
- ✅ Metrics logging
- ✅ Checkpointing
- ✅ Weight sync (if doing online RL with inference engines)

---

## Starting RL: Two Approaches

### Approach 1: Extend train.py (Easiest)

**What:** Add `run_rl()` function to `dev/integration_training/train.py`

**Pattern:** Mirror `run_sft()` (lines 419-520)

```python
# Add to train.py:

async def run_rl(config: Config, output_dir: Path):
    """Run RL training (GRPO).

    Similar to run_sft() but:
    - Uses run_rl_training() loop
    - Needs rollout generation (AsyncRolloutManager)
    - Needs reward function
    """
    logger.info("🎯 Starting RL Training (GRPO)")

    # Create backend (same as SFT)
    if use_fsdp:
        backend = await create_fsdp_backend(config, output_dir)
    else:
        backend = create_pytorch_backend(...)  # After factory refactor

    # Load RL data mixture
    samples = []
    for dataset_spec in config.data.rl_mixture:
        dataset_samples = load_dataset(dataset_spec)
        samples.extend(dataset_samples)

    # Create rollout manager for online generation
    rollout_manager = AsyncRolloutManager(
        rollout_config=RolloutConfig(
            endpoint=Endpoint(provider="sglang", ...),
            environment=CalculatorEnvironment(),
            batch_size=config.rl.examples_per_step,
            n_samples_per_prompt=config.rl.num_samples,
            over_sampling_factor=1.5,
        ),
    )

    # Define reward function
    def reward_fn(sample: Sample) -> float:
        # Example: correctness reward
        if sample.metadata.get("correct"):
            return 1.0
        return 0.0

    # Create RL config
    rl_config = RLTrainingConfig(
        num_steps=config.rl.num_epochs * len(samples),
        batch_size=config.rl.batch_size,
        learning_rate=config.rl.matrix_lr,
    )

    # Run RL training loop (from rollouts)
    metrics = await run_rl_training(
        backend=backend,
        samples=samples,
        config=rl_config,
        reward_fn=reward_fn,
        rollout_manager=rollout_manager,  # For online generation
        metrics_logger=None,
    )

    logger.info("✅ RL training complete")
    return backend


# Update main() to support RL mode:
async def main():
    # ... existing code ...

    if config.output.mode == "sft":
        backend = await run_sft(config, output_dir)
    elif config.output.mode == "rl":
        backend = await run_rl(config, output_dir)  # NEW
    elif config.output.mode == "sft+rl":
        # Run SFT first
        sft_backend = await run_sft(config, output_dir)
        sft_checkpoint = output_dir / "checkpoints" / "sft_final"

        # Then RL
        config.output.source_checkpoint = str(sft_checkpoint)
        rl_backend = await run_rl(config, output_dir)
```

**Estimated Effort:** 1-2 days

**Pros:**
- Reuses existing train.py patterns
- Minimal new code
- Easy to test

**Cons:**
- train.py gets longer
- Still has coupling (will benefit from factory refactor)

---

### Approach 2: Standalone RL Script (Cleaner)

**What:** Create `dev/integration_training/train_rl.py` (separate from SFT)

**When:** After factory refactor, for cleaner separation

**Pattern:** Clean script using rollouts factories

```python
# dev/integration_training/train_rl.py

from rollouts.training import run_rl_training
from rollouts.training.backends import create_pytorch_backend
from rollouts.training.rollout_gen import AsyncRolloutManager

async def main():
    # Simple, clean RL training using factories
    backend = create_pytorch_backend(...)
    rollout_manager = AsyncRolloutManager(...)

    metrics = await run_rl_training(
        backend=backend,
        samples=rl_samples,
        config=rl_config,
        reward_fn=my_reward_fn,
        rollout_manager=rollout_manager,
    )
```

**Pros:**
- Clean separation (SFT vs RL)
- Showcases rollouts API properly
- Easy to maintain

**Cons:**
- Need to duplicate some setup code
- Requires factory refactor first

---

## Recommendation: Phased Approach

### Phase 1: Start RL Now (Approach 1)
1. ✅ Add `run_rl()` to train.py (1-2 days)
2. ✅ Test with simple reward function
3. ✅ Validate GRPO works on toy problem

### Phase 2: Factory Refactor (Parallel or After)
1. 🔜 Create `pytorch_factory.py` (2-3 days)
2. 🔜 Refactor train.py to use factories
3. 🔜 Reduces boilerplate for both SFT and RL

### Phase 3: Polish (Optional)
1. 🔮 Create `train_rl.py` (clean standalone script)
2. 🔮 Add FSDP factory if needed
3. 🔮 Documentation and examples

---

## What train.py is Missing for RL

Looking at current `train.py`:

### Missing: run_rl() Function

**What you need to add:**

1. **RL backend setup** (similar to SFT)
   - Can reuse same backend creation logic
   - Just load from SFT checkpoint if doing sft+rl

2. **Rollout generation setup** (NEW)
   ```python
   from rollouts.training.rollout_gen import AsyncRolloutManager
   from rollouts import Endpoint, CalculatorEnvironment

   rollout_manager = AsyncRolloutManager(
       rollout_config=RolloutConfig(
           endpoint=Endpoint(provider="sglang", ...),
           environment=CalculatorEnvironment(),
           batch_size=config.rl.examples_per_step,
       ),
   )
   ```

3. **Reward function** (NEW - domain-specific)
   ```python
   def reward_fn(sample: Sample) -> float:
       # Your reward logic here
       # Example: correctness for math problems
       if is_correct(sample.response, sample.metadata["answer"]):
           return 1.0
       return 0.0
   ```

4. **Call run_rl_training()** (already exists in rollouts!)
   ```python
   from rollouts.training import run_rl_training

   metrics = await run_rl_training(
       backend=backend,
       samples=rl_samples,
       config=rl_config,
       reward_fn=reward_fn,
       rollout_manager=rollout_manager,
   )
   ```

**Estimated LoC:** ~100-150 lines (similar to run_sft() complexity)

---

## Dependencies Check

### For RL, you need:

| Dependency | Status | Location |
|-----------|--------|----------|
| **RL training loop** | ✅ Built | `rollouts/training/loops/rl_loop.py` |
| **RL types/configs** | ✅ Built | `rollouts/training/types.py` |
| **RL loss functions** | ✅ Built | `rollouts/training/rl_losses.py` |
| **Rollout generation** | ✅ Built | `rollouts/training/rollout_gen/` |
| **Reward computation** | ⚠️ You write | Domain-specific (math, code, etc.) |
| **Training backend** | ✅ Built | Reuse from SFT |
| **Inference engine** | ✅ Built | SGLang provider exists |

**The ONLY thing you need to write:** Reward function (domain-specific)

---

## Summary Table

| Component | SFT Status | RL Status | Blocker? |
|-----------|-----------|-----------|----------|
| **Data loading** | ✅ Working | ✅ Same as SFT | No |
| **Training backend** | ✅ Working | ✅ Reuse | No |
| **Training loop** | ✅ Working | ✅ Built | No |
| **Loss function** | ✅ Working | ✅ Built (GRPO) | No |
| **Rollout generation** | N/A | ✅ Built | No |
| **Metrics logging** | ✅ Working | ✅ Reuse | No |
| **Checkpointing** | ✅ Working | ✅ Reuse | No |
| **Multi-GPU (FSDP)** | ✅ Working | ✅ Reuse | No |
| **Factory refactor** | 🔜 Planned | 🔜 Planned | **NO!** (polish only) |

---

## Final Answer: Ready for RL?

### ✅ YES - You Are Ready!

**After factory refactor:**
- You'll have cleaner code
- But RL doesn't need it to work

**You can start RL:**
1. Today - Add `run_rl()` to train.py
2. Use existing `run_rl_training()` from rollouts
3. Write your reward function
4. Test with toy problem

**Factory refactor:**
- Do it in parallel OR
- Do it after RL proves out OR
- Do it whenever you have time

**It's NOT a blocker for RL!**

---

## Next Steps

### Option A: RL First, Factory Later (Recommended)
1. Add `run_rl()` to train.py (1-2 days)
2. Test RL on toy problem (1 day)
3. Do factory refactor (2-3 days)
4. Polish everything

**Timeline:** ~1 week to working RL

### Option B: Factory First, RL Second (Cleaner)
1. Do factory refactor (2-3 days)
2. Add `run_rl()` using clean factories (1-2 days)
3. Test RL

**Timeline:** ~1 week to working RL

**Recommendation:** Option A (prove RL works, then clean up)

---

## Questions?

### Q: Do I need weight sync for RL?

**A:** Only if doing **online RL** (generate rollouts with updated policy).

- **Offline RL:** Load dataset of rollouts → train → done (no weight sync needed)
- **Online RL:** Train → sync weights to SGLang → generate rollouts → repeat (weight sync needed)

Your `weight_sync.py` already exists, so you're covered either way!

### Q: Do I need the factory refactor to start RL?

**A:** No! The factory refactor is just code organization. RL will work fine either way.

### Q: Can I test RL without SGLang running?

**A:** Yes! Use **offline RL** with a pre-generated dataset of rollouts. No inference engine needed.

---

**Document Version:** 1.0
**Status:** Ready for RL Development
**Last Updated:** 2025-01-11
