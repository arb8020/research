# Gap Analysis: nanochat/SLIME-style Post-Training Run

**Goal**: Implement a complete SFT + RL post-training pipeline (nanochat-like speedrun.sh)

**Date**: 2025-11-09

## What nanochat does

```bash
# nanochat speedrun.sh (4 hours on 8xH100, ~$100)
1. Tokenizer training (custom BPE, 65K vocab)
2. Base model pretraining (560M params, 11.2B tokens)
3. Midtraining (SmolTalk + MMLU + GSM8K, 568K rows)
4. SFT (chat alignment)
5. RL (optional GRPO on GSM8K)
6. Evaluation (CORE benchmarks)
7. Serving (CLI + Web UI)
```

## What SLIME does

```python
# SLIME's 3-component architecture
1. Data Buffer: Manages prompts, handles epoch/sampling
2. Rollout Module (SGLang): Generates responses + rewards
3. Training Module (Megatron): Trains model, syncs weights

# Main loop:
while training:
    prompts = data_buffer.get_batch()
    rollouts = rollout_module.generate(prompts)  # SGLang inference
    metrics = training_module.train(rollouts)     # Megatron training
    training_module.sync_weights(rollout_module)  # Weight update
```

## What WE have (Current State)

### ✅ Completed Components

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| **D1: Rollout Generation** | ✅ Done | `rollouts/agents.py`, `rollouts/environments/` | Full agent framework with tool calling |
| **D2: SFT Sample Prep** | ✅ Done | `rollouts/training/sample_prep.py`, `training/sft.py` | Tokenization, loss masking, JSONL export |
| **D3: Data Buffer** | ✅ Done | `training/data_buffer.py` | SLIME-style stateful prompt iteration |
| **D4: Rollout Manager** | ✅ Done | `training/async_rollout_manager.py` | Async parallel generation, dynamic sampling |
| **D5: Weight Sync** | ✅ Done | `training/weight_sync.py` | SGLang/vLLM weight sync (stateless) |
| **D6v1: Training Backend** | ✅ Done | `rollouts/training/backends/pytorch.py` | PyTorch training with checkpointing |

### ⚠️ Partial Components

| Component | Status | What's Missing |
|-----------|--------|----------------|
| **Tokenizer** | ⚠️ Partial | Have chat template formatting, but no custom BPE training |
| **Reward Functions** | ⚠️ Partial | Have environment grading, but no general reward model |
| **Evaluation** | ⚠️ Partial | Have `rollouts/evaluate.py` for environments, but no MMLU/GSM8K/etc |

### ❌ Missing Components

| Component | Status | Why We Need It |
|-----------|--------|----------------|
| **Training Loop Orchestrator** | ❌ Missing | Ties together D3 (DataBuffer) + D4 (Rollout) + D6 (Training) |
| **RL Loss Functions** | ❌ Missing | PPO/GRPO/DPO losses for RL training |
| **Reward Model** | ❌ Missing | Compute rewards for RL (currently only env-based grading) |
| **Config System** | ❌ Missing | YAML config like nanochat/torchforge |
| **Metrics/Logging** | ❌ Missing | Track loss, reward, grad_norm over training |
| **Dataset Loaders** | ❌ Missing | Load HF datasets (SmolTalk, MMLU, etc.) |

## Detailed Gap Analysis

### 1. SFT Pipeline (What we need for SFT-only)

**nanochat SFT flow:**
```python
# nanochat: scripts/chat_sft.py
1. Load SFT dataset (HF datasets)
2. Apply chat template
3. Tokenize conversations
4. Compute loss mask (don't train on user messages)
5. Training loop:
   - Get batch from dataloader
   - Forward + backward
   - Optimizer step
   - Save checkpoint periodically
6. Evaluation
```

**What we have:**
- ✅ Chat template (in `rollouts/training/sample_prep.py`)
- ✅ Tokenization (`tokenize_conversation`)
- ✅ Loss masking (`compute_loss_mask`)
- ✅ Training backend (D6v1: PyTorchTrainingBackend)
- ✅ Checkpointing (D6v1)

**What we're missing:**
- ❌ **SFT Training Loop Orchestrator** (ties everything together)
- ❌ **Dataset Loader** (load HF datasets like SmolTalk)
- ❌ **Metrics Tracking** (log loss, grad_norm to file/wandb)
- ❌ **Learning Rate Scheduler** (cosine decay, warmup)

**Estimated effort: ~1 day** to implement SFT training loop

### 2. RL Pipeline (What we need for RL)

**SLIME RL flow:**
```python
# SLIME's main loop
1. Initialize: data_buffer, rollout_engines, training_backend
2. For each step:
   a. Get prompts from data_buffer
   b. Generate rollouts (async parallel with SGLang)
   c. Compute rewards (reward model or env grading)
   d. Prepare RL training batch (PPO/GRPO losses)
   e. Training step (forward + backward + optim)
   f. Sync weights to rollout engines (every N steps)
   g. Save checkpoint (every M steps)
```

**What we have:**
- ✅ Data buffer (D3)
- ✅ Rollout generation (D4: AsyncRolloutManager)
- ✅ Training backend (D6v1)
- ✅ Weight sync (D5)
- ✅ Checkpointing (D6v1)

**What we're missing:**
- ❌ **RL Training Loop Orchestrator** (the main loop above)
- ❌ **RL Loss Functions** (PPO, GRPO, DPO)
- ❌ **Reward Model** (or use env grading as rewards)
- ❌ **Advantage Computation** (for PPO/GRPO)
- ❌ **Value Function** (critic, for PPO)

**Estimated effort: ~2-3 days** to implement RL training loop

### 3. Infrastructure Components

| Component | Priority | Effort | Notes |
|-----------|----------|--------|-------|
| **Training Loop Orchestrator** | 🔴 Critical | ~1 day | Glue code for SFT/RL loops |
| **Config System** | 🟡 Nice-to-have | ~0.5 day | YAML configs (use dataclasses for now) |
| **Metrics/Logging** | 🟡 Nice-to-have | ~0.5 day | CSV/JSON logs (TensorBoard later) |
| **LR Scheduler** | 🟡 Nice-to-have | ~0.5 day | Cosine decay + warmup |
| **Dataset Loaders** | 🟢 Can defer | ~0.5 day | Use HF datasets directly for now |
| **Evaluation** | 🟢 Can defer | ~1 day | MMLU/GSM8K eval (have env eval) |

## Minimum Viable SFT Run

**What's the absolute minimum to run SFT?**

```python
# Pseudocode for minimal SFT run
import torch
from pathlib import Path
from rollouts.training.backends import PyTorchTrainingBackend
from rollouts.training import sample_prep

# 1. Load SFT dataset (manual for now)
conversations = [
    [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
    ],
    # ... more conversations
]

# 2. Tokenize and prepare samples
tokenizer = ...  # HuggingFace tokenizer
samples = []
for conv in conversations:
    tokens, user_spans = sample_prep.tokenize_conversation(conv, tokenizer)
    loss_mask = sample_prep.compute_loss_mask(tokens, user_spans)
    samples.append({
        "input_ids": torch.tensor(tokens),
        "labels": torch.tensor(tokens),  # Same as input_ids for causal LM
        "loss_mask": torch.tensor(loss_mask),
    })

# 3. Create model and backend
model = ...  # Your model
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=cross_entropy_loss,  # Your loss function
    checkpoint_dir=Path("/checkpoints"),
)

# 4. Training loop (THIS IS WHAT'S MISSING!)
for step in range(num_steps):
    # Get batch (manual batching for now)
    batch = get_batch(samples, batch_size=4)

    # Train
    metrics = await backend.forward_backward(batch).result()
    await backend.optim_step().result()

    # Log
    print(f"Step {step}: loss={metrics['loss']:.4f}")

    # Save checkpoint
    if step % 100 == 0:
        await backend.save_checkpoint(step, metrics)
```

**What we need to add: ~100 lines of training loop orchestration code**

## Minimum Viable RL Run

**What's the absolute minimum to run RL?**

```python
# Pseudocode for minimal RL run
import trio
from training import DataBuffer, AsyncRolloutManager
from rollouts.training.backends import PyTorchTrainingBackend

# 1. Setup
data_buffer = DataBuffer(prompts=["Solve 2+2", "Calculate 5*7", ...])

config = RolloutConfig(
    batch_size=4,
    generate_fn=my_agent.run,  # Your agent
)

rollout_manager = AsyncRolloutManager(data_buffer, config)

backend = PyTorchTrainingBackend(...)

# 2. Training loop (THIS IS WHAT'S MISSING!)
async def train():
    for step in range(num_steps):
        # Generate rollouts
        batch = await rollout_manager.generate_batch()

        # Compute rewards (environment grading for now)
        for sample in batch.samples:
            reward = grade_answer(sample.metadata["result"])
            sample.metadata["reward"] = reward

        # Prepare RL training batch (MISSING: PPO/GRPO loss computation)
        rl_batch = prepare_rl_batch(batch)  # ← NEED TO IMPLEMENT

        # Train
        metrics = await backend.forward_backward(rl_batch).result()
        await backend.optim_step().result()

        # Sync weights to inference engines
        if step % 10 == 0:
            ckpt_path = await backend.save_checkpoint(step)
            await sync_weights_to_engines(engines, str(ckpt_path))

trio.run(train)
```

**What we need to add:**
1. ~50 lines: RL training loop orchestration
2. ~100 lines: RL loss function (GRPO is simplest)
3. ~50 lines: Reward computation (use env grading)

**Total: ~200 lines of code**

## The Critical Missing Piece: Training Loop Orchestrator

Both SFT and RL need a **Training Loop Orchestrator** that:

1. **Manages the training loop** (steps, epochs)
2. **Batching** (collate samples into batches)
3. **Metrics tracking** (loss, grad_norm, rewards)
4. **Checkpoint scheduling** (save every N steps)
5. **Weight sync scheduling** (sync every M steps for RL)
6. **Early stopping** (optional)

**This is the ONLY major missing piece!**

## Concrete Next Steps

### Option A: SFT-Only Run (Fastest Path, ~1 day)

1. **Create `SFTTrainer` class** (~100 lines)
   - Takes: model, optimizer, dataset, config
   - Implements: training loop with batching/logging/checkpointing
   - Uses: D6v1 PyTorchTrainingBackend

2. **Create simple dataset loader** (~50 lines)
   - Load JSONL file with conversations
   - Use existing `sample_prep` functions

3. **Create example script** (~50 lines)
   - `examples/run_sft.py`
   - Demonstrate full SFT run

**Output**: Can run nanochat-style SFT training on any chat dataset

### Option B: RL-Only Run (Medium Path, ~2-3 days)

1. **Create `RLTrainer` class** (~150 lines)
   - Takes: model, rollout_manager, config
   - Implements: RL training loop with reward computation
   - Uses: D3 + D4 + D5 + D6v1

2. **Implement simple RL loss** (~100 lines)
   - GRPO (simplest) or basic policy gradient
   - Advantage computation

3. **Create example script** (~100 lines)
   - `examples/run_rl.py`
   - Demonstrate full RL run with environment grading

**Output**: Can run SLIME-style RL training

### Option C: Full Pipeline (Comprehensive, ~4-5 days)

1. SFT Trainer (Option A)
2. RL Trainer (Option B)
3. Config system (YAML)
4. Metrics/logging (CSV + optional wandb)
5. Evaluation harness (MMLU, GSM8K)
6. `speedrun.sh` equivalent

**Output**: Complete nanochat-like post-training pipeline

## Recommendation

**Start with Option A (SFT-Only)**:
- Fastest to implement (~1 day)
- Validates all existing components (D1-D6) work together
- Provides immediate value (can fine-tune any model)
- Foundation for RL (Option B builds on it)

**Then add Option B (RL)** when needed:
- Only ~2-3 days additional work
- Unlocks full SLIME-style training
- Environment grading provides rewards (no reward model needed)

## Summary: How Close Are We?

### For SFT:
- **Core components**: ✅ 100% done (D1-D6)
- **Missing**: Training loop orchestrator (~100 lines)
- **Time to MVP**: ~1 day

### For RL:
- **Core components**: ✅ 100% done (D1-D6)
- **Missing**:
  - Training loop orchestrator (~150 lines)
  - RL loss function (~100 lines)
- **Time to MVP**: ~2-3 days

### Overall Assessment:

**We're 90% of the way there!**

All the hard infrastructure is done (D1-D6). We just need ~250 lines of "glue code" to tie it all together into a training loop.

The components are well-designed and composable - they're just waiting for someone to write the orchestration layer.

**Next action**: Implement `SFTTrainer` class (Option A) - should take ~4-6 hours of focused work.
