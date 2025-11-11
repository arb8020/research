# SFT First Approach: Validate Backends Before RL

**TL;DR**: You're right! Test SFT training first to ensure FSDP/PyTorch backends work correctly. Then add RL on top. Much safer.

---

## Why SFT First Is Smarter

### The Problem with Going Straight to RL

If we jump to RL and something breaks, we won't know if it's:
- ❌ FSDP backend broken
- ❌ PyTorch backend broken
- ❌ GRPO loss broken
- ❌ Reward model broken
- ❌ Rollout generation broken
- ❌ Weight sync broken

**Too many variables!**

### SFT First Isolates the Problem

```
Test 1: SFT with PyTorch backend (1 GPU)
  ✅ Pass → Backend works
  ❌ Fail → Fix backend first

Test 2: SFT with FSDP backend (4 GPUs)
  ✅ Pass → FSDP works
  ❌ Fail → Fix FSDP first

Test 3: Add RL components
  Now we KNOW backends work
  Any failure must be in RL-specific code
```

**This is the right approach!** 🎯

---

## Current State: What Works vs What's Missing

### ✅ What We Have (Code Exists)

**1. Training Loops**
- `rollouts/training/loops/sft_loop.py` - SFT training loop
- `rollouts/training/loops/rl_loop.py` - RL training loop (structure)

**2. Backends**
- `rollouts/training/backends/pytorch.py` - Single GPU backend
- `rollouts/training/backends/fsdp.py` - Multi-GPU FSDP backend

**3. Data Types**
- `rollouts/training/types.py` - Sample, RolloutBatch, configs
- `rollouts/training/datasets/sft.py` - SFT data loading

**4. Utilities**
- `rollouts/training/distributed_utils.py` - FSDP helpers
- `rollouts/training/metrics.py` - Logging

### ❌ What's Missing (Need to Create)

**1. Working Examples**
- ❌ No `examples/train_sft_single_gpu.py`
- ❌ No `examples/train_sft_fsdp.py`
- ❌ No test data (small dataset for quick validation)

**2. Loss Function for SFT**
- ⚠️ `PyTorchTrainingBackend` expects `loss_fn` parameter
- ❌ No standard SFT loss function provided

**3. Model Loading**
- ⚠️ Backends expect `model` parameter
- ❌ No helper to load HuggingFace models easily

---

## What We Need to Build (SFT Only)

### Component 1: SFT Loss Function

**File**: `rollouts/training/sft_losses.py`

```python
"""Loss functions for supervised fine-tuning."""

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,       # [batch_size, seq_len, vocab_size]
    labels: torch.Tensor,        # [batch_size, seq_len]
    loss_mask: torch.Tensor,     # [batch_size, seq_len]
) -> torch.Tensor:
    """Standard causal language modeling loss.

    Args:
        logits: Model logits
        labels: Target token IDs
        loss_mask: Mask for valid tokens (1.0 = compute loss, 0.0 = ignore)

    Returns:
        Scalar loss
    """
    # Shift logits and labels for next-token prediction
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_mask = loss_mask[..., 1:].contiguous()

    # Flatten
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)
    shift_mask = shift_mask.view(-1)

    # Compute loss
    loss = F.cross_entropy(
        shift_logits,
        shift_labels,
        reduction='none',
    )

    # Apply mask
    loss = (loss * shift_mask).sum() / shift_mask.sum()

    return loss
```

---

### Component 2: Model Loading Helper

**File**: `rollouts/training/model_utils.py`

```python
"""Utilities for loading models."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path


def load_hf_model(
    model_name: str,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load HuggingFace causal LM model.

    Args:
        model_name: HF model name or path
        device: Device to load on (None = auto)
        dtype: Model dtype (bf16 by default)

    Returns:
        (model, tokenizer)
    """
    print(f"Loading {model_name}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map={"": device} if device else "auto",
    )

    print(f"✅ Model loaded ({model.num_parameters():,} parameters)")

    return model, tokenizer
```

---

### Component 3: Test Dataset

**File**: `examples/data/tiny_sft.jsonl`

```jsonl
{"prompt": "What is 2+2?", "completion": "4"}
{"prompt": "What is the capital of France?", "completion": "Paris"}
{"prompt": "Who wrote Romeo and Juliet?", "completion": "William Shakespeare"}
{"prompt": "What is 10*5?", "completion": "50"}
{"prompt": "What color is the sky?", "completion": "blue"}
```

**File**: `examples/data/load_tiny_dataset.py`

```python
"""Load tiny SFT dataset for testing."""

import json
from pathlib import Path
from rollouts.training.types import Sample


def load_tiny_sft() -> list[Sample]:
    """Load tiny SFT dataset.

    Returns:
        List of Sample objects with tokens and loss masks
    """
    from transformers import AutoTokenizer

    # Load tokenizer (use Qwen for now)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Load data
    data_path = Path(__file__).parent / "tiny_sft.jsonl"
    samples = []

    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            prompt = item["prompt"]
            completion = item["completion"]

            # Tokenize
            full_text = f"Q: {prompt}\nA: {completion}"
            tokens = tokenizer.encode(full_text)

            # Loss mask: 0 for prompt, 1 for completion
            prompt_tokens = tokenizer.encode(f"Q: {prompt}\nA: ")
            loss_mask = [0.0] * len(prompt_tokens) + [1.0] * (len(tokens) - len(prompt_tokens))

            sample = Sample(
                prompt=prompt,
                response=completion,
                tokens=tokens,
                loss_mask=loss_mask,
            )
            samples.append(sample)

    print(f"Loaded {len(samples)} samples")
    return samples
```

---

### Component 4: Single GPU SFT Example

**File**: `examples/train_sft_single_gpu.py`

```python
#!/usr/bin/env python3
"""Test SFT training on single GPU."""

import asyncio
from pathlib import Path

import torch

from rollouts.training.backends.pytorch import PyTorchTrainingBackend
from rollouts.training.loops.sft_loop import run_sft_training
from rollouts.training.model_utils import load_hf_model
from rollouts.training.sft_losses import causal_lm_loss
from rollouts.training.types import SFTTrainingConfig

# Import test data
import sys
sys.path.append(str(Path(__file__).parent))
from data.load_tiny_dataset import load_tiny_sft


async def main():
    # ════════════════════════════════════════════════════════════════
    # CONFIG
    # ════════════════════════════════════════════════════════════════
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    device = torch.device("cuda:0")

    # ════════════════════════════════════════════════════════════════
    # LOAD MODEL
    # ════════════════════════════════════════════════════════════════
    print("Loading model...")
    model, tokenizer = load_hf_model(model_name, device=device)

    # ════════════════════════════════════════════════════════════════
    # SETUP OPTIMIZER
    # ════════════════════════════════════════════════════════════════
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.1,
    )

    # ════════════════════════════════════════════════════════════════
    # CREATE BACKEND
    # ════════════════════════════════════════════════════════════════
    backend = PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=causal_lm_loss,
        checkpoint_dir=Path("checkpoints/sft_test"),
        device=device,
    )

    # ════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ════════════════════════════════════════════════════════════════
    print("\nLoading dataset...")
    samples = load_tiny_sft()

    # ════════════════════════════════════════════════════════════════
    # TRAINING CONFIG
    # ════════════════════════════════════════════════════════════════
    config = SFTTrainingConfig(
        num_steps=50,
        batch_size=2,
        log_every=10,
        checkpoint_every=25,
    )

    # ════════════════════════════════════════════════════════════════
    # RUN TRAINING
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Starting SFT training (single GPU)")
    print("="*60 + "\n")

    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=config,
    )

    # ════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print(f"Final loss: {metrics[-1]['loss']:.4f}")
    print(f"Steps: {len(metrics)}")


if __name__ == "__main__":
    asyncio.run(main())
```

---

### Component 5: FSDP SFT Example

**File**: `examples/train_sft_fsdp.py`

```python
#!/usr/bin/env python3
"""Test SFT training with FSDP (multi-GPU)."""

import asyncio
from pathlib import Path

import torch
import torch.distributed as dist

from rollouts.training.backends.fsdp import FSDPConfig, FSDPTrainingBackend
from rollouts.training.loops.sft_loop import run_sft_training
from rollouts.training.model_utils import load_hf_model
from rollouts.training.sft_losses import causal_lm_loss
from rollouts.training.types import SFTTrainingConfig

# Import test data
import sys
sys.path.append(str(Path(__file__).parent))
from data.load_tiny_dataset import load_tiny_sft


def setup_distributed():
    """Initialize torch.distributed for FSDP."""
    # Use torchrun to launch (handles env vars)
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"[Rank {rank}/{world_size}] Initialized")


async def main():
    # ════════════════════════════════════════════════════════════════
    # SETUP DISTRIBUTED
    # ════════════════════════════════════════════════════════════════
    setup_distributed()

    # ════════════════════════════════════════════════════════════════
    # CONFIG
    # ════════════════════════════════════════════════════════════════
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ════════════════════════════════════════════════════════════════
    # LOAD MODEL (before FSDP)
    # ════════════════════════════════════════════════════════════════
    print("Loading model...")
    model, tokenizer = load_hf_model(
        model_name,
        device=None,  # FSDP will handle device placement
    )

    # ════════════════════════════════════════════════════════════════
    # SETUP OPTIMIZER (after FSDP wrapping in backend)
    # ════════════════════════════════════════════════════════════════
    # Note: Optimizer must be created AFTER FSDP wrapping
    # We'll pass a factory function instead
    def create_optimizer(model):
        return torch.optim.AdamW(
            model.parameters(),
            lr=1e-5,
            weight_decay=0.1,
        )

    # ════════════════════════════════════════════════════════════════
    # CREATE FSDP BACKEND
    # ════════════════════════════════════════════════════════════════
    fsdp_config = FSDPConfig(
        sharding_strategy="FULL_SHARD",
        mixed_precision=True,
    )

    # Create backend (wraps model with FSDP)
    backend = FSDPTrainingBackend(
        model=model,
        optimizer=create_optimizer(model),  # Will use FSDP-wrapped model
        loss_fn=causal_lm_loss,
        checkpoint_dir=Path("checkpoints/sft_fsdp_test"),
        config=fsdp_config,
    )

    # ════════════════════════════════════════════════════════════════
    # LOAD DATA
    # ════════════════════════════════════════════════════════════════
    if dist.get_rank() == 0:
        print("\nLoading dataset...")
    samples = load_tiny_sft()

    # ════════════════════════════════════════════════════════════════
    # TRAINING CONFIG
    # ════════════════════════════════════════════════════════════════
    config = SFTTrainingConfig(
        num_steps=50,
        batch_size=2,
        log_every=10,
        checkpoint_every=25,
    )

    # ════════════════════════════════════════════════════════════════
    # RUN TRAINING
    # ════════════════════════════════════════════════════════════════
    if dist.get_rank() == 0:
        print("\n" + "="*60)
        print(f"Starting SFT training (FSDP, {dist.get_world_size()} GPUs)")
        print("="*60 + "\n")

    metrics = await run_sft_training(
        backend=backend,
        samples=samples,
        config=config,
    )

    # ════════════════════════════════════════════════════════════════
    # RESULTS
    # ════════════════════════════════════════════════════════════════
    if dist.get_rank() == 0:
        print("\n" + "="*60)
        print("Training complete!")
        print("="*60)
        print(f"Final loss: {metrics[-1]['loss']:.4f}")
        print(f"Steps: {len(metrics)}")

    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    asyncio.run(main())
```

**Launch script**: `examples/run_sft_fsdp.sh`

```bash
#!/bin/bash

# Test FSDP on 4 GPUs
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    examples/train_sft_fsdp.py
```

---

## Testing Plan (SFT First)

### Phase 1: Single GPU (Day 1)

```bash
# Step 1: Create files
rollouts/training/sft_losses.py
rollouts/training/model_utils.py
examples/data/tiny_sft.jsonl
examples/data/load_tiny_dataset.py
examples/train_sft_single_gpu.py

# Step 2: Run
cd rollouts
python examples/train_sft_single_gpu.py

# Expected output:
# Loading model...
# ✅ Model loaded (494M parameters)
# Loading dataset...
# Loaded 5 samples
# Starting SFT training (single GPU)
# Step 0: loss=2.3456, grad_norm=1.234, lr=1e-05
# Step 10: loss=1.8234, grad_norm=0.876, lr=1e-05
# ...
# Training complete!
# Final loss: 1.2345
```

**If this works → PyTorch backend is good! ✅**

---

### Phase 2: FSDP (Day 2)

```bash
# Step 1: Create files
examples/train_sft_fsdp.py
examples/run_sft_fsdp.sh

# Step 2: Run
cd rollouts
bash examples/run_sft_fsdp.sh

# Expected output (rank 0):
# [Rank 0/4] Initialized
# Loading model...
# ✅ Model loaded (494M parameters)
# [Rank 0/4] FSDPTrainingBackend initialized
# Loading dataset...
# Loaded 5 samples
# Starting SFT training (FSDP, 4 GPUs)
# Step 0: loss=2.3456, grad_norm=1.234, lr=1e-05
# ...
# Training complete!
# Final loss: 1.2345
```

**If this works → FSDP backend is good! ✅**

---

### Phase 3: Add RL Components (Week 2)

**Only AFTER both SFT tests pass!**

Now we can add RL components with confidence:
1. GRPO loss
2. Reference model
3. Reward models
4. Inference engines
5. GRPO rollout generation

**Any issues now are definitely in RL code, not backends!**

---

## Implementation Checklist

### Day 1: Single GPU SFT
- [ ] Create `rollouts/training/sft_losses.py`
- [ ] Create `rollouts/training/model_utils.py`
- [ ] Create `examples/data/tiny_sft.jsonl`
- [ ] Create `examples/data/load_tiny_dataset.py`
- [ ] Create `examples/train_sft_single_gpu.py`
- [ ] Run and verify it works
- [ ] Fix any issues

### Day 2: FSDP SFT
- [ ] Create `examples/train_sft_fsdp.py`
- [ ] Create `examples/run_sft_fsdp.sh`
- [ ] Run on 4 GPUs
- [ ] Verify loss decreases
- [ ] Verify checkpoints save correctly
- [ ] Fix any FSDP issues

### Day 3-7: RL Components (only if SFT works!)
- [ ] Add GRPO loss
- [ ] Add reference model to backends
- [ ] Add reward models
- [ ] Add inference engines
- [ ] Test end-to-end RL

---

## Summary

**You're absolutely right!** Testing SFT first:

1. ✅ **Validates backends work** (PyTorch + FSDP)
2. ✅ **Isolates problems** (know it's not the backend if RL fails)
3. ✅ **Faster debugging** (fewer variables)
4. ✅ **Incremental progress** (something working after Day 1)
5. ✅ **Safer approach** (foundation before fancy stuff)

**Revised timeline**:
- Day 1: Single GPU SFT working
- Day 2: FSDP SFT working
- Days 3-7: Add RL on top of proven backends

This is much smarter than going straight to RL! 🎯
