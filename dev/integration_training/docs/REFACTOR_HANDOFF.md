# Integration Training Refactor: Handoff Document

**Date:** 2025-01-11
**Status:** Ready for Implementation
**Estimated Effort:** 2-3 days
**Priority:** High (improves maintainability, aligns with design doc)

---

## Executive Summary

**Problem:** `dev/integration_training/train.py` manually assembles training backends instead of using the clean abstractions already built in `rollouts/training/`. This violates Casey Muratori's API design principles and makes the code harder to maintain.

**Solution:** Add convenience factory functions to `rollouts/training/backends/` and refactor `train.py` to use them. This reduces `train.py` from ~700 lines to ~550 lines while improving testability and maintainability.

**Impact:**
- ✅ Aligns with `TRAINING_SYSTEM_DESIGN.md` architecture
- ✅ Reduces boilerplate in `train.py` by ~150 lines
- ✅ Improves testability (can test each tier independently)
- ✅ Makes it trivial to add new backend types (FSDP, DeepSpeed, etc.)
- ✅ No breaking changes (existing code keeps working)

---

## Background

### Current Architecture

```
train.py (700 lines)
  ├─ load_model() - 60 lines
  ├─ create_optimizer() - 35 lines
  ├─ create_loss_fn() - 50 lines
  ├─ Manual device setup - 10 lines
  └─ Manual PyTorchTrainingBackend assembly - 15 lines
       ↓
  PyTorchTrainingBackend (in rollouts/)
       ↓
  TrainingBackend protocol (in rollouts/)
```

**Problem:** train.py has to manually do 5 steps to create a backend, bypassing the abstractions.

### Target Architecture (Casey Muratori's 3-Tier Design)

```
┌─────────────────────────────────────────────────────────────┐
│ TIER 3: Application Code (train.py)                         │
│  backend = create_pytorch_backend(model_name, checkpoint_dir)│
│  - Just 1 line to create backend                            │
├─────────────────────────────────────────────────────────────┤
│ TIER 2: Convenience Factories (NEW - add to rollouts)      │
│  - create_pytorch_backend() - common case (80% usage)       │
│  - create_backend_with_scheduler() - backend + LR schedule  │
├─────────────────────────────────────────────────────────────┤
│ TIER 1: Granular Primitives (NEW - add to rollouts)        │
│  - parse_dtype() - string → torch.dtype                     │
│  - load_hf_model() - explicit model loading                 │
│  - create_adamw_optimizer() - pure optimizer                │
│  - create_warmup_cosine_scheduler() - pure scheduler        │
│  - For power users who need custom control                  │
├─────────────────────────────────────────────────────────────┤
│ TIER 0: Protocol Layer (EXISTS - no changes needed)        │
│  - PyTorchTrainingBackend                                   │
│  - TrainingBackend protocol                                 │
│  - TrainFuture                                              │
└─────────────────────────────────────────────────────────────┘
```

**Casey Muratori's principle:** Provide multiple levels of granularity. Users pick the tier they need.

---

## Deliverables

### D1: Create `rollouts/training/backends/pytorch_factory.py` (NEW FILE)

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/pytorch_factory.py`

**Content:** See [Appendix A: Complete pytorch_factory.py](#appendix-a-complete-pytorch_factorypy)

**Acceptance Criteria:**
- [ ] File created with all Tier 1 and Tier 2 functions
- [ ] All functions have docstrings with examples
- [ ] All functions have Tiger Style assertions
- [ ] Unit tests pass (see [Testing Strategy](#testing-strategy))

**Estimated Effort:** 4-6 hours

---

### D2: Update `rollouts/training/backends/__init__.py`

**Location:** `/Users/chiraagbalu/research/rollouts/rollouts/training/backends/__init__.py`

**Changes:**
```python
# Add to imports:
from rollouts.training.backends.pytorch_factory import (
    # Tier 2: Convenience
    create_pytorch_backend,
    create_backend_with_scheduler,
    # Tier 1: Granular (export for power users)
    parse_dtype,
    compute_device_map_single_gpu,
    load_hf_model,
    create_adamw_optimizer,
    create_cross_entropy_loss,
    create_warmup_cosine_scheduler,
)

# Add to __all__:
__all__ = [
    # ... existing exports ...
    # Tier 2
    "create_pytorch_backend",
    "create_backend_with_scheduler",
    # Tier 1
    "parse_dtype",
    "compute_device_map_single_gpu",
    "load_hf_model",
    "create_adamw_optimizer",
    "create_cross_entropy_loss",
    "create_warmup_cosine_scheduler",
]
```

**Acceptance Criteria:**
- [ ] Imports don't break existing code
- [ ] New functions are accessible via `from rollouts.training.backends import create_pytorch_backend`

**Estimated Effort:** 15 minutes

---

### D3: Refactor `dev/integration_training/train.py`

**Location:** `/Users/chiraagbalu/research/dev/integration_training/train.py`

**Changes:** See [Appendix B: train.py Refactor Diff](#appendix-b-trainpy-refactor-diff)

**Summary of Changes:**

1. **Delete functions** (now in rollouts):
   - `load_model()` (lines 121-182)
   - `create_optimizer()` (lines 185-218)
   - `create_loss_fn()` (lines 221-273)

2. **Simplify `run_sft()`** (lines 419-520):
   ```python
   # BEFORE (lines 444-465):
   model = load_model(...)
   optimizer = create_optimizer(...)
   loss_fn = create_loss_fn()
   device = torch.device(...)
   backend = PyTorchTrainingBackend(
       model=model,
       optimizer=optimizer,
       loss_fn=loss_fn,
       checkpoint_dir=output_dir / "checkpoints",
       device=device,
   )

   # AFTER:
   from rollouts.training.backends import create_pytorch_backend

   backend = create_pytorch_backend(
       model_name=config.model.name,
       checkpoint_dir=output_dir / "checkpoints",
       device_type=config.target.device_type,
       dtype=config.model.dtype,
       gpu_rank=config.target.gpu_ranks[0],
       learning_rate=config.sft.matrix_lr,
       adam_betas=(config.sft.adam_beta1, config.sft.adam_beta2),
       weight_decay=config.sft.weight_decay,
   )
   ```

3. **Simplify scheduler creation** in `create_fsdp_backend()` (lines 366-411):
   ```python
   # BEFORE: 45 lines of manual scheduler setup
   from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
   warmup = LinearLR(...)
   if config.sft.lr_decay_style == "cosine":
       decay = CosineAnnealingLR(...)
   # ... etc ...
   backend.scheduler = SequentialLR(...)

   # AFTER: 5 lines using factory
   from rollouts.training.backends import create_warmup_cosine_scheduler

   backend.scheduler = create_warmup_cosine_scheduler(
       optimizer=optimizer,
       num_warmup_steps=warmup_steps,
       num_training_steps=total_steps,
   )
   ```

**Acceptance Criteria:**
- [ ] train.py runs successfully with test config
- [ ] SFT training works (single GPU)
- [ ] FSDP training works (multi-GPU)
- [ ] No regression in functionality
- [ ] Code is ~150 lines shorter

**Estimated Effort:** 6-8 hours (includes testing)

---

### D4: Update `dev/integration_training/deploy.py` (Optional)

**Location:** `/Users/chiraagbalu/research/dev/integration_training/deploy.py`

**Changes:** None required for core refactor, but could simplify if desired.

**Estimated Effort:** 0 hours (not required)

---

## Testing Strategy

### Unit Tests (rollouts module)

Create: `rollouts/tests/training/backends/test_pytorch_factory.py`

```python
"""Tests for pytorch_factory convenience functions."""

import pytest
import torch
from pathlib import Path
from rollouts.training.backends.pytorch_factory import (
    parse_dtype,
    compute_device_map_single_gpu,
    create_adamw_optimizer,
    create_cross_entropy_loss,
    create_pytorch_backend,
)


class TestTier1Granular:
    """Test Tier 1 (granular) functions."""

    def test_parse_dtype_bfloat16(self):
        dtype = parse_dtype("bfloat16")
        assert dtype == torch.bfloat16

    def test_parse_dtype_float32(self):
        dtype = parse_dtype("float32")
        assert dtype == torch.float32

    def test_parse_dtype_invalid(self):
        with pytest.raises(ValueError, match="Invalid dtype"):
            parse_dtype("invalid")

    def test_compute_device_map_cuda(self):
        device_map = compute_device_map_single_gpu("cuda", 4)
        assert device_map == {"": 4}

    def test_compute_device_map_cpu(self):
        device_map = compute_device_map_single_gpu("cpu", 0)
        assert device_map is None

    def test_create_adamw_optimizer(self):
        model = torch.nn.Linear(10, 10)
        optimizer = create_adamw_optimizer(model, lr=1e-4)
        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.param_groups[0]["lr"] == 1e-4

    def test_create_cross_entropy_loss(self):
        loss_fn = create_cross_entropy_loss()

        # Test with batch dict
        logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        batch = {
            "labels": torch.randint(0, 100, (2, 10)),
            "loss_mask": torch.ones(2, 10),
        }
        loss = loss_fn(logits, batch)
        assert loss.item() > 0


class TestTier2Convenience:
    """Test Tier 2 (convenience) functions."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
    def test_create_pytorch_backend_minimal(self, tmp_path):
        """Test minimal backend creation."""
        backend = create_pytorch_backend(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            checkpoint_dir=tmp_path / "checkpoints",
            gpu_rank=0,
        )

        assert backend is not None
        assert backend.model is not None
        assert backend.optimizer is not None
        assert backend.checkpoint_dir.exists()

    def test_create_pytorch_backend_cpu(self, tmp_path):
        """Test CPU backend creation."""
        backend = create_pytorch_backend(
            model_name="Qwen/Qwen2.5-0.5B-Instruct",
            checkpoint_dir=tmp_path / "checkpoints",
            device_type="cpu",
        )

        assert backend.device.type == "cpu"
```

**Acceptance Criteria:**
- [ ] All unit tests pass
- [ ] Test coverage > 80% for new code
- [ ] Tests run in CI

**Estimated Effort:** 2-3 hours

---

### Integration Tests (train.py)

Run existing integration tests with refactored train.py:

```bash
# Test 1: Single GPU SFT training
python train.py configs/01_debug_sft_rl.py

# Test 2: Multi-GPU FSDP training (if available)
torchrun --nproc_per_node=2 train.py configs/03_debug_sft_fsdp_2gpu.py

# Expected: Both complete successfully with no regression
```

**Acceptance Criteria:**
- [ ] Test 1 completes (loss decreases)
- [ ] Test 2 completes (if multi-GPU available)
- [ ] Checkpoints saved correctly
- [ ] No new warnings/errors

**Estimated Effort:** 1-2 hours

---

## Migration Guide for Users

### For Users of `dev/integration_training/train.py`

**No changes required!** The refactor is internal. Your configs will keep working.

### For Future Code Using rollouts

**Before (manual assembly):**
```python
from rollouts.training.backends.pytorch import PyTorchTrainingBackend
import torch

# Manual setup (5 steps)
model = load_my_model()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = my_loss_function()
device = torch.device("cuda:0")

backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    checkpoint_dir=Path("./checkpoints"),
    device=device,
)
```

**After (convenience factory):**
```python
from rollouts.training.backends import create_pytorch_backend

# One call (Tier 2)
backend = create_pytorch_backend(
    model_name="Qwen/Qwen2.5-0.5B",
    checkpoint_dir=Path("./checkpoints"),
    gpu_rank=0,
    learning_rate=1e-4,
)
```

**Power Users (need custom control):**
```python
from rollouts.training.backends import (
    parse_dtype,
    load_hf_model,
    create_adamw_optimizer,
    PyTorchTrainingBackend,
)

# Tier 1: Granular control
dtype = parse_dtype("bfloat16")
model = load_hf_model("Qwen/Qwen2.5-0.5B", dtype, device_map={"": 4})

# Custom optimizer with different LRs for different layers
optimizer = create_adamw_optimizer(
    model,
    lr=1e-4,
    betas=(0.9, 0.999),  # Different from default
)

# Tier 0: Full control
backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=my_custom_loss,
    checkpoint_dir=Path("./checkpoints"),
    device=torch.device("cuda:4"),
)
```

---

## Risk Assessment

### Low Risk
- ✅ No changes to protocols or existing backends
- ✅ All new code is additive (no breaking changes)
- ✅ Existing `PyTorchTrainingBackend` constructor still works

### Medium Risk
- ⚠️ train.py depends on new factories - must test thoroughly
- ⚠️ New dependencies: none (all stdlib + existing deps)

### Mitigation
- Keep old functions in train.py commented out for 1 week
- Add comprehensive unit tests before refactoring train.py
- Test on both single-GPU and multi-GPU configs

---

## Timeline

| Day | Task | Owner | Hours |
|-----|------|-------|-------|
| Day 1 AM | Create pytorch_factory.py (Tier 1 + Tier 2) | Impl Team | 4-6h |
| Day 1 PM | Write unit tests for factory functions | Impl Team | 2-3h |
| Day 2 AM | Refactor train.py to use factories | Impl Team | 4-5h |
| Day 2 PM | Integration testing (single + multi GPU) | Impl Team | 2-3h |
| Day 3 | Polish, documentation, PR review | Impl Team | 2-4h |

**Total: 14-21 hours (2-3 days)**

---

## Success Criteria

- [ ] All unit tests pass (new factory functions)
- [ ] All integration tests pass (train.py runs successfully)
- [ ] Code coverage > 80% for new code
- [ ] train.py reduced by ~150 lines
- [ ] No regression in functionality
- [ ] Documentation updated (docstrings, README if needed)
- [ ] PR approved and merged

---

## Appendix A: Complete pytorch_factory.py

**File:** `rollouts/training/backends/pytorch_factory.py`

```python
"""Convenience factories for PyTorch backends.

Casey Muratori's 3-Tier API Design:

Tier 1 (Granular - for power users):
    - parse_dtype() - string to torch.dtype
    - compute_device_map_single_gpu() - gpu selection to device_map
    - load_hf_model() - explicit HuggingFace model loading
    - create_adamw_optimizer() - pure optimizer creation
    - create_cross_entropy_loss() - pure loss function
    - create_warmup_cosine_scheduler() - pure LR scheduler

Tier 2 (Convenience - for common cases):
    - create_pytorch_backend() - one-call backend creation
    - create_backend_with_scheduler() - backend + LR schedule

Tier 0 (Protocol - already exists):
    - PyTorchTrainingBackend - low-level backend
    - TrainingBackend protocol - interface

Users pick the tier they need:
- Most users: Tier 2 (one function call)
- Power users: Tier 1 (fine-grained control)
- Framework developers: Tier 0 (direct protocol access)

Example (Tier 2 - common case):
    >>> backend = create_pytorch_backend(
    ...     model_name="Qwen/Qwen2.5-0.5B",
    ...     checkpoint_dir=Path("./checkpoints"),
    ...     gpu_rank=4,
    ... )

Example (Tier 1 - custom control):
    >>> dtype = parse_dtype("bfloat16")
    >>> device_map = compute_device_map_single_gpu("cuda", 4)
    >>> model = load_hf_model("Qwen/Qwen2.5-0.5B", dtype, device_map)
    >>> optimizer = create_adamw_optimizer(model, lr=1e-4)
    >>> backend = PyTorchTrainingBackend(model, optimizer, ...)
"""

from pathlib import Path
from typing import Callable, Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rollouts.training.backends.pytorch import PyTorchTrainingBackend


# ============================================================================
# TIER 1: Granular Primitives (Casey: Fine control, pure functions)
# ============================================================================


def parse_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype.

    Args:
        dtype_str: "bfloat16" | "float32" | "float16"

    Returns:
        torch.dtype

    Raises:
        ValueError: If dtype_str is invalid

    Tiger Style: Explicit validation, clear error messages.

    Example:
        >>> dtype = parse_dtype("bfloat16")
        >>> assert dtype == torch.bfloat16
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float16": torch.float16,
    }

    if dtype_str not in dtype_map:
        valid_options = list(dtype_map.keys())
        raise ValueError(
            f"Invalid dtype: '{dtype_str}'. Must be one of {valid_options}"
        )

    return dtype_map[dtype_str]


def compute_device_map_single_gpu(
    device_type: str,
    gpu_rank: int,
) -> dict[str, int] | None:
    """Compute device_map for single GPU placement.

    Args:
        device_type: "cuda" | "cpu" | "mps"
        gpu_rank: Physical GPU index (e.g., 4 for GPU 4)

    Returns:
        Device map dict for HuggingFace, or None for CPU/MPS

    Tiger Style: Assert preconditions.

    Example:
        >>> device_map = compute_device_map_single_gpu("cuda", 4)
        >>> assert device_map == {"": 4}  # Place entire model on GPU 4
    """
    assert device_type in ["cuda", "cpu", "mps"], (
        f"Invalid device_type: {device_type}. Must be 'cuda', 'cpu', or 'mps'"
    )
    assert gpu_rank >= 0, f"gpu_rank must be >= 0, got {gpu_rank}"

    if device_type == "cuda":
        return {"": gpu_rank}  # HuggingFace: place entire model on this GPU
    else:
        return None  # CPU/MPS don't use device_map


def load_hf_model(
    model_name: str,
    torch_dtype: torch.dtype,
    device_map: dict[str, int] | None,
) -> torch.nn.Module:
    """Load HuggingFace model with explicit parameters.

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B")
        torch_dtype: torch.bfloat16 | torch.float32
        device_map: Device placement (None = CPU)

    Returns:
        Loaded model

    Tiger Style: Assert preconditions, explicit parameters.
    Casey Muratori: This is the "transparent" version - no magic,
    you pass exactly what you want.

    Example:
        >>> model = load_hf_model(
        ...     "Qwen/Qwen2.5-0.5B",
        ...     torch.bfloat16,
        ...     {"": 4},
        ... )
    """
    from transformers import AutoModelForCausalLM

    assert model_name, "model_name cannot be empty"
    assert torch_dtype is not None, "torch_dtype cannot be None"

    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )


def create_adamw_optimizer(
    model: torch.nn.Module,
    lr: float,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> Optimizer:
    """Create AdamW optimizer with explicit parameters.

    Args:
        model: Model to optimize
        lr: Learning rate
        betas: (beta1, beta2) for Adam momentum
        eps: Epsilon for numerical stability
        weight_decay: L2 regularization coefficient

    Returns:
        AdamW optimizer

    Tiger Style: All parameters explicit, assertions for validation.
    Defaults are from SLIME/nanochat (beta2=0.95, not 0.999).

    Example:
        >>> optimizer = create_adamw_optimizer(model, lr=1e-4)
        >>> assert optimizer.param_groups[0]["lr"] == 1e-4
    """
    assert model is not None, "model cannot be None"
    assert lr > 0, f"lr must be positive, got {lr}"
    assert lr < 1.0, f"lr suspiciously high (>1.0): {lr}"
    assert 0 < betas[0] < 1, f"beta1 must be in (0,1), got {betas[0]}"
    assert 0 < betas[1] < 1, f"beta2 must be in (0,1), got {betas[1]}"
    assert eps > 0, f"eps must be positive, got {eps}"
    assert weight_decay >= 0, f"weight_decay must be >= 0, got {weight_decay}"

    return torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def create_cross_entropy_loss() -> Callable:
    """Create standard cross-entropy loss function.

    Returns:
        Loss function matching backend protocol signature:
        loss_fn(logits: torch.Tensor, batch: dict) -> torch.Tensor

    Pattern from train.py:231-272 (works with SLIME batch dict).
    Supports optional loss_mask for per-token weighting.

    Example:
        >>> loss_fn = create_cross_entropy_loss()
        >>> logits = torch.randn(2, 10, 100)  # [batch, seq, vocab]
        >>> batch = {
        ...     "labels": torch.randint(0, 100, (2, 10)),
        ...     "loss_mask": torch.ones(2, 10),
        ... }
        >>> loss = loss_fn(logits, batch)
        >>> assert loss.item() > 0
    """
    import torch.nn.functional as F

    def cross_entropy_loss(logits: torch.Tensor, batch: dict) -> torch.Tensor:
        """Cross-entropy with optional masking.

        Args:
            logits: Model logits [batch, seq_len, vocab_size]
            batch: Training batch dict containing:
                - labels: Target labels [batch, seq_len]
                - loss_mask: Loss mask [batch, seq_len] (optional)

        Returns:
            Scalar loss
        """
        labels = batch["labels"]
        loss_mask = batch.get("loss_mask")

        # Reshape for cross_entropy
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)

        # Compute loss
        loss = F.cross_entropy(logits_flat, labels_flat, reduction="none")
        loss = loss.view(batch_size, seq_len)

        # Apply mask if provided
        if loss_mask is not None:
            loss = loss * loss_mask
            num_valid = loss_mask.sum().clamp(min=1.0)
            return loss.sum() / num_valid
        else:
            return loss.mean()

    return cross_entropy_loss


def create_warmup_cosine_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    """Create warmup + cosine decay scheduler.

    Args:
        optimizer: Optimizer to schedule
        num_warmup_steps: Linear warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR as fraction of base (e.g., 0.1 = 10%)

    Returns:
        LR scheduler (call .step() after each training step)

    Pattern from train.py:366-411 (nanochat + SLIME).
    Phase 1: Linear warmup from 0.1x → 1.0x of base LR
    Phase 2: Cosine decay from 1.0x → min_lr_ratio of base LR

    Tiger Style: Explicit parameters, clear boundaries.

    Example:
        >>> scheduler = create_warmup_cosine_scheduler(
        ...     optimizer,
        ...     num_warmup_steps=100,
        ...     num_training_steps=1000,
        ... )
        >>> for step in range(1000):
        ...     loss = train_step()
        ...     scheduler.step()
    """
    from torch.optim.lr_scheduler import (
        CosineAnnealingLR,
        LinearLR,
        SequentialLR,
    )

    assert num_warmup_steps >= 0, (
        f"num_warmup_steps must be >= 0, got {num_warmup_steps}"
    )
    assert num_training_steps > num_warmup_steps, (
        f"num_training_steps ({num_training_steps}) must be > "
        f"num_warmup_steps ({num_warmup_steps})"
    )
    assert 0 < min_lr_ratio <= 1.0, (
        f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}"
    )

    # Phase 1: Warmup (0.1x → 1.0x of base LR)
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=num_warmup_steps,
    )

    # Phase 2: Cosine decay (1.0x → min_lr_ratio of base LR)
    num_decay_steps = num_training_steps - num_warmup_steps
    base_lr = optimizer.param_groups[0]["lr"]

    decay = CosineAnnealingLR(
        optimizer,
        T_max=num_decay_steps,
        eta_min=base_lr * min_lr_ratio,
    )

    # Combine: warmup then decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, decay],
        milestones=[num_warmup_steps],
    )

    return scheduler


# ============================================================================
# TIER 2: Convenience Factories (Casey: Common cases, 80% of usage)
# ============================================================================


def create_pytorch_backend(
    model_name: str,
    checkpoint_dir: Path,
    device_type: str = "cuda",
    dtype: str = "bfloat16",
    gpu_rank: int = 0,
    learning_rate: float = 1e-4,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
    loss_fn: Optional[Callable] = None,
) -> PyTorchTrainingBackend:
    """Create PyTorch backend with sensible defaults (Tier 2 convenience).

    This is the main entry point for most users. Handles:
    - Loading HuggingFace model
    - Creating AdamW optimizer
    - Setting up loss function
    - Device placement

    Args:
        model_name: HuggingFace model ID (e.g., "Qwen/Qwen2.5-0.5B")
        checkpoint_dir: Where to save checkpoints
        device_type: "cuda" | "cpu" | "mps" (default: "cuda")
        dtype: "bfloat16" | "float32" | "float16" (default: "bfloat16")
        gpu_rank: Physical GPU index (e.g., 4 for GPU 4, default: 0)
        learning_rate: Base learning rate (default: 1e-4)
        adam_betas: (beta1, beta2) for AdamW (default: (0.9, 0.95))
        weight_decay: L2 regularization (default: 0.0)
        loss_fn: Optional custom loss function (default: cross-entropy)

    Returns:
        Ready-to-use PyTorchTrainingBackend

    Casey Muratori: This is the "redundant convenience API".
    Power users can use Tier 1 functions + PyTorchTrainingBackend() directly.

    Example (common case):
        >>> backend = create_pytorch_backend(
        ...     model_name="Qwen/Qwen2.5-0.5B",
        ...     checkpoint_dir=Path("./checkpoints"),
        ...     gpu_rank=4,
        ... )
        >>> # Ready to train!
        >>> future = backend.forward_backward(batch)
        >>> metrics = await future.result()
    """
    # Tier 1: Parse dtype
    torch_dtype = parse_dtype(dtype)

    # Tier 1: Compute device map
    device_map = compute_device_map_single_gpu(device_type, gpu_rank)

    # Tier 1: Load model
    model = load_hf_model(model_name, torch_dtype, device_map)

    # Tier 1: Create optimizer
    optimizer = create_adamw_optimizer(
        model,
        lr=learning_rate,
        betas=adam_betas,
        weight_decay=weight_decay,
    )

    # Tier 1: Create loss function
    if loss_fn is None:
        loss_fn = create_cross_entropy_loss()

    # Create device
    device = (
        torch.device(f"{device_type}:{gpu_rank}")
        if device_type == "cuda"
        else torch.device(device_type)
    )

    # Tier 0: Assemble backend
    return PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )


def create_backend_with_scheduler(
    model_name: str,
    checkpoint_dir: Path,
    num_training_steps: int,
    warmup_ratio: float = 0.03,
    device_type: str = "cuda",
    dtype: str = "bfloat16",
    gpu_rank: int = 0,
    learning_rate: float = 1e-4,
    adam_betas: tuple[float, float] = (0.9, 0.95),
    weight_decay: float = 0.0,
) -> tuple[PyTorchTrainingBackend, LRScheduler]:
    """Create backend + LR scheduler (common pattern from train.py).

    Convenience function that combines backend creation with warmup+cosine
    scheduler setup. Saves ~50 lines of boilerplate.

    Args:
        model_name: HuggingFace model ID
        checkpoint_dir: Checkpoint directory
        num_training_steps: Total training steps
        warmup_ratio: Warmup as fraction of total (e.g., 0.03 = 3%)
        device_type: "cuda" | "cpu" | "mps"
        dtype: "bfloat16" | "float32" | "float16"
        gpu_rank: GPU index
        learning_rate: Base LR
        adam_betas: Adam betas
        weight_decay: L2 regularization

    Returns:
        (backend, scheduler) tuple

    Pattern from train.py:358-411 + lines 444-465.

    Example:
        >>> backend, scheduler = create_backend_with_scheduler(
        ...     model_name="Qwen/Qwen2.5-0.5B",
        ...     checkpoint_dir=Path("./ckpts"),
        ...     num_training_steps=1000,
        ... )
        >>> # Training loop
        >>> for step in range(1000):
        ...     metrics = await backend.forward_backward(batch).result()
        ...     await backend.optim_step().result()
        ...     scheduler.step()  # Update LR
    """
    # Create backend (Tier 2)
    backend = create_pytorch_backend(
        model_name=model_name,
        checkpoint_dir=checkpoint_dir,
        device_type=device_type,
        dtype=dtype,
        gpu_rank=gpu_rank,
        learning_rate=learning_rate,
        adam_betas=adam_betas,
        weight_decay=weight_decay,
    )

    # Create scheduler (Tier 1)
    num_warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    scheduler = create_warmup_cosine_scheduler(
        backend.optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    return backend, scheduler
```

---

## Appendix B: train.py Refactor Diff

### B.1: Delete Functions

**Remove lines 121-273** (functions now in rollouts):

```python
# DELETE THIS ENTIRE SECTION:

def load_model(model_name: str, device_type: str, dtype: str, gpu_ranks: list[int]):
    """Load HuggingFace model.
    ...
    """
    # 60 lines - DELETE

def create_optimizer(model, config: Config, mode: Literal["sft", "rl"]):
    """Create optimizer with nanochat's tiered learning rates.
    ...
    """
    # 35 lines - DELETE

def create_loss_fn():
    """Create loss function for training.
    ...
    """
    # 50 lines - DELETE
```

### B.2: Update Imports

**Add at top of file (after line 35):**

```python
# Import from rollouts module (installed via workspace)
from rollouts.training import (
    JSONLLogger,
    PyTorchTrainingBackend,
    SFTTrainingConfig,
    load_sft_dataset,
    run_sft_training,
)

# NEW: Import convenience factories
from rollouts.training.backends import (
    create_pytorch_backend,
    create_warmup_cosine_scheduler,
)
```

### B.3: Simplify run_sft() Function

**Replace lines 444-465:**

```python
# OLD (21 lines):
    else:
        # Single-GPU PyTorch backend
        model = load_model(
            config.model.name,
            config.target.device_type,
            config.model.dtype,
            config.target.gpu_ranks,
        )
        optimizer = create_optimizer(model, config, mode="sft")
        loss_fn = create_loss_fn()

        # Create backend
        import torch
        device = torch.device(f"{config.target.device_type}:{config.target.gpu_ranks[0]}"
                             if config.target.device_type == "cuda"
                             else config.target.device_type)

        backend = PyTorchTrainingBackend(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            checkpoint_dir=output_dir / "checkpoints",
            device=device,
        )

# NEW (9 lines):
    else:
        # Single-GPU PyTorch backend
        backend = create_pytorch_backend(
            model_name=config.model.name,
            checkpoint_dir=output_dir / "checkpoints",
            device_type=config.target.device_type,
            dtype=config.model.dtype,
            gpu_rank=config.target.gpu_ranks[0],
            learning_rate=config.sft.matrix_lr,
            adam_betas=(config.sft.adam_beta1, config.sft.adam_beta2),
            weight_decay=config.sft.weight_decay,
        )
```

### B.4: Simplify Scheduler Creation in create_fsdp_backend()

**Replace lines 366-411:**

```python
# OLD (45 lines):
    # Create learning rate scheduler with warmup (SLIME pattern)
    # Warmup prevents initial instability from large learning rate
    # Tiger Style: All parameters from config
    total_steps = config.sft.num_iterations
    warmup_ratio = config.sft.warmup_ratio
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = total_steps - warmup_steps

    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

    # Warmup: ramp from 10% to 100% of base LR
    # Note: verbose param added in PyTorch 2.5+, omit for compatibility
    warmup = LinearLR(
        optimizer=optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps,
        last_epoch=-1,
    )

    # Decay: uses lr_decay_style from config
    if config.sft.lr_decay_style == "cosine":
        decay = CosineAnnealingLR(
            optimizer=optimizer,
            T_max=decay_steps,
            eta_min=config.sft.matrix_lr * 0.1,
            last_epoch=-1,
        )
    elif config.sft.lr_decay_style == "linear":
        decay = LinearLR(
            optimizer=optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=decay_steps,
            last_epoch=-1,
        )
    elif config.sft.lr_decay_style == "constant":
        # No decay - just use a dummy scheduler
        from torch.optim.lr_scheduler import LambdaLR
        decay = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: 1.0,
            last_epoch=-1,
        )
    else:
        raise ValueError(f"Unknown lr_decay_style: {config.sft.lr_decay_style}")

    # Combine: warmup then decay
    backend.scheduler = SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, decay],
        milestones=[warmup_steps],
        last_epoch=-1,
    )

# NEW (10 lines):
    # Create learning rate scheduler with warmup (SLIME pattern)
    # Tiger Style: All parameters from config
    total_steps = config.sft.num_iterations
    warmup_steps = max(1, int(total_steps * config.sft.warmup_ratio))

    backend.scheduler = create_warmup_cosine_scheduler(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        min_lr_ratio=0.1,
    )
```

**Note:** This simplification assumes we always use cosine decay. If you need to support `linear` and `constant` from config, we can add those to the factory or keep the conditional logic.

---

## Questions / Clarifications

### Q1: What if we need custom LR schedulers (linear, constant)?

**Option A:** Add `lr_decay_style` parameter to `create_warmup_cosine_scheduler()`:
```python
def create_warmup_scheduler(
    optimizer: Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    decay_style: str = "cosine",  # "cosine" | "linear" | "constant"
    min_lr_ratio: float = 0.1,
) -> LRScheduler:
    # Implementation handles all 3 styles
```

**Option B:** Keep the conditional logic in `create_fsdp_backend()` but use the factory for the common case.

**Recommendation:** Option A (add parameter to factory).

### Q2: What about FSDP backend creation?

FSDP setup (lines 276-416) is more complex and FSDP-specific. Recommend:
- Keep `create_fsdp_backend()` in train.py for now
- But use `create_warmup_cosine_scheduler()` from factory to simplify scheduler setup
- Future: Could add `create_fsdp_backend_from_config()` to rollouts if FSDP becomes common

### Q3: Should we delete the old functions immediately?

**Recommendation:**
1. Comment them out in the refactor PR
2. Delete after 1 week if no issues
3. This gives us a rollback path if needed

---

## References

- **Design Doc:** `~/wafer_stuff/clicker/docs/TRAINING_SYSTEM_DESIGN.md`
- **Code Style - Casey Muratori:** `docs/code_style/code_reuse_casey_muratori.md`
- **Code Style - Tiger Style:** `docs/code_style/tiger_style_safety.md`
- **Current Implementation:** `dev/integration_training/train.py`
- **Rollouts Backend Protocol:** `rollouts/rollouts/training/backends/protocol.py`

---

## Contact

**Questions during implementation?**
- Check Casey Muratori principles in `docs/code_style/code_reuse_casey_muratori.md`
- Check design doc for architecture decisions
- Ping on Slack if unclear

---

**Document Version:** 1.0
**Last Updated:** 2025-01-11
**Status:** Ready for Implementation
