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

from collections.abc import Callable
from pathlib import Path

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
        raise ValueError(f"Invalid dtype: '{dtype_str}'. Must be one of {valid_options}")

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

    assert num_warmup_steps >= 0, f"num_warmup_steps must be >= 0, got {num_warmup_steps}"
    assert num_training_steps > num_warmup_steps, (
        f"num_training_steps ({num_training_steps}) must be > num_warmup_steps ({num_warmup_steps})"
    )
    assert 0 < min_lr_ratio <= 1.0, f"min_lr_ratio must be in (0, 1], got {min_lr_ratio}"

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
    loss_fn: Callable | None = None,
    num_minibatches: int | None = None,
    max_grad_norm: float | None = 1.0,
) -> PyTorchTrainingBackend:
    """Create PyTorch backend with sensible defaults (Tier 2 convenience).

    This is the main entry point for most users. Handles:
    - Loading HuggingFace model
    - Creating AdamW optimizer
    - Setting up loss function
    - Device placement
    - Gradient accumulation (via num_minibatches)

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
        num_minibatches: Split batch into this many pieces for gradient accumulation.
            If None, processes full batch at once (no accumulation).
        max_grad_norm: Clip gradients to this norm. If None, no clipping.
            Default: 1.0 (standard practice for language models)

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
        >>> await backend.optim_step().result()  # Apply gradients
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

    # Create trainer config for gradient accumulation
    from rollouts.training.types import TrainerConfig

    trainer_config = TrainerConfig(
        num_minibatches=num_minibatches,
        max_grad_norm=max_grad_norm,
    )

    # Tier 0: Assemble backend
    return PyTorchTrainingBackend(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        checkpoint_dir=checkpoint_dir,
        device=device,
        trainer_config=trainer_config,
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
