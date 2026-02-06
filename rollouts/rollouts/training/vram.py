"""VRAM preflight check.

Runs one forward+backward pass at worst-case sequence length on the real
backend before training starts.  If peak memory exceeds GPU capacity
(minus what SGLang already grabbed), aborts with a clear breakdown.

No heuristics — we measure the actual peak by running the real model.
The only margin is a configurable safety margin (default 5%) for CUDA
allocator fragmentation.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class VRAMPreflightError(RuntimeError):
    """Raised when the VRAM preflight check fails."""


def preflight_vram_check(
    backend: Any,
    config: Any,
    device: str,
) -> dict[str, float]:
    """Run one dummy forward+backward at worst-case seq_len to measure peak VRAM.

    Must be called after _setup_training_backend (model + optimizer on GPU)
    and after the inference engine has allocated its memory (SGLang grabs
    mem_fraction of the GPU upfront).

    Args:
        backend: PyTorchTrainingBackend (model + optimizer already on GPU).
        config: GRPOConfig with rollout, trainer, and inference sub-configs.
        device: CUDA device string, e.g. "cuda:0".

    Returns:
        Dict with measurement breakdown (for logging).

    Raises:
        VRAMPreflightError: If peak training memory won't fit alongside
            the inference engine, with a breakdown of what's using what.
    """
    import torch

    gpu_index = int(device.split(":")[-1])
    props = torch.cuda.get_device_properties(gpu_index)
    gpu_total_bytes = props.total_memory

    # Memory already allocated before the dry run = weights + optimizer states.
    # This is on the GPU right now (model was loaded in _setup_training_backend).
    torch.cuda.reset_peak_memory_stats(gpu_index)
    already_allocated = torch.cuda.memory_allocated(gpu_index)

    # --- Build worst-case dummy batch ---
    micro_batch_size = _compute_micro_batch_size(config)
    seq_len = config.rollout.max_seq_len
    dummy_batch = _build_dummy_batch(
        micro_batch_size=micro_batch_size,
        seq_len=seq_len,
        loss_type=config.trainer.loss_type,
        device=device,
    )

    # --- Dry run: one forward + backward on a single micro-batch ---
    model = backend.model
    was_training = model.training
    model.train()
    backend.optimizer.zero_grad()

    try:
        output = model(dummy_batch["input_ids"])
        logits = output.logits if hasattr(output, "logits") else output
        loss_result = backend.loss_fn(logits, dummy_batch)
        loss = loss_result[0] if isinstance(loss_result, tuple) else loss_result
        loss.backward()
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated(gpu_index)
        _raise_oom_error(
            peak_bytes=peak,
            gpu_total_bytes=gpu_total_bytes,
            config=config,
            micro_batch_size=micro_batch_size,
            seq_len=seq_len,
        )

    peak_training_bytes = torch.cuda.max_memory_allocated(gpu_index)

    # --- Clean up: zero grads, free activations, reset stats ---
    backend.optimizer.zero_grad(set_to_none=True)
    del dummy_batch, output, logits, loss_result, loss
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(gpu_index)
    if not was_training:
        model.eval()

    # --- Check: does training fit alongside the inference engine? ---
    shared_gpu = set(config.trainer.cuda_device_ids) & set(config.inference.cuda_device_ids)
    if shared_gpu:
        inference_bytes = config.inference.mem_fraction * gpu_total_bytes
    else:
        inference_bytes = 0.0

    safety_margin = config.trainer.vram_safety_margin
    available_bytes = gpu_total_bytes - inference_bytes
    safety_bytes = gpu_total_bytes * safety_margin
    budget_bytes = available_bytes - safety_bytes

    breakdown = {
        "gpu_total_gb": gpu_total_bytes / 1e9,
        "inference_gb": inference_bytes / 1e9,
        "available_gb": available_bytes / 1e9,
        "safety_margin_gb": safety_bytes / 1e9,
        "budget_gb": budget_bytes / 1e9,
        "peak_training_gb": peak_training_bytes / 1e9,
        "weights_optimizer_gb": already_allocated / 1e9,
        "activations_gb": (peak_training_bytes - already_allocated) / 1e9,
        "micro_batch_size": micro_batch_size,
        "seq_len": seq_len,
        "shared_gpu": bool(shared_gpu),
    }

    logger.info(
        f"VRAM preflight: peak_training={peak_training_bytes / 1e9:.1f}GB "
        f"(weights+optim={already_allocated / 1e9:.1f}GB + "
        f"activations={(peak_training_bytes - already_allocated) / 1e9:.1f}GB), "
        f"budget={budget_bytes / 1e9:.1f}GB"
    )

    if peak_training_bytes > budget_bytes:
        _raise_budget_error(
            peak_bytes=peak_training_bytes,
            budget_bytes=budget_bytes,
            gpu_total_bytes=gpu_total_bytes,
            inference_bytes=inference_bytes,
            safety_bytes=safety_bytes,
            already_allocated=already_allocated,
            config=config,
            micro_batch_size=micro_batch_size,
            seq_len=seq_len,
        )

    return breakdown


def _compute_micro_batch_size(config: Any) -> int:
    """Compute the micro-batch size that determines peak memory.

    The backend processes one micro-batch at a time, so peak activation
    memory is micro_batch_size * seq_len, not full_batch * seq_len.
    """
    full_batch_size = config.rollout.batch_size * config.rollout.n_samples_per_prompt
    num_minibatches = config.trainer.num_minibatches
    return full_batch_size // num_minibatches


def _build_dummy_batch(
    micro_batch_size: int,
    seq_len: int,
    loss_type: str,
    device: str,
) -> dict[str, Any]:
    """Build a dummy batch at worst-case dimensions for the dry run."""
    import torch

    input_ids = torch.randint(0, 1000, (micro_batch_size, seq_len), device=device)
    labels = input_ids.clone()
    loss_mask = torch.ones(micro_batch_size, seq_len, device=device)
    advantages = torch.ones(micro_batch_size, device=device)

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "advantages": advantages,
    }

    # Clipped and masked loss functions need old_logprobs
    if loss_type in ("clipped", "masked"):
        batch["old_logprobs"] = torch.zeros(micro_batch_size, device=device)

    return batch


def _raise_oom_error(
    peak_bytes: float,
    gpu_total_bytes: float,
    config: Any,
    micro_batch_size: int,
    seq_len: int,
) -> None:
    """Raise a clear error when the dry run itself OOMs."""
    raise VRAMPreflightError(
        f"VRAM preflight OOM: forward+backward on a single micro-batch "
        f"({micro_batch_size} samples x {seq_len} tokens) exceeded GPU memory.\n"
        f"  GPU total: {gpu_total_bytes / 1e9:.1f}GB\n"
        f"  Inference (mem_fraction={config.inference.mem_fraction}): "
        f"{config.inference.mem_fraction * gpu_total_bytes / 1e9:.1f}GB\n"
        f"  Reduce max_seq_len (currently {config.rollout.max_seq_len}), "
        f"batch_size, or increase num_minibatches.\n"
        f"  To skip this check: TrainerConfig(skip_vram_check=True)"
    )


def _raise_budget_error(
    peak_bytes: float,
    budget_bytes: float,
    gpu_total_bytes: float,
    inference_bytes: float,
    safety_bytes: float,
    already_allocated: float,
    config: Any,
    micro_batch_size: int,
    seq_len: int,
) -> None:
    """Raise a clear error when peak training exceeds the VRAM budget."""
    activation_bytes = peak_bytes - already_allocated
    raise VRAMPreflightError(
        f"VRAM preflight failed: training won't fit alongside inference engine.\n"
        f"  GPU total:           {gpu_total_bytes / 1e9:.1f}GB\n"
        f"  Inference engine:   -{inference_bytes / 1e9:.1f}GB "
        f"(mem_fraction={config.inference.mem_fraction})\n"
        f"  Safety margin:      -{safety_bytes / 1e9:.1f}GB "
        f"({config.trainer.vram_safety_margin:.0%} fragmentation)\n"
        f"  Budget for training: {budget_bytes / 1e9:.1f}GB\n"
        f"  Peak training:       {peak_bytes / 1e9:.1f}GB "
        f"(weights+optim={already_allocated / 1e9:.1f}GB + "
        f"activations={activation_bytes / 1e9:.1f}GB)\n"
        f"  Over by:             {(peak_bytes - budget_bytes) / 1e9:.1f}GB\n"
        f"  Batch shape: {micro_batch_size} micro-batch x {seq_len} seq_len\n"
        f"  Reduce max_seq_len (currently {config.rollout.max_seq_len}), "
        f"batch_size, or use separate GPUs for inference/training.\n"
        f"  To skip this check: TrainerConfig(skip_vram_check=True)"
    )
