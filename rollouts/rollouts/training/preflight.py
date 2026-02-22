"""Pre-flight validation for GRPO training configs.

Validates config against hardware limits before starting a run.
Two phases:
1. Static estimation - quick sanity check before provisioning
2. Dry run - accurate peak memory measurement after provisioning
"""

from dataclasses import dataclass
from typing import Any

# Known GPU VRAM sizes (GB)
GPU_VRAM_GB: dict[str, float] = {
    "RTX A5000": 24.0,
    "RTX 4090": 24.0,
    "RTX 3090": 24.0,
    "A100": 40.0,  # 40GB variant
    "A100-80GB": 80.0,
    "A100 80GB": 80.0,
    "H100": 80.0,
    "H100 SXM": 80.0,
    "L40S": 48.0,
    "A10": 24.0,
    "A6000": 48.0,
}

# Model size estimates (billions of parameters)
MODEL_PARAMS_B: dict[str, float] = {
    # Qwen models
    "Qwen3-0.6B": 0.6,
    "Qwen3-1.5B": 1.5,
    "Qwen3-4B": 4.0,
    "Qwen3-8B": 8.0,
    "Qwen3-14B": 14.0,
    "Qwen3-32B": 32.0,
    "Qwen3-72B": 72.0,
    # Llama models
    "Llama-3-8B": 8.0,
    "Llama-3-70B": 70.0,
    "Llama-3.1-8B": 8.0,
    "Llama-3.1-70B": 70.0,
    # Mistral models
    "Mistral-7B": 7.0,
    "Mixtral-8x7B": 47.0,  # ~47B total, 13B active
}


@dataclass
class MemoryEstimate:
    """Memory usage estimate for a component."""

    model_gb: float  # Model weights
    activations_gb: float  # Forward pass activations
    gradients_gb: float  # Backward pass gradients
    optimizer_gb: float  # Optimizer states (Adam: 2x model size)
    kv_cache_gb: float  # KV cache for inference
    buffer_gb: float  # NCCL buffers, misc

    @property
    def total_gb(self) -> float:
        return (
            self.model_gb
            + self.activations_gb
            + self.gradients_gb
            + self.optimizer_gb
            + self.kv_cache_gb
            + self.buffer_gb
        )

    def __str__(self) -> str:
        return (
            f"model={self.model_gb:.1f}GB, "
            f"activations={self.activations_gb:.1f}GB, "
            f"gradients={self.gradients_gb:.1f}GB, "
            f"optimizer={self.optimizer_gb:.1f}GB, "
            f"kv_cache={self.kv_cache_gb:.1f}GB, "
            f"buffer={self.buffer_gb:.1f}GB, "
            f"total={self.total_gb:.1f}GB"
        )


def get_gpu_vram_gb(gpu_type: str) -> float:
    """Get VRAM for a GPU type. Raises ValueError if unknown."""
    # Try exact match first
    if gpu_type in GPU_VRAM_GB:
        return GPU_VRAM_GB[gpu_type]

    # Try partial match
    for known_gpu, vram in GPU_VRAM_GB.items():
        if known_gpu.lower() in gpu_type.lower():
            return vram

    raise ValueError(f"Unknown GPU type: {gpu_type}. Known types: {list(GPU_VRAM_GB.keys())}")


def estimate_model_params_b(model_name: str) -> float:
    """Estimate model parameters in billions from model name."""
    # Try exact match
    for known_model, params in MODEL_PARAMS_B.items():
        if known_model.lower() in model_name.lower():
            return params

    # Try to extract from name (e.g., "Qwen3-0.6B" -> 0.6)
    import re

    match = re.search(r"(\d+\.?\d*)B", model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))

    # Default to 7B if unknown (conservative estimate)
    return 7.0


def estimate_model_vram_gb(params_b: float, dtype: str = "bfloat16") -> float:
    """Estimate VRAM for model weights.

    Args:
        params_b: Parameters in billions
        dtype: Data type (bfloat16, float16, float32)

    Returns:
        Estimated VRAM in GB
    """
    bytes_per_param = {
        "bfloat16": 2,
        "float16": 2,
        "float32": 4,
        "int8": 1,
        "int4": 0.5,
    }.get(dtype.lower(), 2)

    # params_b billion * bytes_per_param / 1e9 = GB
    return params_b * bytes_per_param


def estimate_inference_vram(
    model_name: str,
    dtype: str,
    mem_fraction: float,
    max_seq_len: int,
    batch_size: int,
) -> MemoryEstimate:
    """Estimate VRAM for inference (SGLang).

    SGLang uses:
    - Model weights
    - KV cache (dynamic, depends on seq len and batch size)
    - Activation buffers
    """
    params_b = estimate_model_params_b(model_name)
    model_gb = estimate_model_vram_gb(params_b, dtype)

    # KV cache estimate: 2 * num_layers * hidden_dim * seq_len * batch_size * 2 bytes
    # Simplified: ~0.5GB per 1B params per 1K seq len at batch=1
    # Scale by batch_size and mem_fraction (SGLang uses this for KV cache)
    kv_cache_gb = params_b * (max_seq_len / 1000) * batch_size * 0.1

    # Activation buffers (small for inference)
    activations_gb = params_b * 0.1

    # NCCL/misc buffers
    buffer_gb = 0.5

    return MemoryEstimate(
        model_gb=model_gb,
        activations_gb=activations_gb,
        gradients_gb=0.0,  # No gradients for inference
        optimizer_gb=0.0,  # No optimizer for inference
        kv_cache_gb=kv_cache_gb,
        buffer_gb=buffer_gb,
    )


def estimate_training_vram(
    model_name: str,
    dtype: str,
    batch_size: int,
    n_samples_per_prompt: int,
    num_minibatches: int,
    max_seq_len: int,
) -> MemoryEstimate:
    """Estimate VRAM for training.

    Training uses:
    - Model weights
    - Gradients (same size as weights)
    - Optimizer states (2x weights for Adam)
    - Activations (depends on batch size and seq len)
    """
    params_b = estimate_model_params_b(model_name)
    model_gb = estimate_model_vram_gb(params_b, dtype)

    # Gradients: same size as model
    gradients_gb = model_gb

    # Optimizer states: Adam uses 2x model size (momentum + variance)
    # With mixed precision, stored in fp32 = 4 bytes per param
    optimizer_gb = params_b * 4 * 2  # 2x for Adam states

    # Activations: depends on batch size, seq len, and model architecture
    # Rough estimate: ~4 bytes per param per token in batch
    # micro_batch_size = total_samples / num_minibatches
    total_samples = batch_size * n_samples_per_prompt
    micro_batch_size = total_samples // num_minibatches
    tokens_per_micro_batch = micro_batch_size * max_seq_len

    # Activation memory scales with tokens and model size
    # ~2GB per 1B params per 1K tokens (rough estimate)
    activations_gb = params_b * (tokens_per_micro_batch / 1000) * 0.5

    # NCCL buffers for weight sync
    buffer_gb = model_gb * 0.1 + 1.0  # 10% of model + 1GB misc

    return MemoryEstimate(
        model_gb=model_gb,
        activations_gb=activations_gb,
        gradients_gb=gradients_gb,
        optimizer_gb=optimizer_gb,
        kv_cache_gb=0.0,  # No KV cache for training
        buffer_gb=buffer_gb,
    )


@dataclass
class PreflightResult:
    """Result of preflight validation."""

    valid: bool
    inference_estimate: MemoryEstimate
    training_estimate: MemoryEstimate
    gpu_vram_gb: float
    warnings: list[str]
    errors: list[str]

    def __str__(self) -> str:
        lines = [
            f"GPU VRAM: {self.gpu_vram_gb:.1f}GB",
            f"Inference estimate: {self.inference_estimate}",
            f"Training estimate: {self.training_estimate}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if self.errors:
            lines.append("Errors:")
            for e in self.errors:
                lines.append(f"  - {e}")
        return "\n".join(lines)


def validate_config(config: Any, gpu_type: str) -> PreflightResult:
    """Validate GRPO config against hardware limits.

    Args:
        config: GRPOConfig instance
        gpu_type: GPU type string (e.g., "RTX A5000")

    Returns:
        PreflightResult with validation status and estimates

    Raises:
        ValueError: If config is fundamentally invalid (fail fast)
    """
    warnings: list[str] = []
    errors: list[str] = []

    # Get GPU VRAM
    try:
        gpu_vram_gb = get_gpu_vram_gb(gpu_type)
    except ValueError as e:
        raise ValueError(str(e)) from e

    # Estimate inference VRAM
    inference_est = estimate_inference_vram(
        model_name=config.model.name,
        dtype=getattr(config.model, "dtype", "bfloat16"),
        mem_fraction=config.inference.mem_fraction,
        max_seq_len=config.rollout.max_seq_len,
        batch_size=config.rollout.batch_size,
    )

    # Estimate training VRAM
    training_est = estimate_training_vram(
        model_name=config.model.name,
        dtype=getattr(config.model, "dtype", "bfloat16"),
        batch_size=config.rollout.batch_size,
        n_samples_per_prompt=config.rollout.n_samples_per_prompt,
        num_minibatches=config.trainer.num_minibatches,
        max_seq_len=config.rollout.max_seq_len,
    )

    # Check inference GPU
    inference_headroom = gpu_vram_gb - inference_est.total_gb
    if inference_headroom < 0:
        errors.append(
            f"Inference requires {inference_est.total_gb:.1f}GB but GPU has {gpu_vram_gb:.1f}GB. "
            f"Reduce mem_fraction from {config.inference.mem_fraction} or use smaller model."
        )
    elif inference_headroom < 2.0:
        warnings.append(
            f"Inference has only {inference_headroom:.1f}GB headroom. "
            f"Consider reducing mem_fraction for stability."
        )

    # Check training GPU
    training_headroom = gpu_vram_gb - training_est.total_gb
    if training_headroom < 0:
        errors.append(
            f"Training requires {training_est.total_gb:.1f}GB but GPU has {gpu_vram_gb:.1f}GB. "
            f"Reduce batch_size, num_minibatches, or max_seq_len."
        )
    elif training_headroom < 2.0:
        warnings.append(
            f"Training has only {training_headroom:.1f}GB headroom. "
            f"Consider reducing batch_size for stability."
        )

    # Check true_pipeline mode (needs extra buffer for concurrent operations)
    if getattr(config.checkpoint, "pipeline_mode", None) == "true_pipeline":
        # In true_pipeline, inference and training run concurrently with weight sync
        # Need extra buffer on both GPUs
        if inference_headroom < 4.0:
            warnings.append(
                f"true_pipeline mode: inference headroom ({inference_headroom:.1f}GB) may be tight. "
                f"Reduce mem_fraction to 0.6-0.7 for concurrent weight sync."
            )
        if training_headroom < 4.0:
            warnings.append(
                f"true_pipeline mode: training headroom ({training_headroom:.1f}GB) may be tight. "
                f"Reduce batch_size or num_minibatches."
            )

    return PreflightResult(
        valid=len(errors) == 0,
        inference_estimate=inference_est,
        training_estimate=training_est,
        gpu_vram_gb=gpu_vram_gb,
        warnings=warnings,
        errors=errors,
    )


def run_preflight_check(config: Any, gpu_type: str) -> None:
    """Run preflight check and raise if invalid.

    Args:
        config: GRPOConfig instance
        gpu_type: GPU type string

    Raises:
        ValueError: If config is invalid for the hardware
    """
    import logging

    logger = logging.getLogger(__name__)

    result = validate_config(config, gpu_type)

    logger.info(f"Preflight check for {gpu_type}:")
    logger.info(f"  Inference: {result.inference_estimate.total_gb:.1f}GB estimated")
    logger.info(f"  Training: {result.training_estimate.total_gb:.1f}GB estimated")
    logger.info(f"  GPU VRAM: {result.gpu_vram_gb:.1f}GB available")

    for warning in result.warnings:
        logger.warning(f"  ⚠ {warning}")

    if not result.valid:
        error_msg = "\n".join(f"  ✗ {e}" for e in result.errors)
        raise ValueError(f"Preflight check failed:\n{error_msg}")

    logger.info("  ✓ Preflight check passed")


# =============================================================================
# Dry Run: Actual Memory Measurement
# =============================================================================


async def measure_peak_memory(
    config: Any,
    backend: Any,
    inference_engine: Any,
    device: Any,
) -> dict[str, float]:
    """Run one training step and measure actual peak memory.

    This is more accurate than estimation because it measures real usage.

    Args:
        config: GRPOConfig
        backend: TrainingBackend
        inference_engine: InferenceEngine
        device: torch.device

    Returns:
        Dict with peak memory measurements
    """
    import torch

    measurements: dict[str, float] = {}

    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)

    # Measure inference (one batch of generations)
    # TODO: Generate one batch and measure

    # Measure training (one forward/backward)
    # TODO: Run one minibatch and measure

    # Get peak memory
    peak_bytes = torch.cuda.max_memory_allocated(device)
    measurements["peak_allocated_gb"] = peak_bytes / 1e9

    peak_reserved = torch.cuda.max_memory_reserved(device)
    measurements["peak_reserved_gb"] = peak_reserved / 1e9

    return measurements
