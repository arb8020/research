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
    "B200": 192.0,
    "L40S": 48.0,
    "A10": 24.0,
    "A6000": 48.0,
}

# Model size estimates (billions of parameters) - fallback cache
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
    # GLM models
    "GLM-4.7-Flash": 30.0,  # 30B total, 3B active (MoE)
    "GLM-Z1-9B": 9.0,
}

# Cache for HuggingFace model configs (config.json contents)
_HF_CONFIG_CACHE: dict[str, dict[str, Any]] = {}

# Cache for HuggingFace model param counts (in billions)
_HF_PARAM_CACHE: dict[str, float] = {}


def _fetch_hf_model_config(model_name: str) -> dict[str, Any] | None:
    """Fetch model config.json from HuggingFace Hub.

    Returns dict with model architecture details, or None if unavailable.
    """
    if model_name in _HF_CONFIG_CACHE:
        return _HF_CONFIG_CACHE[model_name]

    try:
        import json

        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "config.json")
        with open(path) as f:
            config = json.load(f)
        _HF_CONFIG_CACHE[model_name] = config
        return config
    except Exception:
        pass

    return None


def _fetch_hf_param_count(model_name: str) -> float | None:
    """Fetch parameter count from HuggingFace Hub.

    Uses huggingface_hub to read safetensors metadata (fast, no download).
    Returns params in billions, or None if unavailable.
    """
    if model_name in _HF_PARAM_CACHE:
        return _HF_PARAM_CACHE[model_name]

    try:
        from huggingface_hub import get_safetensors_metadata

        meta = get_safetensors_metadata(model_name)
        if meta.parameter_count:
            total = sum(meta.parameter_count.values())
            params_b = total / 1e9
            _HF_PARAM_CACHE[model_name] = params_b
            return params_b
    except Exception:
        pass

    return None


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
    """Estimate model parameters in billions from model name.

    Tries in order:
    1. HuggingFace Hub safetensors metadata (exact count, no download)
    2. Local fallback cache
    3. Regex extraction from model name
    4. Conservative default
    """
    # Try HuggingFace Hub first (most accurate)
    hf_params = _fetch_hf_param_count(model_name)
    if hf_params is not None:
        return hf_params

    # Try local fallback cache
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


def _get_moe_info(model_name: str) -> tuple[int, int] | None:
    """Get MoE info (num_experts, num_active) from HF config if available.

    Returns (total_experts, active_experts) or None if not MoE.
    """
    hf_config = _fetch_hf_model_config(model_name)
    if hf_config is None:
        return None

    # Check for MoE config (different models use different keys)
    num_experts = hf_config.get("num_local_experts") or hf_config.get("n_routed_experts")
    num_active = hf_config.get("num_experts_per_tok") or hf_config.get("num_selected_experts")

    if num_experts and num_active:
        return (num_experts, num_active)
    return None


def estimate_inference_vram(
    model_name: str,
    dtype: str,
    mem_fraction: float,
    max_seq_len: int,
    batch_size: int,
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
) -> MemoryEstimate:
    """Estimate VRAM for inference (SGLang).

    SGLang uses:
    - Model weights
    - KV cache (dynamic, depends on seq len and batch size)
    - Activation buffers

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            Model weights and KV cache are split across these GPUs.
        expert_parallel_size: Number of GPUs for expert parallelism (MoE).
            Experts are sharded across these GPUs.
    """
    params_b = estimate_model_params_b(model_name)
    model_gb = estimate_model_vram_gb(params_b, dtype)

    # Check if MoE and adjust for expert parallelism
    moe_info = _get_moe_info(model_name)
    if moe_info and expert_parallel_size > 1:
        num_experts, _ = moe_info
        # With EP, each GPU only holds 1/EP of the experts
        # But non-expert params (embeddings, attention) are replicated
        # Rough estimate: experts are ~80% of MoE model params
        expert_fraction = 0.8
        non_expert_gb = model_gb * (1 - expert_fraction)
        expert_gb = model_gb * expert_fraction / expert_parallel_size
        model_gb = non_expert_gb + expert_gb

    # Tensor parallelism splits model weights across GPUs
    model_gb = model_gb / tensor_parallel_size

    # KV cache estimate: 2 * num_layers * hidden_dim * seq_len * batch_size * 2 bytes
    # Simplified: ~0.5GB per 1B params per 1K seq len at batch=1
    # Scale by batch_size and mem_fraction (SGLang uses this for KV cache)
    # KV cache is also split across TP GPUs
    # For MoE, KV cache is based on non-expert params (attention layers)
    kv_params_b = params_b
    if moe_info:
        # KV cache only depends on attention, not experts (~20% of MoE model)
        kv_params_b = params_b * 0.2
    kv_cache_gb = kv_params_b * (max_seq_len / 1000) * batch_size * 0.1 / tensor_parallel_size

    # Activation buffers (small for inference)
    activations_gb = params_b * 0.1 / tensor_parallel_size
    if moe_info:
        # For MoE, only active experts contribute to activations
        _, num_active = moe_info
        num_experts, _ = moe_info
        activations_gb = activations_gb * (num_active / num_experts + 0.2)  # +0.2 for non-expert

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
    tensor_parallel_size: int = 1,
    expert_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    use_fsdp: bool = False,
    use_lora: bool = False,
    activation_checkpointing: bool = False,
) -> MemoryEstimate:
    """Estimate VRAM for training.

    Training uses:
    - Model weights
    - Gradients (same size as weights)
    - Optimizer states (2x weights for Adam)
    - Activations (depends on batch size and seq len)

    Args:
        tensor_parallel_size: Number of GPUs for tensor parallelism.
        expert_parallel_size: Number of GPUs for expert parallelism (MoE).
        pipeline_parallel_size: Number of GPUs for pipeline parallelism.
        use_fsdp: Whether using FSDP (ZeRO-3 style sharding).
        use_lora: Whether using LoRA (only train adapter weights).
        activation_checkpointing: Whether using activation checkpointing.
    """
    params_b = estimate_model_params_b(model_name)
    model_gb = estimate_model_vram_gb(params_b, dtype)

    # Check if MoE
    moe_info = _get_moe_info(model_name)

    # Calculate effective parallelism for model weights
    # With EP, each GPU only holds 1/EP of the experts
    if moe_info and expert_parallel_size > 1:
        num_experts, _ = moe_info
        # Experts are ~80% of MoE model params
        expert_fraction = 0.8
        non_expert_gb = model_gb * (1 - expert_fraction)
        expert_gb = model_gb * expert_fraction / expert_parallel_size
        model_gb = non_expert_gb + expert_gb

    # Tensor parallelism splits remaining params across GPUs
    model_gb = model_gb / tensor_parallel_size

    # Pipeline parallelism splits layers across GPUs
    model_gb = model_gb / pipeline_parallel_size

    # FSDP shards optimizer states (and optionally weights/gradients)
    fsdp_shard_factor = 1
    if use_fsdp:
        # Assume FSDP world size = total GPUs / (TP * PP * EP)
        # For simplicity, assume it shards optimizer states across remaining GPUs
        fsdp_shard_factor = max(1, 8 // (tensor_parallel_size * pipeline_parallel_size))

    # Gradients: same size as model (also split by parallelism)
    gradients_gb = model_gb

    # Optimizer states: Adam uses 2x model size (momentum + variance)
    # With mixed precision, stored in fp32 = 4 bytes per param
    trainable_params_b = params_b
    if use_lora:
        # LoRA typically trains <1% of params
        trainable_params_b = params_b * 0.01

    optimizer_gb = trainable_params_b * 4 * 2  # 2x for Adam states in fp32
    optimizer_gb = optimizer_gb / tensor_parallel_size / pipeline_parallel_size
    if moe_info and expert_parallel_size > 1:
        # Expert optimizer states are also sharded
        optimizer_gb = optimizer_gb * (0.2 + 0.8 / expert_parallel_size)
    optimizer_gb = optimizer_gb / fsdp_shard_factor

    # Activations: depends on batch size, seq len, and model architecture
    total_samples = batch_size * n_samples_per_prompt
    micro_batch_size = max(1, total_samples // num_minibatches)

    hf_config = _fetch_hf_model_config(model_name)
    if hf_config is None:
        raise ValueError(
            f"Cannot estimate memory for {model_name}: "
            "unable to fetch config.json from HuggingFace Hub"
        )

    hidden_size = hf_config.get("hidden_size") or hf_config.get("dim") or 4096
    num_layers = hf_config.get("num_hidden_layers") or hf_config.get("n_layers") or 32

    # EleutherAI formula with selective checkpointing
    s = max_seq_len
    b = micro_batch_size
    h = hidden_size
    L = num_layers // pipeline_parallel_size  # Layers per GPU with PP
    t = tensor_parallel_size

    # Base activation formula (with selective checkpointing)
    activation_bytes = s * b * h * L * (10 + 24 / t)

    # Activation checkpointing reduces memory by recomputing
    if activation_checkpointing:
        activation_bytes = activation_bytes * 0.3  # ~70% reduction

    # For MoE, only active experts contribute to activations per token
    if moe_info:
        num_experts, num_active = moe_info
        # Scale down: only num_active/num_experts of expert computation
        moe_scale = (num_active / num_experts) * 0.8 + 0.2  # 0.8 for experts, 0.2 for non-expert
        activation_bytes = activation_bytes * moe_scale

    activations_gb = activation_bytes / 1e9

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

    # Get parallelism config (defaults to 1 if not specified)
    inference_tp = getattr(config.inference, "tensor_parallel_size", 1)
    inference_ep = getattr(config.inference, "expert_parallel_size", 1)
    trainer_tp = getattr(config.trainer, "tensor_parallel_size", 1)
    trainer_ep = getattr(config.trainer, "expert_parallel_size", 1)
    trainer_pp = getattr(config.trainer, "pipeline_parallel_size", 1)
    use_lora = getattr(config.model, "use_lora", False)
    activation_checkpointing = getattr(config.trainer, "activation_checkpointing", False)

    # Check if using FSDP-style backend
    backend = getattr(config.trainer, "backend", "pytorch")
    use_fsdp = backend in ("fsdp", "fsdp2", "torchtitan")

    # Estimate inference VRAM (per GPU after parallelism split)
    inference_est = estimate_inference_vram(
        model_name=config.model.name,
        dtype=getattr(config.model, "dtype", "bfloat16"),
        mem_fraction=config.inference.mem_fraction,
        max_seq_len=config.rollout.max_seq_len,
        batch_size=config.rollout.batch_size,
        tensor_parallel_size=inference_tp,
        expert_parallel_size=inference_ep,
    )

    # Estimate training VRAM (per GPU after parallelism split)
    training_est = estimate_training_vram(
        model_name=config.model.name,
        dtype=getattr(config.model, "dtype", "bfloat16"),
        batch_size=config.rollout.batch_size,
        n_samples_per_prompt=config.rollout.n_samples_per_prompt,
        num_minibatches=config.trainer.num_minibatches,
        max_seq_len=config.rollout.max_seq_len,
        tensor_parallel_size=trainer_tp,
        expert_parallel_size=trainer_ep,
        pipeline_parallel_size=trainer_pp,
        use_fsdp=use_fsdp,
        use_lora=use_lora,
        activation_checkpointing=activation_checkpointing,
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
