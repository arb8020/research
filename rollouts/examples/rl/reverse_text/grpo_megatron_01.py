"""Reverse Text GRPO with Megatron backend + GLM-4.7-Flash.

Based on SLIME's configuration for GLM-4.7-Flash training.
Reference: /tmp/slime/scripts/run-glm4.7-30B-A3B.sh

Hardware Requirements (from SLIME):
- 16x H100-80GB (2 nodes × 8 GPUs) for full SLIME config
- OR 8x H100-80GB with CPU optimizer offload (colocated inference+training)

This config uses 8x H100/A100-80GB with colocated mode.

Memory Budget per GPU (with CPU optimizer offload):
- Model weights (TP=4): 31.2B × 2 bytes / 4 = 15.6 GB
- Gradients (TP=4): 15.6 GB
- Optimizer (CPU offload): 0 GB on GPU
- Activations (checkpointed): ~25 GB for batch=4, seq=4096
- Total: ~56 GB < 80 GB ✓

Parallelism (following SLIME):
- tensor_parallel_size=4
- pipeline_parallel_size=1 (single node)
- expert_parallel_size=8 (all GPUs share expert computation)
- context_parallel_size=1
- sequence_parallel=True

Key optimizations from SLIME:
- --optimizer-cpu-offload (CPU Adam)
- --recompute-granularity full (activation checkpointing)
- --use-dynamic-batch-size
"""

from dataclasses import dataclass

from examples.rl.reverse_text.base_config import train as _base_train
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.grpo import (
    CheckpointConfig,
    GRPOConfig,
    GRPOOutputConfig,
    InferenceConfig,
    ModelConfig,
    RolloutConfig,
    TrainerConfig,
)

# =============================================================================
# Hardware Specification
# =============================================================================


@dataclass
class GPUSpec:
    """GPU hardware specification for memory calculations."""

    name: str
    vram_gb: float


H100_80GB = GPUSpec(name="H100-80GB", vram_gb=80.0)
A100_80GB = GPUSpec(name="A100-80GB", vram_gb=80.0)


@dataclass
class ModelSpec:
    """Model specification for memory calculations."""

    name: str
    total_params_b: float  # Total parameters in billions
    active_params_b: float  # Active parameters per forward (for MoE)
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int  # experts per token


GLM_4_7_FLASH = ModelSpec(
    name="zai-org/GLM-4.7-Flash",
    total_params_b=31.2,
    active_params_b=3.0,  # ~3B active with top-4 of 64
    hidden_size=2048,
    num_layers=47,
    num_experts=64,
    top_k=4,
)


@dataclass
class ParallelismConfig:
    """Parallelism configuration for Megatron."""

    tensor_parallel: int
    pipeline_parallel: int
    expert_parallel: int
    context_parallel: int
    sequence_parallel: bool

    @property
    def total_gpus(self) -> int:
        """Total GPUs needed for this parallelism config."""
        # TP and PP are multiplicative, EP is separate dimension
        return self.tensor_parallel * self.pipeline_parallel


# SLIME's config for GLM-4.7-Flash on 8 GPUs (colocated mode)
PARALLELISM_8GPU = ParallelismConfig(
    tensor_parallel=4,
    pipeline_parallel=1,  # Single node
    expert_parallel=8,  # All 8 GPUs for expert parallelism
    context_parallel=1,
    sequence_parallel=True,
)


def estimate_memory_per_gpu(
    model: ModelSpec,
    parallelism: ParallelismConfig,
    batch_size: int,
    seq_len: int,
    optimizer_cpu_offload: bool = True,
    activation_checkpointing: bool = True,
) -> dict[str, float]:
    """Estimate GPU memory usage per GPU.

    Based on EleutherAI Transformer Math 101 formulas.
    https://blog.eleuther.ai/transformer-math/
    """
    tp = parallelism.tensor_parallel

    # Model weights: total_params × 2 bytes (bf16) / TP
    model_gb = model.total_params_b * 2 / tp

    # Gradients: same as model weights
    gradients_gb = model_gb

    # Optimizer states (Adam): 2 × model_size in fp32 = 8 bytes per param
    # With CPU offload, this is 0 on GPU
    if optimizer_cpu_offload:
        optimizer_gb = 0.0
    else:
        optimizer_gb = model.total_params_b * 8 / tp

    # Activations with checkpointing (EleutherAI formula):
    # activations = s × b × h × L × (10 + 24/t) bytes
    if activation_checkpointing:
        activation_bytes = (
            seq_len * batch_size * model.hidden_size * model.num_layers * (10 + 24 / tp)
        )
    else:
        # Without checkpointing, much higher (includes attention scores)
        activation_bytes = (
            seq_len
            * batch_size
            * model.hidden_size
            * model.num_layers
            * (34 + 5 * seq_len / model.hidden_size)
        )
    activations_gb = activation_bytes / 1e9

    # Buffer for NCCL, misc
    buffer_gb = 2.0

    total_gb = model_gb + gradients_gb + optimizer_gb + activations_gb + buffer_gb

    return {
        "model_gb": model_gb,
        "gradients_gb": gradients_gb,
        "optimizer_gb": optimizer_gb,
        "activations_gb": activations_gb,
        "buffer_gb": buffer_gb,
        "total_gb": total_gb,
    }


# =============================================================================
# Configuration
# =============================================================================

# Megatron-LM commit tested by SLIME for GLM compatibility
MEGATRON_COMMIT = "3714d81d418c9f1bca4594fc35f9e8289f652862"

# Training hyperparameters (matching SLIME)
BATCH_SIZE = 4  # Per minibatch
SEQ_LEN = 4096
N_SAMPLES_PER_PROMPT = 4
NUM_MINIBATCHES = 4

# Verify memory fits
GPU = H100_80GB
MODEL = GLM_4_7_FLASH
PARALLELISM = PARALLELISM_8GPU

memory = estimate_memory_per_gpu(
    model=MODEL,
    parallelism=PARALLELISM,
    batch_size=BATCH_SIZE,
    seq_len=SEQ_LEN,
    optimizer_cpu_offload=True,
    activation_checkpointing=True,
)

# Static assertion - this runs at import time
assert memory["total_gb"] < GPU.vram_gb, (
    f"Configuration does not fit in GPU memory!\n"
    f"Estimated: {memory['total_gb']:.1f} GB, Available: {GPU.vram_gb} GB\n"
    f"Breakdown: model={memory['model_gb']:.1f}, grads={memory['gradients_gb']:.1f}, "
    f"opt={memory['optimizer_gb']:.1f}, activations={memory['activations_gb']:.1f}"
)

# Hardware config
hardware = HardwareConfig(
    gpu_type="H100",  # or A100-80GB
    gpu_count=8,  # Single node, colocated inference+training
    provider="runpod",
    deps=DepsConfig(
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)

# GRPO config
config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_megatron_glm47"),
    model=ModelConfig(
        name=MODEL.name,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        # Colocated mode: all 8 GPUs for both inference and training
        cuda_device_ids=tuple(range(8)),
        lr=1e-6,
        weight_decay=0.1,  # SLIME uses 0.1
        num_minibatches=NUM_MINIBATCHES,
        max_grad_norm=1.0,
        # Megatron parallelism (from SLIME)
        tensor_parallel_size=PARALLELISM.tensor_parallel,
        pipeline_parallel_size=PARALLELISM.pipeline_parallel,
        expert_parallel_size=PARALLELISM.expert_parallel,
        context_parallel_size=PARALLELISM.context_parallel,
        sequence_parallel=PARALLELISM.sequence_parallel,
        seq_length=SEQ_LEN,
        # Memory optimizations (from SLIME)
        optimizer_cpu_offload=True,
        activation_checkpointing=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    ),
    inference=InferenceConfig(
        backend="sglang",
        # Colocated: same GPUs as trainer
        cuda_device_ids=tuple(range(8)),
        port=30000,
        mem_fraction=0.7,  # Lower for colocated mode
        tensor_parallel_size=PARALLELISM.tensor_parallel,
        # SGLang MoE settings (from SLIME)
        expert_parallel_size=PARALLELISM.expert_parallel,
    ),
    rollout=RolloutConfig(
        batch_size=BATCH_SIZE,
        n_samples_per_prompt=N_SAMPLES_PER_PROMPT,
        max_seq_len=SEQ_LEN,
        max_tokens=512,
        temperature=0.8,
    ),
    checkpoint=CheckpointConfig(
        num_steps=100,
        log_every=1,
        checkpoint_every=50,
        sync_weights_every=10,
        weight_sync_mode="nccl",
        pipeline_mode="sync",
    ),
)

# Print memory estimate at import
print(f"Memory estimate per GPU: {memory['total_gb']:.1f} GB / {GPU.vram_gb} GB")
print(f"  Model: {memory['model_gb']:.1f} GB")
print(f"  Gradients: {memory['gradients_gb']:.1f} GB")
print(f"  Optimizer: {memory['optimizer_gb']:.1f} GB (CPU offload)")
print(f"  Activations: {memory['activations_gb']:.1f} GB")


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run Megatron training on reverse text task."""
    return _base_train(config=config, num_samples=100, **kwargs)
