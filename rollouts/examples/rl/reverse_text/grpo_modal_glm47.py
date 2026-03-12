"""Reverse Text GRPO with Megatron backend + GLM-4.7-Flash on Modal.

Based on grpo_megatron_01.py but configured for Modal 8xH100.

Run with:
    python -m argus run --config examples/rl/reverse_text/grpo_modal_glm47.py --gpu H100 --gpu-count 8

Modal advantages:
- GPU snapshotting for fast cold starts (~10x faster)
- Scale to zero when not in use
- Single-node 8xH100 with NVLink
"""

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
# Hardware Configuration for Modal
# =============================================================================

hardware = HardwareConfig(
    gpu_type="H100",
    gpu_count=8,
    provider="modal",
    deps=DepsConfig(
        python_version="3.11",
        system_packages=("git", "curl", "wget", "build-essential"),
        pip_packages=(
            "torch>=2.4",
            "transformers>=5.0",
            "datasets",
            "accelerate",
            "safetensors",
            # Use released sglang instead of git install
            "sglang[all]>=0.4",
            "curl_cffi",
            "peft",
            "huggingface_hub>=1.4.0",
            "ninja",
            "packaging",
            # Flash attention from pre-built wheel
            "flash-attn>=2.7",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
        bootstrap_commands=(
            # Clone Megatron-LM for megatron.core imports
            "git clone --depth 1 https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM",
        ),
    ),
)

# =============================================================================
# Model and Parallelism Configuration
# =============================================================================

MODEL_NAME = "zai-org/GLM-4.7-Flash"

# Parallelism (following SLIME for GLM-4.7-Flash on 8 GPUs)
TENSOR_PARALLEL = 4
PIPELINE_PARALLEL = 1
EXPERT_PARALLEL = 8
CONTEXT_PARALLEL = 1
SEQUENCE_PARALLEL = True

# Training hyperparameters
BATCH_SIZE = 4
SEQ_LEN = 4096
N_SAMPLES_PER_PROMPT = 4
NUM_MINIBATCHES = 4

# =============================================================================
# GRPO Configuration
# =============================================================================

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_modal_glm47"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        cuda_device_ids=tuple(range(8)),
        lr=1e-6,
        weight_decay=0.1,
        num_minibatches=NUM_MINIBATCHES,
        max_grad_norm=1.0,
        # Megatron parallelism
        tensor_parallel_size=TENSOR_PARALLEL,
        pipeline_parallel_size=PIPELINE_PARALLEL,
        expert_parallel_size=EXPERT_PARALLEL,
        context_parallel_size=CONTEXT_PARALLEL,
        sequence_parallel=SEQUENCE_PARALLEL,
        seq_length=SEQ_LEN,
        # Memory optimizations
        optimizer_cpu_offload=True,
        activation_checkpointing=True,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=tuple(range(8)),
        port=30000,
        mem_fraction=0.7,
        tensor_parallel_size=TENSOR_PARALLEL,
        expert_parallel_size=EXPERT_PARALLEL,
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


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run Megatron training on reverse text task via Modal."""
    return _base_train(config=config, num_samples=100, **kwargs)


if __name__ == "__main__":
    train()
