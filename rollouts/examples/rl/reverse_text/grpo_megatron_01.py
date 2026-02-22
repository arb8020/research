"""Reverse Text GRPO with Megatron backend + GLM-4.7-Flash.

Tests Megatron-Core integration with miniray orchestration on a real RL task.
Uses GLM-4.7-Flash (30B-A3B MoE) - small enough for fast iteration.

Architecture:
- 64 routed experts, top-4 routing (A3B = 3B active params of 30B total)
- Multi-Latent Attention (MLA) for efficient KV compression
- Supported via our custom mbridge GLM4MoEBridge

Run with:
    # RunPod (8x A100-80GB)
    python -m rollouts.run --config examples/rl/reverse_text/grpo_megatron_01.py

    # With TUI monitor
    python -m rollouts.run --config examples/rl/reverse_text/grpo_megatron_01.py --tui

Requirements:
    - 8x A100-80GB GPUs (4x inference TP=4, 4x trainer TP=4)
    - mbridge + megatron-core installed on remote
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
# Hardware Configuration
# =============================================================================

# Megatron-LM commit tested by SLIME for GLM compatibility
MEGATRON_COMMIT = "3714d81d418c9f1bca4594fc35f9e8289f652862"

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=8,  # 4x inference TP=4, 4x trainer TP=4
    provider="runpod",  # RunPod for easier debugging
    # RunPod uses run.py bootstrap - deps here are for reference/Modal fallback
    deps=DepsConfig(
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)

# =============================================================================
# Training Configuration
# =============================================================================

# GLM-4.7-Flash: 30B-A3B MoE model
# - 64 routed experts, top-4 routing
# - Multi-Latent Attention (MLA)
# - Supported via custom GLM4MoEBridge in rollouts/training/backends/megatron/mbridge/
MODEL = "THUDM/GLM-4.7-Flash"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="reverse_text_megatron_glm47"),
    model=ModelConfig(
        name=MODEL,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        cuda_device_ids=(4, 5, 6, 7),  # 4 GPUs for trainer
        lr=1e-6,
        weight_decay=0.0,
        num_minibatches=4,
        max_grad_norm=1.0,
        # Megatron parallelism settings for GLM-4.7-Flash
        tensor_parallel_size=4,  # TP=4 for MoE
        pipeline_parallel_size=1,
        expert_parallel_size=1,  # Could increase for multi-node
        seq_length=4096,
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0, 1, 2, 3),  # 4 GPUs for inference
        port=30000,
        mem_fraction=0.85,
        tensor_parallel_size=4,
    ),
    rollout=RolloutConfig(
        batch_size=8,
        n_samples_per_prompt=4,
        max_seq_len=4096,
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
    """Run Megatron training on reverse text task."""
    return _base_train(config=config, num_samples=100, **kwargs)
