"""GLM-4.7-Flash GRPO training experiment.

Uses TorchTitan backend for distributed training with GLM model support.

Run with:
    # RunPod 2×B200 (recommended for GLM-4.7-Flash 30B MoE)
    python -m rollouts.run --config examples/rl/glm/grpo_glm_01.py --provision --provider runpod

    # Local (requires 2x B200 or 2x H100 80GB)
    python -m rollouts.run --config examples/rl/glm/grpo_glm_01.py --local

Note:
    GLM-4.7-Flash is a 30B MoE model with 3.6B active parameters.
    With 2×B200: inference on GPU 0, training on GPU 1.
    Expert parallelism (EP=2) is NOT used here since we have dedicated GPUs
    for inference vs training. EP would only help if we needed to shard the
    model across multiple GPUs for a single role.
"""

from examples.rl.glm.base_config import train as _base_train
from rollouts.training.configs import HardwareConfig
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

hardware = HardwareConfig(
    gpu_type="B200",
    gpu_count=2,  # 1 for inference, 1 for training
    provider="runpod",
)

# =============================================================================
# Training Configuration
# =============================================================================

# GLM-4.7-Flash: 30B MoE, 3.6B active
MODEL_NAME = "zai-org/GLM-4.7-Flash"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="glm_4.7_flash_grpo"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="glm",
        torchtitan_model_size="4.7-flash",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=8,
        loss_type="vanilla",
        # GPU assignment: inference on GPU 0, training on GPU 1
        cuda_device_ids=(1,),
    ),
    inference=InferenceConfig(
        backend="sglang",
        cuda_device_ids=(0,),
        mem_fraction=0.9,  # GLM MoE needs more VRAM
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=8,
        temperature=0.7,
        max_seq_len=2048,
        max_tokens=256,
    ),
    checkpoint=CheckpointConfig(
        num_steps=50,
        checkpoint_every=10,
        sync_weights_every=1,
        weight_sync_mode="disk",  # NCCL weight sync not yet supported for TorchTitan
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run GLM GRPO training."""
    return _base_train(config=config, **kwargs)
