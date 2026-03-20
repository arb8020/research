"""GLM-4.7-Flash GRPO training on Modal with Megatron backend.

Uses Megatron instead of TorchTitan to avoid torch.nn.attention.varlen dependency.
The shared runtime stack is pinned to the Miles-stable Megatron/SGLang baseline
so our witness path does not invent a separate dependency combination.

Run with:
    python -m argus run --config examples/rl/glm/grpo_glm_modal_megatron.py
"""

from examples.rl.base_config import default_remote_megatron_training_deps
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
# Hardware Configuration for Modal (Megatron backend)
# =============================================================================

hardware = HardwareConfig(
    gpu_type="A100-80GB",
    gpu_count=8,
    provider="modal",
    deps=default_remote_megatron_training_deps(),
    use_torchrun=False,  # Training script handles DDP internally
)

# =============================================================================
# Training Configuration
# =============================================================================

MODEL_NAME = "zai-org/GLM-4.7-Flash"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="glm_4.7_flash_grpo_modal_megatron"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=4,  # Reduced for 4 GPUs
        loss_type="vanilla",
        # Split mode: 4 GPUs for training (GPUs 4-7), 4 for inference (GPUs 0-3)
        cuda_device_ids=(4, 5, 6, 7),
        # Megatron parallelism for MoE with 4 GPUs
        # EP=4 distributes 64 experts across 4 GPUs (16 experts per GPU)
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=4,
        sequence_parallel=False,
        activation_checkpointing=True,
    ),
    inference=InferenceConfig(
        backend="sglang",
        realization="slime-sglang",
        # 4 GPUs for inference with TP=4 (or DP=4)
        cuda_device_ids=(0, 1, 2, 3),
        mem_fraction=0.9,
        startup_timeout=600.0,
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
        weight_sync_mode="nccl",
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run GLM GRPO training with Megatron backend."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train(config)
