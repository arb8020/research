"""Mini GLM MoE GRPO training on Modal with Megatron backend.

This is a stack-liveness witness modeled after Prime RL's MoE CI path, which
uses a deliberately small GLM MoE checkpoint instead of full GLM-4.7-Flash.

Run with:
    python -m argus run --config examples/rl/glm/grpo_glm_mini_modal_megatron.py
"""

from dataclasses import replace

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

_megatron_modal_deps = default_remote_megatron_training_deps()
_megatron_modal_deps = replace(
    _megatron_modal_deps,
    runtime_overlay=_megatron_modal_deps.runtime_overlay.extended(
        env={"PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    ),
)

hardware = HardwareConfig(
    gpu_type="A100-80GB",
    gpu_count=8,
    provider="modal",
    deps=_megatron_modal_deps,
    use_torchrun=False,
)

MODEL_NAME = "samsja/mini-glm-moe"

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="mini_glm_moe_grpo_modal_megatron"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="megatron",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=4,
        loss_type="vanilla",
        cuda_device_ids=(4, 5, 6, 7),
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        expert_parallel_size=4,
        sequence_parallel=False,
        activation_checkpointing=True,
        validate_inference_export_in_preflight=False,
    ),
    inference=InferenceConfig(
        backend="sglang",
        realization="slime-sglang",
        cuda_device_ids=(0, 1, 2, 3),
        mem_fraction=0.7,
        startup_timeout=300.0,
    ),
    rollout=RolloutConfig(
        batch_size=4,
        n_samples_per_prompt=4,
        temperature=0.7,
        max_seq_len=1024,
        max_tokens=128,
    ),
    checkpoint=CheckpointConfig(
        num_steps=8,
        checkpoint_every=4,
        sync_weights_every=1,
        weight_sync_mode="nccl",
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    """Run mini GLM MoE GRPO training with Megatron backend."""
    return _base_train(config=config, **kwargs)


if __name__ == "__main__":
    train(config)
