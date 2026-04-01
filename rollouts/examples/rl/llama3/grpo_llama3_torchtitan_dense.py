"""Dense RL witness path using TorchTitan.

This is the canonical dense RL TorchTitan instantiation path.
"""

from examples.rl.base_config import default_remote_training_deps
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

hardware = HardwareConfig(
    gpu_type="H100",
    gpu_count=2,
    provider="runpod",
    deps=default_remote_training_deps(),
    hf_cache_dir="/workspace/.cache/huggingface",
)

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="llama3_8b_torchtitan_dense_grpo"),
    model=ModelConfig(
        name="meta-llama/Llama-3.1-8B",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="llama3",
        torchtitan_model_size="8B",
        lr=1e-6,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=8,
        loss_type="vanilla",
        cuda_device_ids=(1,),
    ),
    inference=InferenceConfig(
        cuda_device_ids=(0,),
        mem_fraction=0.85,
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
        weight_sync_mode="disk",
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    return _base_train(config=config or globals()["config"], **kwargs)
