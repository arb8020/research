"""Offline dense pretraining witness path using TorchTitan + Qwen3-0.6B.

This is the first external-backend pretraining config worth exercising:

- offline dense next-token training
- one training GPU, no inference engine
- shared training runtime factory
- TorchTitan lowering path, not RL orchestration

Before running, point `source_path` at a directory of tokenized `.npy` or `.bin`
shards compatible with `rollouts.pretrain.dataloader.build_loader`.
"""

from examples.rl.base_config import default_remote_training_deps
from rollouts.training import PretrainConfig, PretrainDataConfig, run_pretrain
from rollouts.training.configs import (
    CheckpointConfig,
    HardwareConfig,
    ModelConfig,
    OutputConfig,
    TrainerConfig,
)
from rollouts.training.train import TrainResult

hardware = HardwareConfig(
    gpu_type="A10G",
    gpu_count=1,
    provider="runpod",
    deps=default_remote_training_deps(),
    hf_cache_dir="/workspace/.cache/huggingface",
    use_torchrun=False,
)

config = PretrainConfig(
    data=PretrainDataConfig(
        source_path="/tmp/tokenized_shards",
        seq_len=1024,
        batch_size=8,
    ),
    model=ModelConfig(
        name="Qwen/Qwen3-0.6B",
        dtype="bfloat16",
    ),
    trainer=TrainerConfig(
        backend="torchtitan",
        torchtitan_model="qwen3",
        torchtitan_model_size="0.6B",
        cuda_device_ids=(0,),
        lr=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        num_minibatches=1,
        loss_type="vanilla",
    ),
    checkpoint=CheckpointConfig(
        num_steps=100,
        log_every=1,
        checkpoint_every=20,
    ),
    output=OutputConfig(
        output_dir="results/pretrain",
        experiment_name="qwen3_0_6b_torchtitan_dense_pretrain",
    ),
)


def train(config: PretrainConfig | None = None, **_: object) -> TrainResult:
    return run_pretrain(config or globals()["config"])
