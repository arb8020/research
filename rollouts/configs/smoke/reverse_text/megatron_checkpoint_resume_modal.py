"""Modal smoke that proves Megatron optimizer checkpoint resume semantics."""

from __future__ import annotations

from dataclasses import replace

from configs.prime_ci.reverse_text.rl_megatron import config as _base_config

from examples.rl.base_config import default_remote_megatron_training_deps
from rollouts.training.configs import HardwareConfig
from rollouts.training.smoke import run_megatron_checkpoint_resume_smoke

hardware = HardwareConfig(
    gpu_type="A100-80GB",
    gpu_count=2,
    provider="modal",
    deps=default_remote_megatron_training_deps(),
    use_torchrun=False,
)

config = replace(
    _base_config,
    checkpoint=replace(
        _base_config.checkpoint,
        save_optimizer_state=True,
    ),
    trainer=replace(
        _base_config.trainer,
        cuda_device_ids=(0, 1),
        loss_type="vanilla",
        micro_batch_size=1,
        num_minibatches=1,
        seq_length=128,
        tensor_parallel_size=2,
        pipeline_parallel_size=1,
        expert_parallel_size=1,
        context_parallel_size=1,
    ),
    output=replace(
        _base_config.output,
        experiment_name="smoke_megatron_checkpoint_resume_modal",
    ),
)


def train(config: object = config, **kwargs: object) -> dict:
    return run_megatron_checkpoint_resume_smoke(config=config or globals()["config"], **kwargs)
