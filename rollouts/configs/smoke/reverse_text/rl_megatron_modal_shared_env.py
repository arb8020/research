"""Shared-env Modal smoke for the 3-GPU Megatron reverse-text witness."""

from __future__ import annotations

from dataclasses import replace

from configs.prime_ci.reverse_text.rl_megatron import config as _base_config

from examples.rl.base_config import default_remote_megatron_training_deps
from rollouts.training.configs import HardwareConfig

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=3,
    provider="modal",
    deps=default_remote_megatron_training_deps(),
    use_torchrun=False,
)

config = replace(
    _base_config,
    output=replace(
        _base_config.output,
        experiment_name="smoke_reverse_text_megatron_modal_shared_env",
    ),
)
