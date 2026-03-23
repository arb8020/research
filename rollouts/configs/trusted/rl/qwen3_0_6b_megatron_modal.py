"""Trusted dense Modal RL config: Megatron trainer with Slime-SGLang inference.

Usage:
    python -m argus run --config configs/trusted/rl/qwen3_0_6b_megatron_modal.py
"""

from __future__ import annotations

from examples.rl.qwen.grpo_qwen3_0_6b_modal_megatron_witness import config as _config
from examples.rl.qwen.grpo_qwen3_0_6b_modal_megatron_witness import hardware as _hardware
from examples.rl.qwen.grpo_qwen3_0_6b_modal_megatron_witness import train as _train
from rollouts.config_status import known_good
from rollouts.training.configs import (
    InferenceRoleBinding,
    InferenceWorkerConfig,
    TrainingWorkerConfig,
    WorkerTopologyConfig,
)

config_status = known_good(
    "de0529fa",
    "Matches the Modal dense Qwen3-0.6B Megatron + Slime-SGLang green witness surface.",
)

hardware = _hardware
config = _config
worker_topology = WorkerTopologyConfig(
    hardware=hardware,
    inference_workers=(
        InferenceWorkerConfig(
            worker_id="actor",
            model=config.model.name,
            inference=config.inference,
        ),
    ),
    training_workers=(
        TrainingWorkerConfig(
            worker_id="trainer",
            trainer=config.trainer,
        ),
    ),
    role_bindings=(InferenceRoleBinding(role="actor", worker_id="actor"),),
)


def train(config: object = config, **kwargs: object) -> dict:
    return _train(config=config, **kwargs)
