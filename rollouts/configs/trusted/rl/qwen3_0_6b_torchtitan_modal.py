"""Trusted dense Modal RL config: TorchTitan trainer with QED-vLLM inference.

Usage:
    python -m argus run --config configs/trusted/rl/qwen3_0_6b_torchtitan_modal.py
"""

from __future__ import annotations

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import config as _config
from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import hardware as _hardware
from examples.rl.reverse_text.base_config import reverse_text_score_fn
from examples.training_architecture.shared import make_synthetic_reverse_text_prompts
from rollouts.config_status import known_good
from rollouts.environments.no_tools import BasicEnvironment
from rollouts.training.configs import (
    InferenceRoleBinding,
    InferenceWorkerConfig,
    TrainingWorkerConfig,
    WorkerTopologyConfig,
)
from rollouts.training.grpo import GRPOConfig, grpo_train
from rollouts.training.scoring import FunctionScorer

config_status = known_good(
    "de0529fa",
    "Matches the Modal dense Qwen3-0.6B TorchTitan + QED-vLLM green witness surface.",
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


def train(config: GRPOConfig = config, max_samples: int = 64, **kwargs: object) -> dict:
    prompts = make_synthetic_reverse_text_prompts(max_samples=max_samples)
    return grpo_train(
        config=config,
        prompts=prompts,
        scorer=FunctionScorer(reverse_text_score_fn),
        environment_cls=BasicEnvironment,
        **kwargs,
    )
