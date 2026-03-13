"""Cheap remote smoke for the TorchTitan witness runtime.

Reuses the real witness config and runtime contract, but stops after the
training backend init stage. This catches backend API mismatches before
vLLM startup and before the full RL loop.
"""

from __future__ import annotations

from copy import deepcopy

from rollouts.examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import (
    config as _witness_config,
)
from rollouts.training.smoke import run_torchtitan_backend_init_smoke

config = deepcopy(_witness_config)


def train(config=config, **kwargs):
    return run_torchtitan_backend_init_smoke(config=config or globals()["config"], **kwargs)
