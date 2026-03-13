"""Cheap remote smoke for the TorchTitan witness inference runtime.

This reuses the witness image/runtime but only starts the configured inference
engine(s), waits for health, and exits. It should fail before RL orchestration
or training/inference coordination if the inference stack is broken.
"""

from __future__ import annotations

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import config, hardware
from rollouts.training.smoke import run_inference_startup_smoke


def train(config=config, **kwargs):
    return run_inference_startup_smoke(config=config, **kwargs)

