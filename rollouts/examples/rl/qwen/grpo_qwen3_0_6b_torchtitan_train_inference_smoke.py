"""Cheap remote smoke for the TorchTitan witness training+inference boundary.

This reuses the witness image/runtime, runs the training preflight, then starts
the configured inference engine(s), waits for health, and exits before RL
orchestration.
"""

from __future__ import annotations

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import config, hardware
from rollouts.training.smoke import run_training_and_inference_startup_smoke


def train(config=config, **kwargs):
    return run_training_and_inference_startup_smoke(config=config, **kwargs)

