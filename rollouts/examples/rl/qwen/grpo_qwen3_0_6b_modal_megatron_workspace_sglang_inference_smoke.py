"""Cheap remote smoke for the Megatron + Slime-SGLang witness via workspace source.

This keeps the known-good Modal witness shape but overrides the SGLang import
provenance to come from a checkout synced in the workspace instead of the
ambient installed package. It only starts inference, waits for health, and
exits.
"""

from __future__ import annotations

from dataclasses import replace

from examples.rl.qwen.grpo_qwen3_0_6b_modal_megatron_witness import config as _base_config
from rollouts.training.configs import WorkspaceInferenceSource
from rollouts.training.grpo import GRPOConfig
from rollouts.training.smoke import run_inference_startup_smoke

config = replace(
    _base_config,
    output=replace(
        _base_config.output,
        experiment_name="qwen3_0_6b_megatron_workspace_sglang_inference_smoke",
    ),
    inference=replace(
        _base_config.inference,
        source=WorkspaceInferenceSource(
            path="third_party/sglang",
            subdirectory="python",
        ),
    ),
)


def train(config: GRPOConfig = config, **kwargs: object) -> dict:
    return run_inference_startup_smoke(config=config, **kwargs)
