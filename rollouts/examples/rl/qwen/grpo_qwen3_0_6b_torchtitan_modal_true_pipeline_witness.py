"""Modal true-pipeline witness for TorchTitan + vLLM reverse-text RL.

This starts from the passing blocking witness and changes only the pipeline
controls needed to exercise PipelineRL-style versioned overlap semantics.

Current caveat:
    The direct vLLM NCCL receive/load path is still a blocking serving boundary,
    so the rollout pipeline should be read as "trainer/sampler overlap with
    versioned staleness control", not a proven atomic hot-swap path.
"""

from dataclasses import replace

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import (
    config as base_config,
)
from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import (
    hardware as base_hardware,
)
from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import (
    train as _base_train,
)
from rollouts.training.grpo import GRPOConfig

hardware = base_hardware

config = replace(
    base_config,
    output=replace(
        base_config.output,
        experiment_name="qwen3_0_6b_torchtitan_vllm_modal_true_pipeline_witness",
    ),
    checkpoint=replace(
        base_config.checkpoint,
        pipeline_mode="true_pipeline",
        max_lag=2,
        pipeline_queue_size=0,
    ),
)


def train(config: GRPOConfig | None = None, **kwargs: object) -> dict:
    return _base_train(config=config or globals()["config"], **kwargs)
