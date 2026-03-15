from dataclasses import replace
from typing import Any

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import config as _base_config
from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import hardware as _hardware
from rollouts.training.vllm_nccl_smoke import run_vllm_nccl_smoke

hardware = _hardware

config = replace(
    _base_config,
    checkpoint=replace(
        _base_config.checkpoint,
        inference_sync_realization="vllm_custom_nccl_broadcast",
    ),
)


async def train(config: Any, **kwargs: Any) -> None:
    await run_vllm_nccl_smoke(config, **kwargs)
