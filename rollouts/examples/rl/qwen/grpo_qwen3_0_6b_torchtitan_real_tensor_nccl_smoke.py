from dataclasses import replace
from typing import Any

import trio

from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import config as _base_config
from examples.rl.qwen.grpo_qwen3_0_6b_torchtitan_modal_witness import hardware as _hardware
from rollouts.training.torchtitan_vllm_real_tensor_smoke import (
    run_torchtitan_vllm_real_tensor_smoke,
)

hardware = _hardware

config = replace(
    _base_config,
    output=replace(
        _base_config.output,
        experiment_name="qwen3_0_6b_torchtitan_real_tensor_nccl_smoke",
    ),
    checkpoint=replace(
        _base_config.checkpoint,
        inference_sync_realization="vllm_custom_nccl_broadcast",
    ),
)


async def _train_async(config: Any, kwargs: dict[str, Any]) -> None:
    await run_torchtitan_vllm_real_tensor_smoke(config, **kwargs)


def train(config: Any, **kwargs: Any) -> dict[str, Any]:
    trio.run(_train_async, config, kwargs)
    return {"metrics_history": []}
