"""Planned smoke witness for TensorRT-LLM.

This file is a deliberate stub, not a runnable eval config.

Why it is not wired yet:
- the repo runtime spec already says ``trtllm`` needs a separate inference env
  because its torch/CUDA constraints do not fit the current shared env
- eval's remote launcher does not yet implement that separate-env boundary
- there is no truthful logprob / weight-sync story here yet

Keep this file as the explicit checkpoint for the intended witness shape, not
as a fake runnable config.
"""

from __future__ import annotations

from rollouts.training.configs import DepsConfig, HardwareConfig

MODEL = "Qwen/Qwen3-0.6B"
PORT = 30003

planned_hardware = HardwareConfig(
    provider="modal",
    gpu_type="A100",
    gpu_count=1,
    use_torchrun=False,
    keep_alive=True,
    deps=DepsConfig(
        bootstrap_commands=(
            "# TODO(trtllm-separate-env): install tensorrt_llm in a dedicated env once "
            "# eval remote launchers can target separate runtime environments.",
        ),
    ),
)

NOT_YET_RUNNABLE_REASON = (
    "trtllm smoke witness is stubbed only. The current eval remote launch path "
    "does not implement the separate inference environment that the repo's "
    "trtllm runtime spec requires."
)
