#!/usr/bin/env python3
"""Run a narrow RunPod+Bifrost smoke for the slime-sglang realization.

This is a temporary inference validation path while `modal_runner` is being rewritten.
It provisions a RunPod node, pushes the workspace, bootstraps SGLang, launches the exact
`rollouts.inference.realizations.slime_sglang` entrypoint, then probes:

- `/health`
- one OpenAI-compatible generation request
"""

from __future__ import annotations

import argparse
import json

from runpod_realization_smoke_lib import RunPodSmokeSpec, run_realization_smoke_on_runpod

SLIME_SGLANG_SMOKE = RunPodSmokeSpec(
    name="slime-sglang-smoke",
    launch_module="rollouts.inference.realizations.slime_sglang",
    model_flag="--model-path",
    default_model="Qwen/Qwen2.5-0.5B-Instruct",
    default_gpu_type="A100",
    port=30000,
    bootstrap_steps=(
        (
            "install system deps",
            "apt-get update && apt-get install -y tmux curl git libnuma1 || true",
        ),
        (
            "install uv",
            "curl -LsSf https://astral.sh/uv/install.sh | sh",
        ),
        (
            "sync workspace deps",
            "~/.local/bin/uv python install 3.12 && "
            "~/.local/bin/uv sync --python 3.12 --package rollouts",
        ),
        (
            "install sglang runtime",
            "~/.local/bin/uv pip install --upgrade torch datasets accelerate curl_cffi peft "
            "'sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python' "
            "&& ~/.local/bin/uv pip install --upgrade 'transformers>=5.0.0' "
            "'huggingface_hub>=1.4.0'",
        ),
    ),
    env={
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "AMEM_ENABLE": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_P2P_DISABLE": "1",
        "NCCL_SHM_DISABLE": "1",
        "NCCL_DEBUG": "INFO",
        "NCCL_DEBUG_SUBSYS": "INIT,COLL",
        "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1",
        "ROLLOUTS_SGLANG_FORCE_SYNC_BROADCAST": "1",
    },
)


def main() -> None:
    import trio

    parser = argparse.ArgumentParser(description="Run slime-sglang smoke on RunPod")
    parser.add_argument("--model", default=SLIME_SGLANG_SMOKE.default_model)
    parser.add_argument("--gpu-type", default=SLIME_SGLANG_SMOKE.default_gpu_type)
    parser.add_argument("--keep-alive", action="store_true")
    args = parser.parse_args()

    async def _run() -> dict[str, object]:
        return await run_realization_smoke_on_runpod(
            SLIME_SGLANG_SMOKE,
            model=args.model,
            gpu_type=args.gpu_type,
            keep_alive=args.keep_alive,
        )

    result = trio.run(_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
