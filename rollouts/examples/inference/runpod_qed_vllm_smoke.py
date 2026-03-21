#!/usr/bin/env python3
"""Run a narrow RunPod+Bifrost smoke for the patched qed-vllm realization.

This is a temporary inference validation path while `modal_runner` is being rewritten.
It provisions a RunPod node, pushes the workspace, bootstraps vLLM, launches the exact
`rollouts.inference.realizations.qed_vllm` entrypoint, then probes:

- `/health`
- `/weight_update_schema`
- one OpenAI-compatible generation request
"""

from __future__ import annotations

import argparse
import json
import shlex

from runpod_realization_smoke_lib import RunPodSmokeSpec, run_realization_smoke_on_runpod

STABLE_COMMIT = "9962c39baa6b50bb7389cb63112c1d621fd3480b"
QED_VLLM_VENV = "/root/.venvs/qed-vllm-smoke"
QED_VLLM_SHARED_ENV_PACKAGES = (
    "torch==2.9.0",
    "torchtitan==0.2.0",
    "torchmonarch==0.2.0",
    "torchstore @ git+https://github.com/meta-pytorch/torchstore.git@no-monarch-2026.01.05",
    "datasets>=2.21.0",
    "tokenizers",
    "accelerate>=0.20.0",
    "peft>=0.7.0",
    "hf-transfer",
    "openai",
    "anthropic",
    "dacite",
    "aiohttp",
    "trio",
    "httpx",
    "markdownify",
    "vllm>=0.13.0,<0.14.0",
)

QED_VLLM_INSTALL_COMMAND = (
    f"~/.local/bin/uv venv {shlex.quote(QED_VLLM_VENV)} --python 3.12"
    f" && ~/.local/bin/uv pip install --python {shlex.quote(QED_VLLM_VENV + '/bin/python')}"
    " --no-config --upgrade "
    + " ".join(shlex.quote(package) for package in QED_VLLM_SHARED_ENV_PACKAGES)
)

QED_VLLM_SMOKE = RunPodSmokeSpec(
    name="qed-vllm-smoke",
    launch_module="rollouts.inference.realizations.qed_vllm",
    model_flag="--model",
    default_model="Qwen/Qwen2.5-0.5B-Instruct",
    default_gpu_type="A100",
    port=30001,
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
            "install stable-commit qed-vllm shared env",
            QED_VLLM_INSTALL_COMMAND,
        ),
    ),
    env={
        "VLLM_SERVER_DEV_MODE": "1",
        "HF_HUB_DOWNLOAD_TIMEOUT": "300",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_ASYNC_ERROR_HANDLING": "1",
        "NCCL_P2P_DISABLE": "1",
        "TORCH_DISABLE_SHARE_RDZV_TCP_STORE": "1",
        "PYTHONPATH": ".",
    },
    probe_weight_update_schema=True,
    launch_with_uv=False,
    remote_python_bin=f"{QED_VLLM_VENV}/bin/python",
)


def main() -> None:
    import trio

    parser = argparse.ArgumentParser(
        description=f"Run qed-vllm smoke on RunPod using stable commit {STABLE_COMMIT}"
    )
    parser.add_argument("--model", default=QED_VLLM_SMOKE.default_model)
    parser.add_argument("--gpu-type", default=QED_VLLM_SMOKE.default_gpu_type)
    parser.add_argument("--keep-alive", action="store_true")
    args = parser.parse_args()

    async def _run() -> dict[str, object]:
        return await run_realization_smoke_on_runpod(
            QED_VLLM_SMOKE,
            model=args.model,
            gpu_type=args.gpu_type,
            keep_alive=args.keep_alive,
        )

    result = trio.run(_run)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
