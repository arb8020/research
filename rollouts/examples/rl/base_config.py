"""Shared utilities for RL example remote execution."""

from __future__ import annotations

import base64
from pathlib import Path

from rollouts.image_spec import ImageSpec
from rollouts.training.configs import DepsConfig

MILES_STABLE_SGLANG_COMMIT = "24c91001cf99ba642be791e099d358f4dfe955f5"
MILES_STABLE_MEGATRON_COMMIT = "3714d81d418c9f1bca4594fc35f9e8289f652862"
MILES_STABLE_MBRIDGE_COMMIT = "89eb10887887bc74853f89a4de258c0702932a1c"
MILES_STABLE_TORCH_MEMORY_SAVER_COMMIT = "dc6876905830430b5054325fa4211ff302169c6b"
MILES_STABLE_APEX_COMMIT = "10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4"
MILES_STABLE_CUDA_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu22.04"
MILES_STABLE_PATCH_DIR = (
    Path(__file__).resolve().parents[2] / "third_party" / "miles_patches" / "v0.5.7"
)


def _miles_stable_megatron_runtime_packages() -> tuple[str, ...]:
    """Pinned Megatron/SGLang stack matching `/tmp/miles` stable."""
    return (
        "cuda-python==13.1.0",
        "torch==2.9.1",
        "torchvision==0.24.1",
        "torchaudio==2.9.1",
        "cmake",
        "ninja",
        "flash-attn==2.7.4.post1",
        f"mbridge @ git+https://github.com/ISEEKYAN/mbridge.git@{MILES_STABLE_MBRIDGE_COMMIT}",
        "transformer_engine[pytorch]==2.10.0",
        "flash-linear-attention==0.4.0",
        f"torch_memory_saver @ git+https://github.com/fzyzcjy/torch_memory_saver.git@{MILES_STABLE_TORCH_MEMORY_SAVER_COMMIT}",
        "git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl",
        "nvidia-modelopt[torch]>=0.37.0",
        "numpy<2",
    )


def _rollouts_runtime_packages() -> tuple[str, ...]:
    return (
        "datasets>=4.4.1",
        "accelerate>=0.20.0",
        "safetensors",
        "curl_cffi",
        "peft>=0.7.0",
        "huggingface_hub>=1.4.0",
        "hf-transfer",
        "einops",
        "openai",
        "anthropic",
        "dacite",
        "aiohttp",
        "trio",
        "httpx",
        "markdownify",
    )


def _vendored_patch_bytes(name: str) -> bytes:
    path = MILES_STABLE_PATCH_DIR / name
    return path.read_bytes()


def _write_patch_command(*, patch_name: str, destination: str) -> str:
    payload = base64.b64encode(_vendored_patch_bytes(patch_name)).decode("ascii")
    return (
        "python3 -c "
        f'"import base64; from pathlib import Path; path = Path({destination!r}); '
        "path.parent.mkdir(parents=True, exist_ok=True); "
        f'path.write_bytes(base64.b64decode({payload!r}))"'
    )


def _apply_patch_command(*, repo_dir: str, patch_name: str, patch_dest: str) -> str:
    return (
        f"{_write_patch_command(patch_name=patch_name, destination=patch_dest)} && "
        f"if git -C {repo_dir} apply --reverse --check {patch_dest} >/dev/null 2>&1; then "
        f"echo '{patch_name} already applied to {repo_dir}'; "
        f"else git -C {repo_dir} apply {patch_dest}; fi"
    )


def _runtime_python_bin() -> str:
    return (
        "if [ -x /opt/venvs/rollouts/bin/python ]; then "
        "echo /opt/venvs/rollouts/bin/python; "
        "else echo python3; fi"
    )


def _editable_install_command(repo_dir: str, target: str = ".") -> str:
    runtime_python = f"$({_runtime_python_bin()})"
    return f"cd {repo_dir} && {runtime_python} -m pip install -e '{target}'"


def _apex_install_command() -> str:
    runtime_python = f"$({_runtime_python_bin()})"
    return (
        "cd /tmp && "
        "NVCC_APPEND_FLAGS='--threads 4' "
        f"{runtime_python} -m pip install --no-cache-dir --no-build-isolation "
        "--config-settings='--build-option=--cpp_ext --cuda_ext --parallel 8' "
        f"'apex @ git+https://github.com/NVIDIA/apex.git@{MILES_STABLE_APEX_COMMIT}'"
    )


def _cudnn_install_command() -> str:
    runtime_python = f"$({_runtime_python_bin()})"
    return f"{runtime_python} -m pip install --upgrade --no-deps 'nvidia-cudnn-cu12==9.16.0.29'"


def _miles_stable_source_commands() -> tuple[str, ...]:
    return (
        "if [ ! -d /root/sglang ]; then "
        "git clone https://github.com/sgl-project/sglang.git /root/sglang; "
        "fi",
        f"cd /root/sglang && git fetch --all --tags && "
        f"git reset --hard {MILES_STABLE_SGLANG_COMMIT} && git clean -fd",
        "if [ ! -d /root/Megatron-LM ]; then "
        "git clone https://github.com/NVIDIA/Megatron-LM.git /root/Megatron-LM --recursive; "
        "fi",
        f"cd /root/Megatron-LM && git fetch --all --tags && "
        f"git reset --hard {MILES_STABLE_MEGATRON_COMMIT} && git clean -fd",
        _apply_patch_command(
            repo_dir="/root/sglang",
            patch_name="sglang.patch",
            patch_dest="/tmp/miles-stable-sglang.patch",
        ),
        _apply_patch_command(
            repo_dir="/root/Megatron-LM",
            patch_name="megatron.patch",
            patch_dest="/tmp/miles-stable-megatron.patch",
        ),
        _cudnn_install_command(),
        _apex_install_command(),
        _editable_install_command("/root/sglang", "python[all]"),
        _editable_install_command("/root/Megatron-LM"),
    )


def default_remote_training_deps() -> DepsConfig:
    """Explicit remote training environment for SSH providers.

    This is the canonical non-Modal deps contract for simple RL jobs.
    It avoids the old hidden remote bootstrap behavior by declaring the base
    image and runtime Python dependencies directly in config.
    """
    return DepsConfig(
        image=ImageSpec.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04"),
        pip_packages=(
            "torch>=2.4",
            "transformers>=5.0",
            "datasets",
            "accelerate",
            "safetensors",
            "sglang[all] @ git+https://github.com/sgl-project/sglang.git@main#subdirectory=python",
            "curl_cffi",
            "peft",
            "huggingface_hub>=1.4.0",
        ),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    )


def default_remote_megatron_training_deps() -> DepsConfig:
    """Pinned Megatron/SGLang runtime aligned to `/tmp/miles` stable.

    This matches the package, source commit, and local patch baseline from:
    - `/tmp/miles/build_conda.sh`
    - `/tmp/miles/docker/README.md`
    """
    return DepsConfig(
        python_version="3.12",
        image=ImageSpec.from_registry(MILES_STABLE_CUDA_IMAGE, python_version="3.12"),
        pip_packages=_miles_stable_megatron_runtime_packages() + _rollouts_runtime_packages(),
        pip_index_url="https://download.pytorch.org/whl/cu129",
        pip_extra_index_url="https://pypi.org/simple",
        bootstrap_commands=_miles_stable_source_commands(),
    )


async def run_remote(
    script_path: str,
    keep_alive: bool = False,
    node_id: str | None = None,
    use_tui: bool = False,
    tui_debug: bool = False,
    fire_and_forget: bool = False,
    gpu_count: int = 1,
    gpu_type: str = "A100",
    container_disk_gb: int = 100,
    hf_cache_dir: str = "/workspace/.cache/huggingface",
    persistent_volume_id: str | None = None,
    persistent_volume_mount_path: str = "/workspace",
    persistent_volume_location: str | None = None,
) -> None:
    """Delegate remote execution to `rollouts.run`."""
    from rollouts.run import run_remote as run_remote_impl

    await run_remote_impl(
        script_path=script_path,
        keep_alive=keep_alive,
        node_id=node_id,
        tui=use_tui,
        tail=not use_tui and not tui_debug,
        block=not fire_and_forget,
        gpu_count=gpu_count,
        gpu_type=gpu_type,
        allow_dirty=True,
        skip_hf_token_check=True,
        container_disk_gb=container_disk_gb,
        hf_cache_dir=hf_cache_dir,
        persistent_volume_id=persistent_volume_id,
        persistent_volume_mount_path=persistent_volume_mount_path,
        persistent_volume_location=persistent_volume_location,
        raw_script=True,
    )
