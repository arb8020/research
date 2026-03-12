"""Shared utilities for RL example remote execution."""

from __future__ import annotations

from rollouts.image_spec import ImageSpec, RuntimeOverlay
from rollouts.training.configs import DepsConfig


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
    """Explicit remote deps for Megatron-based SSH workloads."""
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
        runtime_overlay=RuntimeOverlay(
            commands=(
                "if [ ! -d ~/Megatron-LM ]; then "
                "git clone https://github.com/NVIDIA/Megatron-LM.git ~/Megatron-LM --recursive && "
                "cd ~/Megatron-LM && git checkout 3714d81d418c9f1bca4594fc35f9e8289f652862 && "
                "~/.local/bin/uv pip install -e . --no-build-isolation; "
                "fi",
                "~/.local/bin/uv pip install 'transformer_engine[pytorch]>=2.10.0' --no-build-isolation",
                "~/.local/bin/uv pip install --upgrade 'git+https://github.com/ISEEKYAN/mbridge.git' --no-deps",
            ),
        ),
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
