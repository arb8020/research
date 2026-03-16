"""Shared utilities for RL example remote execution."""

from __future__ import annotations

from pathlib import Path

from rollouts.image_spec import ImageSpec
from rollouts.training.configs import DepsConfig

MILES_SGLANG_IMAGE_TAG = "nightly-dev-20260103-24c91001"
MILES_PATCH_VERSION = "v0.5.7"
MILES_MEGATRON_COMMIT = "3714d81d418c9f1bca4594fc35f9e8289f652862"
# Closest recoverable main-branch commit before the published nightly image creation time.
MILES_PINNED_COMMIT = "bfa264385cf9864014b07a985909a20ccc594c13"
MILES_DOCKERFILE = Path(__file__).with_name("miles_v057_pinned.Dockerfile")


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
    """Pinned Megatron/SGLang runtime derived from a stable `miles` patch set.

    Current shared-env launchers realize one runtime contract in `hardware.deps`.
    For Megatron that contract should be image-owned, not reconstructed per run.
    We build that image from a pinned `miles` commit plus the stable `v0.5.7`
    patch set so the receiver/runtime provenance is explicit instead of hidden
    behind an opaque floating registry artifact.
    """
    return DepsConfig(
        python_version="3.12",
        system_packages=(),
        image=ImageSpec.from_dockerfile_path(
            str(MILES_DOCKERFILE),
            context_dir=str(MILES_DOCKERFILE.parent),
            python_version="3.12",
            build_args={
                "SGLANG_IMAGE_TAG": MILES_SGLANG_IMAGE_TAG,
                "PATCH_VERSION": MILES_PATCH_VERSION,
                "MEGATRON_COMMIT": MILES_MEGATRON_COMMIT,
                "MILES_COMMIT": MILES_PINNED_COMMIT,
                "ENABLE_SGLANG_PATCH": "1",
            },
            python_runtime="image_owned",
            python_executable="python3",
            installed_groups=("miles-megatron-runtime",),
        ),
        pip_packages=_rollouts_runtime_packages(),
        pip_extra_index_url="https://pypi.org/simple",
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
