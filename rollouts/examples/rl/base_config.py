"""Shared utilities for RL example remote execution."""

from __future__ import annotations

from rollouts.image_spec import ImageSpec
from rollouts.training.configs import DepsConfig

SLIME_STABLE_TAG = "v0.2.3"
SLIME_STABLE_COMMIT = "6195417d45f272b72d04f619ea334613135d1c1f"
SLIME_STABLE_IMAGE = f"slimerl/slime:{SLIME_STABLE_TAG}"


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
        "mbridge @ git+https://github.com/ISEEKYAN/mbridge.git",
    )


def default_remote_training_deps() -> DepsConfig:
    """Explicit remote training environment for SSH providers.

    This is the canonical non-Modal deps contract for simple RL jobs.
    It avoids the old hidden remote bootstrap behavior by declaring the base
    image and runtime Python dependencies directly in config.
    """
    # TODO(boundary): helpers like this are really named runtime profiles.
    # Promote them into data/product types rather than continuing to hide them
    # behind Python functions.
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
    """Pinned Megatron/SGLang runtime owned by a tagged `slime` image.

    `slimerl/slime:v0.2.3` lines up with the upstream `slime` source tag
    `v0.2.3` at commit `6195417d45f272b72d04f619ea334613135d1c1f`, which is a
    much cleaner source↔image contract than the floating `miles` nightlies.
    """
    return DepsConfig(
        python_version="3.12",
        system_packages=(),
        image=ImageSpec.from_registry(
            SLIME_STABLE_IMAGE,
            python_version="3.12",
            python_runtime="image_owned",
            python_executable="python3",
            installed_groups=("slime-megatron-runtime",),
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
