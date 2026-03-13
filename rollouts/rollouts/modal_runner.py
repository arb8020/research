"""Modal-based training runner.

Run training workloads on Modal sandboxes with GPU access.
Uses Modal's native async APIs via trio_asyncio bridge.

Usage:
    # Public control-plane entrypoint
    python -m argus run --config examples/rl/reverse_text/grpo_modal_01.py

    # Direct workload implementation entrypoint
    python -m rollouts.run --config examples/rl/reverse_text/grpo_modal_01.py

Design:
    Unlike run.py which uses bifrost (SSH-based), this uses Modal sandboxes directly.
    Modal sandboxes have ~10-30s cold start vs RunPod's 2-5 min.

    For GRPO training, everything runs in a single sandbox:
    - SGLang inference server (launched by grpo_train)
    - Training backend (PyTorch)
    - Agent loop (rollout generation)

    Weight sync is local disk since trainer + inference are colocated.
    When we scale to separate sandboxes, we'll use ModalVolumeWeightSync.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from .training.configs import DepsConfig

from ._logging import setup_logging
from .image_spec import (
    ImageManifest,
    image_manifest_for_spec,
    infer_cuda_version,
    manifest_write_command,
    resolve_image_for_provisioning,
)
from .install_probes import (
    apt_install_probe_command,
    command_looks_like_install,
    python_install_probe_command,
    python_runtime_contract_snapshot_command,
    python_runtime_contract_verify_command,
)
from .remote_runtime import (
    MaterializationPlan,
    RuntimeContract,
    SourceSyncPolicy,
    enforce_source_sync_policy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent

# Default Modal app name
MODAL_APP_NAME = "rollouts-training"

# Modal Dict for storing model weight snapshots
# Key: model name (e.g., "zai-org/GLM-4.7-Flash")
# Value: snapshot image ID
MODEL_CACHE_DICT_NAME = "rollouts-model-cache"

# HuggingFace cache directory in sandbox
HF_CACHE_DIR = "/root/.cache/huggingface"
UV_BIN = "/root/.local/bin/uv"
IMAGE_VENV_DIR = "/opt/venvs/rollouts"
IMAGE_VENV_PYTHON = f"{IMAGE_VENV_DIR}/bin/python"
WORKLOAD_ENTRYPOINT_SENTINEL = "__ARGUS_WORKLOAD_ENTRYPOINT_STARTED__"
MODAL_IMAGE_BUILD_HEARTBEAT_S = 15.0
MODAL_IMAGE_BUILD_LOG_LINE_LIMIT = 200
MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT = 1000


@dataclass
class ModalRunConfig:
    """Configuration for a Modal training run.

    This carries the shared runtime/materialization/source-sync contracts
    instead of re-describing them in Modal-specific terms.
    """

    config_path: str
    runtime: RuntimeContract | None = None
    materialization: MaterializationPlan = field(default_factory=MaterializationPlan)
    source_sync_policy: SourceSyncPolicy = field(default_factory=SourceSyncPolicy.committed_only)
    timeout_hours: int = 4
    sandbox_id: str | None = None
    keep_alive: bool = False
    run_name: str | None = None
    event_log: Callable[..., None] | None = None
    model_name: str | None = None  # Model name for weight caching (e.g., "zai-org/GLM-4.7-Flash")
    pruning_recipe: str | None = (
        None  # Path to pruning recipe JSON (if set, model is pruned before caching)
    )

    def __post_init__(self) -> None:
        if self.runtime is None:
            raise ValueError(
                "ModalRunConfig requires a RuntimeContract. Build it from HardwareConfig."
            )

    @property
    def gpu_type(self) -> str:
        assert self.runtime is not None
        return self.runtime.gpu_type

    @property
    def gpu_count(self) -> int:
        assert self.runtime is not None
        return self.runtime.gpu_count

    @property
    def deps(self) -> DepsConfig | None:
        assert self.runtime is not None
        return self.runtime.deps

    @property
    def use_torchrun(self) -> bool:
        assert self.runtime is not None
        return self.runtime.use_torchrun


def _build_modal_image(modal: Any, deps: DepsConfig, gpu_type: str) -> Any:
    """Build a Modal image from the shared image contract."""
    spec = deps.resolved_image(gpu_type)
    overlay = deps.resolved_runtime_overlay()
    cuda_version = infer_cuda_version(gpu_type, spec.pip_index_url)
    if spec.python_runtime == "image_owned":
        image_python = spec.python_executable
        image_path_prefix = (
            "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
    else:
        image_python = IMAGE_VENV_PYTHON
        image_path_prefix = f"{IMAGE_VENV_DIR}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

    if spec.source_type == "registry":
        if spec.python_runtime == "image_owned":
            image = modal.Image.from_registry(spec.source_ref)
        else:
            image = modal.Image.from_registry(spec.source_ref, add_python=spec.python_version)
    elif spec.source_type == "dockerfile_path":
        dockerfile_path = Path(spec.source_ref)
        image = modal.Image.from_dockerfile(
            dockerfile_path,
            context_dir=spec.context_dir or str(dockerfile_path.parent),
            add_python=spec.python_version,
            build_args=spec.build_args,
        )
    else:
        raise ValueError(
            f"Modal runner does not know how to build image source_type={spec.source_type!r}"
        )

    if spec.system_packages:
        image = image.apt_install(*spec.system_packages)
        image = image.run_commands(
            apt_install_probe_command("image-system-packages", packages=spec.system_packages)
        )

    image = image.run_commands(
        "which curl >/dev/null 2>&1 || (apt-get update && apt-get install -y curl)",
        "if [ ! -x /root/.local/bin/uv ]; then curl -LsSf https://astral.sh/uv/install.sh | sh; fi",
    )

    if spec.python_runtime == "managed_venv":
        image = image.run_commands(
            f"{UV_BIN} python install {spec.python_version}",
            f"{UV_BIN} venv {IMAGE_VENV_DIR} --python {spec.python_version}",
        )
    else:
        image = image.run_commands(f"{spec.python_executable} -c 'import sys; print(sys.version)'")

    def _uv_install_command(
        packages: tuple[str, ...],
        *,
        index_url: str | None,
        extra_index_url: str | None,
        pre: bool,
    ) -> str:
        parts = [UV_BIN, "pip", "install", "--compile-bytecode"]
        if spec.python_runtime == "image_owned":
            parts.append("--system")
        else:
            parts.extend(["--python", image_python])
        if index_url:
            parts.extend(["--index-url", index_url])
        if extra_index_url:
            parts.extend(["--extra-index-url", extra_index_url])
        if pre:
            parts.extend(["--prerelease", "allow"])
        parts.extend(packages)
        return shlex.join(parts)

    if spec.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                spec.pip_packages,
                index_url=spec.pip_index_url,
                extra_index_url=spec.pip_extra_index_url,
                pre=spec.pip_prerelease,
            ),
            f"{python_install_probe_command('image-pip-packages', python_bin=image_python, uv_bin=UV_BIN)} && "
            f"{python_runtime_contract_snapshot_command('image-pip-packages', python_bin=image_python)}",
        )

    for cmd in spec.build_commands:
        image = image.run_commands(cmd)
        if command_looks_like_install(cmd):
            image = image.run_commands(
                f"{python_install_probe_command('image-build-command-post-install', python_bin=image_python, uv_bin=UV_BIN)} && "
                f"{python_runtime_contract_verify_command('image-build-command-post-install', python_bin=image_python)}"
            )

    if overlay.system_packages:
        image = image.apt_install(*overlay.system_packages)
        image = image.run_commands(
            apt_install_probe_command("overlay-system-packages", packages=overlay.system_packages)
        )

    if overlay.pip_packages:
        image = image.run_commands(
            _uv_install_command(
                overlay.pip_packages,
                index_url=overlay.pip_index_url or spec.pip_index_url,
                extra_index_url=overlay.pip_extra_index_url or spec.pip_extra_index_url,
                pre=overlay.pip_prerelease or spec.pip_prerelease,
            ),
            f"{python_install_probe_command('overlay-pip-packages', python_bin=image_python, uv_bin=UV_BIN)} && "
            f"{python_runtime_contract_snapshot_command('overlay-pip-packages', python_bin=image_python)}",
        )

    for cmd in overlay.commands:
        image = image.run_commands(cmd)
        if command_looks_like_install(cmd):
            image = image.run_commands(
                f"{python_install_probe_command('overlay-command-post-install', python_bin=image_python, uv_bin=UV_BIN)} && "
                f"{python_runtime_contract_verify_command('overlay-command-post-install', python_bin=image_python)}"
            )

    # Add force rebuild marker (change this to invalidate cache)
    image = image.run_commands("echo 'rollouts-build-v4-uv'")

    env_vars = {
        "HF_HOME": HF_CACHE_DIR,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "PATH": image_path_prefix,
        # Megatron-LM needs to be on PYTHONPATH for megatron.core imports
        "PYTHONPATH": "/root/Megatron-LM:/root",
        # NCCL settings for multi-GPU training
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        **spec.env,
        **overlay.env,
    }
    image = image.env(env_vars)

    manifest: ImageManifest = image_manifest_for_spec(
        spec,
        image_name=f"modal-{gpu_type.lower()}",
        cuda_version=cuda_version,
        resolved_image_ref=(
            resolve_image_for_provisioning(spec).resolved_ref
            if spec.source_type == "registry"
            else None
        ),
        features=overlay.features,
        installed_groups=overlay.installed_groups,
        env=env_vars,
        paths={"megatron_root": "/root/Megatron-LM"},
    )
    image = image.run_commands(manifest_write_command(manifest, spec.manifest_path))

    return image


def _trim_modal_build_log_line(line: str) -> str:
    trimmed = line.rstrip()
    if len(trimmed) <= MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT:
        return trimmed
    return trimmed[: MODAL_IMAGE_BUILD_LOG_CHAR_LIMIT - 3] + "..."


async def _emit_private_modal_image_logs(
    image: Any,
    emit: Callable[..., None],
) -> None:
    """Best-effort image build log capture via Modal private API.

    Modal already prints image build logs under `modal.enable_output()`, but that
    text is not part of our structured run journal. The private `_logs()` stream
    gives us a chance to persist the build log tail into `run.jsonl` as well.
    """

    image_id = getattr(image, "object_id", None)
    if not image_id:
        emit("modal_image_build_logs_unavailable", reason="missing_image_id")
        return

    logs_method = getattr(image, "_logs", None)
    if logs_method is None or not hasattr(logs_method, "aio"):
        emit("modal_image_build_logs_unavailable", image_id=image_id, reason="missing_private_logs")
        return

    emit("modal_image_build_logs_fetch_start", image_id=image_id)

    lines_emitted = 0
    truncated = False
    try:
        import trio_asyncio

        async def _consume_logs() -> tuple[int, bool]:
            nonlocal lines_emitted, truncated
            async for raw_line in logs_method.aio():
                if lines_emitted >= MODAL_IMAGE_BUILD_LOG_LINE_LIMIT:
                    truncated = True
                    break
                line = _trim_modal_build_log_line(raw_line)
                if not line:
                    continue
                logger.info("[modal image] %s", line)
                emit("modal_image_build_log", image_id=image_id, line=line)
                lines_emitted += 1
            return lines_emitted, truncated

        lines_emitted, truncated = await trio_asyncio.aio_as_trio(_consume_logs())
    except Exception as exc:
        emit(
            "modal_image_build_logs_fetch_failed",
            image_id=image_id,
            error=f"{type(exc).__name__}: {exc}",
        )
        logger.warning("Failed to fetch Modal image build logs for %s: %s", image_id, exc)
        return

    emit(
        "modal_image_build_logs_fetch_finished",
        image_id=image_id,
        line_count=lines_emitted,
        truncated=truncated,
    )


async def _eager_build_modal_image(
    image: Any,
    app: Any,
    emit: Callable[..., None],
) -> Any:
    """Build the Modal image explicitly before sandbox creation.

    This separates "image materialization" from "sandbox creation" so hangs are
    attributable to a real phase boundary instead of one opaque `Sandbox.create()`.
    """

    result: dict[str, Any] = {}
    start = trio.current_time()
    image_id = getattr(image, "object_id", None)
    emit("modal_image_build_start", image_id=image_id)

    async def _build_task() -> None:
        try:
            import trio_asyncio

            result["image"] = await trio_asyncio.aio_as_trio(image.build.aio(app))
        except Exception as exc:
            result["error"] = exc

    async with trio.open_nursery() as nursery:
        nursery.start_soon(_build_task)
        while "image" not in result and "error" not in result:
            elapsed = trio.current_time() - start
            emit(
                "modal_image_build_heartbeat",
                image_id=getattr(image, "object_id", None),
                elapsed_sec=round(elapsed, 3),
            )
            await trio.sleep(MODAL_IMAGE_BUILD_HEARTBEAT_S)
        nursery.cancel_scope.cancel()

    if "error" in result:
        exc = result["error"]
        elapsed = trio.current_time() - start
        emit(
            "modal_image_build_failed",
            image_id=getattr(image, "object_id", None),
            elapsed_sec=round(elapsed, 3),
            error=f"{type(exc).__name__}: {exc}",
        )
        raise exc

    built_image = result["image"]
    elapsed = trio.current_time() - start
    emit(
        "modal_image_build_finished",
        image_id=getattr(built_image, "object_id", None),
        elapsed_sec=round(elapsed, 3),
    )

    await _emit_private_modal_image_logs(built_image, emit)
    return built_image


def _get_model_cache_key(model_name: str, pruning_recipe: str | None = None) -> str:
    """Convert model name to a valid cache key.

    If pruning_recipe is set, includes a hash of the recipe in the key
    so pruned models get their own cache entry.
    """
    key = model_name.replace("/", "--").replace(":", "-")
    if pruning_recipe:
        import hashlib

        recipe_hash = hashlib.md5(pruning_recipe.encode()).hexdigest()[:8]
        key = f"{key}--pruned-{recipe_hash}"
    return key


async def _get_cached_snapshot(model_name: str, pruning_recipe: str | None = None) -> Any | None:
    """Look up cached model weights snapshot from Modal Dict.

    Returns the snapshot Image if found, None otherwise.
    """
    import modal
    import trio

    cache_key = _get_model_cache_key(model_name, pruning_recipe)

    try:
        # Dict operations are sync, run in thread
        def _lookup() -> str | None:
            cache_dict = modal.Dict.from_name(MODEL_CACHE_DICT_NAME, create_if_missing=True)
            return cache_dict.get(cache_key)

        snapshot_id = await trio.to_thread.run_sync(_lookup)
        if snapshot_id:
            logger.info(f"Found cached weights snapshot for {model_name}: {snapshot_id}")
            # Reconstruct the Image from the snapshot ID
            snapshot = modal.Image.from_id(snapshot_id)
            return snapshot
    except Exception as e:
        logger.warning(f"Failed to look up model cache: {e}")

    return None


async def _save_snapshot_to_cache(
    model_name: str, snapshot: Any, pruning_recipe: str | None = None
) -> None:
    """Save a model weights snapshot to the Modal Dict cache."""
    import modal
    import trio

    cache_key = _get_model_cache_key(model_name, pruning_recipe)
    snapshot_id = snapshot.object_id

    try:
        # Dict operations are sync, run in thread
        def _save() -> None:
            cache_dict = modal.Dict.from_name(MODEL_CACHE_DICT_NAME, create_if_missing=True)
            cache_dict[cache_key] = snapshot_id

        await trio.to_thread.run_sync(_save)
        logger.info(f"Cached weights snapshot for {model_name}: {snapshot_id}")
    except Exception as e:
        logger.warning(f"Failed to save model cache: {e}")


async def _download_and_snapshot_model(sandbox: Any, model_name: str) -> Any | None:
    """Download model weights and create a directory snapshot.

    Returns the snapshot Image, or None if snapshotting failed.
    """
    import trio
    import trio_asyncio

    logger.info(f"Downloading model weights for {model_name}...")

    # Download using snapshot_download with progress enabled
    # Note: tqdm progress goes to stderr
    download_script = f"""
import sys
from huggingface_hub import snapshot_download
print(f"Starting download of {model_name!r}...", flush=True)
path = snapshot_download("{model_name}")
print(f"Downloaded to: {{path}}", flush=True)
"""
    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio(
            "python",
            "-c",
            download_script.strip(),
            timeout=1800,  # 30 min timeout for large models
        )
    )

    # Stream output using parallel threads (async for doesn't work with trio_asyncio bridge)
    import threading

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info(f"[download] {line.rstrip()}")

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info(f"[download] {line.rstrip()}")  # tqdm progress goes here

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error(f"Model download failed with exit code {exit_code}")
        return None

    logger.info("Model downloaded, creating directory snapshot...")

    # Snapshot the HuggingFace cache directory
    # Note: _experimental_snapshot_directory is the alpha API for directory-only snapshots
    # It's more efficient than snapshot_filesystem for caching just the model weights
    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info(f"Created snapshot: {snapshot.object_id}")
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def _download_prune_and_snapshot_model(
    sandbox: Any, model_name: str, pruning_recipe: str
) -> Any | None:
    """Download model, apply pruning recipe, and create a directory snapshot.

    The pruning runs in the sandbox (requires GPU for model loading).
    Returns the snapshot Image, or None if failed.
    """
    import trio
    import trio_asyncio

    logger.info(f"Downloading and pruning model: {model_name}")

    # Python script to download, prune, and save
    # The pruned model overwrites the HF cache so the snapshot contains pruned weights
    prune_script = f'''
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "{model_name}"
recipe_json = """{pruning_recipe}"""

# Parse recipe
recipe = json.loads(recipe_json)
experts_to_keep = {{int(k): v for k, v in recipe["experts_to_keep"].items()}}

# Download model first
logger.info(f"Downloading {{model_name}}...")
cache_path = snapshot_download(model_name)
logger.info(f"Downloaded to: {{cache_path}}")

# Load model
logger.info("Loading model for pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Get original config
config = model.config
original_num = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
logger.info(f"Original experts: {{original_num}}")

# Apply pruning
for layer_idx, keep_indices in experts_to_keep.items():
    layer = model.model.layers[layer_idx]
    moe_block = getattr(layer, "mlp", None)
    if moe_block is None or not hasattr(moe_block, "experts"):
        continue

    experts = moe_block.experts
    num_experts = len(experts)

    # Prune experts
    new_experts = nn.ModuleList([experts[i] for i in keep_indices])
    moe_block.experts = new_experts

    # Prune router
    router = getattr(moe_block, "gate", None)
    if router is not None and hasattr(router, "weight"):
        weight = router.weight.data
        if weight.shape[0] == num_experts:
            router.weight = nn.Parameter(weight[keep_indices])
        elif weight.shape[1] == num_experts:
            router.weight = nn.Parameter(weight[:, keep_indices])
        if hasattr(router, "bias") and router.bias is not None:
            if router.bias.shape[0] == num_experts:
                router.bias = nn.Parameter(router.bias.data[keep_indices])

    logger.info(f"Layer {{layer_idx}}: {{num_experts}} -> {{len(keep_indices)}} experts")

# Update config
new_num = len(next(iter(experts_to_keep.values())))
if hasattr(config, "n_routed_experts"):
    config.n_routed_experts = new_num
if hasattr(config, "num_experts"):
    config.num_experts = new_num

logger.info(f"Pruning complete: {{original_num}} -> {{new_num}} experts")

# Save back to cache location (overwrites original)
logger.info(f"Saving pruned model to {{cache_path}}...")
model.save_pretrained(cache_path, safe_serialization=True)
tokenizer.save_pretrained(cache_path)

# Also save recipe for reference
with open(f"{{cache_path}}/pruning_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)

logger.info("Pruned model saved!")
'''

    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio(
            "python",
            "-c",
            prune_script,
            timeout=3600,  # 1 hour for download + prune + save
        )
    )

    # Stream output
    import threading

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info(f"[prune] {line.rstrip()}")

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info(f"[prune] {line.rstrip()}")

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error(f"Pruning failed with exit code {exit_code}")
        return None

    logger.info("Pruning complete, creating directory snapshot...")

    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info(f"Created pruned model snapshot: {snapshot.object_id}")
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def _prune_mounted_model_and_snapshot(
    sandbox: Any, model_name: str, pruning_recipe: str
) -> Any | None:
    """Prune already-mounted model weights and create a directory snapshot.

    Use this when the base model is already cached/mounted. Skips download.
    Returns the snapshot Image, or None if failed.
    """
    import trio
    import trio_asyncio

    logger.info(f"Pruning mounted model: {model_name}")

    # Python script to prune (model already in HF cache from mount)
    prune_script = f'''
import json
import logging
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model_name = "{model_name}"
recipe_json = """{pruning_recipe}"""

# Parse recipe
recipe = json.loads(recipe_json)
experts_to_keep = {{int(k): v for k, v in recipe["experts_to_keep"].items()}}

# Get cache path (model should already be there from mount)
cache_path = snapshot_download(model_name, local_files_only=True)
logger.info(f"Using cached model at: {{cache_path}}")

# Load model
logger.info("Loading model for pruning...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Get original config
config = model.config
original_num = getattr(config, "n_routed_experts", None) or getattr(config, "num_experts", 0)
logger.info(f"Original experts: {{original_num}}")

# Apply pruning
for layer_idx, keep_indices in experts_to_keep.items():
    layer = model.model.layers[layer_idx]
    moe_block = getattr(layer, "mlp", None)
    if moe_block is None or not hasattr(moe_block, "experts"):
        continue

    experts = moe_block.experts
    num_experts = len(experts)

    # Prune experts
    new_experts = nn.ModuleList([experts[i] for i in keep_indices])
    moe_block.experts = new_experts

    # Prune router
    router = getattr(moe_block, "gate", None)
    if router is not None and hasattr(router, "weight"):
        weight = router.weight.data
        if weight.shape[0] == num_experts:
            router.weight = nn.Parameter(weight[keep_indices])
        elif weight.shape[1] == num_experts:
            router.weight = nn.Parameter(weight[:, keep_indices])
        if hasattr(router, "bias") and router.bias is not None:
            if router.bias.shape[0] == num_experts:
                router.bias = nn.Parameter(router.bias.data[keep_indices])

    logger.info(f"Layer {{layer_idx}}: {{num_experts}} -> {{len(keep_indices)}} experts")

# Update config
new_num = len(next(iter(experts_to_keep.values())))
if hasattr(config, "n_routed_experts"):
    config.n_routed_experts = new_num
if hasattr(config, "num_experts"):
    config.num_experts = new_num

logger.info(f"Pruning complete: {{original_num}} -> {{new_num}} experts")

# Save back to cache location (overwrites original)
logger.info(f"Saving pruned model to {{cache_path}}...")
model.save_pretrained(cache_path, safe_serialization=True)
tokenizer.save_pretrained(cache_path)

# Also save recipe for reference
with open(f"{{cache_path}}/pruning_recipe.json", "w") as f:
    json.dump(recipe, f, indent=2)

logger.info("Pruned model saved!")
'''

    proc = await trio_asyncio.aio_as_trio(
        sandbox.exec.aio(
            "python",
            "-c",
            prune_script,
            timeout=1800,  # 30 min (no download needed)
        )
    )

    # Stream output
    import threading

    def _read_stdout() -> None:
        for line in proc.stdout:
            logger.info(f"[prune] {line.rstrip()}")

    def _read_stderr() -> None:
        for line in proc.stderr:
            logger.info(f"[prune] {line.rstrip()}")

    def _stream_both() -> None:
        t1 = threading.Thread(target=_read_stdout)
        t2 = threading.Thread(target=_read_stderr)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

    await trio.to_thread.run_sync(_stream_both)

    exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
    if exit_code != 0:
        logger.error(f"Pruning failed with exit code {exit_code}")
        return None

    logger.info("Pruning complete, creating directory snapshot...")

    try:
        snapshot = await trio_asyncio.aio_as_trio(
            sandbox._experimental_snapshot_directory.aio(HF_CACHE_DIR)
        )
        logger.info(f"Created pruned model snapshot: {snapshot.object_id}")
        return snapshot
    except Exception:
        logger.exception("Failed to create snapshot")
        return None


async def _mount_cached_weights(sandbox: Any, snapshot: Any) -> None:
    """Mount cached model weights into the sandbox."""
    import trio_asyncio

    logger.info(f"Mounting cached weights from snapshot {snapshot.object_id}...")
    # Note: _experimental_mount_image is the alpha API for mounting directory snapshots
    await trio_asyncio.aio_as_trio(sandbox._experimental_mount_image.aio(HF_CACHE_DIR, snapshot))
    logger.info("Cached weights mounted")


async def _create_sandbox(
    config: ModalRunConfig,
) -> tuple[Any, str]:
    """Create Modal sandbox using native async API.

    Returns (sandbox, sandbox_id).
    """
    import modal
    import trio_asyncio

    if config.sandbox_id:
        logger.info(f"Reusing sandbox: {config.sandbox_id}")

        def _attach() -> Any:
            return modal.Sandbox.from_id(config.sandbox_id)

        sandbox = await trio.to_thread.run_sync(_attach)
        assert sandbox is not None, f"Failed to reattach to sandbox: {config.sandbox_id}"
        assert sandbox.object_id, "Sandbox missing object_id"
        logger.info(f"Reattached to sandbox: {sandbox.object_id}")
        return sandbox, sandbox.object_id

    logger.info(f"Looking up app: {MODAL_APP_NAME}")
    app = await trio_asyncio.aio_as_trio(
        modal.App.lookup.aio(MODAL_APP_NAME, create_if_missing=True)
    )

    # Clean up any existing sandboxes from this app to avoid hitting limits
    existing = list(modal.Sandbox.list(app_id=app.app_id))
    if existing:
        logger.info(f"Cleaning up {len(existing)} existing sandbox(es)...")
        for sb in existing:
            try:
                sb.terminate()
                logger.info(f"  Terminated {sb.object_id}")
            except Exception as e:
                logger.warning(f"  Failed to terminate {sb.object_id}: {e}")

    # GPU spec
    gpu_count = config.gpu_count
    gpu_type = config.gpu_type
    if gpu_count > 1:
        gpu_spec = f"{gpu_type}:{gpu_count}"
    else:
        gpu_spec = gpu_type

    # Unique name
    ts = int(datetime.now(timezone.utc).timestamp())
    sandbox_name = f"rollouts-{config.gpu_type.lower()}-{ts}"

    timeout_seconds = config.timeout_hours * 3600

    create_timeout_s = 300
    create_heartbeat_s = 15
    create_attempts = 2

    def emit(event: str, **data: Any) -> None:
        if config.event_log is not None:
            config.event_log(
                event,
                provider="modal",
                run_name=config.run_name,
                sandbox_name=sandbox_name,
                gpu_type=config.gpu_type,
                gpu_count=config.gpu_count,
                **data,
            )

    logger.info("Constructing Modal image...")
    assert config.deps is not None  # Validated in __post_init__
    emit("modal_image_construct_start", app_id=app.app_id)
    image = _build_modal_image(modal, config.deps, config.gpu_type)
    emit(
        "modal_image_construct_finished",
        app_id=app.app_id,
        source_ref=getattr(config.deps.resolved_image(config.gpu_type), "source_ref", None),
    )
    logger.info("Eagerly building Modal image...")
    image = await _eager_build_modal_image(image, app, emit)
    logger.info("Modal image ready: %s", getattr(image, "object_id", None))

    async def _create_once(attempt: int) -> Any:
        logger.info(
            f"Creating sandbox: {sandbox_name} (gpu={gpu_spec}) attempt={attempt}/{create_attempts}"
        )
        emit("modal_sandbox_create_attempt_start", attempt=attempt, timeout_sec=create_timeout_s)

        result: dict[str, Any] = {}

        async def _create_task() -> None:
            result["sandbox"] = await trio_asyncio.aio_as_trio(
                modal.Sandbox.create.aio(
                    app=app,
                    image=image,
                    gpu=gpu_spec,
                    timeout=timeout_seconds,
                    name=sandbox_name,
                    verbose=True,  # Enable backend logging for observability
                )
            )

        start = trio.current_time()
        with trio.move_on_after(create_timeout_s) as scope:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(_create_task)
                while "sandbox" not in result:
                    elapsed = trio.current_time() - start
                    emit(
                        "modal_sandbox_create_heartbeat",
                        attempt=attempt,
                        elapsed_sec=round(elapsed, 3),
                    )
                    await trio.sleep(create_heartbeat_s)
                nursery.cancel_scope.cancel()

        if "sandbox" in result:
            elapsed = trio.current_time() - start
            emit(
                "modal_sandbox_create_attempt_succeeded",
                attempt=attempt,
                elapsed_sec=round(elapsed, 3),
            )
            return result["sandbox"]

        assert scope.cancelled_caught
        emit(
            "modal_sandbox_create_attempt_timeout",
            attempt=attempt,
            timeout_sec=create_timeout_s,
        )
        raise TimeoutError(
            f"Modal sandbox creation timed out after {create_timeout_s}s "
            f"(attempt {attempt}/{create_attempts})"
        )

    sandbox = None
    last_error: Exception | None = None
    for attempt in range(1, create_attempts + 1):
        try:
            sandbox = await _create_once(attempt)
            break
        except Exception as exc:
            last_error = exc
            logger.warning(
                "Modal sandbox create attempt %s/%s failed: %s",
                attempt,
                create_attempts,
                exc,
            )
            if attempt == create_attempts:
                break
            emit(
                "modal_sandbox_create_retry_scheduled",
                attempt=attempt,
                next_attempt=attempt + 1,
                error=f"{type(exc).__name__}: {exc}",
            )

    if sandbox is None:
        emit(
            "modal_sandbox_create_failed",
            attempts=create_attempts,
            error=f"{type(last_error).__name__}: {last_error}" if last_error else "unknown",
        )
        raise RuntimeError(
            f"Modal sandbox creation failed after {create_attempts} attempts: {last_error}"
        ) from last_error

    assert sandbox is not None, "Sandbox.create() returned None"
    assert sandbox.object_id, "Sandbox missing object_id"

    logger.info(f"Sandbox created: {sandbox.object_id}")

    return sandbox, sandbox.object_id


def _exec_sync(
    sandbox: Any,
    command: str,
    timeout: int = 300,
    *,
    on_started: Callable[[], None] | None = None,
    on_stdout_line: Callable[[str], None] | None = None,
    on_stderr_line: Callable[[str], None] | None = None,
    on_heartbeat: Callable[[float, float], None] | None = None,
    heartbeat_interval_s: float = 15.0,
) -> tuple[str, str, int]:
    """Execute command in sandbox with interleaved stdout/stderr streaming.

    Uses threads to read stdout and stderr concurrently so output is displayed
    as it arrives instead of waiting for command completion.

    Returns (stdout, stderr, exit_code).
    """
    proc = sandbox.exec("bash", "-c", command, timeout=timeout)
    if on_started is not None:
        on_started()

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []
    stop_heartbeat = threading.Event()
    activity_lock = threading.Lock()
    last_activity_ts = time.monotonic()

    def mark_activity() -> None:
        nonlocal last_activity_ts
        with activity_lock:
            last_activity_ts = time.monotonic()

    def read_stdout() -> None:
        for line in proc.stdout:
            stdout_lines.append(line)
            mark_activity()
            logger.info(f"[sandbox] {line.rstrip()}")
            if on_stdout_line is not None:
                on_stdout_line(line)

    def read_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)
            mark_activity()
            logger.warning(f"[sandbox stderr] {line.rstrip()}")
            if on_stderr_line is not None:
                on_stderr_line(line)

    def emit_heartbeats() -> None:
        if on_heartbeat is None:
            return
        started_ts = time.monotonic()
        while not stop_heartbeat.wait(heartbeat_interval_s):
            with activity_lock:
                silence_sec = time.monotonic() - last_activity_ts
            elapsed_sec = time.monotonic() - started_ts
            on_heartbeat(elapsed_sec, silence_sec)

    # Read both streams concurrently
    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    heartbeat_thread = threading.Thread(target=emit_heartbeats, daemon=True)
    stdout_thread.start()
    stderr_thread.start()
    heartbeat_thread.start()
    stdout_thread.join()
    stderr_thread.join()

    proc.wait()
    stop_heartbeat.set()
    heartbeat_thread.join(timeout=1.0)

    return "".join(stdout_lines), "".join(stderr_lines), proc.returncode


async def _sync_code_to_sandbox(sandbox: Any, local_root: Path) -> str:
    """Sync local code to sandbox via sandbox.open() file API.

    Uses git bundle + sandbox.open() for efficient file transfer.

    Returns workspace path in sandbox (the rollouts subdir within the cloned repo).
    """
    # Git root is ~/research (parent), we clone to /workspace/research
    # The rollouts code is at /workspace/research/rollouts
    clone_dir = "/workspace/research"
    workspace = "/workspace/research/rollouts"

    def _sync() -> None:
        # Create git bundle of current HEAD (fast, includes all needed objects)
        with tempfile.NamedTemporaryFile(suffix=".bundle", delete=False) as f:
            bundle_path = f.name

        try:
            # Get current branch/commit
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(local_root),
                capture_output=True,
                text=True,
                check=True,
            )
            commit = result.stdout.strip()
            logger.info(f"Bundling commit {commit[:8]}...")

            # Create bundle
            subprocess.run(
                ["git", "bundle", "create", bundle_path, "HEAD"],
                cwd=str(local_root),
                check=True,
                capture_output=True,
            )

            bundle_size = os.path.getsize(bundle_path)
            logger.info(f"Bundle size: {bundle_size / 1024 / 1024:.1f} MB")

            # Read bundle data
            with open(bundle_path, "rb") as f:
                bundle_data = f.read()

            # Create workspace directory
            _exec_sync(sandbox, "mkdir -p /workspace", timeout=30)

            # Use sandbox.open() for proper file transfer (Alpha API)
            logger.info("Uploading bundle via sandbox.open()...")
            remote_file = sandbox.open("/tmp/repo.bundle", "wb")
            remote_file.write(bundle_data)
            remote_file.close()
            logger.info(f"Uploaded {len(bundle_data) / 1024 / 1024:.1f} MB")

            logger.info("Extracting bundle...")
            # Clone from bundle - clones parent repo (research) to /workspace/research
            _exec_sync(
                sandbox,
                "cd /workspace && git clone /tmp/repo.bundle research && "
                "cd research && git checkout HEAD",
                timeout=120,
            )

            logger.info(f"Code synced to {workspace}")

        finally:
            os.unlink(bundle_path)

    await trio.to_thread.run_sync(_sync)
    return workspace


async def _run_training_in_sandbox(
    sandbox: Any,
    workspace: str,
    config_path: str,
    run_name: str,
    deps: DepsConfig | None,
    gpu_type: str,
    gpu_count: int = 1,
    use_torchrun: bool = True,
    event_log: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run training script inside Modal sandbox.

    Returns metrics from training.
    """
    # Get relative config path (may already be relative)
    config_p = Path(config_path)
    if config_p.is_absolute():
        config_rel = config_p.relative_to(REPO_ROOT)
    else:
        config_rel = config_p

    image_python = IMAGE_VENV_PYTHON
    image_path_prefix = f"{IMAGE_VENV_DIR}/bin:/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    if deps is not None:
        spec = deps.resolved_image(gpu_type)
        if spec.python_runtime == "image_owned":
            image_python = spec.python_executable
            image_path_prefix = (
                "/root/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
            )

    logger.info("Using prepared image Python environment; skipping per-run dependency install")

    # Run training with PYTHONPATH set to include our code.
    #
    # Important: configs are declarative modules, not executable training entrypoints.
    # Modal should invoke the same local workload entrypoint we would use on a prepared
    # machine, not `python config.py` / `torchrun config.py`.
    logger.info(f"Starting training via argus local entrypoint: {config_rel}")

    # PYTHONPATH includes:
    # - {workspace} for rollouts imports
    # - /workspace/research for miniray (sibling workspace package)
    # - /root/Megatron-LM for megatron.core imports
    env_vars = (
        f"PYTHONUNBUFFERED=1 "
        f"PATH={image_path_prefix} "
        f"PYTHONPATH={workspace}:/workspace/research:/root/Megatron-LM:/root "
        f"ARGUS_EMIT_STARTUP_SENTINEL=1 "
        f"ROLLOUTS_RUN_NAME={run_name} "
        f"ROLLOUTS_OUTPUT_DIR=results/rl/{run_name} "
    )

    # TODO: If we need true multi-process Modal training later, route that through an
    # explicit remote execution/session layer instead of reviving `torchrun config.py`.
    cmd = f"cd {workspace} && {env_vars} {image_python} -m argus.run --local --config {config_rel}"

    startup_seen = threading.Event()
    done = threading.Event()
    results: dict[str, Any] = {}
    start_timeout_s = 60

    def emit(event: str, **data: Any) -> None:
        if event_log is not None:
            event_log(event, **data)

    def _on_started() -> None:
        emit("remote_entrypoint_invoked")
        emit("remote_stdout_stream_open")
        emit("remote_stderr_stream_open")

    def _on_stdout_line(line: str) -> None:
        if WORKLOAD_ENTRYPOINT_SENTINEL in line and not startup_seen.is_set():
            startup_seen.set()
            emit("workload_entrypoint_started")

    def _on_heartbeat(elapsed_sec: float, silence_sec: float) -> None:
        emit(
            "remote_process_heartbeat",
            elapsed_sec=round(elapsed_sec, 3),
            silence_sec=round(silence_sec, 3),
        )

    def _train() -> None:
        try:
            stdout, stderr, exit_code = _exec_sync(
                sandbox,
                cmd,
                timeout=14400,
                on_started=_on_started,
                on_stdout_line=_on_stdout_line,
                on_heartbeat=_on_heartbeat,
            )
            results["stdout"] = stdout
            results["stderr"] = stderr
            results["exit_code"] = exit_code
        finally:
            done.set()

    emit("remote_entrypoint_invoke_start", command=f"{image_python} -m argus.run --local")
    train_thread = threading.Thread(target=_train, daemon=True)
    train_thread.start()

    deadline = trio.current_time() + start_timeout_s
    while not done.is_set() and not startup_seen.is_set():
        if trio.current_time() >= deadline:
            emit("workload_entrypoint_start_timeout", timeout_sec=start_timeout_s)

            def _terminate() -> None:
                sandbox.terminate()

            await trio.to_thread.run_sync(_terminate)
            train_thread.join(timeout=5.0)
            return {
                "success": False,
                "exit_code": 124,
                "stderr": f"Workload entrypoint did not emit startup sentinel within {start_timeout_s}s",
            }
        await trio.sleep(1.0)

    while not done.is_set():
        await trio.sleep(1.0)

    train_thread.join(timeout=1.0)
    stdout = str(results.get("stdout", ""))
    stderr = str(results.get("stderr", ""))
    exit_code = int(results.get("exit_code", 1))
    emit("remote_exit_observed", exit_code=exit_code)

    if exit_code != 0:
        logger.error(f"Training failed with exit code {exit_code}")
        logger.error(f"stderr: {stderr}")
        return {"success": False, "exit_code": exit_code, "stderr": stderr}

    logger.info("Training completed successfully")
    return {"success": True, "exit_code": 0}


async def run_modal(config: ModalRunConfig) -> dict[str, Any]:
    """Run training on Modal.

    Full lifecycle:
    1. Create sandbox with deps
    2. Sync code
    3. Run training
    4. Terminate sandbox

    Returns training results.
    """
    import modal
    import trio_asyncio

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_name = config.run_name or f"modal_{timestamp}"

    def emit(event: str, **data: Any) -> None:
        if config.event_log is not None:
            config.event_log(event, provider="modal", run_name=run_name, **data)

    enforce_source_sync_policy(config.source_sync_policy, repo_root=REPO_ROOT, stream=sys.stderr)
    emit(
        "submit_start",
        config_path=config.config_path,
        gpu_type=config.gpu_type,
        gpu_count=config.gpu_count,
    )

    logger.info("=" * 60)
    logger.info(f"Modal Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {config.config_path}")
    logger.info(f"GPU: {config.gpu_count}x {config.gpu_type}")

    with modal.enable_output():
        async with trio_asyncio.open_loop():
            # Create sandbox
            logger.info("Creating Modal sandbox...")
            emit("modal_sandbox_create_start", gpu_type=config.gpu_type, gpu_count=config.gpu_count)
            sandbox, sandbox_id = await _create_sandbox(config)
            emit("modal_sandbox_created", sandbox_id=sandbox_id)

            try:
                # Test GPU access
                # NOTE: If this hangs, it's likely due to too many pending sandboxes.
                # The cleanup above should prevent this, but if it happens:
                # 1. Check Modal dashboard for pending sandboxes
                # 2. Run: modal.Sandbox.list() and terminate stale ones
                # 3. Modal has per-app sandbox limits that cause exec() to block
                logger.info("Verifying GPU access...")
                emit("modal_gpu_verify_start", sandbox_id=sandbox_id)
                start = trio.current_time()
                proc = await trio_asyncio.aio_as_trio(sandbox.exec.aio("nvidia-smi", timeout=30))
                stdout = await trio_asyncio.aio_as_trio(proc.stdout.read.aio())
                elapsed = trio.current_time() - start
                if elapsed > 10:
                    logger.warning(
                        f"GPU verification took {elapsed:.1f}s (expected <5s). "
                        "If this persists, check for pending sandboxes in Modal dashboard."
                    )
                logger.info(f"[sandbox] {stdout}")
                exit_code = await trio_asyncio.aio_as_trio(proc.wait.aio())
                assert exit_code == 0, f"nvidia-smi failed with exit code {exit_code}"
                logger.info("GPU access verified")
                emit("modal_gpu_verified", sandbox_id=sandbox_id, elapsed_sec=round(elapsed, 3))

                # Model weight caching: check for cached snapshot or download and cache
                # If pruning_recipe is set, the cache key includes a hash of the recipe
                if config.model_name:
                    cached_snapshot = await _get_cached_snapshot(
                        config.model_name, config.pruning_recipe
                    )
                    if cached_snapshot:
                        # Mount cached weights (possibly pruned)
                        await _mount_cached_weights(sandbox, cached_snapshot)
                    else:
                        # Need to download/prune weights
                        if config.pruning_recipe:
                            # Check if base model is cached (can skip download)
                            base_snapshot = await _get_cached_snapshot(config.model_name, None)
                            if base_snapshot:
                                logger.info("Using cached base model, applying pruning...")
                                await _mount_cached_weights(sandbox, base_snapshot)
                                # Prune the mounted weights (they're now in HF_CACHE_DIR)
                                snapshot = await _prune_mounted_model_and_snapshot(
                                    sandbox, config.model_name, config.pruning_recipe
                                )
                            else:
                                logger.info(
                                    f"No cached weights for {config.model_name}, downloading and pruning..."
                                )
                                snapshot = await _download_prune_and_snapshot_model(
                                    sandbox, config.model_name, config.pruning_recipe
                                )
                        else:
                            logger.info(
                                f"No cached weights for {config.model_name}, downloading..."
                            )
                            snapshot = await _download_and_snapshot_model(
                                sandbox, config.model_name
                            )
                        if snapshot:
                            await _save_snapshot_to_cache(
                                config.model_name, snapshot, config.pruning_recipe
                            )

                # Sync code (always uses local git bundle)
                logger.info("Syncing code to sandbox...")
                emit("modal_repo_sync_start", sandbox_id=sandbox_id)
                workspace = await _sync_code_to_sandbox(sandbox, REPO_ROOT)
                logger.info(f"Code synced to {workspace}")
                emit("modal_repo_synced", sandbox_id=sandbox_id, workspace=workspace)

                # Run training
                logger.info("Starting training...")
                emit("modal_training_start", sandbox_id=sandbox_id, workspace=workspace)
                results = await _run_training_in_sandbox(
                    sandbox,
                    workspace,
                    config.config_path,
                    run_name,
                    config.deps,
                    config.gpu_type,
                    config.gpu_count,
                    config.use_torchrun,
                    config.event_log,
                )
                emit(
                    "modal_training_finished",
                    sandbox_id=sandbox_id,
                    success=bool(results.get("success")),
                    exit_code=results.get("exit_code"),
                )

                print(
                    "To reuse: python -m rollouts.modal_runner "
                    f"--sandbox-id {sandbox.object_id} --config {config.config_path}"
                )

                return results

            finally:
                # Terminate sandbox
                if config.keep_alive:
                    logger.info(f"Keeping sandbox alive: {sandbox_id}")
                    emit("modal_sandbox_kept_alive", sandbox_id=sandbox_id)
                else:
                    logger.info(f"Terminating sandbox: {sandbox_id}")
                    emit("modal_sandbox_terminate_start", sandbox_id=sandbox_id)

                    def _terminate() -> None:
                        sandbox.terminate()

                    await trio.to_thread.run_sync(_terminate)
                    logger.info("Sandbox terminated")
                    emit("modal_sandbox_terminated", sandbox_id=sandbox_id)


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_config"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Standalone debug entry point.

    TODO: keep Modal execution guts here for now, but do not treat this module
    as a public control-plane CLI. Public launch entrypoints should live in
    Argus, with deeper SSH/Modal execution unification happening later.
    """
    parser = argparse.ArgumentParser(description="Run training on Modal")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., examples/rl/reverse_text/grpo_01_01.py)",
    )
    parser.add_argument(
        "--gpu",
        default="A100",
        help="GPU type (default: A100)",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs (default: 1)",
    )
    parser.add_argument(
        "--timeout-hours",
        type=int,
        default=4,
        help="Sandbox timeout in hours (default: 4)",
    )
    parser.add_argument(
        "--sandbox-id",
        type=str,
        help="Reuse existing sandbox instead of creating new",
    )
    parser.add_argument(
        "--keep-alive",
        action="store_true",
        help="Keep sandbox running after completion",
    )
    parser.add_argument(
        "--force-deploy-committed",
        action="store_true",
        help="Proceed despite uncommitted changes (only committed code is deployed)",
    )

    args = parser.parse_args()

    # Setup logging with JSONL file output for debugging
    # Creates results/modal_runs/{timestamp}/run.jsonl
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log_dir = REPO_ROOT / "results" / "modal_runs" / f"modal_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "run.jsonl"

    setup_logging(
        level="INFO",
        use_color=True,
        log_file=str(log_file),
        logger_levels={"httpx": "WARNING", "httpcore": "WARNING", "modal": "WARNING"},
    )
    logger.info(f"Log file: {log_file}")

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    assert config_path.exists(), f"Config not found: {config_path}"

    # Load config module to extract hardware config
    config_module = load_config_module(config_path)
    hardware = getattr(config_module, "hardware", None)
    if hardware is None:
        raise ValueError(
            f"Config file must define 'hardware' (HardwareConfig). Got: {dir(config_module)}"
        )

    runtime = runtime_contract_from_hardware(hardware)
    materialization = materialization_plan_from_runtime(runtime)

    if runtime.deps is None:
        raise ValueError(
            "HardwareConfig.deps is required for Modal. "
            "Define deps=DepsConfig(...) in your hardware config."
        )

    # Use hardware config values, allow CLI overrides
    gpu_type = args.gpu if args.gpu != "A100" else runtime.gpu_type
    gpu_count = args.gpu_count if args.gpu_count != 1 else runtime.gpu_count

    # Extract model name and pruning recipe for weight caching (if available)
    # Look for config.model.name and config.model.pruning_recipe (GRPOConfig structure)
    grpo_config = getattr(config_module, "config", None)
    model_name = None
    pruning_recipe = None
    if grpo_config and hasattr(grpo_config, "model"):
        model_config = grpo_config.model
        if hasattr(model_config, "name"):
            model_name = model_config.name
            logger.info(f"Model name for weight caching: {model_name}")
        if hasattr(model_config, "pruning_recipe") and model_config.pruning_recipe:
            # Read the recipe file content
            recipe_path = Path(model_config.pruning_recipe)
            if not recipe_path.is_absolute():
                recipe_path = REPO_ROOT / recipe_path
            if recipe_path.exists():
                with open(recipe_path) as f:
                    pruning_recipe = f.read()
                logger.info(f"Pruning recipe loaded: {model_config.pruning_recipe}")
            else:
                logger.warning(f"Pruning recipe not found: {recipe_path}")

    # Build run config
    run_config = ModalRunConfig(
        config_path=str(config_path),
        runtime=RuntimeContract(
            provider=runtime.provider,
            gpu_type=gpu_type,
            gpu_count=gpu_count,
            deps=runtime.deps,
            container_disk_gb=runtime.container_disk_gb,
            hf_cache_dir=runtime.hf_cache_dir,
            persistent_volume_id=runtime.persistent_volume_id,
            persistent_volume_mount_path=runtime.persistent_volume_mount_path,
            persistent_volume_location=runtime.persistent_volume_location,
            use_torchrun=runtime.use_torchrun,
        ),
        materialization=materialization,
        source_sync_policy=SourceSyncPolicy.committed_only(
            dirty_action="warn" if args.force_deploy_committed else "fail"
        ),
        timeout_hours=args.timeout_hours,
        sandbox_id=args.sandbox_id,
        keep_alive=args.keep_alive,
        model_name=model_name,
        pruning_recipe=pruning_recipe,
    )

    # Run
    results = trio.run(run_modal, run_config)

    if results.get("success"):
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"Training failed: {results}")
        sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Use `python -m argus run --config ...` instead of rollouts.modal_runner.")
