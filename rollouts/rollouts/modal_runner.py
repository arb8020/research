"""Modal-based training runner.

Run training workloads on Modal sandboxes with GPU access.
Uses Modal's native async APIs via trio_asyncio bridge.

Usage:
    # From config file with hardware.provider="modal"
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
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

if TYPE_CHECKING:
    from .training.configs import DepsConfig

from ._logging import setup_logging

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


def _check_uncommitted_changes_warning() -> None:
    """Warn if there are uncommitted changes that won't be deployed.

    Modal runner uses git bundle, which only includes committed code.
    This matches the behavior in run.py (RunPod) which uses bifrost.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return  # Not a git repo or git not available

    lines = [line for line in result.stdout.strip().split("\n") if line]
    if not lines:
        return  # No changes

    # Parse into modified and untracked
    modified = [line[3:] for line in lines if line[:2].strip() in ("M", "MM", "AM", "A")]
    untracked = [line[3:] for line in lines if line.startswith("??")]

    if not modified and not untracked:
        return

    print(
        "\n⚠️  WARNING: Uncommitted changes detected!\n"
        "Modal runner uses git bundle - only committed code is deployed.\n",
        file=sys.stderr,
    )
    total = len(modified) + len(untracked)
    print(f"{total} file(s) will NOT be deployed:\n", file=sys.stderr)
    for f in modified[:5]:
        print(f"   - {f} (modified)", file=sys.stderr)
    if len(modified) > 5:
        print(f"   ... and {len(modified) - 5} more modified", file=sys.stderr)
    for f in untracked[:5]:
        print(f"   - {f} (untracked)", file=sys.stderr)
    if len(untracked) > 5:
        print(f"   ... and {len(untracked) - 5} more untracked", file=sys.stderr)
    print("\nCommit your changes or use '--allow-dirty' to proceed anyway.\n", file=sys.stderr)

    # For now, just warn (not blocking like RunPod)
    # To make this blocking, raise SystemExit(1) here


@dataclass
class ModalRunConfig:
    """Configuration for a Modal training run.

    The deps field comes from HardwareConfig.deps (DepsConfig).
    """

    config_path: str
    gpu_type: str = "A100"
    gpu_count: int = 1
    deps: DepsConfig | None = None  # Required - validated by HardwareConfig
    timeout_hours: int = 4
    use_torchrun: bool = True  # False for torchtitan (handles multi-GPU internally)
    sandbox_id: str | None = None
    keep_alive: bool = False
    model_name: str | None = None  # Model name for weight caching (e.g., "zai-org/GLM-4.7-Flash")
    pruning_recipe: str | None = (
        None  # Path to pruning recipe JSON (if set, model is pruned before caching)
    )

    def __post_init__(self) -> None:
        if self.deps is None:
            raise ValueError(
                "ModalRunConfig requires deps. This should come from HardwareConfig.deps."
            )


def _build_modal_image(modal: Any, deps: DepsConfig, gpu_type: str) -> Any:
    """Build Modal image from DepsConfig specification.

    Uses nvidia/cuda base image instead of debian_slim because Megatron/SGLang
    require nvcc for JIT kernel compilation.
    """
    # GPU-specific torch index and CUDA version
    if gpu_type in ("B200", "GB200"):
        pip_index = "https://download.pytorch.org/whl/nightly/cu128"
        cuda_version = "12.8.0"
    elif deps.pip_index_url:
        pip_index = deps.pip_index_url
        # Infer CUDA version from pip index URL
        if "cu128" in pip_index:
            cuda_version = "12.8.0"
        elif "cu126" in pip_index:
            cuda_version = "12.6.0"
        elif "cu124" in pip_index:
            cuda_version = "12.4.0"
        else:
            cuda_version = "12.4.0"
    else:
        pip_index = "https://download.pytorch.org/whl/cu124"
        cuda_version = "12.4.0"

    # Use CUDA devel image (includes nvcc) for Megatron/SGLang JIT compilation
    cuda_image = f"nvidia/cuda:{cuda_version}-devel-ubuntu22.04"
    image = modal.Image.from_registry(cuda_image, add_python=deps.python_version)

    if deps.system_packages:
        image = image.apt_install(*deps.system_packages)

    if deps.pip_packages:
        # Use uv for faster installs (~5x faster than pip)
        uv_kwargs: dict[str, Any] = {"index_url": pip_index}
        if deps.pip_extra_index_url:
            uv_kwargs["extra_index_url"] = deps.pip_extra_index_url

        image = image.uv_pip_install(*deps.pip_packages, **uv_kwargs)

    for cmd in deps.bootstrap_commands:
        image = image.run_commands(cmd)

    # Add force rebuild marker (change this to invalidate cache)
    image = image.run_commands("echo 'rollouts-build-v4-uv'")

    # Set up HuggingFace cache and Megatron PYTHONPATH
    image = image.env({
        "HF_HOME": "/root/.cache/huggingface",
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Megatron-LM needs to be on PYTHONPATH for megatron.core imports
        "PYTHONPATH": "/root/Megatron-LM:/root",
        # NCCL settings for multi-GPU training
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    })

    return image


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

    logger.info("Building image...")
    assert config.deps is not None  # Validated in __post_init__
    image = _build_modal_image(modal, config.deps, config.gpu_type)
    logger.info("Image built")

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

    logger.info(f"Creating sandbox: {sandbox_name} (gpu={gpu_spec})...")
    sandbox = await trio_asyncio.aio_as_trio(
        modal.Sandbox.create.aio(
            app=app,
            image=image,
            gpu=gpu_spec,
            timeout=timeout_seconds,
            name=sandbox_name,
            verbose=True,  # Enable backend logging for observability
        )
    )

    assert sandbox is not None, "Sandbox.create() returned None"
    assert sandbox.object_id, "Sandbox missing object_id"

    logger.info(f"Sandbox created: {sandbox.object_id}")

    return sandbox, sandbox.object_id


def _exec_sync(sandbox: Any, command: str, timeout: int = 300) -> tuple[str, str, int]:
    """Execute command in sandbox with interleaved stdout/stderr streaming.

    Uses threads to read stdout and stderr concurrently so output is displayed
    as it arrives instead of waiting for command completion.

    Returns (stdout, stderr, exit_code).
    """
    import threading

    proc = sandbox.exec("bash", "-c", command, timeout=timeout)

    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def read_stdout() -> None:
        for line in proc.stdout:
            stdout_lines.append(line)
            logger.info(f"[sandbox] {line.rstrip()}")

    def read_stderr() -> None:
        for line in proc.stderr:
            stderr_lines.append(line)
            logger.warning(f"[sandbox stderr] {line.rstrip()}")

    # Read both streams concurrently
    stdout_thread = threading.Thread(target=read_stdout)
    stderr_thread = threading.Thread(target=read_stderr)
    stdout_thread.start()
    stderr_thread.start()
    stdout_thread.join()
    stderr_thread.join()

    proc.wait()

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
    gpu_count: int = 1,
    use_torchrun: bool = True,
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

    # Install base dependencies (not using editable install to avoid PyPI deps)
    logger.info("Installing dependencies...")

    def _install() -> None:
        # Install just the rollouts deps (skip bifrost/broker from PyPI)
        _exec_sync(
            sandbox,
            f"cd {workspace} && pip install openai anthropic dacite aiohttp trio httpx "
            f"'transformers>=4.50' datasets peft accelerate --quiet",
            timeout=300,
        )

    await trio.to_thread.run_sync(_install)
    logger.info("Dependencies installed")

    # Run training with PYTHONPATH set to include our code
    logger.info(f"Starting training: {config_rel}")

    # PYTHONPATH includes:
    # - {workspace} for rollouts imports
    # - /workspace/research for miniray (sibling workspace package)
    # - /root/Megatron-LM for megatron.core imports
    env_vars = (
        f"PYTHONUNBUFFERED=1 "
        f"PYTHONPATH={workspace}:/workspace/research:/root/Megatron-LM:/root "
        f"ROLLOUTS_RUN_NAME={run_name} "
        f"ROLLOUTS_OUTPUT_DIR=results/rl/{run_name} "
    )

    # Use torchrun for multi-GPU DDP training (unless disabled for backends like torchtitan)
    if gpu_count > 1 and use_torchrun:
        cmd = (
            f"cd {workspace} && {env_vars} "
            f"torchrun --standalone --nproc_per_node={gpu_count} {config_rel}"
        )
    else:
        cmd = f"cd {workspace} && {env_vars} python {config_rel}"

    def _train() -> tuple[str, str, int]:
        return _exec_sync(sandbox, cmd, timeout=14400)  # 4 hour timeout

    stdout, stderr, exit_code = await trio.to_thread.run_sync(_train)

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
    run_name = f"modal_{timestamp}"

    logger.info("=" * 60)
    logger.info(f"Modal Training: {run_name}")
    logger.info("=" * 60)
    logger.info(f"Config: {config.config_path}")
    logger.info(f"GPU: {config.gpu_count}x {config.gpu_type}")

    with modal.enable_output():
        async with trio_asyncio.open_loop():
            # Create sandbox
            logger.info("Creating Modal sandbox...")
            sandbox, sandbox_id = await _create_sandbox(config)

            try:
                # Test GPU access
                # NOTE: If this hangs, it's likely due to too many pending sandboxes.
                # The cleanup above should prevent this, but if it happens:
                # 1. Check Modal dashboard for pending sandboxes
                # 2. Run: modal.Sandbox.list() and terminate stale ones
                # 3. Modal has per-app sandbox limits that cause exec() to block
                logger.info("Verifying GPU access...")
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
                workspace = await _sync_code_to_sandbox(sandbox, REPO_ROOT)
                logger.info(f"Code synced to {workspace}")

                # Run training
                logger.info("Starting training...")
                results = await _run_training_in_sandbox(
                    sandbox,
                    workspace,
                    config.config_path,
                    run_name,
                    config.gpu_count,
                    config.use_torchrun,
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
                else:
                    logger.info(f"Terminating sandbox: {sandbox_id}")

                    def _terminate() -> None:
                        sandbox.terminate()

                    await trio.to_thread.run_sync(_terminate)
                    logger.info("Sandbox terminated")


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
    """CLI entry point."""
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

    args = parser.parse_args()

    # Check for uncommitted changes - git bundle only includes committed code
    _check_uncommitted_changes_warning()

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

    deps = hardware.deps
    if deps is None:
        raise ValueError(
            "HardwareConfig.deps is required for Modal. "
            "Define deps=DepsConfig(...) in your hardware config."
        )

    # Use hardware config values, allow CLI overrides
    gpu_type = args.gpu if args.gpu != "A100" else hardware.gpu_type
    gpu_count = args.gpu_count if args.gpu_count != 1 else hardware.gpu_count

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
        gpu_type=gpu_type,
        gpu_count=gpu_count,
        deps=deps,
        timeout_hours=args.timeout_hours,
        use_torchrun=hardware.use_torchrun,
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
    main()
