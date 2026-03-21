"""Legacy Modal debug shell.

Modal execution now lives in `bifrost.modal_backend`. This module remains only
as a manual debug CLI.
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import trio

from ._logging import setup_logging
from .modal_workload import MODAL_CLEANUP_SCOPES, REPO_ROOT
from .remote_runtime import (
    RuntimeContract,
    SourceSyncPolicy,
    materialization_plan_from_runtime,
    runtime_contract_from_hardware,
)

logger = logging.getLogger(__name__)


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
    """Standalone debug entry point."""
    parser = argparse.ArgumentParser(description="Run training on Modal")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., examples/rl/reverse_text/grpo_01_01.py)",
    )
    parser.add_argument("--gpu", default="A100", help="GPU type (default: A100)")
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
        "--sandbox-id", type=str, help="Reuse existing sandbox instead of creating new"
    )
    parser.add_argument(
        "--keep-alive", action="store_true", help="Keep sandbox running after completion"
    )
    parser.add_argument(
        "--cleanup-scope",
        choices=list(MODAL_CLEANUP_SCOPES),
        default="run",
        help="Pre-create sandbox cleanup scope (default: run)",
    )
    parser.add_argument(
        "--force-deploy-committed",
        action="store_true",
        help="Proceed despite uncommitted changes (only committed code is deployed)",
    )

    args = parser.parse_args()

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
    logger.info("Log file: %s", log_file)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    assert config_path.exists(), f"Config not found: {config_path}"

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

    gpu_type = args.gpu if args.gpu != "A100" else runtime.gpu_type
    gpu_count = args.gpu_count if args.gpu_count != 1 else runtime.gpu_count

    grpo_config = getattr(config_module, "config", None)
    model_name = None
    pruning_recipe = None
    if grpo_config and hasattr(grpo_config, "model"):
        model_config = grpo_config.model
        if hasattr(model_config, "name"):
            model_name = model_config.name
            logger.info("Model name for weight caching: %s", model_name)
        if hasattr(model_config, "pruning_recipe") and model_config.pruning_recipe:
            recipe_path = Path(model_config.pruning_recipe)
            if not recipe_path.is_absolute():
                recipe_path = REPO_ROOT / recipe_path
            if recipe_path.exists():
                with open(recipe_path) as f:
                    pruning_recipe = f.read()
                logger.info("Pruning recipe loaded: %s", model_config.pruning_recipe)
            else:
                logger.warning("Pruning recipe not found: %s", recipe_path)

    from bifrost.modal_backend import ModalExecutionRequest, run_modal_request

    results = trio.run(
        run_modal_request,
        ModalExecutionRequest(
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
            timeout_hours=args.timeout_hours,
            sandbox_id=args.sandbox_id,
            keep_alive=args.keep_alive,
            source_sync_policy=SourceSyncPolicy.committed_only(
                dirty_action="warn" if args.force_deploy_committed else "fail"
            ),
            cleanup_scope=args.cleanup_scope,
            model_name=model_name,
            pruning_recipe=pruning_recipe,
        ),
    )

    if results.get("success"):
        logger.info("Training completed successfully!")
        sys.exit(0)
    logger.error("Training failed: %s", results)
    sys.exit(1)


if __name__ == "__main__":
    raise SystemExit("Use `python -m argus run --config ...` instead of rollouts.modal_runner.")
