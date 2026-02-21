#!/usr/bin/env python3
"""Unified RL training runner.

Usage:
    # Run with config file (reads hardware from config)
    python rollouts/run_rl.py --config examples/rl/kernelbench/grpo_01_01.py

    # Override provider
    python rollouts/run_rl.py --config ... --provider modal
    python rollouts/run_rl.py --config ... --local

    # Override hardware
    python rollouts/run_rl.py --config ... --gpu-type H100 --gpu-count 2

The config file should export:
    - config: Training config (GRPOConfig, SFTConfig, etc.)
    - hardware: HardwareConfig (optional, defaults to local)
    - train(config, **kwargs): Function to run training

Example config:
    from rollouts.training.configs import HardwareConfig
    from rollouts.training.grpo import GRPOConfig, ModelConfig

    hardware = HardwareConfig(
        gpu_type="A100",
        gpu_count=1,
        provider="runpod",
    )

    config = GRPOConfig(
        model=ModelConfig(name="Qwen/Qwen2.5-Coder-7B-Instruct"),
        ...
    )

    def train(config, **kwargs):
        from examples.rl.kernelbench.base_config import train as _train
        return _train(config=config, **kwargs)
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rollouts.training.configs import HardwareConfig

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).parent.parent


def load_config_module(config_path: Path) -> Any:
    """Load a config module from path."""
    spec = importlib.util.spec_from_file_location("_rl_config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["_rl_config"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    from rollouts.training.configs import HardwareConfig

    parser = argparse.ArgumentParser(
        description="Unified RL training runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Config-driven (reads hardware from config file)
    python rollouts/run_rl.py --config examples/rl/kernelbench/grpo_01_01.py

    # Override provider
    python rollouts/run_rl.py --config ... --provider modal
    python rollouts/run_rl.py --config ... --local

    # Override GPU settings
    python rollouts/run_rl.py --config ... --gpu-type H100 --gpu-count 2
        """,
    )
    parser.add_argument("--config", required=True, help="Path to config file")

    # Provider/hardware overrides
    parser.add_argument(
        "--provider",
        type=str,
        choices=["local", "modal", "runpod", "lambdalabs", "vast"],
        help="Override hardware provider",
    )
    parser.add_argument("--local", action="store_true", help="Force local execution")
    parser.add_argument("--gpu-type", type=str, help="Override GPU type")
    parser.add_argument("--gpu-count", type=int, help="Override GPU count")

    # Remote execution options
    parser.add_argument("--node-id", type=str, help="Reuse existing instance (provider:id)")
    parser.add_argument("--tui", action="store_true", help="Launch TUI after submitting")
    parser.add_argument("--tail", action="store_true", help="Stream logs to stdout")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow deploying with uncommitted changes",
    )

    # Training options
    parser.add_argument("--max-samples", type=int, help="Limit dataset size")

    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path

    if not config_path.exists():
        print(f"Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Load config module
    config_module = load_config_module(config_path)

    # Validate required exports
    if not hasattr(config_module, "config"):
        print(f"Config file must export 'config': {config_path}", file=sys.stderr)
        sys.exit(1)

    # Get hardware config (default to local if not specified)
    hardware: HardwareConfig = getattr(config_module, "hardware", HardwareConfig(provider="local"))

    # Apply CLI overrides
    if args.local:
        hardware = replace(hardware, provider="local")
    elif args.provider:
        hardware = replace(hardware, provider=args.provider)

    if args.gpu_type:
        hardware = replace(hardware, gpu_type=args.gpu_type)
    if args.gpu_count:
        hardware = replace(hardware, gpu_count=args.gpu_count)

    # If --node-id provided, infer provider from it
    if args.node_id and hardware.provider == "local":
        provider_from_id = args.node_id.split(":")[0]
        hardware = replace(hardware, provider=provider_from_id)

    print(f"Config: {config_path}")
    print(f"Hardware: {hardware.gpu_count}x {hardware.gpu_type} on {hardware.provider}")

    # Dispatch based on provider
    if hardware.provider == "local":
        _run_local(config_module, args)
    elif hardware.provider == "modal":
        _run_modal(config_path, hardware, args)
    else:
        # SSH-based providers: runpod, lambdalabs, vast
        _run_ssh(config_path, hardware, args)


def _run_local(config_module: Any, args: argparse.Namespace) -> None:
    """Run training locally."""
    if not hasattr(config_module, "train"):
        print("Config file must export 'train' function for local execution", file=sys.stderr)
        sys.exit(1)

    kwargs = {}
    if args.max_samples is not None:
        kwargs["num_samples"] = args.max_samples

    print("Starting local training...")
    results = config_module.train(config=config_module.config, **kwargs)

    steps = len(results.get("metrics_history", [])) if results else 0
    print(f"Training complete. {steps} steps")


def _run_modal(config_path: Path, hardware: HardwareConfig, args: argparse.Namespace) -> None:
    """Run training on Modal."""
    import trio

    from rollouts.modal_runner import ModalRunConfig, run_modal

    modal_config = ModalRunConfig(
        config_path=str(config_path),
        gpu_type=hardware.gpu_type,
        gpu_count=hardware.gpu_count,
    )

    print("Submitting to Modal...")
    results = trio.run(run_modal, modal_config)

    if not results.get("success"):
        print("Modal execution failed", file=sys.stderr)
        sys.exit(1)


def _run_ssh(config_path: Path, hardware: HardwareConfig, args: argparse.Namespace) -> None:
    """Run training via SSH (RunPod, Lambda Labs, Vast.ai)."""
    import trio

    from rollouts.run import run_remote

    print(f"Provisioning on {hardware.provider}...")
    trio.run(
        run_remote,
        str(config_path),
        args.keep_alive,
        args.node_id,
        args.tui,
        hardware.gpu_count,
        hardware.gpu_type,
        args.tail,
        hardware.provider,
        args.allow_dirty,
    )


if __name__ == "__main__":
    main()
