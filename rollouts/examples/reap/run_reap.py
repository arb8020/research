#!/usr/bin/env python3
"""REAP expert pruning entry point.

Run locally:
    python examples/reap/run_reap.py --config examples/reap/configs/qwen3_prune_50.py

Run remotely:
    python examples/reap/configs/qwen3_prune_50.py --provision
    python examples/reap/configs/qwen3_prune_50.py --node-id sf:abc123

Note: For remote execution, run the config file directly (it has the train function).
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
from pathlib import Path

from examples.reap.config import ReapConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config_from_file(config_path: str) -> ReapConfig:
    """Load ReapConfig from a Python config file."""
    path = Path(config_path).resolve()
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "config"):
        raise ValueError(f"Config file {config_path} must define a 'config' variable")

    return module.config


def main() -> None:
    parser = argparse.ArgumentParser(description="REAP expert pruning")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., examples/reap/configs/qwen3_prune_50.py)",
    )

    args = parser.parse_args()

    from examples.reap.base_config import run_reap

    config = load_config_from_file(args.config)
    result = run_reap(config)

    logger.info("REAP complete!")
    logger.info(f"Output: {result['output_path']}")
    logger.info(f"Experts: {result['original_num_experts']} -> {result['pruned_num_experts']}")


if __name__ == "__main__":
    main()
