#!/usr/bin/env python3
"""Unified entry point for KernelBench evaluation.

Usage:
    # Run on Modal (recommended - has GPU for kernel scoring)
    python run_eval.py configs/api_smoke.py --modal

    # Local (requires local CUDA GPU)
    python run_eval.py configs/api_smoke.py
    python run_eval.py configs/sglang_smoke.py --limit 3
    python run_eval.py configs/api_smoke.py --model anthropic/claude-opus-4-20250514

Config files define:
    - endpoint: Endpoint configuration (model, base_url, api_format)
    - levels: KernelBench levels to use [1, 2, 3, 4]
    - backend: "CUDA" or "HIP"
    - max_turns: Maximum turns per problem
    - max_samples: Maximum problems to evaluate (can override with --limit)
    - max_concurrent: Parallel evaluation
"""

from __future__ import annotations

import argparse
import importlib.util
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

import trio

# Add rollouts to path if needed
repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from rollouts.dtypes import Endpoint, EvalConfig, Message
from rollouts.environments.kernelbench_multi import (
    KernelBenchMultiTurnEnvironment,
    configure_sandbox_pool,
)
from rollouts.evaluation import evaluate
from rollouts.gpu_sandbox import SandboxPool

from examples.kernelbench.dataset import load_kernelbench_prompts
from examples.kernelbench.scoring import kernelbench_score_fn

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load config from Python file.

    Config files should define module-level variables like:
        endpoint = Endpoint(...)
        levels = [1]
        backend = "CUDA"
        max_turns = 4
        max_samples = 5
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract config values with defaults
    config = {
        "endpoint": getattr(module, "endpoint", None),
        "levels": getattr(module, "levels", [1]),
        "backend": getattr(module, "backend", "CUDA"),
        "max_turns": getattr(module, "max_turns", 8),
        "max_samples": getattr(module, "max_samples", None),
        "max_concurrent": getattr(module, "max_concurrent", 1),
        "output_dir": getattr(module, "output_dir", None),
        "eval_name": getattr(module, "eval_name", "kernelbench"),
        "verbose": getattr(module, "verbose", True),
        "sandbox_configs": getattr(module, "sandbox_configs", []),
    }

    if config["endpoint"] is None:
        raise ValueError("Config must define 'endpoint'")

    return config


def prepare_messages(sample: dict[str, Any]) -> list[Message]:
    """Convert sample dict to Message list."""
    messages = []
    for msg in sample.get("messages", []):
        messages.append(Message(role=msg["role"], content=msg["content"]))
    return messages


async def run_eval(config_path: str, cli_overrides: dict[str, Any]) -> None:
    """Run evaluation with config and CLI overrides."""
    # Load config
    config = load_config(config_path)

    # Apply CLI overrides
    if cli_overrides.get("limit"):
        config["max_samples"] = cli_overrides["limit"]
    if cli_overrides.get("max_concurrent"):
        config["max_concurrent"] = cli_overrides["max_concurrent"]
    if cli_overrides.get("max_turns"):
        config["max_turns"] = cli_overrides["max_turns"]
    if cli_overrides.get("model"):
        # Parse model string like "anthropic/claude-sonnet-4-20250514"
        model_str = cli_overrides["model"]
        if "/" in model_str:
            provider, model = model_str.split("/", 1)
            config["endpoint"] = Endpoint(
                model=f"{provider}/{model}",
                base_url=_get_base_url(provider),
                api_format=_get_api_format(provider),
            )
        else:
            # Assume anthropic if no provider prefix
            config["endpoint"] = Endpoint(
                model=f"anthropic/{model_str}",
                base_url="https://api.anthropic.com/v1",
                api_format="anthropic-messages",
            )
    if cli_overrides.get("levels"):
        config["levels"] = cli_overrides["levels"]
    if cli_overrides.get("backend"):
        config["backend"] = cli_overrides["backend"]
    if cli_overrides.get("output_dir"):
        config["output_dir"] = Path(cli_overrides["output_dir"])

    # Load dataset
    prompts = load_kernelbench_prompts(
        levels=config["levels"],
        max_samples=config["max_samples"],
        backend=config["backend"],
    )
    logger.info(f"Loaded {len(prompts)} problems from levels {config['levels']}")

    # Configure sandbox pool for kernel evaluation
    pool = SandboxPool(config["sandbox_configs"])
    configure_sandbox_pool(pool)

    # Create environment factory
    async def environment_factory(sample: dict[str, Any]) -> KernelBenchMultiTurnEnvironment:
        return KernelBenchMultiTurnEnvironment(
            ref_code=sample.get("ref_code", ""),
            backend=config["backend"],
            max_turns=config["max_turns"],
        )

    # Build EvalConfig
    eval_config = EvalConfig(
        endpoint=config["endpoint"],
        score_fn=kernelbench_score_fn,
        prepare_messages=prepare_messages,
        environment_factory=environment_factory,
        max_samples=config["max_samples"],
        max_concurrent=config["max_concurrent"],
        output_dir=config["output_dir"],
        eval_name=config["eval_name"],
        verbose=config["verbose"],
        show_progress=True,
    )

    # Start sandbox pool
    await pool.start()

    try:
        # Run evaluation
        report = await evaluate(iter(prompts), eval_config)

        # Print summary
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"Total samples: {report.total_samples}")
        print("Summary metrics:")
        for name, value in report.summary_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {name}: {value:.3f}")
            else:
                print(f"  {name}: {value}")

        if config["output_dir"]:
            print(f"\nResults saved to: {config['output_dir']}")

    finally:
        await pool.stop()


def _get_base_url(provider: str) -> str:
    """Get base URL for provider."""
    urls = {
        "anthropic": "https://api.anthropic.com/v1",
        "openai": "https://api.openai.com/v1",
        "google": "https://generativelanguage.googleapis.com/v1beta",
        "sglang": "http://localhost:30000/v1",
    }
    return urls.get(provider, "http://localhost:30000/v1")


def _get_api_format(provider: str) -> str:
    """Get API format for provider."""
    formats = {
        "anthropic": "anthropic-messages",
        "openai": "openai-chat",
        "google": "google-genai",
        "sglang": "openai-chat",
    }
    return formats.get(provider, "openai-chat")


def run_on_modal(config_path: str, overrides: dict[str, Any], gpu_type: str) -> None:
    """Run evaluation on Modal sandbox with GPU."""
    # Build command for modal_eval.py
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "modal_eval.py"),
        "--config",
        config_path,
        "--gpu-type",
        gpu_type,
    ]

    if overrides.get("limit"):
        cmd.extend(["--limit", str(overrides["limit"])])
    if overrides.get("max_turns"):
        cmd.extend(["--max-turns", str(overrides["max_turns"])])
    if overrides.get("model"):
        cmd.extend(["--model", overrides["model"]])
    if overrides.get("levels"):
        cmd.extend(["--levels"] + [str(l) for l in overrides["levels"]])
    if overrides.get("backend"):
        cmd.extend(["--backend", overrides["backend"]])

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run KernelBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Modal (recommended - has GPU for kernel scoring)
    python run_eval.py configs/api_smoke.py --modal

    # Local
    python run_eval.py configs/api_smoke.py
    python run_eval.py configs/api_smoke.py --limit 3
        """,
    )

    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--modal", action="store_true", help="Run on Modal (has GPU)")
    parser.add_argument("--gpu-type", default="A10G", help="GPU type for Modal (default: A10G)")
    parser.add_argument("--limit", type=int, help="Override max_samples")
    parser.add_argument("--max-concurrent", type=int, help="Override max_concurrent")
    parser.add_argument("--max-turns", type=int, help="Override max_turns")
    parser.add_argument("--model", help="Override model (format: provider/model)")
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        help="Override levels (e.g., --levels 1 2)",
    )
    parser.add_argument("--backend", choices=["CUDA", "HIP"], help="Override backend")
    parser.add_argument("--output-dir", help="Override output directory")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()

    # Setup logging
    level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Collect CLI overrides
    overrides = {}
    if args.limit:
        overrides["limit"] = args.limit
    if args.max_concurrent:
        overrides["max_concurrent"] = args.max_concurrent
    if args.max_turns:
        overrides["max_turns"] = args.max_turns
    if args.model:
        overrides["model"] = args.model
    if args.levels:
        overrides["levels"] = args.levels
    if args.backend:
        overrides["backend"] = args.backend
    if args.output_dir:
        overrides["output_dir"] = args.output_dir

    if args.modal:
        # Run on Modal
        run_on_modal(args.config, overrides, args.gpu_type)
    else:
        # Run locally
        trio.run(run_eval, args.config, overrides)


if __name__ == "__main__":
    main()
