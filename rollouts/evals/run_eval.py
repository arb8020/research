#!/usr/bin/env python3
"""Unified eval runner. Run from rollouts/evals/.

Usage:
    python run_eval.py --config functional_extractor/configs/smoke.py
    python run_eval.py --config functional_extractor/configs/full.py
    python run_eval.py --config functional_extractor/configs/smoke.py --model claude-opus-4-5-20251101
    python run_eval.py --config functional_extractor/configs/smoke.py --limit 2 --allow-dirty

The config path determines which eval to run. The runner:
1. Parses the config path to find the eval name (first directory component)
2. Loads the config file
3. Runs the eval using the EvalSpec pattern with run_eval_from_spec
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

# Add rollouts to path
EVALS_ROOT = Path(__file__).parent
ROLLOUTS_ROOT = EVALS_ROOT.parent
sys.path.insert(0, str(ROLLOUTS_ROOT))
sys.path.insert(0, str(EVALS_ROOT))


def load_module(path: Path, module_name: str | None = None) -> Any:
    """Load a Python module from path."""
    if module_name is None:
        module_name = f"_dynamic_{path.stem}_{id(path)}"

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_eval_module(eval_name: str) -> Any:
    """Load an eval's eval.py module."""
    eval_path = EVALS_ROOT / eval_name / "eval.py"
    if not eval_path.exists():
        raise FileNotFoundError(f"Eval not found: {eval_name} (looked for {eval_path})")

    canonical_name = f"{eval_name.replace('/', '.')}.eval"
    return load_module(eval_path, module_name=canonical_name)


def run_simple_eval(spec: Any, config_module: Any, args: argparse.Namespace) -> dict[str, Any]:
    """Run an eval using the EvalSpec pattern with shared runner."""
    from rollouts.eval_runner import run_eval_from_spec

    kwargs: dict[str, Any] = {}

    # Load config objects if present
    if hasattr(config_module, "endpoint"):
        kwargs["endpoint"] = config_module.endpoint
    if hasattr(config_module, "run"):
        kwargs["run"] = config_module.run
    if hasattr(config_module, "output"):
        kwargs["output"] = config_module.output

    # Load tasks override if present
    if hasattr(config_module, "tasks_override"):
        kwargs["tasks"] = config_module.tasks_override

    # CLI overrides
    if args.model:
        kwargs["model"] = args.model
    if args.provider:
        kwargs["provider"] = args.provider
    if args.max_turns:
        kwargs["max_turns"] = args.max_turns
    if args.max_concurrent:
        kwargs["max_concurrent"] = args.max_concurrent
    if args.limit is not None:
        kwargs["limit"] = args.limit
    if args.no_progress:
        kwargs["show_progress"] = False
    if args.verbose:
        kwargs["verbose"] = True
    if args.resume:
        kwargs["resume_dir"] = args.resume

    return run_eval_from_spec(spec, **kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run evals",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_eval.py --config functional_extractor/configs/smoke.py
    python run_eval.py --config functional_extractor/configs/full.py --limit 2
    python run_eval.py --config functional_extractor/configs/smoke.py --model claude-opus-4-5-20251101
        """,
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config file (e.g., functional_extractor/configs/smoke.py)",
    )

    # Override flags
    parser.add_argument("--model", help="Override model")
    parser.add_argument("--provider", help="Override provider (anthropic/openai)")
    parser.add_argument("--max-turns", type=int, help="Override max turns")
    parser.add_argument("--max-concurrent", type=int, help="Override max concurrent samples")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress display")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--resume", type=Path, help="Resume from previous run directory")

    args = parser.parse_args()

    # Parse config path to get eval name
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = EVALS_ROOT / config_path

    # Extract eval name from path (first component relative to EVALS_ROOT)
    try:
        rel_path = config_path.relative_to(EVALS_ROOT)
        eval_name = rel_path.parts[0]
    except ValueError as e:
        raise ValueError(f"Config path must be under {EVALS_ROOT}") from e

    print(f"Running eval: {eval_name}")
    print(f"Config: {config_path}")

    # Load eval module and config
    eval_module = load_eval_module(eval_name)
    config_module = load_module(config_path)

    # Check for spec
    if hasattr(eval_module, "spec"):
        result = run_simple_eval(eval_module.spec, config_module, args)
    else:
        raise AttributeError(f"Eval {eval_name}/eval.py must define 'spec' (EvalSpec)")

    print("\n" + "=" * 60)
    print("Results:")
    for key, value in result.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
