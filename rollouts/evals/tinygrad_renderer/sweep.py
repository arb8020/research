#!/usr/bin/env python3
"""Run eval sweep across backends and models.

Usage:
    # Run all combinations
    python sweep.py

    # Just one backend
    python sweep.py --backend METAL

    # Just one model
    python sweep.py --model claude-sonnet-4-20250514

    # Dry run (show what would run)
    python sweep.py --dry-run
"""

import argparse
import os
import subprocess
from datetime import datetime
from itertools import product
from pathlib import Path

# Model configurations
MODELS = [
    # Anthropic
    {"model": "claude-sonnet-4-20250514", "provider": "anthropic"},
    {"model": "claude-opus-4-20250514", "provider": "anthropic"},
    # OpenAI
    {"model": "gpt-4.1-2025-04-14", "provider": "openai"},
    {"model": "o3-2025-04-16", "provider": "openai"},  # high thinking
    # Google
    {"model": "gemini-2.5-pro-preview-05-06", "provider": "google"},
]

BACKENDS = ["METAL", "WEBGPU"]


def run_eval(
    backend: str,
    model: str,
    provider: str,
    max_turns: int,
    output_base: Path,
    dry_run: bool = False,
):
    """Run a single eval."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe_model = model.replace("/", "-").replace(":", "-")
    output_dir = output_base / f"{backend.lower()}_{safe_model}_{timestamp}"

    cmd = [
        "uv",
        "run",
        "python",
        "evals/tinygrad_renderer/eval.py",
        "--backend",
        backend,
        "--model",
        model,
        "--provider",
        provider,
        "--max-turns",
        str(max_turns),
        "--output-dir",
        str(output_dir),
    ]

    print(f"\n{'=' * 60}")
    print(f"Backend: {backend}, Model: {model} ({provider})")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    if dry_run:
        print("  [DRY RUN - skipping]")
        return None

    env = os.environ.copy()
    result = subprocess.run(cmd, env=env, cwd=Path(__file__).parent.parent.parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Run tinygrad renderer eval sweep")
    parser.add_argument("--backend", choices=BACKENDS, help="Run only this backend")
    parser.add_argument("--model", help="Run only this model")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "sweep_results")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without running"
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="Number of parallel runs (careful with API limits)"
    )
    args = parser.parse_args()

    # Filter based on args
    backends = [args.backend] if args.backend else BACKENDS
    models = [m for m in MODELS if not args.model or m["model"] == args.model]

    if not models:
        print(f"No models matched '{args.model}'")
        print(f"Available: {[m['model'] for m in MODELS]}")
        return

    # Generate combinations
    combinations = list(product(backends, models))
    print(f"Running {len(combinations)} eval(s):")
    for backend, model_cfg in combinations:
        print(f"  - {backend} x {model_cfg['model']}")

    # Run sequentially (could parallelize with ProcessPoolExecutor)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for backend, model_cfg in combinations:
        ret = run_eval(
            backend=backend,
            model=model_cfg["model"],
            provider=model_cfg["provider"],
            max_turns=args.max_turns,
            output_base=args.output_dir,
            dry_run=args.dry_run,
        )
        results.append({
            "backend": backend,
            "model": model_cfg["model"],
            "provider": model_cfg["provider"],
            "returncode": ret,
        })

    # Summary
    print(f"\n{'=' * 60}")
    print("SWEEP COMPLETE")
    print(f"{'=' * 60}")
    for r in results:
        status = (
            "OK"
            if r["returncode"] == 0
            else f"FAILED ({r['returncode']})"
            if r["returncode"]
            else "SKIPPED"
        )
        print(f"  {r['backend']:8} {r['model']:40} {status}")


if __name__ == "__main__":
    main()
