"""CLI entry point for rlmon.

Usage:
    rlmon results/rl/grpo_20250127/     # Watch specific run
    rlmon --latest                       # Watch most recent run in results/rl/
    rlmon --latest results/sft/          # Most recent in a custom dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .app import make_app


def find_latest_run(base_dir: str = "results/rl") -> Path | None:
    """Find the most recent run directory (by mtime)."""
    base = Path(base_dir)
    if not base.is_dir():
        return None

    dirs = [d for d in base.iterdir() if d.is_dir()]
    if not dirs:
        return None

    # Sort by modification time, newest first
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="btop-style RL training monitor",
    )
    parser.add_argument(
        "output_dir",
        nargs="?",
        help="Path to training output directory (contains metrics.jsonl, config.json, etc.)",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Watch the most recent run directory",
    )
    args = parser.parse_args()

    if args.latest:
        # Use output_dir as base search path, or default to results/rl
        base = args.output_dir or "results/rl"
        latest = find_latest_run(base)
        if latest is None:
            print(f"No run directories found in {base}", file=sys.stderr)
            sys.exit(1)
        output_dir = latest
        print(f"Watching: {output_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)

    assert output_dir is not None
    if not output_dir.is_dir():
        print(f"Error: {output_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    app = make_app(str(output_dir))
    app.run()


if __name__ == "__main__":
    main()
