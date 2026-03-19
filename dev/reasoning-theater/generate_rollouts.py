"""Normalize raw multiple-choice JSONL into reasoning-theater eval tasks."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_tasks_from_jsonl, write_tasks_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw JSONL task file")
    parser.add_argument("--output", type=Path, required=True, help="Normalized JSONL output")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_tasks_from_jsonl(args.input, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_tasks_jsonl(tasks, args.output)
    print(f"Wrote {len(tasks)} normalized tasks to {args.output}")


if __name__ == "__main__":
    main()
