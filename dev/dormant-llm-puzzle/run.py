#!/usr/bin/env python3
"""Run experiments from the command line.

Usage:
    python run.py chat "Hello, how are you?" --model dormant-model-1
    python run.py experiment exp-001
    python run.py list-experiments
    python run.py list-results
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import chat, batch_chat, run_experiment, load_experiment, list_experiments, list_results


async def cmd_chat(args):
    """Send a chat message."""
    response = await chat(
        [{"role": "user", "content": args.prompt}],
        model=args.model,
    )
    print(f"\n{args.model}:")
    print(response)


async def cmd_experiment(args):
    """Run an experiment."""
    config = load_experiment(args.experiment_id)
    print(f"Running experiment: {config['id']}")
    print(f"Hypothesis: {config['hypothesis']}")
    print(f"Model: {config['model']}")
    print(f"Prompts: {len(config['prompts'])}")
    print()

    result = await run_experiment(config)

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)

    if isinstance(result, list):
        for r in result:
            print(f"\n{r.model}:")
            for prompt, response in zip(r.prompts, r.responses):
                print(f"  Q: {prompt[:80]}...")
                print(f"  A: {response[:200]}...")
    else:
        for prompt, response in zip(result.prompts, result.responses):
            print(f"\nQ: {prompt[:80]}...")
            print(f"A: {response[:200]}...")


def cmd_list_experiments(_args):
    """List available experiments."""
    experiments = list_experiments()
    if experiments:
        print("Available experiments:")
        for exp in sorted(experiments):
            print(f"  {exp}")
    else:
        print("No experiments defined yet.")
        print("Create experiments in /experiments/*.json")


def cmd_list_results(_args):
    """List experiment results."""
    results = list_results()
    if results:
        print("Experiment results:")
        for res in sorted(results):
            print(f"  {res}")
    else:
        print("No results yet.")


def main():
    parser = argparse.ArgumentParser(description="Dormant LLM Puzzle runner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Send a chat message")
    chat_parser.add_argument("prompt", help="User message")
    chat_parser.add_argument("--model", default="dormant-model-1", help="Model name")

    # experiment command
    exp_parser = subparsers.add_parser("experiment", help="Run an experiment")
    exp_parser.add_argument("experiment_id", help="Experiment ID")

    # list commands
    subparsers.add_parser("list-experiments", help="List available experiments")
    subparsers.add_parser("list-results", help="List experiment results")

    args = parser.parse_args()

    if args.command == "chat":
        asyncio.run(cmd_chat(args))
    elif args.command == "experiment":
        asyncio.run(cmd_experiment(args))
    elif args.command == "list-experiments":
        cmd_list_experiments(args)
    elif args.command == "list-results":
        cmd_list_results(args)


if __name__ == "__main__":
    main()
