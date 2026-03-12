"""Evaluate multi-turn KernelBench environment with API models or SGLang.

This script evaluates the KernelBenchMultiTurnEnvironment before starting RL training.
It helps derisk the environment setup by testing with capable API models first.

Usage:
    # Evaluate with Kimi K2.5 from OpenCode (recommended for testing)
    python examples/rl/kernelbench/eval_multi_turn.py --model kimi-k2.5 --num-problems 5

    # Evaluate with Claude via OpenCode
    python examples/rl/kernelbench/eval_multi_turn.py --model claude-sonnet-4-6 --num-problems 3

    # Evaluate with local SGLang endpoint (after starting server)
    python examples/rl/kernelbench/eval_multi_turn.py --model sglang --endpoint http://localhost:30000/v1 --num-problems 5

    # Evaluate with specific levels
    python examples/rl/kernelbench/eval_multi_turn.py --model kimi-k2.5 --levels 1 2 --num-problems 10

Environment variables:
    OPENCODE_API_KEY: Required for OpenCode models (get from https://opencode.ai/zen)
    MOONSHOT_API_KEY: Alternative for direct Moonshot/Kimi access
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Add rollouts to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from examples.rl.kernelbench.dataset import load_kernelbench_prompts
from rollouts.agents import AgentState, RunConfig
from rollouts.core import Endpoint
from rollouts.environments.kernelbench_multi import KernelBenchMultiTurnEnvironment
from rollouts.models import MODELS
from rollouts.rollout import run_agent


async def evaluate_single_problem(
    problem: dict[str, Any],
    endpoint: Endpoint,
    max_turns: int = 8,
    verbose: bool = True,
) -> dict[str, Any]:
    """Evaluate a single problem with the multi-turn environment.

    Args:
        problem: Problem dict with messages, ref_code, etc.
        endpoint: API endpoint configuration
        max_turns: Maximum turns allowed
        verbose: Print detailed output

    Returns:
        Results dict with metrics
    """
    problem_id = problem.get("problem_id", "unknown")
    problem_name = problem.get("name", "unknown")
    level = problem.get("level", "unknown")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Problem {problem_id}: {problem_name} (Level {level})")
        print(f"{'=' * 60}")

    # Create environment
    env = KernelBenchMultiTurnEnvironment(
        ref_code=problem.get("ref_code", ""),
        backend=problem.get("backend", "cuda"),
        max_turns=max_turns,
    )

    # Prepare initial messages
    messages = problem.get("messages", [])
    if not messages:
        print(f"Warning: No messages for problem {problem_id}")
        return {
            "problem_id": problem_id,
            "name": problem_name,
            "level": level,
            "error": "No messages",
        }

    # Create initial state
    from dataclasses import dataclass

    from rollouts.core import Trajectory

    @dataclass
    class SimpleActor:
        trajectory: Trajectory

    initial_trajectory = Trajectory(messages=messages)
    initial_state = AgentState(
        actor=SimpleActor(trajectory=initial_trajectory),
        stop=None,
        metadata={},
    )

    # Run agent
    run_config = RunConfig(
        max_turns=max_turns,
        verbose=verbose,
    )

    try:
        final_state = await run_agent(
            initial_state=initial_state,
            endpoint=endpoint,
            environment=env,
            run_config=run_config,
        )

        # Extract results
        metadata = final_state.metadata
        turn_history = metadata.get("turn_history", [])

        results = {
            "problem_id": problem_id,
            "name": problem_name,
            "level": level,
            "turns_used": metadata.get("turns_used", len(turn_history)),
            "best_speedup": metadata.get("best_speedup", 0.0),
            "has_correct_kernel": metadata.get("has_correct_kernel", False),
            "turn_history": turn_history,
            "stop_reason": str(final_state.stop) if final_state.stop else "unknown",
        }

        if verbose:
            print("\nResults:")
            print(f"  Turns used: {results['turns_used']}")
            print(f"  Best speedup: {results['best_speedup']:.2f}x")
            print(f"  Has correct kernel: {results['has_correct_kernel']}")
            print(f"  Stop reason: {results['stop_reason']}")

            # Show per-turn breakdown
            if turn_history:
                print("\n  Turn breakdown:")
                for turn in turn_history:
                    status = "✓" if turn.get("correct") else ("c" if turn.get("compiled") else "✗")
                    speedup = turn.get("speedup", 0.0)
                    print(f"    Turn {turn['turn']}: {status} speedup={speedup:.2f}x")

        return results

    except Exception as e:
        print(f"Error evaluating problem {problem_id}: {e}")
        import traceback

        traceback.print_exc()
        return {
            "problem_id": problem_id,
            "name": problem_name,
            "level": level,
            "error": str(e),
        }


async def evaluate_multi_turn(
    model: str = "kimi-k2.5",
    provider: str = "opencode",
    endpoint_url: str | None = None,
    num_problems: int = 5,
    levels: list[int] | None = None,
    max_turns: int = 8,
    backend: str = "cuda",
    output_file: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run multi-turn evaluation on KernelBench problems.

    Args:
        model: Model identifier (e.g., "kimi-k2.5", "claude-sonnet-4-6")
        provider: Provider name (e.g., "opencode", "moonshot", "sglang")
        endpoint_url: Custom endpoint URL (for SGLang/vLLM)
        num_problems: Number of problems to evaluate
        levels: KernelBench levels to use (default: [1])
        max_turns: Maximum turns per problem
        backend: Kernel backend ("cuda" or "hip")
        output_file: Optional file to save results
        verbose: Print detailed output

    Returns:
        Summary dict with aggregate metrics
    """
    if levels is None:
        levels = [1]

    # Load problems
    print(f"Loading {num_problems} problems from levels {levels}...")
    problems = load_kernelbench_prompts(
        levels=levels,
        max_samples=num_problems,
        backend=backend,
    )
    print(f"Loaded {len(problems)} problems")

    # Setup endpoint
    if provider == "sglang" or endpoint_url:
        # Custom SGLang/vLLM endpoint
        if not endpoint_url:
            endpoint_url = "http://localhost:30000/v1"
        endpoint = Endpoint(
            provider="openai",  # SGLang uses OpenAI-compatible API
            model=model,
            api_base=endpoint_url,
            api_key="dummy",  # Not needed for local SGLang
            temperature=0.7,
            max_tokens=4096,
        )
        print(f"Using SGLang endpoint: {endpoint_url}")
    else:
        # API model from registry
        if provider not in MODELS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(MODELS.keys())}")

        if model not in MODELS[provider]:
            available = list(MODELS[provider].keys())
            raise ValueError(
                f"Unknown model: {model} for provider {provider}. Available: {available}"
            )

        model_meta = MODELS[provider][model]

        # Get API key
        api_key = None
        if provider == "opencode":
            api_key = os.environ.get("OPENCODE_API_KEY")
            if not api_key:
                print("Warning: OPENCODE_API_KEY not set. Get one at https://opencode.ai/zen")
        elif provider == "moonshot":
            api_key = os.environ.get("MOONSHOT_API_KEY")

        if not api_key:
            # Try to get from credentials store
            from rollouts.credentials import get_credential

            api_key = get_credential(provider)

        if not api_key:
            raise ValueError(
                f"No API key found for {provider}. Set {provider.upper()}_API_KEY env var."
            )

        endpoint = Endpoint(
            provider=model_meta.api.replace("-completions", "").replace("-messages", ""),
            model=model_meta.id,
            api_base=model_meta.base_url,
            api_key=api_key,
            temperature=0.7,
            max_tokens=min(4096, model_meta.max_tokens),
        )
        print(f"Using {model_meta.name} via {provider}")
        print(f"  API: {model_meta.api}")
        print(f"  Base URL: {model_meta.base_url}")
        print(f"  Context window: {model_meta.context_window}")
        print(f"  Max tokens: {endpoint.max_tokens}")

    print(f"\nEvaluating with max_turns={max_turns}...")
    print(f"Problems: {len(problems)}")

    # Evaluate each problem
    results = []
    for i, problem in enumerate(problems):
        print(f"\n[{i + 1}/{len(problems)}] ", end="")
        result = await evaluate_single_problem(
            problem=problem,
            endpoint=endpoint,
            max_turns=max_turns,
            verbose=verbose,
        )
        results.append(result)

    # Compute summary
    successful = [r for r in results if "error" not in r]
    correct = [r for r in successful if r.get("has_correct_kernel")]
    speedups = [r.get("best_speedup", 0.0) for r in correct]

    summary = {
        "total": len(results),
        "successful": len(successful),
        "correct": len(correct),
        "correct_rate": len(correct) / len(successful) if successful else 0.0,
        "avg_speedup": sum(speedups) / len(speedups) if speedups else 0.0,
        "max_speedup": max(speedups) if speedups else 0.0,
        "avg_turns": sum(r.get("turns_used", 0) for r in successful) / len(successful)
        if successful
        else 0.0,
        "results": results,
    }

    # Print summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total problems: {summary['total']}")
    print(f"Successful: {summary['successful']}")
    print(f"Correct kernels: {summary['correct']} ({summary['correct_rate']:.1%})")
    print(f"Average speedup (correct only): {summary['avg_speedup']:.2f}x")
    print(f"Max speedup: {summary['max_speedup']:.2f}x")
    print(f"Average turns used: {summary['avg_turns']:.1f}")

    # Save results
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate KernelBench multi-turn environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Kimi K2.5 via OpenCode (recommended)
    python eval_multi_turn.py --model kimi-k2.5 --provider opencode --num-problems 5
    
    # Claude via OpenCode
    python eval_multi_turn.py --model claude-sonnet-4-6 --provider opencode --num-problems 3
    
    # Direct Moonshot/Kimi
    python eval_multi_turn.py --model kimi-k2.5 --provider moonshot --num-problems 5
    
    # Local SGLang (start server first: sglang launch --model ...)
    python eval_multi_turn.py --model Qwen/Qwen2.5-Coder-3B-Instruct --provider sglang --endpoint http://localhost:30000/v1
    
    # Multiple levels
    python eval_multi_turn.py --model kimi-k2.5 --levels 1 2 --num-problems 10
        """,
    )

    parser.add_argument(
        "--model", type=str, default="kimi-k2.5", help="Model identifier (default: kimi-k2.5)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="opencode",
        choices=["opencode", "moonshot", "sglang", "openai", "anthropic"],
        help="API provider (default: opencode)",
    )
    parser.add_argument(
        "--endpoint", type=str, default=None, help="Custom endpoint URL (for SGLang/vLLM)"
    )
    parser.add_argument(
        "--num-problems", type=int, default=5, help="Number of problems to evaluate (default: 5)"
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="+",
        default=[1],
        help="KernelBench levels to use (default: [1])",
    )
    parser.add_argument(
        "--max-turns", type=int, default=8, help="Maximum turns per problem (default: 8)"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cuda",
        choices=["cuda", "hip"],
        help="Kernel backend (default: cuda)",
    )
    parser.add_argument("--output", type=str, default=None, help="Output file for results (JSON)")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")

    args = parser.parse_args()

    # Run evaluation
    try:
        summary = asyncio.run(
            evaluate_multi_turn(
                model=args.model,
                provider=args.provider,
                endpoint_url=args.endpoint,
                num_problems=args.num_problems,
                levels=args.levels,
                max_turns=args.max_turns,
                backend=args.backend,
                output_file=args.output,
                verbose=not args.quiet,
            )
        )

        # Exit with success if we got some correct kernels
        if summary.get("correct", 0) > 0:
            print("\n✓ Evaluation successful - environment is working!")
            return 0
        else:
            print("\n⚠ No correct kernels produced - check environment setup")
            return 1

    except Exception as e:
        print(f"\n✗ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
