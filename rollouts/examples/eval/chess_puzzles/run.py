#!/usr/bin/env python3
"""Chess puzzle evaluation.

Evaluate LLM chess reasoning on Lichess puzzles.

Usage:
    # Basic eval (10 puzzles, linear search)
    uv run python examples/eval/chess_puzzles/run.py

    # With beam search
    uv run python examples/eval/chess_puzzles/run.py --beam-search

    # Custom settings
    uv run python examples/eval/chess_puzzles/run.py \
        --model openai/gpt-4o \
        --num-puzzles 50 \
        --min-rating 1500 \
        --max-rating 2000 \
        --max-concurrent 8

Requires:
    uv add python-chess httpx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import trio

from rollouts import (
    Actor,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    Trajectory,
    run_agent,
)
from rollouts.datasets.lichess_puzzles import (
    get_puzzle_fen_after_opponent_move,
    load_lichess_puzzles,
)
from rollouts.environments.chess_puzzle import ChessPuzzleEnvironment
from rollouts.search import (
    get_best_terminal,
    make_beam_pruner,
    make_expand_n_turns,
    run_search,
    select_all_frontier,
)

# ──────────────────────── Config ────────────────────────────────────────────


@dataclass(frozen=True)
class ChessPuzzleConfig:
    """Chess puzzle evaluation configuration."""

    # Model
    provider: str = "anthropic"
    model: str = "claude-sonnet-4-20250514"
    api_key: str | None = None

    # Dataset
    num_puzzles: int = 10
    min_rating: int = 1000
    max_rating: int = 1500
    themes: list[str] | None = None  # Filter by theme (e.g., ["fork", "pin"])

    # Eval settings
    max_concurrent: int = 4
    max_turns: int = 10

    # Beam search (if enabled)
    beam_search: bool = False
    beam_width: int = 4
    branch_factor: int = 2
    turns_per_step: int = 2
    max_search_steps: int = 5

    # Output
    output_dir: Path = field(default_factory=lambda: Path("results"))
    run_id: str | None = None

    def __post_init__(self):
        if self.run_id is None:
            object.__setattr__(self, "run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

    @property
    def results_dir(self) -> Path:
        return self.output_dir / f"chess_puzzles_{self.run_id}"


# ──────────────────────── Eval Logic ────────────────────────────────────────


async def noop_chunk_handler(chunk):
    """No-op chunk handler."""
    pass


def make_initial_state(puzzle: dict, config: ChessPuzzleConfig) -> AgentState:
    """Create initial agent state for a puzzle."""
    fen = get_puzzle_fen_after_opponent_move(puzzle)

    env = ChessPuzzleEnvironment(
        fen=fen,
        solution=puzzle["solution"],
        move_format="uci",
        render_mode="ascii",
    )

    system_msg = Message(
        role="system",
        content="""You are a chess puzzle solver. Find the best move in this position.

Use the make_move tool to explore candidate moves and see the resulting positions.
When you've found the best move, use submit_answer to submit it.

Think about tactics: forks, pins, skewers, discovered attacks, and checkmate patterns.
Look for forcing moves (checks, captures, threats) first.""",
    )

    user_msg = Message(
        role="user",
        content=f"""Find the best move in this position:

{env.board}

FEN: {fen}

This is a rated {puzzle["rating"]} puzzle with themes: {", ".join(puzzle.get("themes", []))}.""",
    )

    endpoint = Endpoint(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
    )

    actor = Actor(
        trajectory=Trajectory(messages=[system_msg, user_msg]),
        endpoint=endpoint,
        tools=env.get_tools(),
    )

    return AgentState(actor=actor, environment=env)


async def eval_puzzle_linear(
    puzzle: dict,
    config: ChessPuzzleConfig,
) -> dict:
    """Evaluate a single puzzle using linear agent loop."""
    initial_state = make_initial_state(puzzle, config)

    run_config = RunConfig(
        on_chunk=noop_chunk_handler,
        show_progress=False,
    )

    try:
        states = await run_agent(initial_state, run_config)
        final_state = states[-1]

        env = final_state.environment
        submitted = getattr(env, "submitted_answer", None)
        expected = puzzle["solution"][0] if puzzle["solution"] else None

        # Check if correct
        correct = False
        if submitted and expected:
            # Normalize UCI moves for comparison
            correct = submitted.lower().replace(" ", "") == expected.lower().replace(" ", "")

        return {
            "puzzle_id": puzzle["id"],
            "rating": puzzle["rating"],
            "themes": puzzle.get("themes", []),
            "expected": expected,
            "submitted": submitted,
            "correct": correct,
            "turns": final_state.turn_idx,
            "stop_reason": str(final_state.stop) if final_state.stop else None,
            "error": None,
        }
    except Exception as e:
        return {
            "puzzle_id": puzzle["id"],
            "rating": puzzle["rating"],
            "themes": puzzle.get("themes", []),
            "expected": puzzle["solution"][0] if puzzle["solution"] else None,
            "submitted": None,
            "correct": False,
            "turns": 0,
            "stop_reason": None,
            "error": str(e),
        }


async def eval_puzzle_beam(
    puzzle: dict,
    config: ChessPuzzleConfig,
) -> dict:
    """Evaluate a single puzzle using beam search."""
    initial_state = make_initial_state(puzzle, config)

    run_config = RunConfig(
        on_chunk=noop_chunk_handler,
        show_progress=False,
    )

    try:
        tree = await run_search(
            initial_state=initial_state,
            config=run_config,
            select=select_all_frontier,
            expand=make_expand_n_turns(
                n=config.turns_per_step,
                branch_factor=config.branch_factor,
            ),
            prune=make_beam_pruner(beam_width=config.beam_width),
            max_steps=config.max_search_steps,
        )

        best = get_best_terminal(tree)
        expected = puzzle["solution"][0] if puzzle["solution"] else None

        if best:
            env = best.state.environment
            submitted = getattr(env, "submitted_answer", None)

            correct = False
            if submitted and expected:
                correct = submitted.lower().replace(" ", "") == expected.lower().replace(" ", "")

            return {
                "puzzle_id": puzzle["id"],
                "rating": puzzle["rating"],
                "themes": puzzle.get("themes", []),
                "expected": expected,
                "submitted": submitted,
                "correct": correct,
                "turns": best.depth,
                "tree_nodes": len(tree.nodes),
                "stop_reason": str(best.state.stop) if best.state.stop else None,
                "error": None,
            }
        else:
            return {
                "puzzle_id": puzzle["id"],
                "rating": puzzle["rating"],
                "themes": puzzle.get("themes", []),
                "expected": expected,
                "submitted": None,
                "correct": False,
                "turns": 0,
                "tree_nodes": len(tree.nodes),
                "stop_reason": None,
                "error": "No terminal node found",
            }
    except Exception as e:
        return {
            "puzzle_id": puzzle["id"],
            "rating": puzzle["rating"],
            "themes": puzzle.get("themes", []),
            "expected": puzzle["solution"][0] if puzzle["solution"] else None,
            "submitted": None,
            "correct": False,
            "turns": 0,
            "tree_nodes": 0,
            "stop_reason": None,
            "error": str(e),
        }


async def run_eval(config: ChessPuzzleConfig) -> dict:
    """Run full evaluation."""
    print(
        f"Loading {config.num_puzzles} puzzles (rating {config.min_rating}-{config.max_rating})..."
    )

    puzzles = await load_lichess_puzzles(
        num_puzzles=config.num_puzzles,
        min_rating=config.min_rating,
        max_rating=config.max_rating,
    )

    if not puzzles:
        print("No puzzles found!")
        return {"error": "No puzzles loaded"}

    print(f"Loaded {len(puzzles)} puzzles")

    # Setup output
    config.results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = asdict(config)
    config_dict["output_dir"] = str(config.output_dir)
    config_dict["results_dir"] = str(config.results_dir)
    with open(config.results_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    # Choose eval function
    eval_fn = eval_puzzle_beam if config.beam_search else eval_puzzle_linear

    # Run evaluations
    results = []
    results_file = config.results_dir / "results.jsonl"

    send_channel, receive_channel = trio.open_memory_channel[dict](config.max_concurrent)

    async def producer():
        async with send_channel:
            for puzzle in puzzles:
                await send_channel.send(puzzle)

    async def consumer(worker_id: int):
        async for puzzle in receive_channel:
            result = await eval_fn(puzzle, config)
            results.append(result)

            # Stream results to file
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")

            # Progress
            status = "✓" if result["correct"] else "✗"
            print(
                f"[{len(results)}/{len(puzzles)}] {status} "
                f"Puzzle {result['puzzle_id']} (rating {result['rating']}): "
                f"expected={result['expected']}, got={result['submitted']}"
            )

    print(f"\nRunning eval ({'beam search' if config.beam_search else 'linear'})...")
    print("=" * 60)

    async with trio.open_nursery() as nursery:
        nursery.start_soon(producer)
        for i in range(config.max_concurrent):
            nursery.start_soon(consumer, i)

    # Compute metrics
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0

    avg_turns = sum(r["turns"] for r in results) / total if total > 0 else 0

    # Accuracy by rating bucket
    buckets = {}
    for r in results:
        bucket = (r["rating"] // 200) * 200
        if bucket not in buckets:
            buckets[bucket] = {"correct": 0, "total": 0}
        buckets[bucket]["total"] += 1
        if r["correct"]:
            buckets[bucket]["correct"] += 1

    accuracy_by_rating = {
        f"{k}-{k + 199}": v["correct"] / v["total"] if v["total"] > 0 else 0
        for k, v in sorted(buckets.items())
    }

    metrics = {
        "total_puzzles": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_turns": avg_turns,
        "accuracy_by_rating": accuracy_by_rating,
        "errors": sum(1 for r in results if r.get("error")),
    }

    if config.beam_search:
        metrics["avg_tree_nodes"] = (
            sum(r.get("tree_nodes", 0) for r in results) / total if total > 0 else 0
        )

    # Save metrics
    with open(config.results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("=" * 60)
    print(f"\nResults saved to: {config.results_dir}")
    print("\nMetrics:")
    print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Avg turns: {avg_turns:.1f}")
    if accuracy_by_rating:
        print("  By rating:")
        for bucket, acc in accuracy_by_rating.items():
            print(f"    {bucket}: {acc:.1%}")

    return metrics


# ──────────────────────── CLI ───────────────────────────────────────────────


def parse_args() -> ChessPuzzleConfig:
    parser = argparse.ArgumentParser(description="Chess puzzle evaluation")

    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="Model in provider/model format (default: anthropic/claude-sonnet-4-20250514)",
    )
    parser.add_argument(
        "--num-puzzles",
        type=int,
        default=10,
        help="Number of puzzles to evaluate (default: 10)",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1000,
        help="Minimum puzzle rating (default: 1000)",
    )
    parser.add_argument(
        "--max-rating",
        type=int,
        default=1500,
        help="Maximum puzzle rating (default: 1500)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Max concurrent evaluations (default: 4)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Max agent turns per puzzle (default: 10)",
    )

    # Beam search options
    parser.add_argument(
        "--beam-search",
        action="store_true",
        help="Use beam search instead of linear agent loop",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=4,
        help="Beam width for tree search (default: 4)",
    )
    parser.add_argument(
        "--branch-factor",
        type=int,
        default=2,
        help="Branch factor per expansion (default: 2)",
    )

    args = parser.parse_args()

    # Parse provider/model
    if "/" in args.model:
        provider, model = args.model.split("/", 1)
    else:
        provider, model = "anthropic", args.model

    return ChessPuzzleConfig(
        provider=provider,
        model=model,
        num_puzzles=args.num_puzzles,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
        max_concurrent=args.max_concurrent,
        max_turns=args.max_turns,
        beam_search=args.beam_search,
        beam_width=args.beam_width,
        branch_factor=args.branch_factor,
    )


async def main():
    config = parse_args()
    await run_eval(config)


if __name__ == "__main__":
    trio.run(main)
