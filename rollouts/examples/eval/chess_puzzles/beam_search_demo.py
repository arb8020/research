"""Demo: Beam search on a single chess puzzle.

A minimal example showing tree search on chess puzzles.
For full evaluation, use run.py instead.

Usage:
    uv run python examples/eval/chess_puzzles/beam_search_demo.py

Requires:
    uv add python-chess httpx
"""

import trio

from rollouts.datasets.lichess_puzzles import (
    get_puzzle_fen_after_opponent_move,
    load_lichess_puzzles,
)
from rollouts.dtypes import Actor, AgentState, Endpoint, Message, RunConfig, Trajectory
from rollouts.environments.chess_puzzle import ChessPuzzleEnvironment
from rollouts.search import (
    get_best_terminal,
    get_terminal_nodes,
    make_beam_pruner,
    make_expand_n_turns,
    run_search,
    select_all_frontier,
)


async def main():
    # Load a few puzzles (easy ones for testing)
    print("Loading puzzles...", flush=True)
    puzzles = await load_lichess_puzzles(
        num_puzzles=3,
        min_rating=1000,
        max_rating=1200,
    )
    print(f"Loaded {len(puzzles)} puzzles", flush=True)

    if not puzzles:
        print("No puzzles found. Make sure you have internet connection for first download.")
        return

    # Test with first puzzle
    puzzle = puzzles[0]
    print(f"\nPuzzle: {puzzle['id']}")
    print(f"Rating: {puzzle['rating']}")
    print(f"Themes: {puzzle['themes']}")
    print(f"Solution: {puzzle['solution']}")

    # Get position after opponent's move
    fen = get_puzzle_fen_after_opponent_move(puzzle)
    print(f"FEN: {fen}")

    # Create environment
    env = ChessPuzzleEnvironment(
        fen=fen,
        solution=puzzle["solution"],
        move_format="uci",
        render_mode="ascii",
    )

    print(f"\nStarting position:\n{env.board}")

    # Create initial state
    system_msg = Message(
        role="system",
        content="""You are a chess puzzle solver. Find the best move in this position.

Use the make_move tool to explore moves, then submit_answer when you've found the best one.
Think about tactics: forks, pins, discovered attacks, checkmate patterns.""",
    )
    user_msg = Message(
        role="user",
        content=f"Find the best move in this position:\n\n{env.board}\n\nFEN: {fen}",
    )

    trajectory = Trajectory(messages=[system_msg, user_msg])

    # Use a simple model for testing
    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    endpoint = Endpoint(provider="anthropic", model="claude-sonnet-4-20250514", api_key=api_key)

    actor = Actor(
        trajectory=trajectory,
        endpoint=endpoint,
        tools=env.get_tools(),
    )

    initial_state = AgentState(
        actor=actor,
        environment=env,
    )

    # Define value function for chess puzzles
    async def chess_value_fn(state: AgentState) -> float:
        """Value function: use environment's position evaluation."""
        if state.environment and hasattr(state.environment, "get_value"):
            return await state.environment.get_value()
        return 0.0

    # Run beam search
    print("\n" + "=" * 50)
    print("Running beam search...")
    print("  - beam_width=2 (keep top 2)")
    print("  - branch_factor=2 (fork 2x after each step)")
    print("  - n=2 (2 turns per expansion)")
    print("  - max_steps=3")
    print("=" * 50 + "\n")

    config = RunConfig(
        on_chunk=lambda _: trio.lowlevel.checkpoint(),
        show_progress=False,
    )

    tree = await run_search(
        initial_state,
        config,
        select=select_all_frontier,
        expand=make_expand_n_turns(n=2, branch_factor=2),
        value_fn=chess_value_fn,
        prune=make_beam_pruner(beam_width=2),
        max_steps=3,
    )

    # Report results
    print("\nSearch complete!")
    print(f"Total nodes: {len(tree.nodes)}")
    print(f"Frontier size: {len(tree.frontier)}")

    terminals = get_terminal_nodes(tree)
    print(f"Terminal nodes: {len(terminals)}")

    best = get_best_terminal(tree)
    if best:
        print("\nBest terminal node:")
        print(f"  Value: {best.value}")
        print(f"  Depth: {best.depth}")

        # Get the submitted answer from environment
        best_env = best.state.environment
        if hasattr(best_env, "submitted_answer"):
            print(f"  Submitted answer: {best_env.submitted_answer}")
            print(f"  Expected: {puzzle['solution'][0] if puzzle['solution'] else 'N/A'}")
    else:
        print("\nNo terminal nodes found (agent didn't submit an answer)")

        # Show frontier nodes
        print("\nFrontier nodes:")
        for node_id in tree.frontier[:3]:
            node = next(n for n in tree.nodes if n.node_id == node_id)
            print(f"  {node_id}: value={node.value}, depth={node.depth}")


if __name__ == "__main__":
    trio.run(main)
