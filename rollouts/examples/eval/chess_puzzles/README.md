# Chess Puzzle Evaluation

Evaluate LLM chess reasoning using Lichess puzzles with optional tree search.

## Files

- `run.py` - Full evaluation harness with metrics and output
- `beam_search_demo.py` - Minimal single-puzzle demo of tree search

## Overview

- **Dataset**: [Lichess puzzle database](https://database.lichess.org/#puzzles) (~4M puzzles)
- **Task**: Find the best move in a tactical position
- **Modes**: Linear (single trajectory) or beam search (tree exploration)
- **Scoring**: Binary (correct/incorrect) + optional Stockfish centipawn evaluation

## Quick Start

```bash
# Run eval on 10 puzzles with Claude
uv run python examples/eval/chess_puzzles/run.py

# Run with specific model
uv run python examples/eval/chess_puzzles/run.py --model openai/gpt-4o

# Run with beam search
uv run python examples/eval/chess_puzzles/run.py --beam-search

# More puzzles, specific difficulty
uv run python examples/eval/chess_puzzles/run.py --num-puzzles 50 --min-rating 1500 --max-rating 2000
```

## Requirements

```bash
uv add python-chess httpx
# Optional for position evaluation:
uv add stockfish  # Also need stockfish binary installed
```

## Puzzle Format

Each puzzle has:
- **FEN**: Starting position (after opponent's last move)
- **Solution**: Sequence of UCI moves (first move is the puzzle answer)
- **Rating**: Puzzle difficulty (1000-2500+)
- **Themes**: Tactical motifs (fork, pin, mate, etc.)

Example:
```python
{
    "id": "00008",
    "fen": "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
    "solution": ["f2g3", "e6e7", "b2b1", "b3c1", "b1c1", "h6c1"],
    "rating": 1853,
    "themes": ["crushing", "hangingPiece", "long", "middlegame"],
}
```

## Evaluation Modes

### Linear (Default)
Standard agent loop - model explores one trajectory:
```bash
uv run python examples/eval/chess_puzzles/run.py
```

### Beam Search
Tree search with branching and pruning:
```bash
uv run python examples/eval/chess_puzzles/run.py --beam-search --beam-width 4 --branch-factor 2
```

## Output

Results saved to `results/chess_puzzles_<timestamp>/`:
- `config.json` - Experiment configuration
- `results.jsonl` - Per-puzzle results
- `metrics.json` - Aggregate metrics

## Metrics

- **accuracy**: Fraction of puzzles solved correctly
- **avg_turns**: Average agent turns per puzzle
- **solve_by_rating**: Accuracy bucketed by puzzle rating

## How Puzzles Work

1. Opponent makes a move (included in FEN setup)
2. Agent must find the best response
3. For multi-move puzzles, only first move is evaluated

The agent has two tools:
- `make_move(move)`: Play a move (for exploration)
- `submit_answer(move, reasoning)`: Submit final answer

## Data Source

Puzzles from https://database.lichess.org/#puzzles (CC0 license).

First run downloads and caches the database (~300MB compressed → ~2GB uncompressed).
Cached at `~/.cache/rollouts/lichess/`.

## Notes

- About 6% of Lichess games include Stockfish evaluations
- Puzzle ratings use Glicko-2 (like player ratings)
- Themes help filter for specific tactics (fork, pin, mate, etc.)
