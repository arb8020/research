"""Dataset transformations for rollouts.

Each dataset module provides a transformation function that converts
dataset rows into initial Trajectory objects for agent execution.
"""

from .lichess_puzzles import (
    MATE_THEMES,
    TACTICAL_THEMES,
    get_puzzle_fen_after_opponent_move,
    load_lichess_puzzles,
    load_lichess_puzzles_sync,
)

__all__ = [
    "load_lichess_puzzles",
    "load_lichess_puzzles_sync",
    "get_puzzle_fen_after_opponent_move",
    "TACTICAL_THEMES",
    "MATE_THEMES",
]
