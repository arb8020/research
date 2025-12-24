#!/usr/bin/env python3
"""Unit tests for ChessPuzzleEnvironment.

Minimal tests for the chess puzzle environment.
For full evaluation, see examples/eval/chess_puzzles/

Requires: python-chess (uv add python-chess)
"""

import pytest
import trio

from rollouts.dtypes import ToolCall

# Skip all tests if python-chess not available
pytest.importorskip("chess")

from rollouts.environments.chess_puzzle import ChessPuzzleEnvironment

# Scholar's mate position - white to move, Qxf7# wins
SCHOLARS_MATE_FEN = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"
SCHOLARS_MATE_SOLUTION = ["h5f7"]  # Qxf7#


async def test_make_move_legal():
    """make_move accepts legal moves."""
    env = ChessPuzzleEnvironment(fen=SCHOLARS_MATE_FEN, solution=SCHOLARS_MATE_SOLUTION)
    result = await env.exec_tool(
        ToolCall(id="1", name="make_move", args={"move": "h5f7"}),
        current_state=None,
        run_config=None,
    )
    assert not result.is_error
    assert len(env.moves_made) == 1
    print("✓ make_move accepts legal moves")


async def test_make_move_illegal():
    """make_move rejects illegal moves."""
    env = ChessPuzzleEnvironment(fen=SCHOLARS_MATE_FEN, solution=SCHOLARS_MATE_SOLUTION)
    result = await env.exec_tool(
        ToolCall(id="1", name="make_move", args={"move": "e1e8"}),
        current_state=None,
        run_config=None,
    )
    assert result.is_error
    print("✓ make_move rejects illegal moves")


async def test_submit_answer():
    """submit_answer works correctly."""
    env = ChessPuzzleEnvironment(fen=SCHOLARS_MATE_FEN, solution=SCHOLARS_MATE_SOLUTION)

    # Correct answer
    result = await env.exec_tool(
        ToolCall(id="1", name="submit_answer", args={"move": "h5f7"}),
        current_state=None,
        run_config=None,
    )
    assert not result.is_error
    assert "correct" in result.content.lower()
    print("✓ submit_answer works")


async def test_serialize_deserialize():
    """Environment can be serialized and deserialized."""
    env = ChessPuzzleEnvironment(fen=SCHOLARS_MATE_FEN, solution=SCHOLARS_MATE_SOLUTION)

    # Make a move (h5g5 is legal from starting position)
    result = await env.exec_tool(
        ToolCall(id="1", name="make_move", args={"move": "h5g5"}),
        current_state=None,
        run_config=None,
    )
    assert not result.is_error, f"Move failed: {result.error}"

    # Serialize and deserialize
    data = await env.serialize()
    env2 = await ChessPuzzleEnvironment.deserialize(data)

    assert env2.moves_made == ["h5g5"]
    assert env2.board.fen() == env.board.fen()

    # Verify they're independent (for tree search branching)
    # After h5g5, it's black's turn. f8e7 is a legal response.
    result2 = await env2.exec_tool(
        ToolCall(id="2", name="make_move", args={"move": "f8e7"}),
        current_state=None,
        run_config=None,
    )
    assert not result2.is_error, f"Move failed: {result2.error}"

    assert len(env.moves_made) == 1  # Original unchanged
    assert len(env2.moves_made) == 2  # Fork has new move
    print("✓ serialize/deserialize works")


if __name__ == "__main__":

    async def main():
        print("\n" + "=" * 50)
        print("Chess Puzzle Environment Tests")
        print("=" * 50 + "\n")

        await test_make_move_legal()
        await test_make_move_illegal()
        await test_submit_answer()
        await test_serialize_deserialize()

        print("\n" + "=" * 50)
        print("✅ All tests passed!")
        print("=" * 50 + "\n")

    trio.run(main)
