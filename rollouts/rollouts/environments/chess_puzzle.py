"""Chess puzzle environment for testing tree search.

Requires: uv add python-chess stockfish

The environment presents a chess position and asks the model to find the best move.
Scoring is based on Stockfish centipawn evaluation.

Example:
    env = ChessPuzzleEnvironment(
        fen="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        solution=["Qxf7"],  # Scholar's mate
    )
"""

from dataclasses import dataclass, field
from typing import Literal

import trio

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    StopReason,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)
from ..search import VerifyResult

try:
    import chess
    import chess.engine

    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False


@dataclass
class ChessPuzzleEnvironment:
    """Chess puzzle environment with Stockfish evaluation.

    Tools:
        - make_move: Play a move on the board
        - submit_answer: Submit final answer for the puzzle

    The environment tracks moves made and can evaluate positions using Stockfish.
    """

    # Puzzle configuration
    fen: str  # Starting position
    solution: list[str] = field(default_factory=list)  # Expected solution moves (UCI or SAN)

    # State
    board: "chess.Board | None" = field(default=None, repr=False)
    moves_made: list[str] = field(default_factory=list)
    submitted_answer: str | None = None

    # Configuration
    move_format: Literal["uci", "san"] = "uci"
    render_mode: Literal["ascii", "fen", "both"] = "ascii"

    # Stockfish for evaluation (optional)
    stockfish_path: str | None = None
    _engine: "chess.engine.SimpleEngine | None" = field(default=None, repr=False)

    def __post_init__(self):
        if not CHESS_AVAILABLE:
            raise ImportError("python-chess is required: uv add python-chess")

        if self.board is None:
            self.board = chess.Board(self.fen)

    async def serialize(self) -> dict:
        return {
            "fen": self.fen,
            "solution": self.solution,
            "current_fen": self.board.fen() if self.board else self.fen,
            "moves_made": self.moves_made,
            "submitted_answer": self.submitted_answer,
            "move_format": self.move_format,
            "render_mode": self.render_mode,
            "stockfish_path": self.stockfish_path,
        }

    @staticmethod
    async def deserialize(data: dict) -> "ChessPuzzleEnvironment":
        env = ChessPuzzleEnvironment(
            fen=data["fen"],
            solution=data.get("solution", []),
            move_format=data.get("move_format", "uci"),
            render_mode=data.get("render_mode", "ascii"),
            stockfish_path=data.get("stockfish_path"),
        )
        # Restore current position
        if "current_fen" in data:
            env.board = chess.Board(data["current_fen"])
        env.moves_made = data.get("moves_made", [])
        env.submitted_answer = data.get("submitted_answer")
        return env

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="make_move",
                    description=f"Play a move on the board. Use {self.move_format.upper()} notation.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "move": {
                                "type": "string",
                                "description": f"The move in {self.move_format.upper()} notation (e.g., {'e2e4' if self.move_format == 'uci' else 'e4'})",
                            },
                        },
                    ),
                    required=["move"],
                ),
            ),
            Tool(
                type="function",
                function=ToolFunction(
                    name="submit_answer",
                    description="Submit your final answer for the puzzle. This is the best move to play.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "move": {
                                "type": "string",
                                "description": f"Your answer in {self.move_format.upper()} notation",
                            },
                            "reasoning": {
                                "type": "string",
                                "description": "Brief explanation of why this is the best move",
                            },
                        },
                    ),
                    required=["move"],
                ),
            ),
        ]

    def _render_board(self) -> str:
        """Render board according to render_mode."""
        if self.board is None:
            return "No board"

        parts = []
        if self.render_mode in ("ascii", "both"):
            parts.append(str(self.board))
        if self.render_mode in ("fen", "both"):
            parts.append(f"FEN: {self.board.fen()}")
        return "\n".join(parts)

    def _parse_move(self, move_str: str) -> "chess.Move | None":
        """Parse move string to chess.Move."""
        if self.board is None:
            return None

        try:
            if self.move_format == "uci":
                return chess.Move.from_uci(move_str)
            else:  # san
                return self.board.parse_san(move_str)
        except (ValueError, chess.InvalidMoveError):
            return None

    def _format_move(self, move: "chess.Move") -> str:
        """Format move according to move_format."""
        if self.move_format == "uci":
            return move.uci()
        else:  # san
            return self.board.san(move) if self.board else move.uci()

    async def _get_evaluation(self) -> int | None:
        """Get Stockfish centipawn evaluation of current position.

        Returns None if Stockfish is not available.
        Positive = white is better, negative = black is better.
        """
        if self.board is None or not self.stockfish_path:
            return None

        try:
            # Use stockfish package if available
            try:
                from stockfish import Stockfish

                sf = Stockfish(path=self.stockfish_path)
                sf.set_fen_position(self.board.fen())
                evaluation = sf.get_evaluation()
                if evaluation["type"] == "cp":
                    return evaluation["value"]
                elif evaluation["type"] == "mate":
                    # Convert mate to large centipawn value
                    mate_in = evaluation["value"]
                    return 10000 if mate_in > 0 else -10000
            except ImportError:
                pass

            # Fall back to python-chess engine
            transport, engine = await chess.engine.popen_uci(self.stockfish_path)
            info = await engine.analyse(self.board, chess.engine.Limit(time=0.1))
            await engine.quit()

            score = info.get("score")
            if score:
                cp = score.relative.score(mate_score=10000)
                return cp
        except Exception:
            pass

        return None

    async def verify(self, state: AgentState) -> VerifyResult:
        """Verify current state for tree search scoring.

        Returns centipawn evaluation normalized to 0-1 range.
        """
        # Check if puzzle is solved
        if self.submitted_answer:
            is_correct = self._check_solution(self.submitted_answer)
            return VerifyResult(
                score=1.0 if is_correct else 0.0,
                terminal=True,
            )

        # Get position evaluation
        centipawns = await self._get_evaluation()
        if centipawns is None:
            return VerifyResult(score=0.5)  # No evaluation available

        # Normalize centipawns to 0-1 using sigmoid
        # 100 centipawns ≈ 0.73, 300 centipawns ≈ 0.95
        import math

        normalized = 1 / (1 + math.exp(-centipawns / 200))

        # Flip if it's black to move (we want score from perspective of side to move)
        if self.board and not self.board.turn:
            normalized = 1 - normalized

        return VerifyResult(score=normalized)

    def _check_solution(self, move_str: str) -> bool:
        """Check if move matches the puzzle solution."""
        if not self.solution:
            return False

        move = self._parse_move(move_str)
        if move is None:
            return False

        # Check against first expected solution move
        # (puzzles typically have one correct first move)
        for sol in self.solution:
            sol_move = self._parse_move(sol)
            if sol_move and move == sol_move:
                return True

        return False

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No feedback needed for chess environment."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute tool call."""
        try:
            if tool_call.name == "make_move":
                return await self._exec_make_move(tool_call)
            elif tool_call.name == "submit_answer":
                return await self._exec_submit_answer(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except trio.Cancelled:
            # Re-raise cancellation so agent loop can handle it
            raise
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=str(e),
            )

    async def _exec_make_move(self, tool_call: ToolCall) -> ToolResult:
        """Execute make_move tool."""
        if self.board is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No board initialized",
            )

        move_str = tool_call.args.get("move", "")
        move = self._parse_move(move_str)

        if move is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Invalid move format: {move_str}",
            )

        if move not in self.board.legal_moves:
            legal_moves = [self._format_move(m) for m in self.board.legal_moves]
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Illegal move: {move_str}. Legal moves: {', '.join(legal_moves[:10])}{'...' if len(legal_moves) > 10 else ''}",
            )

        # Make the move
        self.board.push(move)
        self.moves_made.append(self._format_move(move))

        # Get evaluation if available
        eval_str = ""
        centipawns = await self._get_evaluation()
        if centipawns is not None:
            eval_str = f"\nEvaluation: {centipawns / 100:+.2f}"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Move played: {move_str}\n\n{self._render_board()}{eval_str}",
        )

    async def _exec_submit_answer(self, tool_call: ToolCall) -> ToolResult:
        """Execute submit_answer tool."""
        move_str = tool_call.args.get("move", "")
        reasoning = tool_call.args.get("reasoning", "")

        self.submitted_answer = move_str

        is_correct = self._check_solution(move_str)

        content = f"Answer submitted: {move_str}"
        if reasoning:
            content += f"\nReasoning: {reasoning}"
        content += f"\n\nResult: {'CORRECT!' if is_correct else 'Incorrect'}"
        if not is_correct and self.solution:
            content += f"\nExpected: {self.solution[0]}"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=content,
            stop_reason=StopReason.TASK_COMPLETED,
        )
