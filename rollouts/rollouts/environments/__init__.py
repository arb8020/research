from ..dtypes import Environment
from .ask_user import AskUserQuestionEnvironment
from .binary_search import BinarySearchEnvironment
from .calculator import CalculatorEnvironment
from .coding import LocalFilesystemEnvironment
from .compose import ComposedEnvironment, compose
from .git_worktree import GitWorktreeEnvironment
from .no_tools import BasicEnvironment, NoToolsEnvironment
from .repl import MessageParsingREPLEnvironment, REPLEnvironment

__all__ = [
    "Environment",
    "AskUserQuestionEnvironment",
    "CalculatorEnvironment",
    "BinarySearchEnvironment",
    "BasicEnvironment",
    "NoToolsEnvironment",
    "LocalFilesystemEnvironment",
    "GitWorktreeEnvironment",
    "BrowsingEnvironment",
    "ChessPuzzleEnvironment",
    "REPLEnvironment",
    "MessageParsingREPLEnvironment",
    "ComposedEnvironment",
    "compose",
]


def __getattr__(name: str) -> type:
    """Lazy imports for environments with heavy dependencies."""
    if name == "BrowsingEnvironment":
        from .browsing import BrowsingEnvironment

        return BrowsingEnvironment
    if name == "ChessPuzzleEnvironment":
        from .chess_puzzle import ChessPuzzleEnvironment

        return ChessPuzzleEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
