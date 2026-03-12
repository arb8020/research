from ..dtypes import Environment
from .binary_search import BinarySearchEnvironment
from .calculator import CalculatorEnvironment
from .coding import CodingEnvironment, LocalFilesystemEnvironment
from .compose import ComposedEnvironment, compose
from .factory import (
    EnvironmentBuildConfig,
    EnvironmentFactory,
    build_environment,
    get_environment_factories,
    get_environment_names,
    infer_environment_name,
    register_environment_factory,
)
from .git_worktree import GitWorktreeEnvironment
from .handoff import HandoffEnvironment
from .no_tools import BasicEnvironment, NoToolsEnvironment
from .repl import MessageParsingREPLEnvironment, REPLEnvironment

__all__ = [
    "Environment",
    "CalculatorEnvironment",
    "BinarySearchEnvironment",
    "BasicEnvironment",
    "NoToolsEnvironment",
    "CodingEnvironment",
    "LocalFilesystemEnvironment",
    "GitWorktreeEnvironment",
    "HandoffEnvironment",
    "BrowsingEnvironment",
    "ChessPuzzleEnvironment",
    "REPLEnvironment",
    "MessageParsingREPLEnvironment",
    "ComposedEnvironment",
    "compose",
    "EnvironmentBuildConfig",
    "EnvironmentFactory",
    "build_environment",
    "get_environment_factories",
    "get_environment_names",
    "infer_environment_name",
    "register_environment_factory",
    "TerminalBenchEnvironment",
]


def __getattr__(name: str) -> type:
    """Lazy imports for environments with heavy dependencies."""
    if name == "BrowsingEnvironment":
        from .browsing import BrowsingEnvironment

        return BrowsingEnvironment
    if name == "ChessPuzzleEnvironment":
        from .chess_puzzle import ChessPuzzleEnvironment

        return ChessPuzzleEnvironment
    if name == "TerminalBenchEnvironment":
        from .terminal_bench import TerminalBenchEnvironment

        return TerminalBenchEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
