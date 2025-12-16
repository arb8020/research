from ..dtypes import Environment
from .binary_search import BinarySearchEnvironment
from .calculator import CalculatorEnvironment
from .coding import LocalFilesystemEnvironment
from .git_worktree import GitWorktreeEnvironment
from .no_tools import BasicEnvironment, NoToolsEnvironment

__all__ = [
    "Environment",
    "CalculatorEnvironment",
    "BinarySearchEnvironment",
    "BasicEnvironment",
    "NoToolsEnvironment",
    "LocalFilesystemEnvironment",
    "GitWorktreeEnvironment",
    "BrowsingEnvironment",
]


def __getattr__(name: str):
    """Lazy imports for environments with heavy dependencies."""
    if name == "BrowsingEnvironment":
        from .browsing import BrowsingEnvironment

        return BrowsingEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
