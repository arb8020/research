"""
refactor-snipe: Fast, deterministic AST-based refactoring tool.

A sniper-style refactoring tool that uses TreeSitter for precise,
single-operation refactors without LLM roundtrips.
"""

from .refactors import ExtractFunction
from .scope import ScopeAnalyzer
from .treesitter import Region, TreeSitterParser

__all__ = [
    "ExtractFunction",
    "ScopeAnalyzer",
    "TreeSitterParser",
    "Region",
]
