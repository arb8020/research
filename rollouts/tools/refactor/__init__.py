# refactor - Multi-file AI refactoring tool
# Follows imports recursively, builds context, applies <write>/<patch> commands

from .imports import collect_context
from .commands import parse_commands, apply_commands
from .refactor import refactor

__all__ = ["collect_context", "parse_commands", "apply_commands", "refactor"]
