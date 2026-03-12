"""Core types for csearch."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Location:
    """A location in a file."""

    path: Path
    line_start: int
    line_end: int

    def __post_init__(self) -> None:
        assert str(self.path), "path cannot be empty"
        assert self.line_start > 0, "line_start must be positive"
        assert self.line_end > 0, "line_end must be positive"
        assert self.line_end >= self.line_start, "line_end must be >= line_start"

    def __str__(self) -> str:
        if self.line_start == self.line_end:
            return f"{self.path}:{self.line_start}"
        return f"{self.path}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class SearchResult:
    """A search result with location and optional snippet."""

    location: Location
    name: str
    kind: str  # "function", "class", "method", "variable", etc.
    snippet: str | None = None

    def __post_init__(self) -> None:
        assert self.name.strip(), "name cannot be empty"
        assert self.kind.strip(), "kind cannot be empty"

    def format_compact(self) -> str:
        """Single line: path:start-end kind name"""
        return f"{self.location} {self.kind} {self.name}"

    def format_with_snippet(self) -> str:
        """Location + snippet block."""
        lines = [str(self.location)]
        if self.snippet:
            lines.append(self.snippet)
        return "\n".join(lines)


@dataclass(frozen=True)
class SearchOptions:
    """Options for search operations."""

    path: Path = Path(".")
    backend: str = "auto"  # "tree-sitter", "ctags", "auto"
    include_snippet: bool = False
    limit: int | None = None  # None means no limit
