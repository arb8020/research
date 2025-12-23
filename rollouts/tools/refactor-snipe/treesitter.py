"""
TreeSitter parsing layer for refactor-snipe.

Provides AST parsing and node querying capabilities using py-tree-sitter.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser, Query, QueryCursor, Tree

# Language registry - add more as needed
LANGUAGES: dict[str, Language] = {
    "python": Language(tspython.language()),
}

EXTENSION_TO_LANG: dict[str, str] = {
    ".py": "python",
}


@dataclass
class Position:
    """A position in a file (0-indexed line and column)."""

    row: int
    col: int

    @classmethod
    def from_tuple(cls, t: tuple[int, int]) -> Position:
        return cls(row=t[0], col=t[1])

    def to_tuple(self) -> tuple[int, int]:
        return (self.row, self.col)


@dataclass
class Region:
    """A region in a file defined by start and end positions."""

    start: Position
    end: Position

    @classmethod
    def from_node(cls, node: Node) -> Region:
        return cls(
            start=Position.from_tuple(node.start_point),
            end=Position.from_tuple(node.end_point),
        )

    @classmethod
    def from_lines(cls, start_line: int, end_line: int, source: bytes) -> Region:
        """Create a region spanning full lines (1-indexed, inclusive)."""
        lines = source.split(b"\n")
        start_row = start_line - 1  # Convert to 0-indexed
        end_row = end_line - 1

        # End column is end of the last line
        end_col = len(lines[end_row]) if end_row < len(lines) else 0

        return cls(
            start=Position(row=start_row, col=0),
            end=Position(row=end_row, col=end_col),
        )

    def first_code_column(self, source: bytes) -> int:
        """Find the column of the first non-whitespace character on the start line."""
        lines = source.split(b"\n")
        if self.start.row < len(lines):
            line = lines[self.start.row]
            for i, ch in enumerate(line):
                if ch not in (ord(" "), ord("\t")):
                    return i
        return 0

    def contains(self, pos: Position) -> bool:
        if pos.row < self.start.row or pos.row > self.end.row:
            return False
        if pos.row == self.start.row and pos.col < self.start.col:
            return False
        if pos.row == self.end.row and pos.col > self.end.col:
            return False
        return True

    def contains_node(self, node: Node) -> bool:
        """Check if this region fully contains a node."""
        node_region = Region.from_node(node)
        return self.contains(node_region.start) and self.contains(node_region.end)

    def intersects(self, node: Node) -> bool:
        """Check if this region intersects with a node."""
        node_region = Region.from_node(node)
        # No intersection if one is entirely before/after the other
        if self.end.row < node_region.start.row:
            return False
        if self.start.row > node_region.end.row:
            return False
        if self.end.row == node_region.start.row and self.end.col < node_region.start.col:
            return False
        if self.start.row == node_region.end.row and self.start.col > node_region.end.col:
            return False
        return True

    def is_after(self, node: Node) -> bool:
        """Check if this region is entirely after a node."""
        node_region = Region.from_node(node)
        if self.start.row > node_region.end.row:
            return True
        if self.start.row == node_region.end.row and self.start.col > node_region.end.col:
            return True
        return False

    def get_text(self, source: bytes) -> bytes:
        """Extract the text covered by this region from source."""
        lines = source.split(b"\n")
        if self.start.row == self.end.row:
            return lines[self.start.row][self.start.col : self.end.col]

        result = [lines[self.start.row][self.start.col :]]
        for row in range(self.start.row + 1, self.end.row):
            result.append(lines[row])
        result.append(lines[self.end.row][: self.end.col])
        return b"\n".join(result)


class TreeSitterParser:
    """Wrapper around tree-sitter parser with query capabilities."""

    def __init__(self, language: str):
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language: {language}")

        self.language = language
        self.lang = LANGUAGES[language]
        self.parser = Parser(self.lang)
        self._tree: Tree | None = None
        self._source: bytes | None = None

    @classmethod
    def from_file(cls, path: Path) -> TreeSitterParser:
        ext = path.suffix
        if ext not in EXTENSION_TO_LANG:
            raise ValueError(f"Unsupported file extension: {ext}")

        lang = EXTENSION_TO_LANG[ext]
        parser = cls(lang)
        parser.parse_file(path)
        return parser

    def parse(self, source: bytes) -> Tree:
        self._source = source
        self._tree = self.parser.parse(source)
        return self._tree

    def parse_file(self, path: Path) -> Tree:
        return self.parse(path.read_bytes())

    @property
    def tree(self) -> Tree:
        if self._tree is None:
            raise RuntimeError("No tree parsed yet")
        return self._tree

    @property
    def source(self) -> bytes:
        if self._source is None:
            raise RuntimeError("No source parsed yet")
        return self._source

    @property
    def root(self) -> Node:
        return self.tree.root_node

    def query(self, query_str: str) -> Query:
        """Create a query for the current language."""
        return Query(self.lang, query_str)

    def query_captures(self, query_str: str, node: Node | None = None) -> list[tuple[Node, str]]:
        """Run a query and return all captures as (node, capture_name) pairs."""
        q = self.query(query_str)
        cursor = QueryCursor(q)
        target = node or self.root

        result = []
        for _, captures in cursor.matches(target):
            for name, nodes in captures.items():
                for n in nodes:
                    result.append((n, name))
        return result

    def get_text(self, node: Node) -> str:
        """Get the text of a node."""
        return node.text.decode("utf-8") if node.text else ""

    def find_node_at(self, row: int, col: int) -> Node | None:
        """Find the smallest node containing the given position."""
        return self.root.descendant_for_point_range((row, col), (row, col))

    def find_nodes_in_region(self, region: Region) -> Iterator[Node]:
        """Find all nodes that intersect with a region."""

        def walk(node: Node) -> Iterator[Node]:
            if region.intersects(node):
                yield node
                for child in node.children:
                    yield from walk(child)

        yield from walk(self.root)


def get_language_for_file(path: Path) -> str:
    """Get the language name for a file based on extension."""
    ext = path.suffix
    if ext not in EXTENSION_TO_LANG:
        raise ValueError(f"Unsupported file extension: {ext}")
    return EXTENSION_TO_LANG[ext]
