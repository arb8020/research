"""
Inline Variable refactoring (Fowler refactoring #123).

Replaces all occurrences of a variable with its assigned value,
then removes the variable declaration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    from ..langs import get_generator
    from ..scope import ScopeAnalyzer
    from ..treesitter import Region, TreeSitterParser, get_language_for_file
except ImportError:
    from langs import get_generator
    from scope import ScopeAnalyzer
    from treesitter import Region, TreeSitterParser, get_language_for_file


@dataclass
class TextEdit:
    """A text edit to apply to a file."""

    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    start_col: int  # 0-indexed
    end_col: int  # 0-indexed
    new_text: str


@dataclass
class InlineVariableResult:
    """Result of inline variable refactoring."""

    variable_name: str
    value: str
    occurrences_replaced: int
    edits: list[TextEdit]


class InlineVariable:
    """Inline Variable refactoring operation."""

    def __init__(self, path: Path, line: int, col: int | None = None) -> None:
        """
        Initialize the refactoring.

        Args:
            path: Path to the file
            line: Line number where the variable is (1-indexed)
            col: Optional column number (0-indexed). If not provided, finds first assignment on line.
        """
        self.path = path
        self.line = line
        self.col = col

        # Parse the file
        self.parser = TreeSitterParser.from_file(path)
        self.language = get_language_for_file(path)
        self.codegen = get_generator(self.language)
        self.analyzer = ScopeAnalyzer(self.parser)

    def execute(self) -> InlineVariableResult:
        """Execute the refactoring and return the result."""
        # Find the assignment at the given line
        row = self.line - 1  # Convert to 0-indexed

        # Find an assignment node on this line
        assignment_node = self._find_assignment_on_line(row)
        if assignment_node is None:
            raise ValueError(f"No variable assignment found on line {self.line}")

        # Get the variable name and value
        left = assignment_node.child_by_field_name("left")
        right = assignment_node.child_by_field_name("right")

        if left is None or right is None:
            raise ValueError("Could not parse assignment")

        # Handle simple identifier assignment
        if left.type != "identifier":
            raise ValueError(f"Cannot inline complex assignment (left side is {left.type})")

        var_name = self.parser.get_text(left)
        var_value = self.parser.get_text(right)

        # Find the containing scope
        scope = self.analyzer.get_containing_scope(assignment_node)
        if scope is None:
            scope = self.parser.root

        # Find all references to this variable in the scope
        references = self._find_references(var_name, scope, assignment_node)

        if not references:
            raise ValueError(f"No references to '{var_name}' found to inline")

        # Create edits: replace all references with the value, then delete the assignment
        edits = []

        # Replace references (in reverse order to preserve positions)
        for ref_node in sorted(
            references, key=lambda n: (n.start_point[0], n.start_point[1]), reverse=True
        ):
            edits.append(
                TextEdit(
                    start_line=ref_node.start_point[0] + 1,
                    end_line=ref_node.end_point[0] + 1,
                    start_col=ref_node.start_point[1],
                    end_col=ref_node.end_point[1],
                    new_text=var_value,
                )
            )

        # Delete the assignment line
        edits.append(
            TextEdit(
                start_line=assignment_node.start_point[0] + 1,
                end_line=assignment_node.end_point[0] + 1,
                start_col=0,
                end_col=0,  # Special marker for full line deletion
                new_text="",
            )
        )

        return InlineVariableResult(
            variable_name=var_name,
            value=var_value,
            occurrences_replaced=len(references),
            edits=edits,
        )

    def _find_assignment_on_line(self, row: int):
        """Find an assignment node on the given line."""
        query = "(assignment) @assign"
        for node, _ in self.parser.query_captures(query):
            if node.start_point[0] == row:
                return node
        return None

    def _find_references(self, var_name: str, scope, exclude_node):
        """Find all references to a variable, excluding the definition."""
        references = []
        query = "(identifier) @ref"

        exclude_region = Region.from_node(exclude_node)

        for node, _ in self.parser.query_captures(query, scope):
            if self.parser.get_text(node) != var_name:
                continue

            # Skip the definition itself
            if exclude_region.contains_node(node):
                continue

            # Skip if it's on the left side of an assignment (a redefinition)
            parent = node.parent
            if parent and parent.type == "assignment":
                left = parent.child_by_field_name("left")
                if left and self._node_contains(left, node):
                    continue

            references.append(node)

        return references

    def _node_contains(self, container, target) -> bool:
        """Check if container node contains target node."""

        def walk(node) -> bool:
            if node.id == target.id:
                return True
            for child in node.children:
                if walk(child):
                    return True
            return False

        return walk(container)


def apply_edits(path: Path, edits: list[TextEdit]) -> str:
    """Apply a list of text edits to a file and return the result."""
    content = path.read_text()
    lines = content.split("\n")

    # Sort edits by position, descending (apply from bottom to top, right to left)
    sorted_edits = sorted(
        edits,
        key=lambda e: (e.start_line, e.start_col),
        reverse=True,
    )

    for edit in sorted_edits:
        line_idx = edit.start_line - 1

        if edit.start_col == 0 and edit.end_col == 0 and edit.new_text == "":
            # Full line deletion
            del lines[line_idx]
        else:
            # In-line replacement
            line = lines[line_idx]
            new_line = line[: edit.start_col] + edit.new_text + line[edit.end_col :]
            lines[line_idx] = new_line

    return "\n".join(lines)
