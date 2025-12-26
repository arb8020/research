"""
Extract Variable refactoring (Fowler refactoring #119).

Extracts an expression into a variable, replacing all occurrences
of that expression with the variable name.
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
    from treesitter import TreeSitterParser, get_language_for_file


@dataclass
class TextEdit:
    """A text edit to apply to a file."""

    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    start_col: int  # 0-indexed
    end_col: int  # 0-indexed
    new_text: str
    is_insertion: bool = False  # True for line insertions


@dataclass
class ExtractVariableResult:
    """Result of extract variable refactoring."""

    variable_name: str
    expression: str
    occurrences_replaced: int
    edits: list[TextEdit]


class ExtractVariable:
    """Extract Variable refactoring operation."""

    def __init__(
        self,
        path: Path,
        line: int,
        start_col: int,
        end_col: int,
        variable_name: str,
    ) -> None:
        """
        Initialize the refactoring.

        Args:
            path: Path to the file
            line: Line number of the expression (1-indexed)
            start_col: Start column of the expression (0-indexed)
            end_col: End column of the expression (0-indexed)
            variable_name: Name for the extracted variable
        """
        self.path = path
        self.line = line
        self.start_col = start_col
        self.end_col = end_col
        self.variable_name = variable_name

        # Parse the file
        self.parser = TreeSitterParser.from_file(path)
        self.language = get_language_for_file(path)
        self.codegen = get_generator(self.language)
        self.analyzer = ScopeAnalyzer(self.parser)

    def execute(self) -> ExtractVariableResult:
        """Execute the refactoring and return the result."""
        row = self.line - 1  # Convert to 0-indexed

        # Find the node at the given position
        node = self.parser.find_node_at(row, self.start_col)
        if node is None:
            raise ValueError(f"No expression found at line {self.line}, column {self.start_col}")

        # Walk up to find a meaningful expression node
        expression_node = self._find_expression_node(node)
        if expression_node is None:
            raise ValueError("Could not find a valid expression to extract")

        expression_text = self.parser.get_text(expression_node)

        # Find the containing scope
        scope = self.analyzer.get_containing_scope(expression_node)
        if scope is None:
            scope = self.parser.root

        # Find all occurrences of this expression in the scope
        occurrences = self._find_occurrences(expression_text, scope)

        if not occurrences:
            occurrences = [expression_node]

        # Sort occurrences by position
        occurrences = sorted(occurrences, key=lambda n: (n.start_point[0], n.start_point[1]))

        # Find the statement containing the first occurrence (where we'll insert the variable)
        first_occurrence = occurrences[0]
        insert_line = self._find_statement_line(first_occurrence)

        # Get indentation of that line
        lines = self.parser.source.split(b"\n")
        target_line = lines[insert_line] if insert_line < len(lines) else b""
        indent = ""
        for ch in target_line:
            if ch in (ord(" "), ord("\t")):
                indent += chr(ch)
            else:
                break

        # Create edits
        edits = []

        # Replace all occurrences with variable name (in reverse order)
        for occ in sorted(
            occurrences, key=lambda n: (n.start_point[0], n.start_point[1]), reverse=True
        ):
            edits.append(
                TextEdit(
                    start_line=occ.start_point[0] + 1,
                    end_line=occ.end_point[0] + 1,
                    start_col=occ.start_point[1],
                    end_col=occ.end_point[1],
                    new_text=self.variable_name,
                )
            )

        # Insert variable assignment before the first occurrence's statement
        assignment = f"{indent}{self.variable_name} = {expression_text}"
        edits.append(
            TextEdit(
                start_line=insert_line + 1,  # Convert to 1-indexed
                end_line=insert_line + 1,
                start_col=0,
                end_col=0,
                new_text=assignment,
                is_insertion=True,
            )
        )

        return ExtractVariableResult(
            variable_name=self.variable_name,
            expression=expression_text,
            occurrences_replaced=len(occurrences),
            edits=edits,
        )

    def _find_expression_node(self, node):
        """Walk up to find a meaningful expression node."""
        # Expression types we can extract
        extractable = {
            "binary_operator",
            "call",
            "subscript",
            "attribute",
            "list",
            "dictionary",
            "tuple",
            "set",
            "list_comprehension",
            "dictionary_comprehension",
            "set_comprehension",
            "generator_expression",
            "concatenated_string",
            "string",
            "integer",
            "float",
            "true",
            "false",
            "none",
            "identifier",
            "unary_operator",
            "comparison_operator",
            "boolean_operator",
            "conditional_expression",
            "lambda",
        }

        current = node
        best = None

        while current is not None:
            if current.type in extractable:
                best = current
            # Stop at statement level
            if current.type in ("expression_statement", "assignment", "return_statement"):
                break
            current = current.parent

        return best

    def _find_occurrences(self, expression_text: str, scope):
        """Find all occurrences of an expression in a scope."""
        occurrences = []

        def walk(node) -> None:
            if self.parser.get_text(node) == expression_text:
                occurrences.append(node)
            else:
                for child in node.children:
                    walk(child)

        walk(scope)
        return occurrences

    def _find_statement_line(self, node) -> int:
        """Find the line of the statement containing a node."""
        current = node
        while current is not None:
            if current.type in (
                "expression_statement",
                "assignment",
                "return_statement",
                "if_statement",
                "for_statement",
                "while_statement",
                "with_statement",
            ):
                return current.start_point[0]
            current = current.parent
        return node.start_point[0]


def apply_edits(path: Path, edits: list[TextEdit]) -> str:
    """Apply a list of text edits to a file and return the result."""
    content = path.read_text()
    lines = content.split("\n")

    # Separate insertions from replacements
    insertions = [e for e in edits if e.is_insertion]
    replacements = [e for e in edits if not e.is_insertion]

    # Apply replacements first (in reverse order)
    for edit in sorted(replacements, key=lambda e: (e.start_line, e.start_col), reverse=True):
        line_idx = edit.start_line - 1
        line = lines[line_idx]
        new_line = line[: edit.start_col] + edit.new_text + line[edit.end_col :]
        lines[line_idx] = new_line

    # Apply insertions (in reverse order by line)
    for edit in sorted(insertions, key=lambda e: e.start_line, reverse=True):
        line_idx = edit.start_line - 1
        lines.insert(line_idx, edit.new_text)

    return "\n".join(lines)
