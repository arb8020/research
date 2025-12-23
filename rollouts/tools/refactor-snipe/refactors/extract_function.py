"""
Extract Function refactoring (Fowler refactoring #106).

Extracts a selected region of code into a new function, automatically:
- Detecting parameters (variables used but defined outside the region)
- Detecting return values (variables defined in region and used after)
- Generating the function definition
- Replacing the original code with a function call
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
    new_text: str


@dataclass
class ExtractFunctionResult:
    """Result of extract function refactoring."""

    # The new function definition to insert
    function_definition: str
    # Line number where to insert the function (1-indexed, insert before this line)
    insert_line: int
    # The function call that replaces the extracted code
    function_call: str
    # Lines to replace (1-indexed, inclusive)
    replace_start: int
    replace_end: int
    # Full list of edits to apply
    edits: list[TextEdit]


class ExtractFunction:
    """Extract Function refactoring operation."""

    def __init__(self, path: Path, start_line: int, end_line: int, function_name: str):
        """
        Initialize the refactoring.

        Args:
            path: Path to the file
            start_line: Start line of selection (1-indexed, inclusive)
            end_line: End line of selection (1-indexed, inclusive)
            function_name: Name for the extracted function
        """
        self.path = path
        self.start_line = start_line
        self.end_line = end_line
        self.function_name = function_name

        # Parse the file
        self.parser = TreeSitterParser.from_file(path)
        self.language = get_language_for_file(path)
        self.codegen = get_generator(self.language)
        self.analyzer = ScopeAnalyzer(self.parser)

        # Create region for selected code
        self.region = Region.from_lines(start_line, end_line, self.parser.source)

    def execute(self) -> ExtractFunctionResult:
        """Execute the refactoring and return the result."""
        # Find the scope containing the selection
        # Start from a node in the region (use first code column to skip whitespace)
        first_code_col = self.region.first_code_column(self.parser.source)
        start_node = self.parser.find_node_at(self.region.start.row, first_code_col)
        if start_node is None:
            raise ValueError("Could not find node at selection start")

        scope = self.analyzer.get_containing_scope(start_node)
        if scope is None:
            # Use module as scope
            scope = self.parser.root

        # Analyze variables
        params = sorted(self.analyzer.find_free_variables_in_region(self.region, scope))
        return_vars = self.analyzer.find_return_variables(self.region, scope)

        # Get the selected code
        selected_text = self.region.get_text(self.parser.source).decode("utf-8")
        body_lines = selected_text.split("\n")

        # Determine if we're inside a class method
        is_method = self.analyzer.is_inside_class(start_node)

        # Get indentation info
        scope_indent = self.analyzer.get_scope_indentation(scope)

        # For the new function body, use a standard 4-space indent
        func_body_indent = "    "

        # Normalize body indentation: strip existing indent, add standard indent
        normalized_body = self.codegen.normalize_body_indent(body_lines, func_body_indent)

        # Add return statement if needed
        if return_vars:
            return_stmt = self.codegen.return_statement(return_vars)
            normalized_body.append(func_body_indent + return_stmt)

        # Generate the function definition
        func_def = self.codegen.function(
            name=self.function_name,
            params=params,
            body=normalized_body,
            is_method=is_method,
        )

        # Add scope-level indentation to the entire function
        func_lines = func_def.split("\n")
        indented_func_lines = []
        for line in func_lines:
            if line.strip():
                indented_func_lines.append(scope_indent + line)
            else:
                indented_func_lines.append("")

        func_def_indented = "\n".join(indented_func_lines)

        # Generate the function call
        call_expr = self.codegen.function_call(
            name=self.function_name,
            args=params,
            is_method=is_method,
        )

        # Wrap with assignment if there are return values
        if return_vars:
            call_stmt = self.codegen.assignment(return_vars, call_expr)
        else:
            call_stmt = call_expr

        # Get the indentation of the first selected line
        first_line = body_lines[0] if body_lines else ""
        call_indent = ""
        for ch in first_line:
            if ch in " \t":
                call_indent += ch
            else:
                break

        call_stmt_indented = call_indent + call_stmt

        # Determine where to insert the function
        # Insert before the containing function/scope
        if scope.type == "function_definition":
            insert_line = scope.start_point[0] + 1  # Convert to 1-indexed
        elif scope.type == "class_definition":
            # Find the containing function within the class
            func_scope = self.analyzer.get_function_scope(start_node)
            if func_scope:
                insert_line = func_scope.start_point[0] + 1
            else:
                insert_line = self.start_line
        else:
            # Module level - insert before selection
            insert_line = self.start_line

        # Create the edits
        edits = [
            # Insert function definition
            TextEdit(
                start_line=insert_line,
                end_line=insert_line,
                new_text=func_def_indented + "\n\n",
            ),
            # Replace selection with function call
            TextEdit(
                start_line=self.start_line,
                end_line=self.end_line,
                new_text=call_stmt_indented,
            ),
        ]

        return ExtractFunctionResult(
            function_definition=func_def_indented,
            insert_line=insert_line,
            function_call=call_stmt_indented,
            replace_start=self.start_line,
            replace_end=self.end_line,
            edits=edits,
        )


def apply_edits(path: Path, edits: list[TextEdit]) -> str:
    """
    Apply a list of text edits to a file and return the result.

    Edits are applied in reverse order (bottom to top) to preserve line numbers.
    """
    content = path.read_text()
    lines = content.split("\n")

    # Sort edits by start line, descending (apply from bottom to top)
    sorted_edits = sorted(edits, key=lambda e: e.start_line, reverse=True)

    for edit in sorted_edits:
        # Convert to 0-indexed
        start_idx = edit.start_line - 1
        end_idx = edit.end_line  # end is inclusive, so we go to end_line

        # For insertion (start == end), insert before the line
        if edit.start_line == edit.end_line and edit.new_text.endswith("\n\n"):
            # This is an insertion, insert before start_idx
            new_lines = edit.new_text.rstrip("\n").split("\n")
            lines = lines[:start_idx] + new_lines + [""] + lines[start_idx:]
        else:
            # This is a replacement
            new_lines = edit.new_text.split("\n")
            lines = lines[:start_idx] + new_lines + lines[end_idx:]

    return "\n".join(lines)
