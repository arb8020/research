"""
Inline Function refactoring (Fowler refactoring #115).

Replaces all calls to a function with its body, then removes the function definition.
This is the inverse of Extract Function.
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
    delete_lines: bool = False  # True for full line range deletion


@dataclass
class InlineFunctionResult:
    """Result of inline function refactoring."""

    function_name: str
    calls_inlined: int
    edits: list[TextEdit]


class InlineFunction:
    """Inline Function refactoring operation."""

    def __init__(self, path: Path, line: int):
        """
        Initialize the refactoring.

        Args:
            path: Path to the file
            line: Line number where the function is defined (1-indexed)
        """
        self.path = path
        self.line = line

        # Parse the file
        self.parser = TreeSitterParser.from_file(path)
        self.language = get_language_for_file(path)
        self.codegen = get_generator(self.language)
        self.analyzer = ScopeAnalyzer(self.parser)

    def execute(self) -> InlineFunctionResult:
        """Execute the refactoring and return the result."""
        row = self.line - 1  # Convert to 0-indexed

        # Find the function definition at the given line
        func_node = self._find_function_on_line(row)
        if func_node is None:
            raise ValueError(f"No function definition found on line {self.line}")

        # Get function name
        name_node = func_node.child_by_field_name("name")
        if name_node is None:
            raise ValueError("Could not find function name")
        func_name = self.parser.get_text(name_node)

        # Get function parameters
        params = self.analyzer.get_function_parameters(func_node)

        # Get function body
        body_node = func_node.child_by_field_name("body")
        if body_node is None:
            raise ValueError("Could not find function body")

        body_lines, return_value = self._extract_body_and_return(body_node)

        # Check for multiple return statements
        return_count = self._count_returns(body_node)
        if return_count > 1:
            raise ValueError("Cannot inline function with multiple return statements")

        # Find all calls to this function
        calls = self._find_calls(func_name)

        if not calls:
            raise ValueError(f"No calls to '{func_name}' found to inline")

        edits = []

        # Process each call site (in reverse order)
        for call_node in sorted(
            calls, key=lambda n: (n.start_point[0], n.start_point[1]), reverse=True
        ):
            call_edits = self._inline_call(call_node, params, body_lines, return_value)
            edits.extend(call_edits)

        # Delete the function definition
        edits.append(
            TextEdit(
                start_line=func_node.start_point[0] + 1,
                end_line=func_node.end_point[0] + 1,
                start_col=0,
                end_col=0,
                new_text="",
                delete_lines=True,
            )
        )

        return InlineFunctionResult(
            function_name=func_name,
            calls_inlined=len(calls),
            edits=edits,
        )

    def _find_function_on_line(self, row: int):
        """Find a function definition on the given line."""
        query = "(function_definition) @func"
        for node, _ in self.parser.query_captures(query):
            if node.start_point[0] == row:
                return node
        return None

    def _extract_body_and_return(self, body_node):
        """Extract body lines and return value from function body."""
        body_lines = []
        return_value = None

        for child in body_node.children:
            if child.type == "return_statement":
                # Extract return value
                for sub in child.children:
                    if sub.type not in ("return",):
                        return_value = self.parser.get_text(sub)
            elif child.type == "pass_statement":
                continue  # Skip pass statements
            else:
                # Get the line text
                line_text = self.parser.get_text(child)
                body_lines.append(line_text)

        return body_lines, return_value

    def _count_returns(self, body_node) -> int:
        """Count return statements in a function body."""
        count = 0
        query = "(return_statement) @ret"
        for node, _ in self.parser.query_captures(query, body_node):
            count += 1
        return count

    def _find_calls(self, func_name: str):
        """Find all calls to a function by name."""
        calls = []
        query = "(call function: (identifier) @name) @call"

        # Get all calls
        all_captures = self.parser.query_captures(query)

        # Group by call - find calls where the function name matches
        i = 0
        while i < len(all_captures):
            node, capture_name = all_captures[i]
            if capture_name == "name" and self.parser.get_text(node) == func_name:
                # Find the parent call node
                parent = node.parent
                if parent and parent.type == "call":
                    calls.append(parent)
            i += 1

        return calls

    def _inline_call(
        self, call_node, params: list[str], body_lines: list[str], return_value: str | None
    ):
        """Generate edits to inline a single function call."""
        edits = []

        # Get the arguments from the call
        args = self._get_call_arguments(call_node)

        # Find the statement containing this call
        stmt_node = self._find_containing_statement(call_node)

        # Get indentation
        lines = self.parser.source.split(b"\n")
        stmt_line = (
            lines[stmt_node.start_point[0]] if stmt_node.start_point[0] < len(lines) else b""
        )
        indent = ""
        for ch in stmt_line:
            if ch in (ord(" "), ord("\t")):
                indent += chr(ch)
            else:
                break

        # Check if the call is part of an assignment
        is_assignment = stmt_node.type == "assignment" or (
            stmt_node.type == "expression_statement"
            and stmt_node.child_count > 0
            and stmt_node.children[0].type == "assignment"
        )

        # Build parameter assignments if args differ from params
        param_assignments = []
        for i, param in enumerate(params):
            if i < len(args):
                arg = args[i]
                if arg != param:
                    param_assignments.append(f"{indent}{param} = {arg}")

        # Case 1: Simple return value replacement (no body, just return)
        if not body_lines and return_value:
            # Substitute parameters in return value
            substituted_return = self._substitute_params(return_value, params, args)

            if is_assignment:
                # Replace the call with the return value
                edits.append(
                    TextEdit(
                        start_line=call_node.start_point[0] + 1,
                        end_line=call_node.end_point[0] + 1,
                        start_col=call_node.start_point[1],
                        end_col=call_node.end_point[1],
                        new_text=substituted_return,
                    )
                )
            else:
                # Replace the whole statement
                edits.append(
                    TextEdit(
                        start_line=stmt_node.start_point[0] + 1,
                        end_line=stmt_node.end_point[0] + 1,
                        start_col=0,
                        end_col=0,
                        new_text="",
                        delete_lines=True,
                    )
                )

        # Case 2: Body with optional return
        else:
            # Substitute parameters in body lines
            substituted_body = [self._substitute_params(line, params, args) for line in body_lines]

            if return_value:
                substituted_return = self._substitute_params(return_value, params, args)

                if is_assignment:
                    # Insert body before, replace call with return value
                    for line in reversed(substituted_body):
                        edits.append(
                            TextEdit(
                                start_line=stmt_node.start_point[0] + 1,
                                end_line=stmt_node.start_point[0] + 1,
                                start_col=0,
                                end_col=0,
                                new_text=indent + line + "\n",
                            )
                        )

                    edits.append(
                        TextEdit(
                            start_line=call_node.start_point[0] + 1,
                            end_line=call_node.end_point[0] + 1,
                            start_col=call_node.start_point[1],
                            end_col=call_node.end_point[1],
                            new_text=substituted_return,
                        )
                    )
                else:
                    # Replace statement with body (no return needed)
                    combined = "\n".join(indent + line for line in substituted_body)
                    edits.append(
                        TextEdit(
                            start_line=stmt_node.start_point[0] + 1,
                            end_line=stmt_node.end_point[0] + 1,
                            start_col=0,
                            end_col=len(lines[stmt_node.end_point[0]])
                            if stmt_node.end_point[0] < len(lines)
                            else 0,
                            new_text=combined,
                        )
                    )
            else:
                # No return value, just inline the body
                combined = "\n".join(indent + line for line in substituted_body)
                edits.append(
                    TextEdit(
                        start_line=stmt_node.start_point[0] + 1,
                        end_line=stmt_node.end_point[0] + 1,
                        start_col=0,
                        end_col=len(lines[stmt_node.end_point[0]])
                        if stmt_node.end_point[0] < len(lines)
                        else 0,
                        new_text=combined,
                    )
                )

        return edits

    def _get_call_arguments(self, call_node) -> list[str]:
        """Get the argument values from a call node."""
        args = []
        args_node = call_node.child_by_field_name("arguments")
        if args_node:
            for child in args_node.children:
                if child.type not in ("(", ")", ","):
                    args.append(self.parser.get_text(child))
        return args

    def _find_containing_statement(self, node):
        """Find the statement containing a node."""
        current = node
        while current is not None:
            if current.type in (
                "expression_statement",
                "assignment",
                "return_statement",
            ):
                return current
            current = current.parent
        return node

    def _substitute_params(self, text: str, params: list[str], args: list[str]) -> str:
        """Substitute parameter names with argument values in text."""
        result = text
        for param, arg in zip(params, args, strict=False):
            if param != arg:
                # Simple word-boundary replacement (not perfect but works for most cases)
                import re

                result = re.sub(rf"\b{re.escape(param)}\b", arg, result)
        return result


def apply_edits(path: Path, edits: list[TextEdit]) -> str:
    """Apply a list of text edits to a file and return the result."""
    content = path.read_text()
    lines = content.split("\n")

    # Separate line deletions from other edits
    line_deletions = [e for e in edits if e.delete_lines]
    other_edits = [e for e in edits if not e.delete_lines]

    # Apply non-deletion edits first (in reverse order)
    for edit in sorted(other_edits, key=lambda e: (e.start_line, e.start_col), reverse=True):
        line_idx = edit.start_line - 1

        if edit.new_text.endswith("\n"):
            # Line insertion
            lines.insert(line_idx, edit.new_text.rstrip("\n"))
        else:
            # In-line replacement
            line = lines[line_idx]
            new_line = line[: edit.start_col] + edit.new_text + line[edit.end_col :]
            lines[line_idx] = new_line

    # Apply line deletions (in reverse order)
    for edit in sorted(line_deletions, key=lambda e: e.start_line, reverse=True):
        start_idx = edit.start_line - 1
        end_idx = edit.end_line  # end is inclusive
        del lines[start_idx:end_idx]

    return "\n".join(lines)
