"""
Python code generation for refactor-snipe.

Ported from refactoring.nvim's code_generation/langs/python.lua.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class CodeGenerator(Protocol):
    """Protocol for language-specific code generators."""

    def function(
        self,
        name: str,
        params: list[str],
        body: list[str],
        return_type: str | None = None,
        is_method: bool = False,
    ) -> str: ...

    def function_call(
        self,
        name: str,
        args: list[str],
        is_method: bool = False,
    ) -> str: ...

    def assignment(
        self,
        names: list[str],
        value: str,
    ) -> str: ...

    def return_statement(self, values: list[str]) -> str: ...

    def indent_lines(self, lines: list[str], indent: str) -> list[str]: ...


@dataclass
class PythonCodeGen:
    """Python code generator."""

    indent_str: str = "    "  # 4 spaces by default

    def function(
        self,
        name: str,
        params: list[str],
        body: list[str],
        return_type: str | None = None,
        is_method: bool = False,
    ) -> str:
        """Generate a Python function definition."""
        # Build parameter list
        if is_method:
            all_params = ["self"] + params
        else:
            all_params = params

        param_str = ", ".join(all_params)

        # Build signature
        if return_type:
            sig = f"def {name}({param_str}) -> {return_type}:"
        else:
            sig = f"def {name}({param_str}):"

        # Build function
        lines = [sig]
        if body:
            lines.extend(body)
        else:
            lines.append(f"{self.indent_str}pass")

        return "\n".join(lines)

    def function_call(
        self,
        name: str,
        args: list[str],
        is_method: bool = False,
    ) -> str:
        """Generate a function call expression."""
        arg_str = ", ".join(args)
        if is_method:
            return f"self.{name}({arg_str})"
        return f"{name}({arg_str})"

    def assignment(
        self,
        names: list[str],
        value: str,
    ) -> str:
        """Generate an assignment statement."""
        if len(names) == 1:
            return f"{names[0]} = {value}"
        return f"{', '.join(names)} = {value}"

    def return_statement(self, values: list[str]) -> str:
        """Generate a return statement."""
        if not values:
            return "return"
        if len(values) == 1:
            return f"return {values[0]}"
        return f"return {', '.join(values)}"

    def indent_lines(self, lines: list[str], indent: str) -> list[str]:
        """Add indentation to lines."""
        result = []
        for line in lines:
            if line.strip():  # Only indent non-empty lines
                result.append(indent + line)
            else:
                result.append(line)
        return result

    def dedent_lines(self, lines: list[str], levels: int = 1) -> list[str]:
        """Remove indentation from lines."""
        dedent_amount = len(self.indent_str) * levels
        result = []
        for line in lines:
            # Count leading whitespace
            stripped = line.lstrip()
            if not stripped:
                result.append("")
                continue

            leading = len(line) - len(stripped)
            new_leading = max(0, leading - dedent_amount)
            result.append(" " * new_leading + stripped)

        return result

    def normalize_body_indent(self, lines: list[str], target_indent: str) -> list[str]:
        """
        Normalize body lines to have consistent indentation.
        First line's indentation is used as the base, then normalized to target.
        """
        if not lines:
            return []

        # Find minimum indentation (excluding empty lines)
        min_indent = float("inf")
        for line in lines:
            if line.strip():
                leading = len(line) - len(line.lstrip())
                min_indent = min(min_indent, leading)

        if min_indent == float("inf"):
            min_indent = 0

        # Normalize
        result = []
        for line in lines:
            if line.strip():
                stripped = line.lstrip()
                current_leading = len(line) - len(stripped)
                extra_indent = current_leading - min_indent
                result.append(target_indent + " " * extra_indent + stripped)
            else:
                result.append("")

        return result
