#!/usr/bin/env python3
"""
refactor-snipe CLI - Fast, deterministic AST-based refactoring.

Usage:
    snipe extract-function <file>:<start>-<end> --name <function_name>
    snipe extract-function <file>:<start>-<end> --name <function_name> --dry-run
    snipe extract-function <file>:<start>-<end> --name <function_name> --diff

Examples:
    # Extract lines 10-15 into a function called "process_data"
    snipe extract-function src/main.py:10-15 --name process_data

    # Preview what would change (don't modify file)
    snipe extract-function src/main.py:10-15 --name process_data --dry-run

    # Show unified diff of changes
    snipe extract-function src/main.py:10-15 --name process_data --diff
"""

from __future__ import annotations

import difflib
import re
from pathlib import Path

import typer

try:
    from .refactors.extract_function import ExtractFunction
    from .refactors.extract_function import apply_edits as apply_extract_function_edits
    from .refactors.extract_variable import ExtractVariable
    from .refactors.extract_variable import apply_edits as apply_extract_variable_edits
    from .refactors.inline_function import InlineFunction
    from .refactors.inline_function import apply_edits as apply_inline_function_edits
    from .refactors.inline_variable import InlineVariable
    from .refactors.inline_variable import apply_edits as apply_inline_variable_edits
except ImportError:
    from refactors.extract_function import ExtractFunction
    from refactors.extract_function import apply_edits as apply_extract_function_edits
    from refactors.extract_variable import ExtractVariable
    from refactors.extract_variable import apply_edits as apply_extract_variable_edits
    from refactors.inline_function import InlineFunction
    from refactors.inline_function import apply_edits as apply_inline_function_edits
    from refactors.inline_variable import InlineVariable
    from refactors.inline_variable import apply_edits as apply_inline_variable_edits

app = typer.Typer(
    name="snipe",
    help="Fast, deterministic AST-based refactoring tool.",
    no_args_is_help=True,
)


def parse_file_range(spec: str) -> tuple[Path, int, int]:
    """
    Parse a file:start-end specification.

    Examples:
        "foo.py:10-15" -> (Path("foo.py"), 10, 15)
        "src/bar.py:5-5" -> (Path("src/bar.py"), 5, 5)
    """
    match = re.match(r"^(.+):(\d+)-(\d+)$", spec)
    if not match:
        raise typer.BadParameter(
            f"Invalid format: {spec!r}. Expected format: <file>:<start>-<end> (e.g., foo.py:10-15)"
        )

    path = Path(match.group(1))
    start = int(match.group(2))
    end = int(match.group(3))

    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    if start < 1:
        raise typer.BadParameter(f"Start line must be >= 1, got {start}")

    if end < start:
        raise typer.BadParameter(f"End line ({end}) must be >= start line ({start})")

    return path, start, end


@app.command("extract-function")
def extract_function(
    file_range: str = typer.Argument(
        ...,
        help="File and line range in format <file>:<start>-<end> (e.g., foo.py:10-15)",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name for the extracted function",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Preview changes without modifying the file",
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Show unified diff of changes",
    ),
) -> None:
    """
    Extract a code region into a new function.

    Automatically detects:
    - Parameters: variables used in the region but defined outside
    - Return values: variables defined in the region and used after

    Examples:
        snipe extract-function src/main.py:10-15 --name process_data
        snipe extract-function src/main.py:10-15 -n process_data --diff
    """
    path, start, end = parse_file_range(file_range)

    try:
        refactor = ExtractFunction(path, start, end, name)
        result = refactor.execute()
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None from None

    original = path.read_text()
    modified = apply_extract_function_edits(path, result.edits)

    if diff:
        # Show unified diff
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
        diff_text = "".join(diff_lines)

        if diff_text:
            # Colorize diff output
            for line in diff_text.split("\n"):
                if line.startswith("+") and not line.startswith("+++"):
                    typer.secho(line, fg=typer.colors.GREEN)
                elif line.startswith("-") and not line.startswith("---"):
                    typer.secho(line, fg=typer.colors.RED)
                elif line.startswith("@@"):
                    typer.secho(line, fg=typer.colors.CYAN)
                else:
                    typer.echo(line)
        else:
            typer.echo("No changes")
        return

    if dry_run:
        # Show what would happen
        typer.secho("=== Dry Run ===", fg=typer.colors.YELLOW, bold=True)
        typer.echo()
        typer.secho("New function:", fg=typer.colors.GREEN)
        typer.echo(result.function_definition)
        typer.echo()
        typer.secho(f"Inserted at line {result.insert_line}", fg=typer.colors.BLUE)
        typer.echo()
        typer.secho("Replacement:", fg=typer.colors.GREEN)
        typer.echo(result.function_call)
        typer.echo()
        typer.secho(
            f"Replaces lines {result.replace_start}-{result.replace_end}", fg=typer.colors.BLUE
        )
        return

    # Apply changes
    path.write_text(modified)
    typer.secho(f"Extracted function '{name}' from lines {start}-{end}", fg=typer.colors.GREEN)


def parse_file_line(spec: str) -> tuple[Path, int]:
    """Parse a file:line specification."""
    match = re.match(r"^(.+):(\d+)$", spec)
    if not match:
        raise typer.BadParameter(
            f"Invalid format: {spec!r}. Expected format: <file>:<line> (e.g., foo.py:10)"
        )

    path = Path(match.group(1))
    line = int(match.group(2))

    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    if line < 1:
        raise typer.BadParameter(f"Line must be >= 1, got {line}")

    return path, line


def parse_file_line_cols(spec: str) -> tuple[Path, int, int, int]:
    """Parse a file:line:start_col-end_col specification."""
    match = re.match(r"^(.+):(\d+):(\d+)-(\d+)$", spec)
    if not match:
        raise typer.BadParameter(
            f"Invalid format: {spec!r}. Expected format: <file>:<line>:<start_col>-<end_col> (e.g., foo.py:10:5-15)"
        )

    path = Path(match.group(1))
    line = int(match.group(2))
    start_col = int(match.group(3))
    end_col = int(match.group(4))

    if not path.exists():
        raise typer.BadParameter(f"File not found: {path}")

    if line < 1:
        raise typer.BadParameter(f"Line must be >= 1, got {line}")

    if end_col < start_col:
        raise typer.BadParameter(f"End column ({end_col}) must be >= start column ({start_col})")

    return path, line, start_col, end_col


def show_diff(original: str, modified: str, path: Path) -> None:
    """Show a colorized unified diff."""
    diff_lines = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    diff_text = "".join(diff_lines)

    if diff_text:
        for line in diff_text.split("\n"):
            if line.startswith("+") and not line.startswith("+++"):
                typer.secho(line, fg=typer.colors.GREEN)
            elif line.startswith("-") and not line.startswith("---"):
                typer.secho(line, fg=typer.colors.RED)
            elif line.startswith("@@"):
                typer.secho(line, fg=typer.colors.CYAN)
            else:
                typer.echo(line)
    else:
        typer.echo("No changes")


@app.command("inline-variable")
def inline_variable(
    file_line: str = typer.Argument(
        ...,
        help="File and line in format <file>:<line> (e.g., foo.py:10)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Preview changes without modifying the file",
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Show unified diff of changes",
    ),
) -> None:
    """
    Inline a variable at the given line.

    Replaces all uses of the variable with its assigned value,
    then removes the variable declaration.

    Examples:
        snipe inline-variable src/main.py:5
        snipe inline-variable src/main.py:5 --diff
    """
    path, line = parse_file_line(file_line)

    try:
        refactor = InlineVariable(path, line)
        result = refactor.execute()
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    original = path.read_text()
    modified = apply_inline_variable_edits(path, result.edits)

    if diff:
        show_diff(original, modified, path)
        return

    if dry_run:
        typer.secho("=== Dry Run ===", fg=typer.colors.YELLOW, bold=True)
        typer.echo()
        typer.secho(f"Variable: {result.variable_name}", fg=typer.colors.GREEN)
        typer.secho(f"Value: {result.value}", fg=typer.colors.GREEN)
        typer.secho(f"Occurrences to replace: {result.occurrences_replaced}", fg=typer.colors.BLUE)
        return

    path.write_text(modified)
    typer.secho(
        f"Inlined variable '{result.variable_name}' ({result.occurrences_replaced} occurrences)",
        fg=typer.colors.GREEN,
    )


@app.command("extract-variable")
def extract_variable(
    file_range: str = typer.Argument(
        ...,
        help="File and column range in format <file>:<line>:<start_col>-<end_col>",
    ),
    name: str = typer.Option(
        ...,
        "--name",
        "-n",
        help="Name for the extracted variable",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Preview changes without modifying the file",
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Show unified diff of changes",
    ),
) -> None:
    """
    Extract an expression into a variable.

    Replaces all occurrences of the expression with the variable name
    and inserts a variable assignment.

    Examples:
        snipe extract-variable src/main.py:10:5-20 --name result
        snipe extract-variable src/main.py:10:5-20 -n result --diff
    """
    path, line, start_col, end_col = parse_file_line_cols(file_range)

    try:
        refactor = ExtractVariable(path, line, start_col, end_col, name)
        result = refactor.execute()
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    original = path.read_text()
    modified = apply_extract_variable_edits(path, result.edits)

    if diff:
        show_diff(original, modified, path)
        return

    if dry_run:
        typer.secho("=== Dry Run ===", fg=typer.colors.YELLOW, bold=True)
        typer.echo()
        typer.secho(f"Variable: {result.variable_name}", fg=typer.colors.GREEN)
        typer.secho(f"Expression: {result.expression}", fg=typer.colors.GREEN)
        typer.secho(f"Occurrences to replace: {result.occurrences_replaced}", fg=typer.colors.BLUE)
        return

    path.write_text(modified)
    typer.secho(
        f"Extracted variable '{result.variable_name}' ({result.occurrences_replaced} occurrences)",
        fg=typer.colors.GREEN,
    )


@app.command("inline-function")
def inline_function(
    file_line: str = typer.Argument(
        ...,
        help="File and line in format <file>:<line> where function is defined",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        "-d",
        help="Preview changes without modifying the file",
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Show unified diff of changes",
    ),
) -> None:
    """
    Inline a function, replacing all calls with its body.

    Finds all calls to the function and replaces them with the function body,
    then removes the function definition.

    Examples:
        snipe inline-function src/main.py:5
        snipe inline-function src/main.py:5 --diff
    """
    path, line = parse_file_line(file_line)

    try:
        refactor = InlineFunction(path, line)
        result = refactor.execute()
    except Exception as e:
        typer.secho(f"Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1) from None

    original = path.read_text()
    modified = apply_inline_function_edits(path, result.edits)

    if diff:
        show_diff(original, modified, path)
        return

    if dry_run:
        typer.secho("=== Dry Run ===", fg=typer.colors.YELLOW, bold=True)
        typer.echo()
        typer.secho(f"Function: {result.function_name}", fg=typer.colors.GREEN)
        typer.secho(f"Calls to inline: {result.calls_inlined}", fg=typer.colors.BLUE)
        return

    path.write_text(modified)
    typer.secho(
        f"Inlined function '{result.function_name}' ({result.calls_inlined} calls)",
        fg=typer.colors.GREEN,
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
