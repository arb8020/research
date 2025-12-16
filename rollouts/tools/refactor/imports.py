"""
Import parsing and context collection.

Follows imports recursively to build a context map of all relevant files.
Supports Python, JavaScript/TypeScript, and Lua.
"""

import re
from pathlib import Path


def detect_language(file_path: Path) -> str:
    """Detect language from file extension."""
    suffix = file_path.suffix.lower()

    if suffix == ".py":
        return "python"
    elif suffix in (".js", ".jsx"):
        return "javascript"
    elif suffix in (".ts", ".tsx"):
        return "typescript"
    elif suffix == ".lua":
        return "lua"
    elif suffix in (".c", ".cpp", ".h", ".hpp", ".cc"):
        return "cpp"
    else:
        return "unknown"


def parse_imports(content: str, language: str, file_path: Path) -> list[Path]:
    """
    Extract import paths from file content.
    Returns list of resolved absolute paths.
    """
    imports: list[Path] = []
    file_dir = file_path.parent

    if language == "python":
        imports.extend(_parse_python_imports(content, file_dir))
    elif language in ("javascript", "typescript"):
        imports.extend(_parse_js_imports(content, file_dir))
    elif language == "lua":
        imports.extend(_parse_lua_imports(content, file_dir))
    elif language == "cpp":
        imports.extend(_parse_cpp_imports(content, file_dir))

    # Also check for Victor's custom markers (works in any language)
    imports.extend(_parse_custom_markers(content, file_dir))

    return imports


def _parse_python_imports(content: str, file_dir: Path) -> list[Path]:
    """
    Parse Python imports. Only follows relative imports (from . or from ..).
    Absolute imports (from foo import bar) are skipped as they're likely stdlib/packages.
    """
    imports = []

    # from .foo import bar  OR  from ..foo import bar
    # Group 1: the dots + module path
    relative_pattern = re.compile(r"^from\s+(\.+[\w.]*)\s+import", re.MULTILINE)

    for match in relative_pattern.finditer(content):
        import_path = match.group(1)
        resolved = _resolve_python_relative_import(import_path, file_dir)
        if resolved and resolved.exists():
            imports.append(resolved)

    return imports


def _resolve_python_relative_import(import_path: str, file_dir: Path) -> Path | None:
    """Resolve a Python relative import like '.foo' or '..foo.bar'."""
    # Count leading dots
    dots = 0
    for char in import_path:
        if char == ".":
            dots += 1
        else:
            break

    # Navigate up directories
    base_dir = file_dir
    for _ in range(dots - 1):  # -1 because . means current dir
        base_dir = base_dir.parent

    # Get module path after dots
    module_part = import_path[dots:]
    if not module_part:
        return None

    # Convert module.submodule to path
    path_parts = module_part.split(".")
    resolved = base_dir / "/".join(path_parts)

    # Try as file
    if (resolved.with_suffix(".py")).exists():
        return resolved.with_suffix(".py")

    # Try as package
    if (resolved / "__init__.py").exists():
        return resolved / "__init__.py"

    return None


def _parse_js_imports(content: str, file_dir: Path) -> list[Path]:
    """Parse JavaScript/TypeScript imports. Only follows relative imports."""
    imports = []

    patterns = [
        # import x from './path'  OR  import x from "../path"
        re.compile(r"""import\s+.*?\s+from\s+['"](\.[^'"]+)['"]"""),
        # import('./path')
        re.compile(r"""import\s*\(\s*['"](\.[^'"]+)['"]\s*\)"""),
        # require('./path')
        re.compile(r"""require\s*\(\s*['"](\.[^'"]+)['"]\s*\)"""),
    ]

    for pattern in patterns:
        for match in pattern.finditer(content):
            import_path = match.group(1)
            resolved = _resolve_js_import(import_path, file_dir)
            if resolved:
                imports.append(resolved)

    return imports


def _resolve_js_import(import_path: str, file_dir: Path) -> Path | None:
    """Resolve a JS/TS import path."""
    base = file_dir / import_path

    # Try exact path
    if base.exists() and base.is_file():
        return base

    # Try with extensions
    extensions = [".ts", ".tsx", ".js", ".jsx"]
    for ext in extensions:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate

    # Try as directory with index
    for ext in extensions:
        candidate = base / f"index{ext}"
        if candidate.exists():
            return candidate

    return None


def _parse_lua_imports(content: str, file_dir: Path) -> list[Path]:
    """Parse Lua require statements."""
    imports = []

    # require('path') or require "path"
    pattern = re.compile(r"""require\s*\(?['"]([^'"]+)['"]\)?""")

    for match in pattern.finditer(content):
        module_path = match.group(1)

        # Skip if looks like a package (no ./ prefix and no dots suggesting path)
        if not module_path.startswith(".") and "/" not in module_path:
            # Could be 'foo.bar' style - convert to path
            path_str = module_path.replace(".", "/")
        else:
            path_str = module_path

        resolved = file_dir / path_str

        # Try with .lua extension
        if (resolved.with_suffix(".lua")).exists():
            imports.append(resolved.with_suffix(".lua"))
        elif (resolved / "init.lua").exists():
            imports.append(resolved / "init.lua")

    return imports


def _parse_cpp_imports(content: str, file_dir: Path) -> list[Path]:
    """Parse C/C++ #include statements (only quoted, not angle brackets)."""
    imports = []

    # #include "path"
    pattern = re.compile(r'^#include\s+"([^"]+)"', re.MULTILINE)

    for match in pattern.finditer(content):
        include_path = match.group(1)
        resolved = file_dir / include_path
        if resolved.exists():
            imports.append(resolved)

    return imports


def _parse_custom_markers(content: str, file_dir: Path) -> list[Path]:
    """
    Parse Victor's custom import markers:
    #[./path]
    //[./path]
    --[./path]
    """
    imports = []

    patterns = [
        re.compile(r"^#\[(\./[^\]]+)\]$", re.MULTILINE),
        re.compile(r"^//\[(\./[^\]]+)\]$", re.MULTILINE),
        re.compile(r"^--\[(\./[^\]]+)\]$", re.MULTILINE),
    ]

    for pattern in patterns:
        for match in pattern.finditer(content):
            import_path = match.group(1)
            resolved = file_dir / import_path
            if resolved.exists():
                imports.append(resolved)

    return imports


def collect_context(
    root_file: Path,
    workspace_root: Path | None = None,
) -> dict[str, str]:
    """
    Recursively collect all files imported by root_file.

    Returns a dict mapping relative paths to file contents.
    """
    root_file = Path(root_file).resolve()

    if workspace_root is None:
        workspace_root = root_file.parent
    else:
        workspace_root = Path(workspace_root).resolve()

    context: dict[str, str] = {}
    visited: set[Path] = set()

    def visit(file_path: Path) -> None:
        file_path = file_path.resolve()

        # Skip if already visited
        if file_path in visited:
            return
        visited.add(file_path)

        # Skip if outside workspace
        try:
            file_path.relative_to(workspace_root)
        except ValueError:
            return

        # Read file
        try:
            content = file_path.read_text()
        except (OSError, UnicodeDecodeError):
            return

        # Add to context
        rel_path = file_path.relative_to(workspace_root)
        context[str(rel_path)] = content

        # Parse and follow imports
        language = detect_language(file_path)
        imports = parse_imports(content, language, file_path)

        for import_path in imports:
            visit(import_path)

    visit(root_file)
    return context
