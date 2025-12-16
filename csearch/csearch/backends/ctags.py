"""Ctags backend for symbol search.

Uses universal-ctags to find symbol definitions.
Ctags is broader (more languages) but less precise than tree-sitter.

Two modes:
1. Use pre-built tags file (.csearch.tags) if available
2. Fall back to running ctags on-demand (slower)
"""

import subprocess
from pathlib import Path

from csearch.types import Location, SearchOptions, SearchResult

TAGS_FILENAME = ".csearch.tags"


def ctags_available() -> bool:
    """Check if universal-ctags is available.

    We need universal-ctags, not BSD ctags (which ships with macOS).
    Universal-ctags supports --version and -R flags.
    """
    try:
        result = subprocess.run(
            ["ctags", "--version"],
            capture_output=True,
            text=True,
        )
        # BSD ctags fails on --version, universal-ctags succeeds
        if result.returncode != 0:
            return False
        # Also check it's actually universal-ctags
        return "Universal Ctags" in result.stdout or "Exuberant Ctags" in result.stdout
    except FileNotFoundError:
        return False


def get_tags_path(root: Path) -> Path:
    """Get path to tags file for a directory."""
    return root / TAGS_FILENAME


def build_ctags_index(root: Path) -> dict:
    """Build ctags index for a directory.

    Returns stats dict with 'tags' count.
    """
    if not ctags_available():
        return {"tags": 0, "error": "ctags not available"}

    tags_path = get_tags_path(root)

    try:
        result = subprocess.run(
            [
                "ctags",
                "-R",
                "--fields=+n",  # include line numbers
                "-f", str(tags_path),
                ".",  # use relative path from cwd
            ],
            capture_output=True,
            text=True,
            cwd=root,
        )

        if result.returncode != 0:
            return {"tags": 0, "error": result.stderr}

        # Count tags
        tag_count = 0
        if tags_path.exists():
            with open(tags_path) as f:
                for line in f:
                    if not line.startswith("!"):
                        tag_count += 1

        return {"tags": tag_count}

    except Exception as e:
        return {"tags": 0, "error": str(e)}


def parse_tags_file(tags_path: Path, query: str) -> list[dict]:
    """Parse tags file and find matching entries.

    Tags file format (tab-separated):
    {tagname}<Tab>{tagfile}<Tab>{tagaddress}[;<Tab>{kind}[<Tab>{field}...]]

    Example:
    MyClass	src/foo.py	/^class MyClass:$/;"	c	line:10
    """
    if not tags_path.exists():
        return []

    results = []
    with open(tags_path) as f:
        for line in f:
            if line.startswith("!"):
                # Skip metadata lines
                continue

            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue

            name = parts[0]
            if name != query:
                continue

            file_path = parts[1]

            # Parse kind and line number from remaining fields
            kind = "unknown"
            line_num = 1

            for part in parts[3:]:
                if part.startswith("line:"):
                    try:
                        line_num = int(part[5:])
                    except ValueError:
                        pass
                elif len(part) == 1 or (len(part) > 1 and part[1] == "\t"):
                    # Single char kind like 'f' for function, 'c' for class
                    kind = expand_ctags_kind(part[0])

            results.append({
                "name": name,
                "path": file_path,
                "line": line_num,
                "kind": kind,
            })

    return results


def expand_ctags_kind(short: str) -> str:
    """Expand single-char ctags kind to readable name."""
    kinds = {
        "c": "class",
        "f": "function",
        "m": "method",
        "v": "variable",
        "s": "struct",
        "t": "type",
        "e": "enum",
        "i": "interface",
        "n": "namespace",
        "p": "property",
    }
    return kinds.get(short, short)


def extract_snippet(file_path: Path, line: int, context: int = 10) -> str:
    """Extract lines around a match.

    Ctags only gives us a line number, not the full extent.
    We extract some context.
    """
    try:
        lines = file_path.read_text().splitlines()
        start = max(0, line - 1)
        end = min(len(lines), line + context)
        return "\n".join(lines[start:end])
    except Exception:
        return ""


def search_definitions_ctags(query: str, options: SearchOptions) -> list[SearchResult]:
    """Search for symbol definitions using ctags.

    Uses pre-built tags file if available, otherwise runs ctags on-demand.
    """
    tags_path = get_tags_path(options.path)

    if tags_path.exists():
        # Use indexed tags file
        tags = parse_tags_file(tags_path, query)
    else:
        # Fall back to on-demand ctags (slower)
        tags = run_ctags_ondemand(options.path, query)

    results: list[SearchResult] = []

    for tag in tags:
        name = tag.get("name", "")
        file_path = tag.get("path", "")
        line = tag.get("line", 1)
        kind = tag.get("kind", "unknown")

        if not file_path:
            continue

        full_path = options.path / file_path

        snippet = None
        if options.include_snippet:
            snippet = extract_snippet(full_path, line)

        results.append(
            SearchResult(
                location=Location(
                    path=Path(file_path),
                    line_start=line,
                    line_end=line,  # ctags doesn't give us end line
                ),
                name=name,
                kind=kind,
                snippet=snippet,
            )
        )

        if options.limit is not None and len(results) >= options.limit:
            break

    return results


def run_ctags_ondemand(path: Path, query: str) -> list[dict]:
    """Run ctags on-demand and filter results.

    This is slow - prefer using a pre-built tags file.
    """
    if not ctags_available():
        return []

    try:
        # Run ctags with JSON output
        result = subprocess.run(
            [
                "ctags",
                "-R",
                "--output-format=json",
                "--fields=+n",
                str(path),
            ],
            capture_output=True,
            text=True,
            cwd=path,
        )

        if result.returncode != 0:
            return []

        import json

        tags = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            try:
                tag = json.loads(line)
                if tag.get("name") == query:
                    tags.append(tag)
            except json.JSONDecodeError:
                continue

        return tags

    except Exception:
        return []
