"""Tree-sitter backend for symbol search."""

from pathlib import Path

from csearch.types import Location, SearchOptions, SearchResult

# Language extension mapping
LANG_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# Tree-sitter queries for finding definitions by language
# These find function/class/method definitions
DEFINITION_QUERIES: dict[str, str] = {
    "python": """
        (function_definition name: (identifier) @name) @def
        (class_definition name: (identifier) @name) @def
    """,
    "javascript": """
        (function_declaration name: (identifier) @name) @def
        (class_declaration name: (identifier) @name) @def
        (method_definition name: (property_identifier) @name) @def
        (variable_declarator name: (identifier) @name value: (arrow_function)) @def
    """,
    "typescript": """
        (function_declaration name: (identifier) @name) @def
        (class_declaration name: (identifier) @name) @def
        (method_definition name: (property_identifier) @name) @def
        (variable_declarator name: (identifier) @name value: (arrow_function)) @def
    """,
    "go": """
        (function_declaration name: (identifier) @name) @def
        (method_declaration name: (field_identifier) @name) @def
        (type_declaration (type_spec name: (type_identifier) @name)) @def
    """,
    "rust": """
        (function_item name: (identifier) @name) @def
        (struct_item name: (type_identifier) @name) @def
        (impl_item type: (type_identifier) @name) @def
        (enum_item name: (type_identifier) @name) @def
    """,
}

# Queries for finding references (call sites)
REFERENCE_QUERIES: dict[str, str] = {
    "python": """
        (call function: (identifier) @name) @ref
        (call function: (attribute attribute: (identifier) @name)) @ref
    """,
    "javascript": """
        (call_expression function: (identifier) @name) @ref
        (call_expression function: (member_expression property: (property_identifier) @name)) @ref
    """,
    "typescript": """
        (call_expression function: (identifier) @name) @ref
        (call_expression function: (member_expression property: (property_identifier) @name)) @ref
    """,
    "go": """
        (call_expression function: (identifier) @name) @ref
        (call_expression function: (selector_expression field: (field_identifier) @name)) @ref
    """,
    "rust": """
        (call_expression function: (identifier) @name) @ref
        (call_expression function: (field_expression field: (field_identifier) @name)) @ref
    """,
}


def get_parser(lang: str):
    """Get tree-sitter parser for language. Lazy load to avoid import cost."""
    import tree_sitter

    if lang == "python":
        import tree_sitter_python as ts_python

        return tree_sitter.Parser(tree_sitter.Language(ts_python.language()))
    elif lang == "javascript":
        import tree_sitter_javascript as ts_js

        return tree_sitter.Parser(tree_sitter.Language(ts_js.language()))
    elif lang == "typescript":
        import tree_sitter_typescript as ts_ts

        return tree_sitter.Parser(tree_sitter.Language(ts_ts.language_typescript()))
    elif lang == "go":
        import tree_sitter_go as ts_go

        return tree_sitter.Parser(tree_sitter.Language(ts_go.language()))
    elif lang == "rust":
        import tree_sitter_rust as ts_rust

        return tree_sitter.Parser(tree_sitter.Language(ts_rust.language()))
    else:
        return None


def get_language(lang: str):
    """Get tree-sitter Language object."""
    import tree_sitter

    if lang == "python":
        import tree_sitter_python as ts_python

        return tree_sitter.Language(ts_python.language())
    elif lang == "javascript":
        import tree_sitter_javascript as ts_js

        return tree_sitter.Language(ts_js.language())
    elif lang == "typescript":
        import tree_sitter_typescript as ts_ts

        return tree_sitter.Language(ts_ts.language_typescript())
    elif lang == "go":
        import tree_sitter_go as ts_go

        return tree_sitter.Language(ts_go.language())
    elif lang == "rust":
        import tree_sitter_rust as ts_rust

        return tree_sitter.Language(ts_rust.language())
    else:
        return None


IGNORED_DIRS = {
    ".venv",
    "venv",
    "node_modules",
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "build",
    "dist",
    ".tox",
    ".eggs",
}


def find_files(root: Path, extensions: set[str]) -> list[Path]:
    """Find all files with given extensions under root, skipping ignored dirs."""
    files = []
    for ext in extensions:
        for path in root.rglob(f"*{ext}"):
            # Skip if any parent is in ignored dirs
            if not any(part in IGNORED_DIRS for part in path.parts):
                files.append(path)
    return files


def extract_snippet(file_path: Path, start_line: int, end_line: int) -> str:
    """Extract lines from file."""
    lines = file_path.read_text().splitlines()
    # Convert to 0-indexed
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    return "\n".join(lines[start_idx:end_idx])


def run_query(language, query_str: str, root_node) -> list[tuple[int, dict]]:
    """Run a tree-sitter query and return matches.

    Returns list of (pattern_index, {capture_name: [nodes]})
    """
    import tree_sitter

    query = tree_sitter.Query(language, query_str)
    cursor = tree_sitter.QueryCursor(query)
    return list(cursor.matches(root_node))


def get_files_to_search(query: str, options: SearchOptions) -> list[Path]:
    """Get files to search, using trigram index if available."""
    from csearch.index import query_index

    # Try to use index
    candidate_files = query_index(options.path, query)

    if candidate_files is not None:
        # Filter to supported extensions
        supported_exts = set(LANG_EXTENSIONS.keys())
        return [f for f in candidate_files if f.suffix in supported_exts]

    # Fall back to full scan
    supported_exts = set(LANG_EXTENSIONS.keys())
    return find_files(options.path, supported_exts)


def search_definitions_tree_sitter(query: str, options: SearchOptions) -> list[SearchResult]:
    """Search for symbol definitions using tree-sitter.

    Args:
        query: Symbol name to search for (exact match)
        options: Search options

    Returns:
        List of matching definitions
    """
    results: list[SearchResult] = []

    files = get_files_to_search(query, options)

    for file_path in files:
        ext = file_path.suffix
        lang = LANG_EXTENSIONS.get(ext)
        if not lang or lang not in DEFINITION_QUERIES:
            continue

        parser = get_parser(lang)
        language = get_language(lang)
        if not parser or not language:
            continue

        try:
            source = file_path.read_bytes()
            tree = parser.parse(source)

            matches = run_query(language, DEFINITION_QUERIES[lang], tree.root_node)

            for _pattern_idx, captures in matches:
                name_nodes = captures.get("name", [])
                def_nodes = captures.get("def", [])

                if not name_nodes or not def_nodes:
                    continue

                name_node = name_nodes[0]
                def_node = def_nodes[0]

                name_text = name_node.text.decode("utf-8")
                if name_text != query:
                    continue

                start_line = def_node.start_point[0] + 1  # 1-indexed
                end_line = def_node.end_point[0] + 1

                kind = def_node.type.replace("_definition", "").replace("_declaration", "")

                snippet = None
                if options.include_snippet:
                    snippet = extract_snippet(file_path, start_line, end_line)

                results.append(
                    SearchResult(
                        location=Location(
                            path=file_path.relative_to(options.path),
                            line_start=start_line,
                            line_end=end_line,
                        ),
                        name=name_text,
                        kind=kind,
                        snippet=snippet,
                    )
                )

                if options.limit is not None and len(results) >= options.limit:
                    return results

        except Exception:
            # Skip files that fail to parse
            continue

    return results


def search_references_tree_sitter(query: str, options: SearchOptions) -> list[SearchResult]:
    """Search for symbol references (call sites) using tree-sitter.

    Args:
        query: Symbol name to search for
        options: Search options

    Returns:
        List of matching references
    """
    results: list[SearchResult] = []

    files = get_files_to_search(query, options)

    for file_path in files:
        ext = file_path.suffix
        lang = LANG_EXTENSIONS.get(ext)
        if not lang or lang not in REFERENCE_QUERIES:
            continue

        parser = get_parser(lang)
        language = get_language(lang)
        if not parser or not language:
            continue

        try:
            source = file_path.read_bytes()
            tree = parser.parse(source)

            matches = run_query(language, REFERENCE_QUERIES[lang], tree.root_node)

            for _pattern_idx, captures in matches:
                name_nodes = captures.get("name", [])
                ref_nodes = captures.get("ref", [])

                if not name_nodes or not ref_nodes:
                    continue

                name_node = name_nodes[0]
                ref_node = ref_nodes[0]

                name_text = name_node.text.decode("utf-8")
                if name_text != query:
                    continue

                start_line = ref_node.start_point[0] + 1
                end_line = ref_node.end_point[0] + 1

                snippet = None
                if options.include_snippet:
                    snippet = extract_snippet(file_path, start_line, end_line)

                results.append(
                    SearchResult(
                        location=Location(
                            path=file_path.relative_to(options.path),
                            line_start=start_line,
                            line_end=end_line,
                        ),
                        name=name_text,
                        kind="call",
                        snippet=snippet,
                    )
                )

                if options.limit is not None and len(results) >= options.limit:
                    return results

        except Exception:
            continue

    return results
