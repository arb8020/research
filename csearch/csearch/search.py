"""Search functions - pure functions that return results."""

from csearch.types import SearchOptions, SearchResult
from csearch.backends.tree_sitter import search_definitions_tree_sitter, search_references_tree_sitter
from csearch.backends.ctags import search_definitions_ctags


def search_definitions(query: str, options: SearchOptions) -> list[SearchResult]:
    """Find symbol definitions matching query.

    Pure function: takes query + options, returns results.
    Dispatches to appropriate backend based on options.
    """
    if options.backend == "ctags":
        return search_definitions_ctags(query, options)
    elif options.backend == "tree-sitter":
        return search_definitions_tree_sitter(query, options)
    else:
        # auto: try tree-sitter first, fall back to ctags
        results = search_definitions_tree_sitter(query, options)
        if not results:
            results = search_definitions_ctags(query, options)
        return results


def search_references(query: str, options: SearchOptions) -> list[SearchResult]:
    """Find references to a symbol.

    Pure function: takes query + options, returns results.
    References are harder - tree-sitter can find call sites,
    but true reference finding needs LSP.
    """
    if options.backend == "ctags":
        # ctags doesn't support references, fall back to tree-sitter
        return search_references_tree_sitter(query, options)
    else:
        return search_references_tree_sitter(query, options)
