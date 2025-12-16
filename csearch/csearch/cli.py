"""CLI interface for csearch."""

from pathlib import Path

import typer

from csearch.search import search_definitions, search_references
from csearch.types import SearchOptions

app = typer.Typer(
    name="csearch",
    help="Code search CLI - find symbol definitions and references.",
    no_args_is_help=True,
)


@app.command()
def defs(
    query: str,
    path: Path = typer.Option(Path("."), "--path", "-p", help="Directory to search"),
    backend: str = typer.Option("auto", "--backend", "-b", help="tree-sitter, ctags, or auto"),
    snippet: bool = typer.Option(False, "--snippet", "-s", help="Include code snippets"),
    limit: int = typer.Option(None, "--limit", "-l", help="Maximum results"),
    no_limit: bool = typer.Option(False, "--no-limit", "-L", help="Return all results"),
) -> None:
    """Find symbol definitions (functions, classes, methods)."""
    # Resolve limit: --no-limit overrides --limit
    resolved_limit = None if no_limit else limit

    options = SearchOptions(
        path=path.resolve(),
        backend=backend,
        include_snippet=snippet,
        limit=resolved_limit,
    )

    results = search_definitions(query, options)

    if not results:
        typer.echo(f"No definitions found for '{query}'", err=True)
        raise typer.Exit(1)

    for result in results:
        if snippet:
            typer.echo(result.format_with_snippet())
            typer.echo("")  # blank line between results
        else:
            typer.echo(result.format_compact())


@app.command()
def refs(
    query: str,
    path: Path = typer.Option(Path("."), "--path", "-p", help="Directory to search"),
    backend: str = typer.Option("auto", "--backend", "-b", help="tree-sitter, ctags, or auto"),
    snippet: bool = typer.Option(False, "--snippet", "-s", help="Include code snippets"),
    limit: int = typer.Option(None, "--limit", "-l", help="Maximum results"),
    no_limit: bool = typer.Option(False, "--no-limit", "-L", help="Return all results"),
) -> None:
    """Find references to a symbol (call sites, usages)."""
    resolved_limit = None if no_limit else limit

    options = SearchOptions(
        path=path.resolve(),
        backend=backend,
        include_snippet=snippet,
        limit=resolved_limit,
    )

    results = search_references(query, options)

    if not results:
        typer.echo(f"No references found for '{query}'", err=True)
        raise typer.Exit(1)

    for result in results:
        if snippet:
            typer.echo(result.format_with_snippet())
            typer.echo("")
        else:
            typer.echo(result.format_compact())


@app.command()
def semantic(
    query: str,
    path: Path = typer.Option(Path("."), "--path", "-p", help="Directory to search"),
    snippet: bool = typer.Option(False, "--snippet", "-s", help="Include code snippets"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum results"),
) -> None:
    """Semantic search using embeddings (requires 'semantic' extra)."""
    typer.echo("Semantic search not yet implemented", err=True)
    raise typer.Exit(1)


@app.command()
def index(
    path: Path = typer.Argument(Path("."), help="Directory to index"),
    status: bool = typer.Option(False, "--status", help="Show index status"),
) -> None:
    """Build or check search index (trigram + ctags)."""
    from csearch.index import build_index, get_index_status
    from csearch.backends.ctags import build_ctags_index, ctags_available

    if status:
        status_info = get_index_status(path.resolve())
        typer.echo(status_info)
    else:
        resolved = path.resolve()
        typer.echo(f"Indexing {resolved}...")

        # Build trigram index
        stats = build_index(resolved)
        typer.echo(f"  Trigrams: {stats['files']} files, {stats['trigrams']} trigrams")

        # Build ctags index
        if ctags_available():
            ctags_stats = build_ctags_index(resolved)
            if "error" in ctags_stats:
                typer.echo(f"  Ctags: {ctags_stats['error']}", err=True)
            else:
                typer.echo(f"  Ctags: {ctags_stats['tags']} tags")
        else:
            typer.echo("  Ctags: not available (install universal-ctags for broader language support)")


if __name__ == "__main__":
    app()
