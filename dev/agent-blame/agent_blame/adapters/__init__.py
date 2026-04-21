"""Adapters: source-specific transcript parsers producing FileEdit streams.

Each adapter in this package exports a single top-level function:

    def iter_file_edits(session_path: Path) -> Iterable[FileEdit]

The adapter does not filter, sort, or fold. Its job is to normalize a
source's on-disk transcript format into the common FileEdit type, nothing
more. Fold and reconcile run downstream on the merged stream from all
adapters.
"""
