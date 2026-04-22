"""Provenance index over edits' add sets.

An edit's "add set" is the list of lines it actually introduced (as
opposed to context lines it merely preserved). See tool_diff.added_lines
for the precise definition.

The provenance index maps line text to the edits whose add set contains
that text. Reconcile queries this index per line of the current file:
"which edit added this exact text?"

Index is keyed by (path, text) so same-path matches are preferred — an
agent that added `    return None` to file A doesn't get credit for
`    return None` in file B.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .effects import FileEdit
from .tool_diff import added_lines


@dataclass
class ProvenanceIndex:
    """Maps line text to edits whose add set contains that text.

    by_text: any edit, across all paths (for cross-file moves)
    by_path_and_text: edits that added this line to *this specific path*.
                      Always preferred over cross-file matches.

    Both lists sorted latest-first.
    """

    by_text: dict[str, list[FileEdit]] = field(default_factory=lambda: defaultdict(list))
    by_path_and_text: dict[tuple[str, str], list[FileEdit]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def lookup(
        self, *, path: str, text: str, cross_file_allowed: bool,
    ) -> list[FileEdit]:
        same = self.by_path_and_text.get((path, text), [])
        if same:
            return same
        if cross_file_allowed:
            return self.by_text.get(text, [])
        return []


def build_provenance(edits: Iterable[FileEdit]) -> ProvenanceIndex:
    """Index each edit's add set only. Context lines don't get credited.

    This is the key change vs. the old index-everything-in-new_content
    behavior: an Edit that quotes 95 unchanged lines and adds 5 only
    claims those 5. The 95 context lines fall to whoever first wrote
    them (a prior Write, or an earlier Edit that added them).
    """
    idx = ProvenanceIndex()
    for edit in edits:
        for line in added_lines(edit):
            idx.by_text[line].append(edit)
            idx.by_path_and_text[(edit.path, line)].append(edit)
    for bucket in idx.by_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    for bucket in idx.by_path_and_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    return idx
