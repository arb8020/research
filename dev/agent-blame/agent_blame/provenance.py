"""Flat line-provenance index: which FileEdits ever emitted a given line text.

Independent of fold. Where fold tracks 'the current state of this file if
we replay every edit in order' (and breaks on cross-session drift), the
provenance index simply asks: 'across every `new_content` of every edit
we've ever seen, who wrote a line with this exact text?'

Reconcile uses both. The virtual-state attribution (via fold) is preferred
when it hits, because it carries intra-session coherence: consecutive
unchanged lines get attributed to the session that wrote the surrounding
block, not to whoever else happened to emit the same line text elsewhere.

When virtual state has no hit — because the session that wrote this line
went stale mid-chain, or because the line came from a shell-edit we
couldn't fold — the provenance index is the honest fallback: if some
agent ever wrote this exact text, credit them (latest by timestamp wins).

## What this buys us

- Stale edits still contribute to attribution. Their `new_content` lines
  are in the provenance index even though fold couldn't apply them.
- Non-fold-able edit sources (shell heredocs, `sed -i`, python one-liners
  emitted by Bash tool calls) plug in cleanly: just feed their FileEdits
  into the same index. Fold may ignore them; provenance doesn't care.
- No fake positions. Provenance match is content-only — we never invent
  a line number that wasn't observed.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

from .effects import FileEdit


@dataclass
class ProvenanceIndex:
    """Maps line text to FileEdits that emitted that text in their new_content.

    Two indices:
        by_text          — any edit, across all paths
        by_path_and_text — keyed (abs_path, text); edits that emitted this
                           line to *this specific path*. Always preferred
                           over cross-file matches since a same-path hit
                           is a strong signal of authorship.

    Both lists are sorted latest-first so `[0]` is the most recent claim.
    """

    by_text: dict[str, list[FileEdit]] = field(default_factory=lambda: defaultdict(list))
    by_path_and_text: dict[tuple[str, str], list[FileEdit]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def lookup(
        self, *, path: str, text: str, cross_file_allowed: bool,
    ) -> list[FileEdit]:
        """Return candidate edits for `text` at `path`, same-path preferred.

        If `cross_file_allowed` is False, only same-path matches are returned
        — used to suppress short-line cross-file noise at reconcile time.
        """
        same = self.by_path_and_text.get((path, text), [])
        if same:
            return same
        if cross_file_allowed:
            return self.by_text.get(text, [])
        return []


def build_provenance(edits: Iterable[FileEdit]) -> ProvenanceIndex:
    """Index every line of every edit's `new_content`.

    For `write` edits, every line of the new file is indexed.
    For `edit` edits, every line of the replacement text is indexed.

    We intentionally do not try to subtract 'context' lines that appeared
    in old_content too — those lines *did* appear in the agent's output,
    and if they survive in the repo the agent's involvement with them is
    real (even if weaker than a pure add). Treating context as authored
    is a conservative overcount we accept for v0.
    """
    idx = ProvenanceIndex()
    for edit in edits:
        lines = edit.new_content.split("\n")
        # Strip trailing empty for files-ending-in-newline, same convention
        # as reconcile. Avoids counting the synthetic "" as authored content.
        if lines and lines[-1] == "":
            lines = lines[:-1]
        for line in lines:
            idx.by_text[line].append(edit)
            idx.by_path_and_text[(edit.path, line)].append(edit)
    # Sort each bucket latest-first.
    for bucket in idx.by_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    for bucket in idx.by_path_and_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    return idx
