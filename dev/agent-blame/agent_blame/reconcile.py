"""Reconcile virtual file state with the current repo.

The fold produced a per-file virtual state: "if you'd replayed every agent
edit, here's what each file would look like and who wrote each line." The
repo on disk is the ground truth of what files *actually* look like today.

Reconciliation is the join. For each line in the current repo file:

    - If the line's text matches a line in the virtual state, we attribute
      it to that line's edit.
    - If no match, we attribute it to "unknown" (human-authored, or
      pre-agent, or the agent-edit didn't survive rebase/formatter).

We match by line content only, not position. This handles the common
cases: auto-formatters reshuffle whitespace, humans insert lines above
agent code, git rebase merges branches. As long as the *text of the line*
is unique enough, we can still find it.

## Ambiguity

Short lines like `}`, ``, `return` appear many times across virtual
states. When a repo line matches multiple candidates, we pick the most
recent edit by timestamp. If the match is across different sessions with
similar timestamps, the result is under-specified — reconcile returns the
pick but flags ambiguity, leaving the UI layer to decide whether to show
"multiple candidates."

## Why not `git blame`

`git blame` gives us commit SHAs, not session IDs. Unless we committed
session metadata into commit trailers (the approach we deliberately did
not take), blame SHAs are not useful for joining with transcripts. We
use current file content directly and let content hash carry the join.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .effects import FileEdit
from .fold import FileState


@dataclass(frozen=True)
class LineAttribution:
    """Attribution for one line of a current repo file."""

    line_number: int  # 1-indexed, like git blame
    text: str
    edit: FileEdit | None  # None means "unknown" (human or unmatched)
    ambiguous: bool = False
    """True if multiple virtual-state lines matched this line's text."""


@dataclass
class FileAttribution:
    """Per-file line-by-line attribution, ready to print or serialize."""

    repo_path: Path
    """Path relative to repo root."""
    lines: list[LineAttribution]

    @property
    def attributed_count(self) -> int:
        return sum(1 for l in self.lines if l.edit is not None)

    @property
    def unknown_count(self) -> int:
        return sum(1 for l in self.lines if l.edit is None)


def _index_virtual_state(
    states: dict[str, FileState],
) -> dict[str, list[FileEdit]]:
    """Flatten fold output to `line_text -> [edits that produced this text]`.

    Sorted latest-first so the first match in reconcile is the most recent.
    """
    by_text: dict[str, list[FileEdit]] = defaultdict(list)
    for state in states.values():
        for attr in state.lines:
            # Seeded (unattributed) lines don't help us attribute anything —
            # they represent content whose origin we do not know. Exclude.
            if attr.edit is None:
                continue
            by_text[attr.text].append(attr.edit)
    for text, edits in by_text.items():
        edits.sort(key=lambda e: e.timestamp, reverse=True)
    return by_text


def reconcile_repo(
    repo_root: Path,
    virtual_states: dict[str, FileState],
    repo_files: Iterable[Path],
) -> list[FileAttribution]:
    """Attribute every line of every repo file to a FileEdit or None.

    Args:
        repo_root: Absolute path to the repo root. Used to produce
            repo-relative paths in the output.
        virtual_states: Fold output, keyed by the absolute path the agent
            wrote to.
        repo_files: Paths (absolute or repo-relative) to reconcile. Binary
            or non-text files should be filtered upstream.

    Returns:
        One FileAttribution per file, in the order `repo_files` yielded them.
    """
    by_text = _index_virtual_state(virtual_states)

    # Secondary index: same line text might repeat within one file's virtual
    # state; we want to prefer matches from a virtual state for the *same*
    # file path over cross-file matches, to reduce cross-file noise.
    virtual_by_path_and_text: dict[tuple[str, str], list[FileEdit]] = defaultdict(list)
    for path, state in virtual_states.items():
        for attr in state.lines:
            if attr.edit is None:
                continue  # seeded lines don't attribute
            virtual_by_path_and_text[(path, attr.text)].append(attr.edit)
    for k, edits in virtual_by_path_and_text.items():
        edits.sort(key=lambda e: e.timestamp, reverse=True)

    out: list[FileAttribution] = []
    for fpath in repo_files:
        abs_path = (repo_root / fpath).resolve() if not fpath.is_absolute() else fpath
        try:
            text = abs_path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        rel = abs_path.relative_to(repo_root) if abs_path.is_absolute() else fpath
        lines = text.split("\n")
        # A file ending in "\n" splits to a trailing "" — don't attribute it.
        if lines and lines[-1] == "":
            lines = lines[:-1]

        abs_str = str(abs_path)
        attributions: list[LineAttribution] = []
        for i, line_text in enumerate(lines, start=1):
            # Same-path match is always acceptable — the agent wrote to this
            # exact path, so matching line text in the same virtual-state
            # file is a strong signal.
            same_path_candidates = virtual_by_path_and_text.get((abs_str, line_text), [])
            # Cross-file match is only acceptable for distinctive lines.
            # Short/whitespace-only lines (`}`, ``, `    return`) match
            # ubiquitously and would inflate attribution. Threshold chosen
            # empirically: 20 non-whitespace chars filters most glue lines
            # while retaining signatures, docstrings, and most real code.
            stripped = line_text.strip()
            cross_file_allowed = len(stripped) >= 20
            cross_file_candidates = (
                by_text.get(line_text, []) if cross_file_allowed else []
            )
            candidates = same_path_candidates or cross_file_candidates
            if not candidates:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=None,
                ))
                continue
            # Latest by timestamp wins; flag ambiguity if multiple distinct
            # sessions competed for the same line text.
            distinct_sessions = {e.session_id for e in candidates}
            attributions.append(LineAttribution(
                line_number=i,
                text=line_text,
                edit=candidates[0],
                ambiguous=len(distinct_sessions) > 1,
            ))
        out.append(FileAttribution(repo_path=rel, lines=attributions))
    return out


def summary_stats(attributions: list[FileAttribution]) -> dict[str, int]:
    """Aggregate across all files."""
    total = sum(len(a.lines) for a in attributions)
    attributed = sum(a.attributed_count for a in attributions)
    return {
        "total_lines": total,
        "attributed": attributed,
        "unknown": total - attributed,
    }


def top_sessions_by_lines(
    attributions: list[FileAttribution], n: int = 10,
) -> list[tuple[str, str, int]]:
    """Return top-N (source, session_id, line_count) by lines attributed.

    Ties broken by source+session_id for determinism.
    """
    counter: Counter[tuple[str, str]] = Counter()
    for fa in attributions:
        for line in fa.lines:
            if line.edit is not None:
                counter[(line.edit.source, line.edit.session_id)] += 1
    items = sorted(
        counter.items(),
        key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
    )[:n]
    return [(source, sid, count) for (source, sid), count in items]
