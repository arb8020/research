"""Reconcile attribution against a source (working tree, git SHA, ...).

The fold produces a per-file virtual state. The provenance index maps every
line text ever emitted by an agent to the edits that emitted it. Reconcile
joins both against a `source` — a callable that returns the current text
of a file — and emits per-line attributions for every file the source
knows about.

## Attribution order

For each line of each source file we try, in order:

    1. Virtual-state same-path match   (strongest; fold's intra-session coherence)
    2. Provenance same-path match       (any agent wrote this line to this file)
    3. Virtual-state cross-file match   (only if line is distinctive enough)
    4. Provenance cross-file match      (same distinctiveness guard)
    5. Unknown                          (no agent we know of wrote this text)

Virtual-state matches are preferred over provenance because they benefit
from intra-session chaining: an unchanged line in a session's virtual
state keeps its original author across later edits, whereas provenance
only sees the raw `new_content` of individual edits.

## Distinctiveness

Short or whitespace-heavy lines (`}`, ``, `    return`) match ubiquitously
across files. We require ≥20 non-whitespace characters for cross-file
matches in both virtual-state and provenance. Same-path matches have no
such threshold — being at the same path is already a strong signal.

## Why no `git blame`

We do not join against `git blame` SHAs. Unless commits are stamped with
session IDs (the approach this project deliberately avoids), SHAs are
not useful for joining with transcripts. We match by content instead.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .effects import FileEdit
from .fold import FileState
from .provenance import ProvenanceIndex
from .sources import SourceReader

CROSS_FILE_MIN_CHARS = 20
"""Minimum stripped length for a line to be eligible for cross-file match.

Below this threshold, only same-path matches are allowed. This keeps
`}`, blank lines, and short boilerplate from getting attributed en masse
to any agent that ever wrote such a line.
"""


@dataclass(frozen=True)
class LineAttribution:
    line_number: int  # 1-indexed, like git blame
    text: str
    edit: FileEdit | None  # None means 'unknown'
    ambiguous: bool = False
    match_kind: str = "unknown"
    """Which attribution strategy hit:
        'virtual_same_path' | 'virtual_cross_file'
        | 'provenance_same_path' | 'provenance_cross_file'
        | 'unknown'"""


@dataclass
class FileAttribution:
    repo_path: Path  # repo-relative
    lines: list[LineAttribution]

    @property
    def attributed_count(self) -> int:
        return sum(1 for l in self.lines if l.edit is not None)

    @property
    def unknown_count(self) -> int:
        return sum(1 for l in self.lines if l.edit is None)


# --------------------------------------------------------------------------
# Virtual-state indices
# --------------------------------------------------------------------------


def _index_virtual_state(
    states: dict[str, FileState],
) -> tuple[dict[str, list[FileEdit]], dict[tuple[str, str], list[FileEdit]]]:
    """Build (by_text, by_path_and_text) from the fold output.

    Seeded (edit=None) lines are excluded — they represent content whose
    origin we don't know, and we must not fabricate attribution for them.
    """
    by_text: dict[str, list[FileEdit]] = defaultdict(list)
    by_path_and_text: dict[tuple[str, str], list[FileEdit]] = defaultdict(list)
    for path, state in states.items():
        for attr in state.lines:
            if attr.edit is None:
                continue
            by_text[attr.text].append(attr.edit)
            by_path_and_text[(path, attr.text)].append(attr.edit)
    for bucket in by_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    for bucket in by_path_and_text.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    return by_text, by_path_and_text


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------


def reconcile(
    *,
    repo_root: Path,
    virtual_states: dict[str, FileState],
    provenance: ProvenanceIndex,
    source: SourceReader,
    files: Iterable[Path],
) -> list[FileAttribution]:
    """Attribute every line of every file in `files` via the source reader.

    Args:
        repo_root: for producing repo-relative paths in output.
        virtual_states: fold output.
        provenance: provenance index (built from the same edits as fold).
        source: callable(abs_path) -> text | None. Working-tree or git-sha.
        files: absolute or repo-relative paths to reconcile.

    Returns: one FileAttribution per readable file, in input order.
    """
    v_by_text, v_by_path_and_text = _index_virtual_state(virtual_states)

    out: list[FileAttribution] = []
    for fpath in files:
        abs_path = (repo_root / fpath).resolve() if not fpath.is_absolute() else fpath
        text = source(str(abs_path))
        if text is None:
            continue
        rel = abs_path.relative_to(repo_root) if abs_path.is_absolute() else fpath
        lines = text.split("\n")
        # File ending in newline -> trailing "" which is not a real line.
        if lines and lines[-1] == "":
            lines = lines[:-1]

        abs_str = str(abs_path)
        attributions: list[LineAttribution] = []
        for i, line_text in enumerate(lines, start=1):
            stripped = line_text.strip()
            cross_file_allowed = len(stripped) >= CROSS_FILE_MIN_CHARS

            # 1. Virtual same-path
            cands = v_by_path_and_text.get((abs_str, line_text), [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "virtual_same_path"))
                continue

            # 2. Provenance same-path
            cands = provenance.by_path_and_text.get((abs_str, line_text), [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "provenance_same_path"))
                continue

            if not cross_file_allowed:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=None,
                ))
                continue

            # 3. Virtual cross-file
            cands = v_by_text.get(line_text, [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "virtual_cross_file"))
                continue

            # 4. Provenance cross-file
            cands = provenance.by_text.get(line_text, [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "provenance_cross_file"))
                continue

            # 5. Unknown
            attributions.append(LineAttribution(
                line_number=i, text=line_text, edit=None,
            ))
        out.append(FileAttribution(repo_path=rel, lines=attributions))
    return out


def _make_attr(
    line_number: int, text: str, cands: list[FileEdit], match_kind: str,
) -> LineAttribution:
    distinct_sessions = {e.session_id for e in cands}
    return LineAttribution(
        line_number=line_number,
        text=text,
        edit=cands[0],  # latest-first sort done at index time
        ambiguous=len(distinct_sessions) > 1,
        match_kind=match_kind,
    )


# --------------------------------------------------------------------------
# Summary helpers
# --------------------------------------------------------------------------


def summary_stats(attributions: list[FileAttribution]) -> dict[str, int]:
    total = sum(len(a.lines) for a in attributions)
    attributed = sum(a.attributed_count for a in attributions)
    # Per-match-kind breakdown is useful to judge whether fold or
    # provenance is carrying the attribution.
    kinds: Counter[str] = Counter()
    for fa in attributions:
        for line in fa.lines:
            kinds[line.match_kind] += 1
    return {
        "total_lines": total,
        "attributed": attributed,
        "unknown": total - attributed,
        **{f"kind_{k}": v for k, v in kinds.items()},
    }


def top_sessions_by_lines(
    attributions: list[FileAttribution], n: int = 10,
) -> list[tuple[str, str, int]]:
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
