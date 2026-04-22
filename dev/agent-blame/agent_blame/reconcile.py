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
from .tool_diff import added_lines

CROSS_FILE_MIN_CHARS = 20
"""Minimum stripped length for a line to be eligible for cross-file match.

Below this threshold, only same-path matches are allowed. This keeps
`}`, blank lines, and short boilerplate from getting attributed en masse
to any agent that ever wrote such a line.
"""

BLOCK_SHINGLE_K = 4
"""Length of line-shingles used to seed block matches.

Lifted from MOSS-style winnowing (Schleimer, Wilkerson, Aiken, 2003): an
exact K-line match guarantees the two sources share a run of at least K
consecutive identical lines. K=4 is the sweet spot for source code —
short enough that small agent edits still match, long enough that
coincidental matches (a 3-line `if x: /    y = z /    return`) stay
below the threshold.
"""

SHINGLE_MIN_SUBSTANTIAL_LINES = 2
"""How many of the K lines in a shingle must be "substantial" (stripped
length >= 8) for the shingle to be admissible.

Prevents the classic duplicate-line bug: a shingle of four blank lines
matches anywhere four blanks happen to appear. Requiring two substantial
anchor lines inside the shingle ties every shingle to real content. This
is patience-diff's insight adapted for shingles — anchor on rare lines,
tolerate blanks *between* anchors.
"""

SHINGLE_SUBSTANTIAL_CHARS = 8
"""Stripped-length threshold for a line to be 'substantial' in a shingle.

Less strict than CROSS_FILE_MIN_CHARS (20) because a substantial line
inside a shingle only needs to be distinctive enough to rule out
blank/bracket-only shingles — the full K-line match provides the other
discriminating signal.
"""


@dataclass(frozen=True)
class LineAttribution:
    line_number: int  # 1-indexed, like git blame
    text: str
    edit: FileEdit | None  # None means 'unknown'
    ambiguous: bool = False
    match_kind: str = "unknown"
    """Which attribution strategy hit:
        'block_same_path'   — the line is in a multi-line run of add-set
                              lines from a single edit, found contiguously
                              at this path. Highest confidence.
        'provenance_same_path' — per-line match: the edit's add set contains
                              this line text, scoped to this path.
        'provenance_cross_file' — per-line match across files; only fires
                              for distinctive lines (CROSS_FILE_MIN_CHARS).
        'unknown' — no edit's add set contains this line. The line
                   predates our session history, or was written by a
                   source we didn't index (human, a tool call we can't
                   parse, etc.)"""


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
# Note: virtual-state indices removed.
#
# The old pipeline used fold's per-file virtual state (post-apply line
# attribution) as a primary attribution source. Under add-set semantics
# it's redundant — the provenance index already captures what fold could
# attribute, and without the context-lines-get-credited confusion that
# fold's virtual state had. Fold is still used for staleness reporting
# (the CLI prints "N stale edits"), but not for attribution.
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Block matching via shingle hashing (patience-diff-inspired)
# --------------------------------------------------------------------------
#
# The prior implementation used difflib.SequenceMatcher, which picks
# globally-optimal LCS alignments. That's the wrong objective for
# attribution: with duplicate lines (blanks, `}`, repeated patch context)
# it happily claims a blank line in row 2 against a blank line in the
# middle of a 600-line patch, because the 1-line "match" extends some
# other run's total score. The fix is patience-diff's insight: anchor on
# distinctive K-line windows, not on individual lines, and only extend
# runs outward from those anchors where the line-by-line match is exact.
#
# The algorithm below is `git blame -M`-adjacent: every edit's
# new_content is indexed as overlapping K-line shingles. For each
# shingle the current file has that matches, extend up and down to find
# the maximal contiguous-identical run. Greedy-claim longest-first, with
# per-line "already-claimed" tracking so each file line is attributed at
# most once. Ties broken by edit recency (latest wins).


def _edits_by_path(
    virtual_states: dict[str, FileState],
    provenance: ProvenanceIndex,
) -> dict[str, list[FileEdit]]:
    """Return `abs_path -> [edits that touched this path]`, latest-first."""
    by_path: dict[str, list[FileEdit]] = defaultdict(list)
    seen: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for path, state in virtual_states.items():
        for e in state.history:
            key = (e.source, e.session_id, e.tool_call_id)
            if key not in seen[path]:
                seen[path].add(key)
                by_path[path].append(e)
    for bucket in by_path.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)
    return by_path


def _is_substantial(line: str) -> bool:
    """A line is 'substantial' if stripped it's at least
    SHINGLE_SUBSTANTIAL_CHARS long. Used as a discriminator to reject
    shingles that are mostly whitespace/brackets."""
    return len(line.strip()) >= SHINGLE_SUBSTANTIAL_CHARS


def _shingle_admissible(window: tuple[str, ...]) -> bool:
    """True if this K-line window is distinctive enough to be an anchor.

    The hard-won invariant: at least SHINGLE_MIN_SUBSTANTIAL_LINES of
    the K lines must be substantial. A window of `{}` / blanks / bare
    `return` lines matches too many places to be useful as an anchor.
    """
    n_substantial = sum(1 for line in window if _is_substantial(line))
    return n_substantial >= SHINGLE_MIN_SUBSTANTIAL_LINES


def _build_shingle_index(
    edits: list[FileEdit],
) -> dict[tuple[str, ...], list[tuple[FileEdit, int]]]:
    """`shingle -> [(edit, offset_in_add_lines), ...]`.

    One shingle = K consecutive lines of an edit's ADD SET — lines the
    edit actually introduced, not context. This is the central shift:
    we only claim a block of current-file lines if they form a
    contiguous run of lines a single edit *added*, not merely quoted.

    Edits whose add set is shorter than K contribute nothing here (they
    fall back to per-line matching via the provenance index, which is
    also add-set-only).

    Note on discontinuities: an edit that adds two separate non-adjacent
    regions (e.g. a fix in two places in one Edit) would split into two
    runs in its add_lines list. Any K-window spanning that split would
    be a false shingle. We don't model this yet — the adapter emits
    each hunk of apply_patch as its own FileEdit, and Claude Code
    Edit/Write are single-region, so the assumption "add_lines is one
    contiguous sequence the edit introduced" holds in practice today.
    If MultiEdit wasn't expanded at adapter time (or Codex grouped
    hunks), we'd need to track hunk boundaries here.
    """
    idx: dict[tuple[str, ...], list[tuple[FileEdit, int]]] = defaultdict(list)
    for edit in edits:
        adds = added_lines(edit)
        n = len(adds)
        if n < BLOCK_SHINGLE_K:
            continue
        for i in range(n - BLOCK_SHINGLE_K + 1):
            window = tuple(adds[i : i + BLOCK_SHINGLE_K])
            if not _shingle_admissible(window):
                continue
            idx[window].append((edit, i))
    return idx


@dataclass
class _Match:
    """A candidate block match: edit wrote lines[file_start..file_end] of the file."""
    edit: FileEdit
    file_start: int  # inclusive, 0-based
    file_end: int    # inclusive, 0-based
    @property
    def length(self) -> int:
        return self.file_end - self.file_start + 1


def _extend_match(
    current_lines: list[str],
    add_lines: list[str],
    file_pos: int,
    add_pos: int,
) -> tuple[int, int]:
    """From an anchor (file_pos, add_pos) where K consecutive lines are
    known equal, extend up and down while add-set lines continue to match
    current-file lines.

    `add_lines` is the edit's add set (tool_diff.added_lines), not its
    full new_content. This restricts extension to lines the edit
    actually introduced — we won't grow a run into context the edit
    merely preserved.

    Returns `(file_start, file_end)` inclusive 0-based bounds of the
    maximal equal run covering the anchor.
    """
    start = file_pos
    a = add_pos
    while start > 0 and a > 0 and current_lines[start - 1] == add_lines[a - 1]:
        start -= 1
        a -= 1
    end = file_pos + BLOCK_SHINGLE_K - 1
    a = add_pos + BLOCK_SHINGLE_K - 1
    while end + 1 < len(current_lines) and a + 1 < len(add_lines) \
            and current_lines[end + 1] == add_lines[a + 1]:
        end += 1
        a += 1
    return start, end


def _apply_block_pass(
    *,
    current_lines: list[str],
    edits: list[FileEdit],
) -> list[tuple[int, FileEdit]]:
    """Claim line ranges via shingle anchors + exact-run extension.

    Returns `[(line_index_0based, edit)]`. Each file line appears at most
    once in the output (greedy claim). Lines not covered here fall through
    to the per-line pass downstream.

    Determinism: edits are iterated latest-first; for a given file line,
    the first (most recent) edit whose extended run covers it wins.
    """
    claimed: list[bool] = [False] * len(current_lines)
    claims: list[tuple[int, FileEdit]] = []

    # Build the shingle index once.
    shingle_index = _build_shingle_index(edits)
    if not shingle_index:
        return claims

    # For each K-window in the current file, find all anchor matches,
    # extend to maximal runs over the edit's ADD SET (not full
    # new_content), and collect candidates.
    candidates: list[_Match] = []
    n = len(current_lines)
    add_lines_cache: dict[int, list[str]] = {}
    def add_lines_for(edit: FileEdit) -> list[str]:
        key = id(edit)
        if key not in add_lines_cache:
            add_lines_cache[key] = added_lines(edit)
        return add_lines_cache[key]

    for i in range(n - BLOCK_SHINGLE_K + 1):
        window = tuple(current_lines[i : i + BLOCK_SHINGLE_K])
        hits = shingle_index.get(window)
        if not hits:
            continue
        for edit, add_pos in hits:
            adds = add_lines_for(edit)
            start, end = _extend_match(current_lines, adds, i, add_pos)
            candidates.append(_Match(edit=edit, file_start=start, file_end=end))

    # Greedy claim: longest first; within equal length, latest edit first.
    # `edits` is already latest-first; convert to recency rank for ties.
    recency_rank = {id(e): rank for rank, e in enumerate(edits)}
    candidates.sort(key=lambda m: (-m.length, recency_rank.get(id(m.edit), 10**9)))

    for m in candidates:
        if all(claimed[i] for i in range(m.file_start, m.file_end + 1)):
            # Fully covered by earlier claims; skip.
            continue
        for i in range(m.file_start, m.file_end + 1):
            if not claimed[i]:
                claimed[i] = True
                claims.append((i, m.edit))
    return claims


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
    edits_by_path = _edits_by_path(virtual_states, provenance)

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

        # Block pass first. For every line the block pass claims, we skip
        # the per-line logic entirely — block match wins. block_claims is
        # a dict mapping line-index-0based -> FileEdit.
        block_claims: dict[int, FileEdit] = {
            idx: edit
            for idx, edit in _apply_block_pass(
                current_lines=lines,
                edits=edits_by_path.get(abs_str, []),
            )
        }

        attributions: list[LineAttribution] = []
        for i, line_text in enumerate(lines, start=1):
            # 1. Block match (most trustworthy — multi-line run of adds)
            be = block_claims.get(i - 1)
            if be is not None:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=be,
                    ambiguous=False, match_kind="block_same_path",
                ))
                continue

            stripped = line_text.strip()
            # Per-line passes require the line to be substantial.
            # Without this, blank/bracket-only lines get claimed anywhere
            # the agent happened to add one. If you want the blank
            # attributed, let the block pass handle it; otherwise
            # 'unknown' is the honest answer.
            per_line_allowed = len(stripped) >= SHINGLE_SUBSTANTIAL_CHARS
            cross_file_allowed = len(stripped) >= CROSS_FILE_MIN_CHARS

            if not per_line_allowed:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=None,
                ))
                continue

            # 2. Provenance same-path (per-line, from add sets)
            cands = provenance.by_path_and_text.get((abs_str, line_text), [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "provenance_same_path"))
                continue

            if not cross_file_allowed:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=None,
                ))
                continue

            # 3. Provenance cross-file
            cands = provenance.by_text.get(line_text, [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "provenance_cross_file"))
                continue

            # 4. Unknown
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
