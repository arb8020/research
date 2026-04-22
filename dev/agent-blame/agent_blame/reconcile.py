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
        'block_same_path'   — the line sits inside a multi-line run of the
                              edit's new_content that appears contiguously
                              in the current file at this path
        'virtual_same_path' — per-line match via fold's virtual state
        'provenance_same_path' — per-line match via flat provenance index
        'virtual_cross_file'
        'provenance_cross_file'
        'unknown'"""


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
    """`shingle -> [(edit, offset_in_edit_lines), ...]`.

    One shingle = K consecutive lines of an edit's new_content. We skip
    shingles that fail _shingle_admissible so ambiguous windows never
    seed a claim. Edits shorter than K lines contribute nothing here
    (the per-line fallback passes will still see them).
    """
    idx: dict[tuple[str, ...], list[tuple[FileEdit, int]]] = defaultdict(list)
    for edit in edits:
        lines = edit.new_content.split("\n")
        if lines and lines[-1] == "":
            lines = lines[:-1]
        n = len(lines)
        if n < BLOCK_SHINGLE_K:
            continue
        for i in range(n - BLOCK_SHINGLE_K + 1):
            window = tuple(lines[i : i + BLOCK_SHINGLE_K])
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
    edit_lines: list[str],
    file_pos: int,
    edit_pos: int,
) -> tuple[int, int]:
    """From an anchor (file_pos, edit_pos) where K consecutive lines are
    known equal, extend up and down while lines continue to match.

    Returns `(file_start, file_end)` inclusive 0-based bounds of the
    maximal equal run covering the anchor.
    """
    # Extend left
    start = file_pos
    e = edit_pos
    while start > 0 and e > 0 and current_lines[start - 1] == edit_lines[e - 1]:
        start -= 1
        e -= 1
    # Extend right (starting from the end of the K-anchor)
    end = file_pos + BLOCK_SHINGLE_K - 1
    e = edit_pos + BLOCK_SHINGLE_K - 1
    while end + 1 < len(current_lines) and e + 1 < len(edit_lines) \
            and current_lines[end + 1] == edit_lines[e + 1]:
        end += 1
        e += 1
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
    # extend to maximal runs, and collect candidates.
    candidates: list[_Match] = []
    n = len(current_lines)
    # Cache per-edit line splits so _extend_match doesn't re-split.
    edit_lines_cache: dict[int, list[str]] = {}
    def edit_lines_for(edit: FileEdit) -> list[str]:
        key = id(edit)
        if key not in edit_lines_cache:
            ls = edit.new_content.split("\n")
            if ls and ls[-1] == "":
                ls = ls[:-1]
            edit_lines_cache[key] = ls
        return edit_lines_cache[key]

    for i in range(n - BLOCK_SHINGLE_K + 1):
        window = tuple(current_lines[i : i + BLOCK_SHINGLE_K])
        hits = shingle_index.get(window)
        if not hits:
            continue
        for edit, edit_pos in hits:
            edit_lines = edit_lines_for(edit)
            start, end = _extend_match(current_lines, edit_lines, i, edit_pos)
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
    v_by_text, v_by_path_and_text = _index_virtual_state(virtual_states)
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
            # 0. Block match (most trustworthy — requires multi-line context)
            be = block_claims.get(i - 1)
            if be is not None:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=be,
                    ambiguous=False, match_kind="block_same_path",
                ))
                continue

            stripped = line_text.strip()
            # Per-line passes all require the line to be substantial.
            # Without this, blank/bracket-only lines get claimed anywhere
            # the agent happened to emit a blank in its new_content — the
            # exact false-positive we hit on grpo.py line 2. If you want
            # the blank attributed, let the block pass handle it; otherwise
            # "unknown" is the honest answer.
            per_line_allowed = len(stripped) >= SHINGLE_SUBSTANTIAL_CHARS
            cross_file_allowed = len(stripped) >= CROSS_FILE_MIN_CHARS

            if not per_line_allowed:
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=None,
                ))
                continue

            # 1. Virtual same-path (per-line)
            cands = v_by_path_and_text.get((abs_str, line_text), [])
            if cands:
                attributions.append(_make_attr(i, line_text, cands, "virtual_same_path"))
                continue

            # 2. Provenance same-path (per-line)
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
