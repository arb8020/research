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
from difflib import SequenceMatcher
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

MIN_BLOCK_LINES = 2
"""Minimum block length (in lines) to claim via block matching.

Single-line "blocks" are just per-line matching in disguise and carry no
additional signal. Two-line blocks already imply continuity (the line
above/below was also written by this edit), which is the whole point of
the block pass — so this is the lowest useful threshold.
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
# Block matching
# --------------------------------------------------------------------------


def _edits_by_path(
    virtual_states: dict[str, FileState],
    provenance: ProvenanceIndex,
) -> dict[str, list[FileEdit]]:
    """Return `abs_path -> [edits that touched this path]`, latest-first.

    We source from both fold's history (complete — every edit we parsed
    went through fold) and, as a safety net, provenance's same-path index.
    In practice fold.history is the authoritative list; provenance only
    adds edits that somehow slipped through without entering fold.
    """
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


def _apply_block_pass(
    *,
    abs_path: str,
    current_lines: list[str],
    edits: list[FileEdit],
) -> list[tuple[int, FileEdit]]:
    """Claim line ranges by matching multi-line runs of edit.new_content
    against the current file, greedily, latest-edit-first.

    Returns a list of `(line_index_0based, edit)` tuples for every line
    attributed via the block pass. Lines not covered here fall through
    to the per-line pass downstream.

    Algorithm:
        claimed = bitmap of line indices already attributed
        for each edit in `edits` (latest first):
            a = current_lines (as seen today)
            b = split(edit.new_content)
            for each matching block (ai, bj, k) from SequenceMatcher:
                if k < MIN_BLOCK_LINES: skip
                claim every line ai..ai+k-1 that is still unclaimed,
                    attributing to this edit
        return claims

    `SequenceMatcher.get_matching_blocks` returns non-overlapping
    monotonically-increasing matches (the LCS decomposition). We walk
    them in order and claim each whole run. A later edit's match cannot
    steal lines an earlier (more recent) edit already claimed.

    We deliberately do NOT check "is this block long enough to be
    distinctive" — a 2-line block already implies intra-session
    continuity, which is exactly the signal we care about. Short-line
    noise is filtered at the per-line fallback (CROSS_FILE_MIN_CHARS),
    not here.
    """
    claims: list[tuple[int, FileEdit]] = []
    claimed: list[bool] = [False] * len(current_lines)

    for edit in edits:
        edit_lines = edit.new_content.split("\n")
        # Trailing-newline convention: if edit.new_content ends with "\n"
        # the split yields a trailing "" we should drop.
        if edit_lines and edit_lines[-1] == "":
            edit_lines = edit_lines[:-1]
        if len(edit_lines) < MIN_BLOCK_LINES:
            continue

        # SequenceMatcher builds a hash index over `b` (the second arg).
        # Putting edit_lines as `b` keeps the per-edit hashed side small
        # while the larger `current_lines` is walked. This is the fast
        # direction.
        matcher = SequenceMatcher(a=current_lines, b=edit_lines, autojunk=False)
        for ai, bj, k in matcher.get_matching_blocks():
            if k < MIN_BLOCK_LINES:
                continue
            for off in range(k):
                idx = ai + off
                if 0 <= idx < len(claimed) and not claimed[idx]:
                    claimed[idx] = True
                    claims.append((idx, edit))
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
            for idx, edit
            in _apply_block_pass(
                abs_path=abs_str,
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
            cross_file_allowed = len(stripped) >= CROSS_FILE_MIN_CHARS

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
