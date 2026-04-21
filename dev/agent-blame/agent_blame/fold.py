"""Fold a stream of FileEdits into a virtual per-file state with line attribution.

This is the core of the blame story. Given an ordered list of edits (from
potentially many sessions), we replay them against an in-memory model of each
file. Every line currently in the model carries a pointer back to the edit
that introduced it. Later edits overwrite attribution as they overwrite
content.

The output is *not* reconciled with the repo yet — that is reconcile.py's job.
The output is: "if you applied these edits in order to an empty filesystem,
which edit is responsible for each line of each file?" Reconcile then joins
that against what the repo actually looks like on disk today.

## Ordering

Edits are sorted by `(timestamp, session_id, adapter-emit-order)`. Timestamp
alone is insufficient because edits inside the same assistant message share
a timestamp, and MultiEdit expansions all share one. The adapter emits in
log order; we preserve that as a tiebreak via a stable sort on an explicit
sequence number assigned at read time.

## The virtual file state

A file is a list of `LineAttr` records, one per line. Each `LineAttr` knows:
  - the line text
  - which FileEdit wrote it

`write` replaces the whole list. `edit` finds `old_content` in the current
text and splices in `new_content`.

## Edit matching

For `edit` ops, we locate `old_content` by exact string match in the joined
file text. If it does not match, the edit is stale — the file state has
drifted from what the agent saw (either a prior edit failed, or a human
intervened between agent turns, or we missed a write). We record the miss
and skip, rather than guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable

from .effects import FileEdit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LineAttr:
    """One line of a virtual file, tagged with the edit that produced it.

    `edit` is `None` when the line was seeded from the repo's current state
    (because our earliest observed edit was an `edit`, not a `write`, so we
    had to bootstrap from somewhere). These unattributed seed lines end up
    as "unknown" in reconcile output — correct, since we genuinely do not
    know which agent (or human) wrote them.
    """

    text: str
    edit: FileEdit | None


@dataclass
class FileState:
    """Virtual per-file state built by folding edits.

    `lines` is the current content. `history` records every edit applied to
    this file (for debugging coverage); not every history entry is reflected
    in `lines` since later edits overwrite earlier ones.
    """

    path: str
    lines: list[LineAttr] = field(default_factory=list)
    history: list[FileEdit] = field(default_factory=list)
    stale_edits: list[FileEdit] = field(default_factory=list)
    """Edits whose `old_content` did not match the current virtual state.

    These are informational: they indicate our view of the filesystem
    diverged from the agent's. Common causes: human edits between agent
    sessions, edits we failed to parse upstream, reordering bugs.
    """


def _split_lines(text: str) -> list[str]:
    """Split preserving trailing-newline semantics.

    A file ending in `\\n` has a final empty "line" we deliberately include,
    because reinserting it is how we keep the trailing newline invariant
    under join-with-`\\n`. Callers should not treat that empty tail as
    attributable content — reconcile strips it.
    """
    # splitlines() drops the trailing empty; split("\n") keeps it. We want the latter.
    return text.split("\n")


def _apply_write(state: FileState, edit: FileEdit) -> None:
    """`write` replaces the file wholesale. Every line is attributed to this edit."""
    state.lines = [LineAttr(text=line, edit=edit) for line in _split_lines(edit.new_content)]


def _apply_edit(state: FileState, edit: FileEdit) -> None:
    """`edit` splices new_content in place of the first match of old_content.

    The invariant we preserve: unchanged lines keep their prior attribution;
    only lines that actually change (or are introduced) point to `edit`.
    This is what makes the later "top sessions by lines" breakdown honest.

    Implementation:

        1. Serialize current lines to a single string `current` (with the
           same line-by-line attribution as `state.lines`, by index).
        2. Find `old_content` as a substring of `current`. If absent,
           record stale and return — do not guess.
        3. Compute the line span in `state.lines` that the match covers.
           The span is `[first_line, last_line]` where `first_line` is the
           index of the line containing character `idx`, and `last_line`
           is the index of the line containing `idx + len(old_content) - 1`.
        4. Build the replacement lines from
               prefix_of_first_line + new_content + suffix_of_last_line
           (prefix/suffix are the portions of the boundary lines that
           were NOT part of the match).
        5. Replace `state.lines[first_line:last_line+1]` with the
           replacement, attributed to `edit`.

    If the match boundaries fall exactly at line breaks, the prefix/suffix
    are empty and we replace whole lines cleanly. If they fall mid-line,
    we concatenate with the unchanged portions of the boundary lines
    — those merged lines are attributed to `edit` since at least part
    of them changed.
    """
    assert edit.old_content is not None  # guaranteed by FileEdit.__post_init__
    # Build cumulative character offsets: `line_starts[i]` is the offset of
    # the first character of `state.lines[i]` in the joined representation.
    # Joined representation is `"\n".join(line.text for line in state.lines)`.
    line_starts: list[int] = []
    cursor = 0
    for i, attr in enumerate(state.lines):
        line_starts.append(cursor)
        cursor += len(attr.text)
        if i < len(state.lines) - 1:
            cursor += 1  # the join-`\n`
    total_len = cursor
    current = "\n".join(a.text for a in state.lines)
    assert len(current) == total_len, (
        f"offset accounting drift: computed {total_len}, actual {len(current)}"
    )

    idx = current.find(edit.old_content)
    if idx == -1:
        state.stale_edits.append(edit)
        logger.debug(
            "stale edit for %s (session=%s, tool=%s): old_content not in virtual state",
            edit.path, edit.session_id, edit.tool_call_id,
        )
        return
    end_idx = idx + len(edit.old_content)  # exclusive

    # Locate the line containing `idx` (first_line) and `end_idx - 1` (last_line).
    # Special case: empty files have state.lines == [] and line_starts == [];
    # old_content must then be "" to match, which is meaningless — caller
    # probably wanted a Write. We treat this as stale.
    if not state.lines:
        state.stale_edits.append(edit)
        logger.debug(
            "edit into empty virtual file %s (session=%s, tool=%s)",
            edit.path, edit.session_id, edit.tool_call_id,
        )
        return

    first_line = _locate_line(line_starts, idx, len(state.lines))
    # For end_idx: we want the line containing the last matched character.
    # If old_content is empty (zero-width match), treat it as an insertion
    # at `idx`, spanning only `first_line`.
    if edit.old_content == "":
        last_line = first_line
        prefix = state.lines[first_line].text[: idx - line_starts[first_line]]
        suffix = state.lines[first_line].text[idx - line_starts[first_line]:]
    else:
        last_line = _locate_line(line_starts, end_idx - 1, len(state.lines))
        prefix = state.lines[first_line].text[: idx - line_starts[first_line]]
        suffix = state.lines[last_line].text[end_idx - line_starts[last_line]:]

    # Construct replacement text and split into lines, attributed to `edit`.
    replacement_text = prefix + edit.new_content + suffix
    replacement_lines = [
        LineAttr(text=line, edit=edit) for line in _split_lines(replacement_text)
    ]
    state.lines[first_line : last_line + 1] = replacement_lines


def _locate_line(line_starts: list[int], char_idx: int, n_lines: int) -> int:
    """Return the index of the line containing character offset `char_idx`.

    `line_starts[i]` is the start offset of line i. We find the largest i
    such that `line_starts[i] <= char_idx`. Linear scan is fine for v0;
    files rarely exceed a few thousand lines and edits are infrequent.
    """
    assert n_lines > 0, "should not locate lines in an empty file"
    # Walk from the end for typical append-heavy edit patterns.
    for i in range(n_lines - 1, -1, -1):
        if line_starts[i] <= char_idx:
            return i
    raise AssertionError(
        f"char_idx {char_idx} is before line 0 (starts at {line_starts[0]})"
    )


def fold_edits(
    edits: Iterable[FileEdit],
    *,
    seed_reader: "SeedReader | None" = None,
) -> dict[str, FileState]:
    """Replay `edits` in order; return per-path FileState.

    If `seed_reader` is provided, it is called when a path's first edit is
    an `edit` (not a `write`) — the returned text is used as the initial
    virtual state, attributed to `None` (unknown). This lets us handle
    agent edits against files whose original `write` is not in our session
    history (e.g. human-authored code, or sessions deleted before indexing).

    Without a seed_reader, such paths start empty and every `edit` against
    them is stale.

    We sort edits as a safety net in case the caller merged streams
    naively; the stable sort preserves adapter emit order within equal
    timestamps.
    """
    indexed = list(enumerate(edits))
    indexed.sort(key=lambda pair: (pair[1].timestamp, pair[1].session_id, pair[0]))

    states: dict[str, FileState] = {}
    for _seq, edit in indexed:
        state = states.get(edit.path)
        if state is None:
            state = FileState(path=edit.path)
            states[edit.path] = state
            # First-touch seeding: only if we don't have a Write kicking
            # off this file's history. Writes overwrite anyway; seeding
            # would be wasted work.
            if edit.op == "edit" and seed_reader is not None:
                seed_text = seed_reader(edit.path)
                if seed_text is not None:
                    state.lines = [
                        LineAttr(text=line, edit=None)
                        for line in _split_lines(seed_text)
                    ]
        state.history.append(edit)
        if edit.op == "write":
            _apply_write(state, edit)
        elif edit.op == "edit":
            _apply_edit(state, edit)
            # A stale edit means our virtual state has drifted from what
            # the agent saw (usually: cross-session drift where humans or
            # uncaptured sessions modified the file in between). We do not
            # attempt to recover by re-seeding — that would wipe prior
            # attribution for *all* lines of the file. Instead, we accept
            # the loss of this one edit's attribution and keep going.
            # Reconcile via content-match often still recovers most of
            # what this edit produced, since it matches by text not
            # position.
        else:
            raise AssertionError(f"unknown op {edit.op!r} in fold")
    return states


# Callable protocol for `seed_reader`. Kept as an alias rather than a class
# to avoid a heavy abstraction for a one-parameter callable.
SeedReader = object  # spec: Callable[[str], str | None]

