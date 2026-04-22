"""Fold a stream of FileEdits into a virtual per-file state with line attribution.

The core invariant:

    Every line in the virtual state is tagged with the edit that FIRST
    INTRODUCED that line content at that position.

Not "last touched," not "inside the patch window of," not "appeared in
the new_content of." First introduced. A Write introduces every line of
its content. An Edit introduces only the lines of its new_content that
did not previously exist in the spliced region — unchanged lines keep
their prior originating_edit.

## Haiku example (the canonical test)

    EDIT_A writes:
        "haiku line 1\\nhakiuline2\\nhaiku line 3"

    EDIT_B edits old="...all three..." new="haiku line 1\\nhaiku line 2\\nhaiku line 3"
        (fixes the typo on line 2 only)

    After fold:
        line 0 "haiku line 1"  originating_edit = A  (unchanged)
        line 1 "haiku line 2"  originating_edit = B  (changed)
        line 2 "haiku line 3"  originating_edit = A  (unchanged)

## How Edit handles unchanged lines

An Edit replaces a span of lines [first..last] in the virtual state with
a new list of lines from its new_content. To attribute correctly we run
a line-level diff between the OLD-side lines (the span being replaced)
and the NEW-side lines (the replacement):

    for each opcode (tag, i1, i2, j1, j2):
        if tag == 'equal':
            new-side line inherits originating_edit from old-side line
        if tag in ('insert', 'replace'):
            new-side lines get this edit as originating_edit
        if tag == 'delete':
            nothing (those lines go away)

Context lines inside an Edit's old/new (which differ from virtual
state's lines if the Edit's context diverges from what we think it
should be) are still "equal" as far as this fold is concerned — the
Edit claims it's quoting them, and we take that on faith. The real
check is whether a line is in `new_content` that was NOT in
`old_content`.

## Stale edits

If `old_content` cannot be found in the virtual state, the edit is
stale — our view of the filesystem has drifted from what the agent
saw (human edits between sessions, uncaptured sessions, etc). We
record it and skip. No silent re-seeding — that would retroactively
credit non-agents for lines.

## Boundary handling

If `old_content` starts or ends mid-line (doesn't align to newline
boundaries), the boundary line is partially changed. We treat it as
changed overall — the whole boundary line is attributed to the
editing edit. This is a simplification; if the non-match portion of
a boundary line dominates, it's still "owned" by this edit. Rare in
practice: agents nearly always submit line-aligned edits.

## Ordering

Edits are sorted by (timestamp, session_id, adapter-emit-order). Timestamp
alone is insufficient because edits inside the same assistant message
share a timestamp. Adapter emit order breaks the tie.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from difflib import SequenceMatcher
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
    """Splice `edit.new_content` in place of the first match of `edit.old_content`.

    Attribution rule: within the spliced region, a line in the new side
    inherits the originating_edit of the matching old-side line if the
    line-level diff says 'equal'. Otherwise (insert/replace), the line
    is attributed to `edit`.

    Stages:
      1. Find `old_content` as a substring of the joined virtual state.
      2. Map the character range to a line range [first_line..last_line]
         in state.lines. If the match starts or ends mid-line, handle
         the boundary by attributing the (partially-changed) line to
         `edit`; the lines fully inside the match get diff-based
         attribution.
      3. Extract the "old-side lines" (the lines inside the match,
         including their current originating_edit).
      4. Split `new_content` into new-side lines.
      5. Diff old-side vs new-side at line granularity. For each opcode,
         emit LineAttr entries with the right originating_edit.
      6. Splice into state.lines.
    """
    assert edit.old_content is not None  # guaranteed by FileEdit.__post_init__

    # Build cumulative character offsets
    line_starts: list[int] = []
    cursor = 0
    for i, attr in enumerate(state.lines):
        line_starts.append(cursor)
        cursor += len(attr.text)
        if i < len(state.lines) - 1:
            cursor += 1  # the join-'\n'
    current = "\n".join(a.text for a in state.lines)

    idx = current.find(edit.old_content)
    if idx == -1:
        state.stale_edits.append(edit)
        logger.debug(
            "stale edit for %s (session=%s, tool=%s): old_content not in virtual state",
            edit.path, edit.session_id, edit.tool_call_id,
        )
        return

    end_idx = idx + len(edit.old_content)  # exclusive

    if not state.lines:
        # Empty virtual file; an edit with old="" is semantically weird.
        # Treat as stale rather than fabricate.
        state.stale_edits.append(edit)
        return

    first_line = _locate_line(line_starts, idx, len(state.lines))
    if edit.old_content == "":
        # Zero-width insertion at char idx. Treat as "insert between
        # lines" — attribute inserted new-content lines to `edit` and
        # leave the existing line alone (minus any prefix/suffix split).
        last_line = first_line
    else:
        last_line = _locate_line(line_starts, end_idx - 1, len(state.lines))

    # Mid-line boundary prefix/suffix — the portions of first/last line
    # that are OUTSIDE the match. If non-empty, those portions stay;
    # the portion inside the match is what gets spliced out.
    prefix = state.lines[first_line].text[: idx - line_starts[first_line]]
    suffix = state.lines[last_line].text[end_idx - line_starts[last_line]:]

    # The full replacement text (what goes in where old_content was).
    replacement_text = prefix + edit.new_content + suffix
    new_side_line_texts = _split_lines(replacement_text)

    # The old-side lines — with their current attribution — are exactly
    # state.lines[first_line..last_line+1].
    old_side_lines: list[LineAttr] = state.lines[first_line : last_line + 1]
    old_side_texts: list[str] = [a.text for a in old_side_lines]

    # Diff old-side texts against new-side texts. `equal` opcodes let
    # us inherit originating_edit from the matched old-side line.
    # SequenceMatcher is safe at this scale — spliced regions are
    # typically dozens of lines, not thousands; duplicate-line ambiguity
    # is not a concern because both sides come from the same edit.
    sm = SequenceMatcher(a=old_side_texts, b=new_side_line_texts, autojunk=False)
    replacement_lines: list[LineAttr] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            # Each equal-new-line inherits from corresponding old-line.
            # equal ranges have i2-i1 == j2-j1.
            for off in range(j2 - j1):
                old_attr = old_side_lines[i1 + off]
                new_text = new_side_line_texts[j1 + off]
                # Sanity: texts must actually match in 'equal' opcodes.
                replacement_lines.append(LineAttr(text=new_text, edit=old_attr.edit))
        elif tag in ("insert", "replace"):
            for off in range(j2 - j1):
                replacement_lines.append(
                    LineAttr(text=new_side_line_texts[j1 + off], edit=edit)
                )
        # 'delete' contributes nothing to the new side.

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
    timestamped_seeder=None,  # TimestampedSeeder; avoids a hard import here
) -> dict[str, FileState]:
    """Replay `edits` in order; return per-path FileState.

    Seeding strategy (two mechanisms, composable):

    1. `seed_reader(path) -> str | None` is called on first touch of a
       path when the first edit is an `edit` op (Writes overwrite anyway,
       so seeding would be wasted). Populates virtual state with
       file-as-of-now, all lines attributed to None.

    2. `timestamped_seeder(path, ts) -> str | None` is called when an
       edit goes stale (its `old_content` isn't in our virtual state —
       we've drifted from what the agent saw). It returns the file as
       it was in git at the edit's timestamp. We reset the virtual
       state to that content (attributed None) and retry the edit.

       The tradeoff for re-seeding: we discard previously-applied edits'
       attribution on this file. Accepted because a stale edit means
       everything chained after it would also go stale; by re-seeding
       we recover this edit's chain at the cost of earlier edits that
       failed to survive forward anyway.

    Without either seeder, paths start empty, first edits on
    pre-existing files go stale, and cross-session drift is silently
    lost.

    Ordering: stable sort by (timestamp, session_id, adapter-emit-order).
    """
    indexed = list(enumerate(edits))
    indexed.sort(key=lambda pair: (pair[1].timestamp, pair[1].session_id, pair[0]))

    states: dict[str, FileState] = {}
    for _seq, edit in indexed:
        state = states.get(edit.path)
        if state is None:
            state = FileState(path=edit.path)
            states[edit.path] = state
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
            continue

        if edit.op != "edit":
            raise AssertionError(f"unknown op {edit.op!r} in fold")

        # Apply the edit. Staleness goes into state.stale_edits.
        stale_before = len(state.stale_edits)
        _apply_edit(state, edit)
        stale_after = len(state.stale_edits)

        if stale_after > stale_before and timestamped_seeder is not None:
            # The edit couldn't locate its old_content in our virtual
            # state. Only re-seed if we have NO attributed lines yet on
            # this file — re-seeding mid-chain would discard all prior
            # attribution we'd already built up for the file (e.g. a
            # previous Write or successful Edit), which is usually a
            # net loss of credited lines.
            #
            # Mid-chain stale edits are left as stale. Reconcile's
            # same-path content match still gives the edit a chance to
            # be credited at reconcile time if its new-content lines
            # survive to the current file, but we don't gamble the
            # virtual state to chase it.
            has_attributed = any(a.edit is not None for a in state.lines)
            if not has_attributed:
                seed_text = timestamped_seeder(edit.path, edit.timestamp)
                if seed_text is not None:
                    logger.debug(
                        "re-seeding empty state for %s from git at %s",
                        edit.path, edit.timestamp.isoformat(),
                    )
                    state.lines = [
                        LineAttr(text=line, edit=None)
                        for line in _split_lines(seed_text)
                    ]
                    state.stale_edits.pop()
                    _apply_edit(state, edit)

    return states


# Callable protocol for `seed_reader`. Kept as an alias rather than a class
# to avoid a heavy abstraction for a one-parameter callable.
SeedReader = object  # spec: Callable[[str], str | None]

