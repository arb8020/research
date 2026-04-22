"""Reconcile the virtual file state with the actual file on disk.

Fold produces, per file, an ordered list of (text, originating_edit)
pairs — the "virtual state." Reconcile's job: for each line of the
current file, look up that line's originating_edit in the virtual
state.

Ideal case: the virtual state and the current file match line-for-line.
Then every current line maps cleanly to its originating_edit.

Realistic case: they differ. Humans edit between sessions. Agents we
don't have transcripts for touched files. Stale edits got skipped.
We handle drift via two rules:

1. Same-path content match. If the current file has a line whose text
   appears exactly in this file's virtual state, and that virtual-state
   line is attributed, use that attribution. Positional alignment is
   not required — if the agent wrote a line that survived to HEAD
   (possibly moved within the file), the agent still gets credit.

2. If no match, the line is unknown. It either predates our sessions
   or was written by a source we didn't index.

## Intentionally NOT done

- No cross-file credit. An agent that wrote `return None` in file A
  does not get credit for `return None` in file B. Cross-file moves
  are out of scope — they produce too many false positives, and the
  correct source ("the agent that ACTUALLY wrote this line") is rarely
  findable without writing a real code-movement detector.
- No substantial-line threshold. If a line's text (including blanks)
  appears in the virtual state with a real originating_edit, we
  attribute honestly. Blanks end up unattributed naturally when no
  edit introduced them into this specific file's virtual state.
- No provenance index. It was a workaround for fold being buggy.
- No block/shingle matching. Same.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .effects import FileEdit
from .fold import FileState
from .sources import SourceReader


@dataclass(frozen=True)
class LineAttribution:
    line_number: int  # 1-indexed, like git blame
    text: str
    edit: FileEdit | None  # None means 'unknown'
    ambiguous: bool = False
    match_kind: str = "unknown"
    """One of:
        'virtual_same_position' — virtual state had an attributed line at
            the same index as the current file, with the same text.
        'virtual_same_path' — virtual state had this line text somewhere
            in the same file, attributed.
        'unknown' — no hit.
    """


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


def reconcile(
    *,
    repo_root: Path,
    virtual_states: dict[str, FileState],
    source: SourceReader,
    files: Iterable[Path],
    provenance=None,  # kept for API compat; ignored
) -> list[FileAttribution]:
    """Attribute every line of every file in `files` via fold's virtual state.

    The provenance kwarg is kept for backward-compat with callers that
    still pass it; it's ignored under the add-set-less model.
    """
    # Build a same-path-text index over virtual states for O(1) fallback
    # when position-aligned match fails.
    path_text_index: dict[tuple[str, str], list[FileEdit]] = defaultdict(list)
    for path, state in virtual_states.items():
        for attr in state.lines:
            if attr.edit is None:
                continue
            path_text_index[(path, attr.text)].append(attr.edit)
    # Sort latest-first so [0] is the most recent of several candidates.
    for bucket in path_text_index.values():
        bucket.sort(key=lambda e: e.timestamp, reverse=True)

    out: list[FileAttribution] = []
    for fpath in files:
        abs_path = (repo_root / fpath).resolve() if not fpath.is_absolute() else fpath
        text = source(str(abs_path))
        if text is None:
            continue
        rel = abs_path.relative_to(repo_root) if abs_path.is_absolute() else fpath
        current_lines = text.split("\n")
        if current_lines and current_lines[-1] == "":
            current_lines = current_lines[:-1]

        abs_str = str(abs_path)
        vstate = virtual_states.get(abs_str)

        attributions: list[LineAttribution] = []
        for i, line_text in enumerate(current_lines, start=1):
            # 1. Position-aligned match: virtual_state[i-1] has this text?
            if vstate is not None and (i - 1) < len(vstate.lines):
                va = vstate.lines[i - 1]
                if va.text == line_text and va.edit is not None:
                    attributions.append(LineAttribution(
                        line_number=i, text=line_text, edit=va.edit,
                        match_kind="virtual_same_position",
                    ))
                    continue

            # 2. Same-path content match: the virtual state has this line
            #    text somewhere, attributed. Handles line moves within a
            #    file (e.g. function reordered).
            cands = path_text_index.get((abs_str, line_text), [])
            if cands:
                distinct_sessions = {e.session_id for e in cands}
                attributions.append(LineAttribution(
                    line_number=i, text=line_text, edit=cands[0],
                    ambiguous=len(distinct_sessions) > 1,
                    match_kind="virtual_same_path",
                ))
                continue

            # 3. Unknown.
            attributions.append(LineAttribution(
                line_number=i, text=line_text, edit=None,
            ))

        out.append(FileAttribution(repo_path=rel, lines=attributions))
    return out


# --------------------------------------------------------------------------
# Summary helpers — used by the CLI and the server.
# --------------------------------------------------------------------------


def summary_stats(attributions: list[FileAttribution]) -> dict[str, int]:
    total = sum(len(a.lines) for a in attributions)
    attributed = sum(a.attributed_count for a in attributions)
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
