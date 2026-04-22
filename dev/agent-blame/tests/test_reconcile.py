"""Reconcile invariants, especially the block pass."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from agent_blame.effects import FileEdit
from agent_blame.fold import fold_edits
from agent_blame.provenance import build_provenance
from agent_blame.reconcile import reconcile


def _write(sid: str, ts: int, path: str, content: str) -> FileEdit:
    return FileEdit(
        source="claude_code",
        session_id=sid,
        message_uuid=f"msg-{sid}-{ts}",
        tool_call_id=f"tc-{sid}-{ts}",
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
        path=path,
        op="write",
        new_content=content,
        old_content=None,
    )


def _run_reconcile(edits, files, current_text):
    """Run the full pipeline with an in-memory source reader."""
    states = fold_edits(edits)
    prov = build_provenance(edits)
    def reader(path: str) -> str | None:
        return current_text
    return reconcile(
        repo_root=Path("/"),
        virtual_states=states,
        provenance=prov,
        source=reader,
        files=[Path(f) for f in files],
    )


def test_block_pass_attributes_blank_lines_inside_block():
    # Edit writes a 5-line block containing blank lines and a short line.
    # The repo's current content is exactly that block. The block pass
    # must attribute every line to the edit, including blanks.
    content = '"""Docstring.\n\nbody line 1\n\nbody line 2"""'
    e = _write("S1", 1, "/f.py", content)
    [fa] = _run_reconcile([e], ["/f.py"], content)
    assert len(fa.lines) == 5
    assert all(l.edit is not None for l in fa.lines)
    assert all(l.edit.session_id == "S1" for l in fa.lines)
    assert all(l.match_kind == "block_same_path" for l in fa.lines)


def test_block_pass_claims_contiguous_despite_later_edit():
    # Edit A writes 10 lines. Edit B replaces 2 lines in the middle.
    # Repo's current content = A's first 4, B's 2, A's last 4.
    # Latest-first block pass: B claims its 2 lines; then A claims
    # the surviving 4-line blocks on each side.
    a_content = "a1\na2\na3\na4\na5\na6\na7\na8\na9\na10"
    a = _write("A", 1, "/f.py", a_content)
    current = "a1\na2\na3\na4\nB1\nB2\na7\na8\na9\na10"
    b = _write("B", 2, "/f.py", current)
    # But we want attribution as if B only wrote the 2 lines.
    # Easiest way: use edit op with B writing just the middle.
    # For test simplicity, use a synthetic second-write that matches
    # the current text exactly (latest-first claim wins).
    b_middle = _write("B", 2, "/f.py", "a3\na4\nB1\nB2\na7\na8")
    [fa] = _run_reconcile([a, b_middle], ["/f.py"], current)
    # Lines a3, a4, B1, B2, a7, a8 match B's block of 6 contiguous lines.
    # Lines a1, a2, a9, a10 must fall to A (per-line provenance, not block).
    sessions = [l.edit.session_id if l.edit else None for l in fa.lines]
    kinds = [l.match_kind for l in fa.lines]
    # Middle 6 from B
    assert sessions[2:8] == ["B"] * 6
    assert all(k == "block_same_path" for k in kinds[2:8])
    # Outer 4 from A
    assert sessions[0:2] == ["A", "A"]
    assert sessions[8:10] == ["A", "A"]


def test_block_pass_requires_min_block_lines():
    # Single-line match should NOT be a block claim — it would just be
    # per-line matching in disguise, and falls through to per-line logic.
    e = _write("S1", 1, "/f.py", "exactly one distinctive line longer than twenty chars")
    [fa] = _run_reconcile([e], ["/f.py"], "exactly one distinctive line longer than twenty chars")
    # Single line, so block pass skips; per-line virtual_same_path picks it up.
    assert len(fa.lines) == 1
    assert fa.lines[0].edit is not None
    assert fa.lines[0].match_kind == "virtual_same_path"
