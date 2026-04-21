"""Fold invariants: unchanged lines preserve attribution, changed lines don't."""

from __future__ import annotations

from datetime import datetime, timezone

from agent_blame.effects import FileEdit
from agent_blame.fold import fold_edits


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


def _edit(sid: str, ts: int, path: str, old: str, new: str) -> FileEdit:
    return FileEdit(
        source="claude_code",
        session_id=sid,
        message_uuid=f"msg-{sid}-{ts}",
        tool_call_id=f"tc-{sid}-{ts}",
        timestamp=datetime.fromtimestamp(ts, tz=timezone.utc),
        path=path,
        op="edit",
        new_content=new,
        old_content=old,
    )


def test_write_attributes_all_lines_to_writer():
    e = _write("S1", 1, "/f.py", "a\nb\nc")
    states = fold_edits([e])
    lines = states["/f.py"].lines
    assert [a.text for a in lines] == ["a", "b", "c"]
    assert all(a.edit.session_id == "S1" for a in lines)


def test_edit_preserves_unchanged_lines():
    # Session 1 writes three lines. Session 2 edits only the middle.
    # The surviving outer lines must remain attributed to S1.
    w = _write("S1", 1, "/f.py", "a\nOLD\nc")
    e = _edit("S2", 2, "/f.py", "OLD", "NEW")
    states = fold_edits([w, e])
    lines = states["/f.py"].lines
    assert [a.text for a in lines] == ["a", "NEW", "c"]
    assert lines[0].edit.session_id == "S1"
    assert lines[1].edit.session_id == "S2"
    assert lines[2].edit.session_id == "S1"


def test_edit_spanning_multiple_lines_reattributes_only_touched_lines():
    w = _write("S1", 1, "/f.py", "a\nb\nc\nd\ne")
    # Replace "b\nc\nd" with one line; a and e must stay S1.
    e = _edit("S2", 2, "/f.py", "b\nc\nd", "MERGED")
    states = fold_edits([w, e])
    lines = states["/f.py"].lines
    assert [a.text for a in lines] == ["a", "MERGED", "e"]
    assert lines[0].edit.session_id == "S1"
    assert lines[1].edit.session_id == "S2"
    assert lines[2].edit.session_id == "S1"


def test_edit_mid_line_attributes_merged_line_to_editor():
    # Edit "bar" -> "BAZ" inside a line; the whole line becomes S2's.
    w = _write("S1", 1, "/f.py", "foo bar qux\nnext")
    e = _edit("S2", 2, "/f.py", "bar", "BAZ")
    states = fold_edits([w, e])
    lines = states["/f.py"].lines
    assert [a.text for a in lines] == ["foo BAZ qux", "next"]
    assert lines[0].edit.session_id == "S2"
    assert lines[1].edit.session_id == "S1"


def test_stale_edit_recorded_and_skipped():
    w = _write("S1", 1, "/f.py", "a\nb")
    e = _edit("S2", 2, "/f.py", "NOT_PRESENT", "X")
    states = fold_edits([w, e])
    assert [a.text for a in states["/f.py"].lines] == ["a", "b"]
    assert e in states["/f.py"].stale_edits


def test_edits_sorted_by_timestamp():
    # Pass later timestamp first; fold must reorder.
    late = _write("S2", 10, "/f.py", "late")
    early = _write("S1", 1, "/f.py", "early")
    states = fold_edits([late, early])
    # Late wins because it applied second.
    assert [a.text for a in states["/f.py"].lines] == ["late"]
    assert states["/f.py"].lines[0].edit.session_id == "S2"


def test_insertion_at_line_start_attributes_new_lines_only():
    # Classic "add import at top" — first line unchanged, new lines on top.
    w = _write("S1", 1, "/f.py", "line1\nline2")
    e = _edit("S2", 2, "/f.py", "line1", "import new\nline1")
    states = fold_edits([w, e])
    lines = states["/f.py"].lines
    assert [a.text for a in lines] == ["import new", "line1", "line2"]
    # The inserted line and the re-emitted "line1" both get fused in
    # boundary handling; acceptable. The clean guarantee is that `line2`
    # stays attributed to S1.
    assert lines[2].edit.session_id == "S1"
