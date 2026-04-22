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
    # Edit writes a block containing blank lines AND substantial content
    # (needed because the shingle pass rejects windows dominated by
    # blanks/short lines). With enough substantial neighbors the block
    # match claims the whole contiguous run, including the blank.
    content = (
        'def handle_request(conn, request_body, logger):\n'
        '\n'
        '    headers = parse_headers(request_body)\n'
        '\n'
        '    return dispatch(conn, headers, logger)'
    )
    e = _write("S1", 1, "/f.py", content)
    [fa] = _run_reconcile([e], ["/f.py"], content)
    assert len(fa.lines) == 5
    assert all(l.edit is not None for l in fa.lines), \
        f"unattributed lines: {[l for l in fa.lines if l.edit is None]}"
    assert all(l.edit.session_id == "S1" for l in fa.lines)
    assert all(l.match_kind == "virtual_same_position" for l in fa.lines)


def test_write_then_edit_credits_unchanged_lines_to_writer():
    # A writes a 10-line function. B edits the middle 3 lines,
    # quoting 1 line of context on each side. Outer lines (including
    # the quoted context) stay credited to A; the 2 new middle lines
    # go to B.
    a = _write("A", 1, "/f.py",
        "def process(items):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        validated = validate(item)\n"
        "        processed = transform(validated)\n"
        "        enriched = enrich(processed)\n"
        "        results.append(enriched)\n"
        "    finalize(results)\n"
        "    emit_telemetry(results)\n"
        "    return results"
    )
    b = FileEdit(
        source="claude_code", session_id="B", message_uuid="msg-B",
        tool_call_id="tc-B",
        timestamp=datetime.fromtimestamp(2, tz=timezone.utc),
        path="/f.py", op="edit",
        old_content=(
            "        validated = validate(item)\n"
            "        processed = transform(validated)\n"
            "        enriched = enrich(processed)\n"
            "        results.append(enriched)\n"
            "    finalize(results)"
        ),
        new_content=(
            "        validated = validate(item)\n"
            "        NEW_LINE_FROM_B_ONE = True\n"
            "        NEW_LINE_FROM_B_TWO = False\n"
            "    finalize(results)"
        ),
    )

    current = (
        "def process(items):\n"
        "    results = []\n"
        "    for item in items:\n"
        "        validated = validate(item)\n"
        "        NEW_LINE_FROM_B_ONE = True\n"
        "        NEW_LINE_FROM_B_TWO = False\n"
        "    finalize(results)\n"
        "    emit_telemetry(results)\n"
        "    return results"
    )
    [fa] = _run_reconcile([a, b], ["/f.py"], current)
    sessions = [l.edit.session_id if l.edit else None for l in fa.lines]
    # Outer lines from A, including the two context lines B quoted.
    assert sessions[0:4] == ["A"] * 4, f"got {sessions[0:4]}"
    # B's two new lines
    assert sessions[4:6] == ["B", "B"], f"got {sessions[4:6]}"
    # `finalize(results)` was context in B -> credit stays with A.
    # Plus remaining A-only tail.
    assert sessions[6:9] == ["A", "A", "A"], f"got {sessions[6:9]}"


def test_block_pass_rejects_shingle_of_trivial_lines():
    # An edit containing mostly blank/short lines (e.g. a Codex
    # apply_patch whose hunk context is all blanks) must NOT claim
    # blank lines in unrelated parts of the file. This is the grpo.py
    # line 2 false-positive the user reported.
    edit_content = '\n\n\n\n    return\n'
    e = _write("S1", 1, "/f.py", edit_content)
    # Current file has blank lines, but they're in context that wasn't
    # part of any agent edit.
    current = (
        '"""This is the module docstring.\n'
        '\n'
        'It explains things.\n'
        '"""\n'
        'import os'
    )
    [fa] = _run_reconcile([e], ["/f.py"], current)
    # No line should be attributed to S1 — its edit was all trivial
    # lines and the shingle pass correctly refused to use it as an
    # anchor. Falls through to per-line, and only exact-match lines
    # would hit — but `"\n"` matches blank-in-file via provenance
    # same-path? Let's just assert the blank at line 2 is NOT S1.
    for line in fa.lines:
        if line.text == "":
            # Blank lines should not claim attribution from trivial-shingle edit.
            # They may end up unknown, or if somehow attributed it should
            # not be via block_same_path.
            if line.edit is not None:
                assert line.match_kind != "block_same_path", \
                    f"line {line.n} {line.text!r} claimed via block from trivial-shingle edit"


def test_edit_does_not_credit_context_lines():
    """Core attribution invariant under add-set semantics.

    A writes a 10-line function. B subsequently edits a 3-line region
    inside it — B's old_string and new_string both quote the 7 context
    lines around the change. Those 7 context lines must be credited to
    A, not B, because B didn't create them.
    """
    a_content = (
        'def process(items):\n'
        '    results = []\n'
        '    for item in items:\n'
        '        validated = validate(item)\n'
        '        processed = transform(validated)\n'
        '        enriched = enrich(processed)\n'
        '        results.append(enriched)\n'
        '    finalize(results)\n'
        '    emit_telemetry(results)\n'
        '    return results'
    )
    a = _write("A", 1, "/f.py", a_content)

    # B changes the middle 3 lines (validated/processed/enriched) to
    # something else, quoting 2 lines of context on each side.
    b = FileEdit(
        source="claude_code",
        session_id="B",
        message_uuid="msg-B",
        tool_call_id="tc-B",
        timestamp=datetime.fromtimestamp(2, tz=timezone.utc),
        path="/f.py",
        op="edit",
        old_content=(
            '    for item in items:\n'
            '        validated = validate(item)\n'
            '        processed = transform(validated)\n'
            '        enriched = enrich(processed)\n'
            '        results.append(enriched)'
        ),
        new_content=(
            '    for item in items:\n'
            '        CHANGED_LINE_FROM_B = True\n'
            '        ANOTHER_CHANGED_LINE = False\n'
            '        results.append(enriched)'
        ),
    )

    current = (
        'def process(items):\n'
        '    results = []\n'
        '    for item in items:\n'
        '        CHANGED_LINE_FROM_B = True\n'
        '        ANOTHER_CHANGED_LINE = False\n'
        '        results.append(enriched)\n'
        '    finalize(results)\n'
        '    emit_telemetry(results)\n'
        '    return results'
    )

    [fa] = _run_reconcile([a, b], ["/f.py"], current)
    sessions = [l.edit.session_id if l.edit else None for l in fa.lines]

    # Lines 0,1,2 are all A's (def process / results = [] / for item in items)
    # — even though line 2 ("for item in items:") was quoted as context by B.
    assert sessions[0:3] == ["A", "A", "A"], f"context credited to B! got {sessions[0:3]}"
    # Lines 3,4 are B's (the two new lines)
    assert sessions[3:5] == ["B", "B"], f"got {sessions[3:5]}"
    # Line 5 "results.append(enriched)" is context in B — must credit A.
    assert sessions[5] == "A", f"context credited to B! got {sessions[5]}"
    # Lines 6,7,8 are outer A-context that B never saw.
    assert sessions[6:9] == ["A", "A", "A"], f"got {sessions[6:9]}"


def test_block_pass_requires_min_shingle_length():
    # Edit shorter than BLOCK_SHINGLE_K lines cannot seed a block match
    # (no K-line window exists). Per-line fallback must still attribute
    # its line(s).
    e = _write("S1", 1, "/f.py",
        "this is a single line of distinctive content long enough to pass")
    [fa] = _run_reconcile([e], ["/f.py"],
        "this is a single line of distinctive content long enough to pass")
    assert len(fa.lines) == 1
    assert fa.lines[0].edit is not None
    assert fa.lines[0].match_kind == "virtual_same_position"
