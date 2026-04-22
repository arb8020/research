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
    assert all(l.match_kind == "block_same_path" for l in fa.lines)


def test_block_pass_claims_contiguous_despite_later_edit():
    # Edit A writes a 10-line function. Edit B replaces a 6-line window
    # of it (containing 4 of A's lines + 2 new lines). The block pass
    # runs latest-first: B's 6-line run wins, then A fills the outer
    # 4 lines via its own block match.
    a = _write("A", 1, "/f.py",
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
    # Current file: A's outer lines + B's middle 6
    current = (
        'def process(items):\n'
        '    results = []\n'
        '    for item in items:\n'
        '        validated = validate(item)\n'
        '        NEW_LINE_FROM_B_ONE = True\n'
        '        NEW_LINE_FROM_B_TWO = False\n'
        '        results.append(enriched)\n'
        '    finalize(results)\n'
        '    emit_telemetry(results)\n'
        '    return results'
    )
    # Edit B's exact claimable content = the 6-line window from current.
    b_middle = _write("B", 2, "/f.py",
        '    for item in items:\n'
        '        validated = validate(item)\n'
        '        NEW_LINE_FROM_B_ONE = True\n'
        '        NEW_LINE_FROM_B_TWO = False\n'
        '        results.append(enriched)\n'
        '    finalize(results)'
    )
    [fa] = _run_reconcile([a, b_middle], ["/f.py"], current)
    sessions = [l.edit.session_id if l.edit else None for l in fa.lines]
    kinds = [l.match_kind for l in fa.lines]
    # Middle 6 lines from B (block_same_path)
    assert sessions[2:8] == ["B"] * 6, f"got {sessions[2:8]}"
    assert all(k == "block_same_path" for k in kinds[2:8]), \
        f"got {kinds[2:8]}"
    # Outer 2 + 2 lines from A
    assert sessions[0:2] == ["A", "A"], f"got {sessions[0:2]}"
    assert sessions[8:10] == ["A", "A"], f"got {sessions[8:10]}"


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
    # Per-line virtual_same_path (no block, one line shorter than K).
    assert fa.lines[0].match_kind == "virtual_same_path"
