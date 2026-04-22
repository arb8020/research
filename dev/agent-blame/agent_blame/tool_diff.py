"""Render tool-call inputs as structured diffs for the UI.

For tool calls that represent file mutations, we produce a list of file-
scoped hunks that the UI can render in a GitHub-style diff layout. This
avoids the `<pre>JSON.stringify</pre>` fallback, which is unreadable for
anything longer than a few lines.

Supported inputs:
- Claude Code Write       -> one "add file" hunk (new_content only)
- Claude Code Edit        -> one hunk with old_string/new_string diffed
- Claude Code MultiEdit   -> one hunk per sub-edit, all same path
- Codex apply_patch       -> one or more hunks per file, parsed from
                             OpenAI's apply_patch format (reused from
                             agent_blame.adapters.codex._parse_apply_patch)

Output schema (JSON-friendly):

    render = {
      "renderable": True,
      "kind": "write" | "edit" | "apply_patch",
      "files": [
        {
          "path": str,
          "op": "add" | "update" | "delete",
          "hunks": [
            {
              "rows": [
                {"op": "context"|"add"|"remove", "text": str,
                 "before_line": int|None, "after_line": int|None}
              ]
            }, ...
          ]
        }, ...
      ]
    }

If a tool call is not renderable as a diff, return {"renderable": False}.
UI falls back to JSON rendering.

Line numbers (before_line / after_line) are 1-indexed and relative to the
start of the synthesized diff, not the real file — agents don't give us
the real base file contents, and we don't want to lie about it. The
numbers are still useful for orientation within the hunk.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from .adapters.codex import ApplyPatchParseError, _parse_apply_patch


def render_tool_call(name: str, input_: Any) -> dict:
    """Return a UI-renderable diff shape, or {"renderable": False}."""
    if not isinstance(name, str):
        return {"renderable": False}

    if name == "Write":
        return _render_cc_write(input_)
    if name == "Edit":
        return _render_cc_edit(input_)
    if name == "MultiEdit":
        return _render_cc_multiedit(input_)
    if name == "apply_patch":
        return _render_apply_patch(input_)

    return {"renderable": False}


# --------------------------------------------------------------------------
# Claude Code
# --------------------------------------------------------------------------


def _render_cc_write(input_: Any) -> dict:
    if not isinstance(input_, dict):
        return {"renderable": False}
    path = input_.get("file_path")
    content = input_.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        return {"renderable": False}
    lines = content.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    rows = [
        {"op": "add", "text": line, "before_line": None, "after_line": i + 1}
        for i, line in enumerate(lines)
    ]
    return {
        "renderable": True,
        "kind": "write",
        "files": [{
            "path": path,
            "op": "add",
            "hunks": [{"rows": rows}],
        }],
    }


def _render_cc_edit(input_: Any) -> dict:
    if not isinstance(input_, dict):
        return {"renderable": False}
    path = input_.get("file_path")
    old = input_.get("old_string")
    new = input_.get("new_string")
    if not isinstance(path, str) or not isinstance(old, str) or not isinstance(new, str):
        return {"renderable": False}
    hunk = _diff_strings(old, new)
    return {
        "renderable": True,
        "kind": "edit",
        "files": [{
            "path": path,
            "op": "update",
            "hunks": [hunk],
        }],
    }


def _render_cc_multiedit(input_: Any) -> dict:
    if not isinstance(input_, dict):
        return {"renderable": False}
    path = input_.get("file_path")
    edits = input_.get("edits")
    if not isinstance(path, str) or not isinstance(edits, list):
        return {"renderable": False}
    hunks = []
    for e in edits:
        if not isinstance(e, dict):
            continue
        old = e.get("old_string")
        new = e.get("new_string")
        if isinstance(old, str) and isinstance(new, str):
            hunks.append(_diff_strings(old, new))
    if not hunks:
        return {"renderable": False}
    return {
        "renderable": True,
        "kind": "edit",
        "files": [{
            "path": path,
            "op": "update",
            "hunks": hunks,
        }],
    }


def _diff_strings(old: str, new: str) -> dict:
    """Unified-diff rows for two strings. SequenceMatcher is fine at this
    scale — the inputs are individual edit pieces, not whole files.

    Duplicate-line ambiguity (the reason we abandoned SequenceMatcher in
    reconcile.py) doesn't bite here because we're diffing *against* the
    exact old_string the agent provided, not hunting for it in a larger
    corpus. The alignment is constrained to a single small pair.
    """
    old_lines = old.split("\n")
    new_lines = new.split("\n")
    # Don't strip trailing empty here — Edit old/new strings can legitimately
    # end with/without a newline, and preserving that is useful for the diff
    # consumer.
    sm = SequenceMatcher(a=old_lines, b=new_lines, autojunk=False)
    rows: list[dict] = []
    before_line = 1
    after_line = 1
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                rows.append({
                    "op": "context",
                    "text": old_lines[i1 + off],
                    "before_line": before_line,
                    "after_line": after_line,
                })
                before_line += 1
                after_line += 1
        elif tag == "delete":
            for off in range(i2 - i1):
                rows.append({
                    "op": "remove",
                    "text": old_lines[i1 + off],
                    "before_line": before_line,
                    "after_line": None,
                })
                before_line += 1
        elif tag == "insert":
            for off in range(j2 - j1):
                rows.append({
                    "op": "add",
                    "text": new_lines[j1 + off],
                    "before_line": None,
                    "after_line": after_line,
                })
                after_line += 1
        elif tag == "replace":
            for off in range(i2 - i1):
                rows.append({
                    "op": "remove",
                    "text": old_lines[i1 + off],
                    "before_line": before_line,
                    "after_line": None,
                })
                before_line += 1
            for off in range(j2 - j1):
                rows.append({
                    "op": "add",
                    "text": new_lines[j1 + off],
                    "before_line": None,
                    "after_line": after_line,
                })
                after_line += 1
    return {"rows": rows}


# --------------------------------------------------------------------------
# Codex apply_patch
# --------------------------------------------------------------------------


def _render_apply_patch(input_: Any) -> dict:
    """Parse a Codex apply_patch input and expand into hunk rows.

    Reuses adapters.codex._parse_apply_patch which returns
    [(path, op, old, new), ...] tuples — one per file-level hunk for
    Update File, or one whole-file "write" for Add File. We then convert
    each tuple into a row-list using _diff_strings for Update hunks and
    a pure-add row list for Write.
    """
    if not isinstance(input_, str):
        return {"renderable": False}
    try:
        parsed = _parse_apply_patch(input_)
    except ApplyPatchParseError:
        return {"renderable": False}
    if not parsed:
        return {"renderable": False}

    # Group tuples by (path, op). One file may have multiple update hunks
    # but only one add/write.
    by_file: dict[tuple[str, str], list[dict]] = {}
    for path, op, old, new in parsed:
        key = (path, op)
        if op == "edit":
            hunk = _diff_strings(old or "", new)
        elif op == "write":
            lines = new.split("\n")
            if lines and lines[-1] == "":
                lines = lines[:-1]
            rows = [
                {"op": "add", "text": line, "before_line": None, "after_line": i + 1}
                for i, line in enumerate(lines)
            ]
            hunk = {"rows": rows}
        else:
            continue
        by_file.setdefault(key, []).append(hunk)

    files = []
    for (path, op), hunks in by_file.items():
        files.append({
            "path": path,
            "op": "update" if op == "edit" else "add",
            "hunks": hunks,
        })
    if not files:
        return {"renderable": False}
    return {"renderable": True, "kind": "apply_patch", "files": files}
