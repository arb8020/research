"""Workspace snapshot reconstruction from agent trajectories.

Reconstructs filesystem state and bash history at each turn of an agent
trajectory, enabling the eval viewer to show a workspace panel alongside
the conversation.

Two paths:

  Live (native rollouts agents):
    Environment emits workspace_snapshot events into events.jsonl after each
    exec_tool call. Server reads them directly.

  Reconstructed (external drivers — Claude Code, Codex):
    No live hooks available. Walk the trajectory tool calls, apply their effects
    to a virtual filesystem, synthesize workspace_snapshot events per turn.
    Best-effort: bash side effects are parsed where possible, flagged "uncertain"
    otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

# ── Types ─────────────────────────────────────────────────────────────────────


@dataclass
class BashHistoryEntry:
    turn: int
    cmd: str
    stdout: str
    stderr: str
    exit_code: int
    uncertain_fs_effects: bool = False  # True if cmd may have changed files in unknown ways


@dataclass
class WorkspaceSnapshot:
    turn: int
    files: dict[str, str]  # path -> contents
    cwd: str
    bash_history: list[BashHistoryEntry]
    source: str = "reconstructed"  # "live" | "reconstructed"


@dataclass
class LineEdit:
    turn: int
    type: str  # "edit" | "write" | "bash_sed" | "bash_redirect"
    diff: str  # unified diff fragment or description
    message_index: int  # index into trajectory messages list
    cmd: str = ""  # for bash-derived edits


# line_history[filename][line_number_str] = list[LineEdit]
LineHistory = dict[str, dict[str, list[LineEdit]]]


# ── Main reconstruction entry point ───────────────────────────────────────────


def reconstruct_workspace_snapshots(
    messages: list[dict[str, Any]],
    initial_files: dict[str, str] | None = None,
    cwd: str = "/workspace",
) -> tuple[list[WorkspaceSnapshot], LineHistory]:
    """Reconstruct workspace snapshots from a trajectory message list.

    Args:
        messages: List of message dicts from the trajectory (role/content).
        initial_files: Optional known initial file state. If None, files are
            discovered as they appear in read/write/edit tool calls.
            Keys should be bare filenames (e.g. "perf_takehome.py") — they
            will be normalized to absolute paths under cwd internally.
        cwd: Working directory to use as root for relative paths.

    Returns:
        (snapshots, line_history) where snapshots is one entry per turn
        (assistant message boundary) and line_history maps filename → line
        number → list of edits.
    """
    # Normalize initial_files keys to absolute paths under cwd
    normalized_initial: dict[str, str] = {}
    for path, content in (initial_files or {}).items():
        normalized_initial[_resolve(cwd, path)] = content

    state = _ReconstructionState(
        files=normalized_initial,
        cwd=cwd,
        bash_history=[],
    )
    line_history: LineHistory = {}
    snapshots: list[WorkspaceSnapshot] = []
    turn = 0

    for msg_index, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant":
            # Process tool calls in this assistant message
            blocks = content if isinstance(content, list) else []
            for block in blocks:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "toolCall":
                    _apply_tool_call(
                        block,
                        state,
                        line_history,
                        turn=turn,
                        message_index=msg_index,
                    )

            # Turn boundary: emit snapshot after each assistant message that
            # contained tool calls (or always, for completeness)
            snapshots.append(
                WorkspaceSnapshot(
                    turn=turn,
                    files=dict(state.files),
                    cwd=state.cwd,
                    bash_history=list(state.bash_history),
                )
            )
            turn += 1

        elif role == "tool":
            # Tool results: find the most recent bash history entry without
            # stdout and fill it in. tool_call_id isn't tracked per-entry yet
            # so we match by recency within the current turn window.
            result_content = content if isinstance(content, str) else str(content)
            # Strip Codex's metadata header from exec_command output:
            # "Chunk ID: ...\nWall time: ...\nProcess exited with code N\n...Output:\n<actual>"
            if "Output:\n" in result_content:
                result_content = result_content.split("Output:\n", 1)[1]
            for entry in reversed(state.bash_history):
                if not entry.stdout:
                    entry.stdout = result_content.strip()
                    # Detect non-zero exit from Codex output header
                    if "Process exited with code " in (content if isinstance(content, str) else ""):
                        m = re.search(
                            r"Process exited with code (\d+)",
                            content if isinstance(content, str) else "",
                        )
                        if m and m.group(1) != "0":
                            entry.exit_code = int(m.group(1))
                    break

    # Always emit a final snapshot
    if not snapshots or snapshots[-1].turn != turn - 1:
        snapshots.append(
            WorkspaceSnapshot(
                turn=turn,
                files=dict(state.files),
                cwd=state.cwd,
                bash_history=list(state.bash_history),
            )
        )

    return snapshots, line_history


# ── Internal reconstruction state ─────────────────────────────────────────────


@dataclass
class _ReconstructionState:
    files: dict[str, str]
    cwd: str
    bash_history: list[BashHistoryEntry]


def _resolve(cwd: str, path: str) -> str:
    """Resolve a possibly-relative path against cwd, return normalized string."""
    if not path:
        return cwd
    p = PurePosixPath(path)
    if p.is_absolute():
        return str(p)
    return str(PurePosixPath(cwd) / p)


def _apply_tool_call(
    block: dict[str, Any],
    state: _ReconstructionState,
    line_history: LineHistory,
    turn: int,
    message_index: int,
) -> None:
    name = block.get("name", "")
    args = block.get("arguments") or {}
    if not isinstance(args, dict):
        return

    match name:
        case "write" | "create":
            _apply_write(block, state, line_history, turn, message_index)
        case "edit" | "str_replace_editor" | "str_replace_based_edit_tool":
            _apply_edit(block, state, line_history, turn, message_index)
        case "read" | "view":
            # Read confirms the file exists; no mutation
            pass
        case "bash" | "shell" | "exec_command" | "run_command" | "computer":
            _apply_bash(block, state, line_history, turn, message_index)
        case "apply_patch":
            _apply_patch(block, state, line_history, turn, message_index)


def _apply_write(
    block: dict[str, Any],
    state: _ReconstructionState,
    line_history: LineHistory,
    turn: int,
    message_index: int,
) -> None:
    args = block.get("arguments") or {}
    path = _resolve(state.cwd, args.get("path") or args.get("file_path") or "")
    content = args.get("content") or args.get("file_text") or ""
    if not path:
        return

    old_content = state.files.get(path, "")
    state.files[path] = content

    # Record which lines changed
    _record_line_edits(
        line_history=line_history,
        filename=path,
        old_content=old_content,
        new_content=content,
        edit=LineEdit(
            turn=turn,
            type="write",
            diff=_make_diff_summary(old_content, content),
            message_index=message_index,
        ),
    )


def _apply_edit(
    block: dict[str, Any],
    state: _ReconstructionState,
    line_history: LineHistory,
    turn: int,
    message_index: int,
) -> None:
    args = block.get("arguments") or {}
    path = _resolve(state.cwd, args.get("path") or args.get("file_path") or "")
    old_str = args.get("old_str") or args.get("old_string") or ""
    new_str = args.get("new_str") or args.get("new_string") or ""
    if not path or path not in state.files:
        # File not yet known — still record the edit intent
        if path:
            state.files[path] = new_str
        return

    old_content = state.files[path]
    if old_str in old_content:
        new_content = old_content.replace(old_str, new_str, 1)
        state.files[path] = new_content
        _record_line_edits(
            line_history=line_history,
            filename=path,
            old_content=old_content,
            new_content=new_content,
            edit=LineEdit(
                turn=turn,
                type="edit",
                diff=f"-{old_str[:120]}\n+{new_str[:120]}",
                message_index=message_index,
            ),
        )


def _apply_bash(
    block: dict[str, Any],
    state: _ReconstructionState,
    line_history: LineHistory,
    turn: int,
    message_index: int,
) -> None:
    args = block.get("arguments") or {}
    cmd = args.get("command") or args.get("cmd") or args.get("input") or ""
    if not cmd:
        return

    uncertain = False

    # ── mv <src> <dst> ────────────────────────────────────────────────────────
    m = re.match(r"^\s*mv\s+(-\S+\s+)*(\S+)\s+(\S+)\s*$", cmd)
    if m:
        src = _resolve(state.cwd, m.group(2))
        dst = _resolve(state.cwd, m.group(3))
        if src in state.files:
            state.files[dst] = state.files.pop(src)

    # ── cp <src> <dst> ────────────────────────────────────────────────────────
    m = re.match(r"^\s*cp\s+(-\S+\s+)*(\S+)\s+(\S+)\s*$", cmd)
    if m:
        src = _resolve(state.cwd, m.group(2))
        dst = _resolve(state.cwd, m.group(3))
        if src in state.files:
            state.files[dst] = state.files[src]

    # ── rm <path> ─────────────────────────────────────────────────────────────
    m = re.match(r"^\s*rm\s+(-\S+\s+)*(.+)$", cmd)
    if m:
        paths_str = m.group(2).strip()
        for p in paths_str.split():
            resolved = _resolve(state.cwd, p)
            state.files.pop(resolved, None)

    # ── mkdir / rmdir ─────────────────────────────────────────────────────────
    # No file content effect, skip

    # ── touch <path> ─────────────────────────────────────────────────────────
    m = re.match(r"^\s*touch\s+(.+)$", cmd)
    if m:
        for p in m.group(1).split():
            resolved = _resolve(state.cwd, p)
            if resolved not in state.files:
                state.files[resolved] = ""

    # ── output redirect: cmd > file or cmd >> file ────────────────────────────
    m = re.search(r"(>>?)\s*(\S+)\s*$", cmd)
    if m:
        mode = m.group(1)
        path = _resolve(state.cwd, m.group(2))
        # Can't know content without running the command — mark uncertain
        if mode == ">":
            state.files[path] = ""  # file will exist but we don't know contents
        uncertain = True

    # ── tee <path> ───────────────────────────────────────────────────────────
    m = re.search(r"\btee\s+(-a\s+)?(\S+)", cmd)
    if m:
        path = _resolve(state.cwd, m.group(2))
        state.files[path] = ""
        uncertain = True

    # ── sed -i 's/pattern/replacement/flags' <file> ──────────────────────────
    # Handle both: sed -i 's/.../.../' file  and  sed -i -e 's/.../' file
    for sed_match in re.finditer(
        r"sed\s+(?:-\w+\s+)*'s/(.+?)/(.*?)/([gimg]*)'\s+(\S+)",
        cmd,
        re.DOTALL,
    ):
        pattern = sed_match.group(1)
        replacement = sed_match.group(2)
        flags_str = sed_match.group(3)
        filepath = _resolve(state.cwd, sed_match.group(4))

        if filepath in state.files:
            old_content = state.files[filepath]
            re_flags = re.MULTILINE if "m" in flags_str else 0
            count = 0 if "g" in flags_str else 1
            try:
                new_content = re.sub(
                    pattern,
                    replacement,
                    old_content,
                    count=count,
                    flags=re_flags,
                )
                state.files[filepath] = new_content
                _record_line_edits(
                    line_history=line_history,
                    filename=filepath,
                    old_content=old_content,
                    new_content=new_content,
                    edit=LineEdit(
                        turn=turn,
                        type="bash_sed",
                        diff=f"sed: {cmd[:100]}",
                        message_index=message_index,
                        cmd=cmd,
                    ),
                )
            except re.error:
                uncertain = True

    # ── python -c / heredoc writes ────────────────────────────────────────────
    # Too complex to parse reliably — flag uncertain
    if re.search(r"\bpython\b.*-c\b|cat\s*<<|tee\s+/", cmd):
        uncertain = True

    state.bash_history.append(
        BashHistoryEntry(
            turn=turn,
            cmd=cmd,
            stdout="",  # filled in when tool result message is processed
            stderr="",
            exit_code=0,
            uncertain_fs_effects=uncertain,
        )
    )


def _apply_patch(
    block: dict[str, Any],
    state: _ReconstructionState,
    line_history: LineHistory,
    turn: int,
    message_index: int,
) -> None:
    """Apply an apply_patch tool call (Codex custom_tool_call format).

    Patch format (Codex apply_patch):
        *** Begin Patch
        *** Update File: path/to/file.py
        @@
         context line
        -removed line
        +added line
         context line
        *** End Patch

    Lines prefixed with ' ' are context (unchanged), '-' are removed, '+' are added.
    """
    args = block.get("arguments") or {}
    patch_text = args.get("input") or ""
    if not patch_text:
        return

    current_file: str | None = None
    affected_files: list[str] = []

    # Split into per-file sections
    sections: list[tuple[str, list[str]]] = []
    current_lines: list[str] = []

    for line in patch_text.splitlines():
        if (
            line.startswith("*** Update File:")
            or line.startswith("*** Add File:")
            or line.startswith("*** Delete File:")
        ):
            if current_file is not None:
                sections.append((current_file, current_lines))
            current_file = _resolve(state.cwd, line.split(":", 1)[1].strip())
            current_lines = []
        elif line in ("*** Begin Patch", "*** End Patch"):
            continue
        elif current_file is not None:
            current_lines.append(line)

    if current_file is not None:
        sections.append((current_file, current_lines))

    for filepath, hunk_lines in sections:
        affected_files.append(filepath)
        old_content = state.files.get(filepath, "")
        new_content = _apply_patch_hunks(old_content, hunk_lines)
        if new_content != old_content:
            state.files[filepath] = new_content
            _record_line_edits(
                line_history=line_history,
                filename=filepath,
                old_content=old_content,
                new_content=new_content,
                edit=LineEdit(
                    turn=turn,
                    type="patch",
                    diff=_make_diff_summary(old_content, new_content),
                    message_index=message_index,
                    cmd="apply_patch",
                ),
            )

    names = ", ".join(Path(f).name for f in affected_files)
    state.bash_history.append(
        BashHistoryEntry(
            turn=turn,
            cmd=f"apply_patch ({names})" if names else "apply_patch",
            stdout="",
            stderr="",
            exit_code=0,
            uncertain_fs_effects=False,
        )
    )


def _apply_patch_hunks(old_content: str, hunk_lines: list[str]) -> str:
    """Apply Codex-format patch hunks to old_content, return new content.

    Hunk lines:
        @@           - hunk separator (ignored, we use context matching)
         context     - unchanged line (leading space)
        -removed     - line to remove
        +added       - line to add
    """
    old_lines = old_content.splitlines(keepends=True)
    result = list(old_lines)

    # Process hunks separated by @@
    i = 0
    while i < len(hunk_lines):
        line = hunk_lines[i]
        if line.startswith("@@"):
            i += 1
            continue

        # Collect a contiguous block of edits
        context: list[str] = []
        removes: list[str] = []
        adds: list[str] = []

        while i < len(hunk_lines) and not hunk_lines[i].startswith("@@"):
            hl = hunk_lines[i]
            if hl.startswith("-"):
                removes.append(hl[1:])
            elif hl.startswith("+"):
                adds.append(hl[1:])
            elif hl.startswith(" "):
                if removes or adds:
                    # Apply the accumulated removes/adds before this context
                    result = _splice_lines(result, context, removes, adds)
                    removes = []
                    adds = []
                    context = []
                context.append(hl[1:])
            i += 1

        if removes or adds:
            result = _splice_lines(result, context, removes, adds)

    return "".join(result)


def _splice_lines(
    lines: list[str],
    context: list[str],
    removes: list[str],
    adds: list[str],
) -> list[str]:
    """Find the context+removes sequence in lines and replace with context+adds."""
    target = [c if c.endswith("\n") else c + "\n" for c in context]
    remove_lines = [r if r.endswith("\n") else r + "\n" for r in removes]
    add_lines = [a if a.endswith("\n") else a + "\n" for a in adds]
    search = target + remove_lines

    if not search:
        return lines

    # Find search sequence in lines
    for i in range(len(lines) - len(search) + 1):
        # Match ignoring trailing whitespace differences
        if all(lines[i + j].rstrip() == search[j].rstrip() for j in range(len(search))):
            return lines[: i + len(target)] + add_lines + lines[i + len(search) :]

    # Fallback: context-free search for removes only
    if remove_lines and not target:
        for i in range(len(lines) - len(remove_lines) + 1):
            if all(
                lines[i + j].rstrip() == remove_lines[j].rstrip() for j in range(len(remove_lines))
            ):
                return lines[:i] + add_lines + lines[i + len(remove_lines) :]

    # Could not find — return unchanged (patch failed to apply)
    return lines


def _apply_line_diff(
    state: _ReconstructionState,
    line_history: LineHistory,
    filepath: str,
    old_lines: list[str],
    new_lines: list[str],
    turn: int,
    message_index: int,
) -> None:
    old_content = "".join(old_lines)
    new_content = "".join(new_lines)
    state.files[filepath] = new_content
    _record_line_edits(
        line_history=line_history,
        filename=filepath,
        old_content=old_content,
        new_content=new_content,
        edit=LineEdit(
            turn=turn,
            type="patch",
            diff="patch applied",
            message_index=message_index,
        ),
    )


# ── Line history helpers ──────────────────────────────────────────────────────


def _record_line_edits(
    line_history: LineHistory,
    filename: str,
    old_content: str,
    new_content: str,
    edit: LineEdit,
) -> None:
    """Find which lines changed between old and new content, record the edit."""
    if old_content == new_content:
        return

    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()

    # Find lines that differ (simple approach: compare line by line up to
    # the shorter length, then mark all new lines beyond that)
    changed_lines: set[int] = set()

    # Lines that exist in new content and differ from old
    for i, new_line in enumerate(new_lines):
        line_num = i + 1  # 1-indexed
        old_line = old_lines[i] if i < len(old_lines) else None
        if old_line != new_line:
            changed_lines.add(line_num)

    # Lines that were deleted (mark the line before deletion)
    if len(old_lines) > len(new_lines) and new_lines:
        changed_lines.add(len(new_lines))

    if filename not in line_history:
        line_history[filename] = {}

    for line_num in changed_lines:
        key = str(line_num)
        if key not in line_history[filename]:
            line_history[filename][key] = []
        line_history[filename][key].append(edit)


def _make_diff_summary(old: str, new: str) -> str:
    """Make a short diff summary string."""
    old_lines = old.splitlines()
    new_lines = new.splitlines()
    added = len(new_lines) - len(old_lines)
    changed = sum(1 for o, n in zip(old_lines, new_lines, strict=False) if o != n)
    parts = []
    if changed:
        parts.append(f"{changed} lines changed")
    if added > 0:
        parts.append(f"+{added} lines")
    elif added < 0:
        parts.append(f"{added} lines")
    return ", ".join(parts) or "modified"


# ── Serialization ─────────────────────────────────────────────────────────────


def snapshots_to_api_response(
    snapshots: list[WorkspaceSnapshot],
    line_history: LineHistory,
    source: str = "reconstructed",
) -> dict[str, Any]:
    """Convert snapshots and line_history to the API response format."""
    return {
        "snapshots": [
            {
                "turn": s.turn,
                "files": s.files,
                "cwd": s.cwd,
                "bash_history": [
                    {
                        "turn": e.turn,
                        "cmd": e.cmd,
                        "stdout": e.stdout,
                        "stderr": e.stderr,
                        "exit_code": e.exit_code,
                        "uncertain_fs_effects": e.uncertain_fs_effects,
                    }
                    for e in s.bash_history
                ],
            }
            for s in snapshots
        ],
        "line_history": {
            filename: {
                line_num: [
                    {
                        "turn": e.turn,
                        "type": e.type,
                        "diff": e.diff,
                        "message_index": e.message_index,
                        "cmd": e.cmd,
                    }
                    for e in edits
                ]
                for line_num, edits in lines.items()
            }
            for filename, lines in line_history.items()
        },
        "source": source,
    }
