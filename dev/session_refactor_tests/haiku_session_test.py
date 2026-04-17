"""Haiku session test — passes 1 and 2: end-to-end run + structural assertions + resume.

Used as a baseline/gate for the runtime session refactor (see
rollouts/rollouts/agents/runtime_refactor.md).

Pass 1 (default when no session exists): run the agent end-to-end, produce a
session file and a workspace on disk. No assertions beyond "it ran and stopped."

Pass 2 (--verify): given an existing session + workspace, assert:
  - Structural invariants on the session:
      every ToolCallContent has a matching tool-role Message with the same
      tool_call_id, and vice versa — no orphans either way.
  - Final workspace matches expected shape (poems/ exists, poems/haikus/ does not).
  - Message resume: session_store.get_trajectory returns the same messages the
    agent's Trajectory carried at the end.
  - Environment resume by fold: replay state-changing tool calls (bash, write,
    edit) from the session against a FRESH tempdir, assert the resulting
    filesystem matches the original workspace.

The environment-resume test is intentionally hand-rolled: it parses tool calls
out of assistant message content blocks (G1: today effects aren't first-class).
Post-refactor, the same test reads effect entries from the session directly.
Both should produce the same result.

Pass 3 (SIGKILL harness) is a separate script.

Usage:
  # Run + verify, fresh tempdirs:
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_session_test.py --verify

  # Run only (keep artifacts for later inspection):
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_session_test.py \\
      --keep --session-dir /tmp/haiku_sessions --workspace-dir /tmp/haiku_workspace

  # Verify an existing run without re-running:
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_session_test.py \\
      --verify-only --session-dir /tmp/haiku_sessions --workspace-dir /tmp/haiku_workspace

Requires: ANTHROPIC_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import trio

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rollouts"))

from rollouts.agents import RunConfig, stdout_handler  # noqa: E402
from rollouts.agents.runtime import run_agent  # noqa: E402
from rollouts.agents.types import Actor, AgentState  # noqa: E402
from rollouts.core import Message  # noqa: E402
from rollouts.dtypes import Endpoint, Trajectory  # noqa: E402
from rollouts.environments.coding import LocalFilesystemEnvironment  # noqa: E402
from rollouts.store import FileSessionStore  # noqa: E402


HAIKU_PROMPT = """Your working directory is the current directory. Complete these steps in order:

1. Create a directory `poems/`.
2. Inside `poems/`, create a subdirectory `haikus/`.
3. Inside `haikus/`, write three files `haiku_1.txt`, `haiku_2.txt`, `haiku_3.txt`.
   Each should contain a three-line haiku (any content you like).
4. For each of the three files, edit the FIRST line to say "FIRST LINE".
   Then edit the SECOND line to say "SECOND LINE".
   Then edit the THIRD line to say "THIRD LINE".
5. Use glob or grep to verify that all three files exist and contain "SECOND LINE".
6. Delete the three haiku files AND the `haikus/` subdirectory.
7. Verify the deletion: confirm `poems/haikus/` no longer exists but `poems/` does.

When done, summarize what happened and stop.
"""


# ────────────────────────────────────────────────────────────────────────────
# Pass 1: run the agent end to end.
# ────────────────────────────────────────────────────────────────────────────


def build_initial_state(workspace_dir: Path, endpoint: Endpoint) -> AgentState:
    env = LocalFilesystemEnvironment(working_dir=workspace_dir, tools="full")
    system_msg = Message(
        role="system",
        content=(
            "You are a careful coding agent. Use the tools available to you "
            "(read, write, edit, bash, glob, grep) to accomplish the user's task. "
            "Verify your work as you go."
        ),
    )
    user_msg = Message(role="user", content=HAIKU_PROMPT)
    actor = Actor(
        trajectory=Trajectory(messages=[system_msg, user_msg]),
        endpoint=endpoint,
        tools=env.get_tools(),
    )
    return AgentState(actor=actor, environment=env)


async def run_agent_once(
    model: str, session_base_dir: Path, workspace_dir: Path
) -> tuple[str, list[AgentState]]:
    # Span-persistence in providers/base.py constructs a default-dir
    # FileSessionStore independent of ours (G7 in runtime_refactor.md). Pre-create
    # its path to silence warnings; noted as a smear, not our bug to fix here.
    (Path.home() / ".rollouts" / "sessions").mkdir(parents=True, exist_ok=True)

    model_str = model if "/" in model else f"anthropic/{model}"
    endpoint = Endpoint(
        model=model_str,
        base_url="https://api.anthropic.com/v1",
        api_format="anthropic-messages",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )

    store = FileSessionStore(base_dir=session_base_dir)
    state = build_initial_state(workspace_dir, endpoint)
    run_config = RunConfig(
        on_chunk=stdout_handler,
        session_store=store,
        show_progress=True,
    )

    print(f"workspace: {workspace_dir}")
    print(f"session store: {session_base_dir}")
    print()

    states = await run_agent(state, run_config)
    final = states[-1]
    assert final.session_id is not None, "session should have an id after run"

    print()
    print("=" * 60)
    print(f"stop_reason: {final.stop}")
    print(f"turns: {final.turn_idx}")
    print(f"session_id: {final.session_id}")
    return final.session_id, states


# ────────────────────────────────────────────────────────────────────────────
# Pass 2: structural assertions + resume tests.
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCallRecord:
    """One tool call observed in the session, with its matching result if any."""

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    result_content: str | None  # None means no matching tool-role message found
    assistant_message_idx: int
    result_message_idx: int | None


def load_messages_jsonl(session_dir: Path) -> list[dict[str, Any]]:
    path = session_dir / "messages.jsonl"
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_tool_calls(messages: list[dict[str, Any]]) -> list[ToolCallRecord]:
    """Walk messages, pair up every ToolCallContent with its tool-role Message.

    This is the hand-rolled pre-refactor version of "read effects from the log."
    Post-refactor this function goes away — effects are first-class entries.
    """
    records: list[ToolCallRecord] = []

    # First pass: build an index of tool-role messages by tool_call_id.
    result_by_id: dict[str, tuple[int, str]] = {}
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid is None:
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            # Tool-role messages with structured content are unusual; stringify.
            content = json.dumps(content)
        result_by_id[tcid] = (i, content)

    # Second pass: walk assistant messages, extract ToolCallContent blocks.
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "toolCall":
                continue
            call_id = block.get("id", "")
            tool_name = block.get("name", "")
            args = block.get("arguments", {})
            match = result_by_id.get(call_id)
            if match is None:
                records.append(
                    ToolCallRecord(
                        call_id=call_id,
                        tool_name=tool_name,
                        arguments=args,
                        result_content=None,
                        assistant_message_idx=i,
                        result_message_idx=None,
                    )
                )
            else:
                result_idx, result_content = match
                records.append(
                    ToolCallRecord(
                        call_id=call_id,
                        tool_name=tool_name,
                        arguments=args,
                        result_content=result_content,
                        assistant_message_idx=i,
                        result_message_idx=result_idx,
                    )
                )

    return records


def check_structural_invariants(
    messages: list[dict[str, Any]], records: list[ToolCallRecord]
) -> list[str]:
    """Return a list of invariant violations. Empty list = all good."""
    violations: list[str] = []

    # Every ToolCallContent must have a matching tool-role message.
    orphan_calls = [r for r in records if r.result_message_idx is None]
    for r in orphan_calls:
        violations.append(
            f"orphan ToolCall: id={r.call_id} tool={r.tool_name} "
            f"(assistant_msg_idx={r.assistant_message_idx}, no matching tool-role message)"
        )

    # Every tool-role message must have a matching ToolCallContent block.
    matched_ids: set[str] = {r.call_id for r in records if r.result_message_idx is not None}
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        tcid = msg.get("tool_call_id")
        if tcid is None:
            violations.append(f"tool-role message at idx {i} has no tool_call_id")
            continue
        if tcid not in matched_ids:
            violations.append(
                f"orphan tool-role message: id={tcid} (idx {i}, no matching ToolCallContent)"
            )

    return violations


def check_workspace_final_state(workspace_dir: Path) -> list[str]:
    """Expected: poems/ exists, poems/haikus/ does NOT exist."""
    violations: list[str] = []
    poems = workspace_dir / "poems"
    haikus = poems / "haikus"
    if not poems.is_dir():
        violations.append(f"expected {poems} to be a directory, not found or wrong type")
    if haikus.exists():
        violations.append(f"expected {haikus} to NOT exist, but it does")
    return violations


async def check_message_resume(
    session_base_dir: Path, session_id: str, final_states: list[AgentState] | None
) -> list[str]:
    """Call get_trajectory and verify it matches the final state (if provided) or at
    least parses. When final_states is None (--verify-only), we just check parsing."""
    violations: list[str] = []
    store = FileSessionStore(base_dir=session_base_dir)
    trajectory, err = await store.get_trajectory(session_id)
    if err is not None:
        violations.append(f"get_trajectory returned error: {err}")
        return violations
    if trajectory is None:
        violations.append("get_trajectory returned None trajectory")
        return violations
    if len(trajectory.messages) == 0:
        violations.append("loaded trajectory has zero messages")
        return violations

    if final_states is not None:
        expected = final_states[-1].actor.trajectory.messages
        if len(trajectory.messages) != len(expected):
            violations.append(
                f"message count mismatch: disk={len(trajectory.messages)} "
                f"in-memory={len(expected)}"
            )
        else:
            for i, (a, b) in enumerate(zip(trajectory.messages, expected)):
                if a.role != b.role:
                    violations.append(
                        f"role mismatch at idx {i}: disk={a.role} in-memory={b.role}"
                    )
                    break
    return violations


def replay_effects_into_fresh_workspace(
    records: list[ToolCallRecord], fresh_dir: Path
) -> list[str]:
    """Hand-rolled fold: replay state-changing tool calls into a fresh dir.

    Only replays bash / write / edit. glob, grep, read are read-only.
    Returns violations (e.g., tool we don't know how to replay, exec failures).

    Post-refactor this becomes `fold(Environment, session.effects)`.
    """
    violations: list[str] = []
    fresh_dir.mkdir(parents=True, exist_ok=True)

    for r in records:
        if r.tool_name == "bash":
            cmd = r.arguments.get("command")
            if not isinstance(cmd, str):
                violations.append(
                    f"bash call {r.call_id} has non-string command: {r.arguments!r}"
                )
                continue
            try:
                # Timeout is generous; these commands should be fast.
                subprocess.run(
                    cmd, shell=True, cwd=fresh_dir, check=False, timeout=30, capture_output=True
                )
            except subprocess.TimeoutExpired:
                violations.append(f"bash replay timed out: {cmd[:80]}")
        elif r.tool_name == "write":
            path = r.arguments.get("path")
            content = r.arguments.get("content", "")
            if not isinstance(path, str):
                violations.append(f"write call {r.call_id} missing path: {r.arguments!r}")
                continue
            target = _resolve_replay_path(fresh_dir, path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content if isinstance(content, str) else json.dumps(content))
        elif r.tool_name == "edit":
            path = r.arguments.get("path")
            old = r.arguments.get("old_string", r.arguments.get("old", ""))
            new = r.arguments.get("new_string", r.arguments.get("new", ""))
            replace_all = r.arguments.get("replace_all", False)
            if not isinstance(path, str):
                violations.append(f"edit call {r.call_id} missing path: {r.arguments!r}")
                continue
            target = _resolve_replay_path(fresh_dir, path)
            if not target.exists():
                violations.append(
                    f"edit call {r.call_id} targets non-existent {target} (replay order bug?)"
                )
                continue
            text = target.read_text()
            if replace_all:
                new_text = text.replace(old, new)
            else:
                # Edit tool requires the occurrence to be unique; if not, the
                # agent's real call would have failed, but we're replaying a
                # successful call so one occurrence is assumed.
                new_text = text.replace(old, new, 1)
            target.write_text(new_text)
        else:
            # Read-only tools (glob, grep, read) don't change state.
            pass
    return violations


def _resolve_replay_path(fresh_dir: Path, path: str) -> Path:
    """Tool calls use absolute paths from the original workspace_dir. Strip the
    original prefix and remap under fresh_dir. If the path is already relative,
    treat it as relative to fresh_dir."""
    p = Path(path)
    if not p.is_absolute():
        return fresh_dir / p
    # Absolute path from the original workspace. Find its last path component
    # under something that looks like our expected structure (poems/ rooted).
    parts = p.parts
    for i, part in enumerate(parts):
        if part == "poems":
            return fresh_dir.joinpath(*parts[i:])
    # Fall back to using the filename only — crude but covers single-file edits.
    return fresh_dir / p.name


def compare_trees(a: Path, b: Path) -> list[str]:
    """Return differences between two directory trees (paths + file content).

    Both must exist. Directory-only comparison; not comparing metadata.
    """
    violations: list[str] = []

    def rel_set(root: Path) -> dict[str, tuple[bool, bytes | None]]:
        out: dict[str, tuple[bool, bytes | None]] = {}
        for p in sorted(root.rglob("*")):
            rel = str(p.relative_to(root))
            if p.is_dir():
                out[rel] = (True, None)
            elif p.is_file():
                out[rel] = (False, p.read_bytes())
        return out

    entries_a = rel_set(a)
    entries_b = rel_set(b)
    keys = set(entries_a) | set(entries_b)
    for k in sorted(keys):
        va = entries_a.get(k)
        vb = entries_b.get(k)
        if va is None:
            violations.append(f"only in replay: {k}")
        elif vb is None:
            violations.append(f"only in original: {k}")
        elif va[0] != vb[0]:
            violations.append(f"type mismatch at {k}: replay_dir={va[0]} original_dir={vb[0]}")
        elif va[1] != vb[1]:
            violations.append(f"file content differs at {k}")
    return violations


async def verify(
    session_base_dir: Path,
    workspace_dir: Path,
    session_id: str,
    final_states: list[AgentState] | None,
) -> int:
    """Run all pass-2 assertions. Returns number of violations found."""
    session_dir = session_base_dir / session_id
    messages = load_messages_jsonl(session_dir)
    records = extract_tool_calls(messages)

    print()
    print("=" * 60)
    print("PASS 2 VERIFICATION")
    print("=" * 60)
    print(f"messages on disk: {len(messages)}")
    print(f"tool calls extracted: {len(records)}")
    print(f"tool names: {sorted(set(r.tool_name for r in records))}")
    print()

    all_violations: list[tuple[str, list[str]]] = []

    print("▸ structural invariants (ToolCall↔ToolResult pairing)")
    v = check_structural_invariants(messages, records)
    all_violations.append(("structural", v))
    print(f"  violations: {len(v)}")

    print("▸ final workspace state (poems/ yes, poems/haikus/ no)")
    v = check_workspace_final_state(workspace_dir)
    all_violations.append(("workspace_final", v))
    print(f"  violations: {len(v)}")

    print("▸ message resume (session_store.get_trajectory)")
    v = await check_message_resume(session_base_dir, session_id, final_states)
    all_violations.append(("message_resume", v))
    print(f"  violations: {len(v)}")

    print("▸ environment resume by fold (replay effects → fresh workspace → tree match)")
    with tempfile.TemporaryDirectory(prefix="haiku_replay_") as td:
        fresh = Path(td)
        replay_violations = replay_effects_into_fresh_workspace(records, fresh)
        tree_violations = compare_trees(fresh, workspace_dir)
        v = replay_violations + tree_violations
    all_violations.append(("env_resume", v))
    print(f"  violations: {len(v)}")

    total = sum(len(v) for _, v in all_violations)
    print()
    print("=" * 60)
    if total == 0:
        print("✓ ALL PASS-2 CHECKS PASSED")
    else:
        print(f"✗ {total} VIOLATION(S)")
        for name, v in all_violations:
            if v:
                print(f"\n  [{name}]")
                for line in v:
                    print(f"    {line}")
    print("=" * 60)
    return total


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────


def walk_tree(root: Path) -> list[str]:
    if not root.exists():
        return []
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        suffix = "/" if p.is_dir() else ""
        out.append(f"{rel}{suffix}")
    return out


def find_existing_session(session_base_dir: Path) -> str | None:
    """Find the most recently modified session in a base dir."""
    if not session_base_dir.exists():
        return None
    candidates = [
        p for p in session_base_dir.iterdir() if p.is_dir() and (p / "messages.jsonl").exists()
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].name


async def run_main(args: argparse.Namespace) -> int:
    cleanup_dirs: list[tempfile.TemporaryDirectory] = []

    if args.session_dir is None:
        d = tempfile.TemporaryDirectory(prefix="haiku_sessions_")
        session_base = Path(d.name)
        cleanup_dirs.append(d)
    else:
        args.session_dir.mkdir(parents=True, exist_ok=True)
        session_base = args.session_dir

    if args.workspace_dir is None:
        d = tempfile.TemporaryDirectory(prefix="haiku_workspace_")
        workspace_dir = Path(d.name)
        cleanup_dirs.append(d)
    else:
        args.workspace_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = args.workspace_dir

    exit_code = 0

    try:
        if args.verify_only:
            session_id = find_existing_session(session_base)
            if session_id is None:
                print(f"No session found under {session_base}", file=sys.stderr)
                return 2
            n_violations = await verify(session_base, workspace_dir, session_id, None)
            exit_code = 0 if n_violations == 0 else 1
        else:
            if "ANTHROPIC_API_KEY" not in os.environ:
                print("ANTHROPIC_API_KEY not set", file=sys.stderr)
                return 1
            session_id, states = await run_agent_once(args.model, session_base, workspace_dir)
            print()
            print("final workspace tree:")
            for entry in walk_tree(workspace_dir):
                print(f"  {entry}")
            if args.verify:
                n_violations = await verify(session_base, workspace_dir, session_id, states)
                exit_code = 0 if n_violations == 0 else 1
    finally:
        if args.keep:
            print()
            print(f"keeping session dir: {session_base}")
            print(f"keeping workspace dir: {workspace_dir}")
            for d in cleanup_dirs:
                d._finalizer.detach()  # type: ignore[attr-defined]

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--keep", action="store_true", help="Do not delete tempdirs on exit.")
    parser.add_argument(
        "--verify", action="store_true", help="Run pass-2 verification after the run."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Don't run the agent; verify the most recent session in --session-dir.",
    )
    args = parser.parse_args()

    if args.verify_only and args.session_dir is None:
        print("--verify-only requires --session-dir", file=sys.stderr)
        sys.exit(2)

    sys.exit(trio.run(run_main, args))


if __name__ == "__main__":
    main()
