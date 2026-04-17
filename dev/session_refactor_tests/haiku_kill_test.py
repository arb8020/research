"""Haiku SIGKILL harness — pass 3 for the runtime session refactor.

What this does:
  1. Launch haiku_session_test.py as a subprocess, writing to a known session
     dir and workspace dir.
  2. Tail the session's messages.jsonl as it grows. Count ToolCallContent
     blocks in assistant messages as they land.
  3. After the Nth tool call lands (configurable), send SIGKILL to the
     subprocess. No cleanup runs — it's a hard kill.
  4. Verify post-kill state:
       - messages.jsonl parses (tolerant of trailing partial line)
       - session is loadable via FileSessionStore.get_trajectory
       - known-good effects can be replayed by fold into a fresh tempdir
         and that tree matches the workspace as it stood at kill time.

This is the pre-refactor baseline. Once effects are first-class session
entries, the same test should still pass — the data we're asserting on is
semantic (what happened + what's recoverable), not structural (where in the
messages list the data lives).

Usage:
  export ANTHROPIC_API_KEY=...
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_kill_test.py --kill-after-calls 3
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import trio

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rollouts"))

# Import pass-2 helpers for fold-based verification.
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
from haiku_session_test import (  # noqa: E402
    compare_trees,
    extract_tool_calls,
    replay_effects_into_fresh_workspace,
)

from rollouts.store import FileSessionStore  # noqa: E402


def load_messages_tolerant(path: Path) -> tuple[list[dict[str, Any]], bool]:
    """Read messages.jsonl, skipping a trailing partial line if present.

    Returns (messages, had_partial_trailing_line).

    A partial trailing line can happen if the writer was killed mid-flush.
    Any interior line that fails to parse is still a hard error (that would
    be a real corruption bug).
    """
    if not path.exists():
        return [], False
    text = path.read_text()
    if not text:
        return [], False
    lines = text.split("\n")
    # Last element is "" if the file ended with a newline (normal case).
    trailing_partial = False
    if lines and lines[-1] == "":
        lines = lines[:-1]
    # If the file didn't end with a newline, the last line is suspect.
    elif lines:
        last = lines[-1]
        try:
            json.loads(last)
        except json.JSONDecodeError:
            trailing_partial = True
            lines = lines[:-1]

    out: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"interior parse error at line {i}: {e}") from None
    return out, trailing_partial


def count_tool_calls_so_far(messages: list[dict[str, Any]]) -> int:
    """Count ToolCallContent blocks across all assistant messages."""
    n = 0
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "toolCall":
                n += 1
    return n


def find_session_dir(session_base_dir: Path, started_after: float) -> Path | None:
    """Find the session dir created by the subprocess after `started_after`."""
    if not session_base_dir.exists():
        return None
    for p in sorted(session_base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if p.is_dir() and p.stat().st_mtime >= started_after - 1:
            if (p / "messages.jsonl").exists() or (p / "session.json").exists():
                return p
    return None


async def kill_at_boundary(
    session_base_dir: Path, workspace_dir: Path, model: str, kill_after_calls: int
) -> tuple[Path, int, Path]:
    """Launch the haiku script, kill after the Nth tool call lands.

    Returns (session_dir, observed_tool_call_count, workspace_snapshot_dir).

    The workspace_snapshot_dir is a copy of workspace_dir taken just before
    the kill — this is the ground truth for "what state had been achieved at
    the moment of kill."
    """
    start_time = time.time()
    script = str(THIS_DIR / "haiku_session_test.py")

    env = os.environ.copy()
    # Drop progress bar in the child so its stdout isn't a carnival.
    cmd = [
        sys.executable,
        script,
        "--keep",
        "--session-dir",
        str(session_base_dir),
        "--workspace-dir",
        str(workspace_dir),
        "--model",
        model,
    ]
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.DEVNULL,  # quiet; we watch the files
        stderr=subprocess.DEVNULL,
    )

    print(f"launched subprocess pid={proc.pid}")

    session_dir: Path | None = None
    last_count = 0
    poll_deadline = time.time() + 180  # 3 minute overall safety

    try:
        while time.time() < poll_deadline:
            if session_dir is None:
                session_dir = find_session_dir(session_base_dir, start_time)
                if session_dir is None:
                    await trio.sleep(0.2)
                    continue
                print(f"detected session dir: {session_dir}")

            msg_path = session_dir / "messages.jsonl"
            messages, _partial = load_messages_tolerant(msg_path)
            count = count_tool_calls_so_far(messages)
            if count != last_count:
                print(f"  tool calls observed: {count}")
                last_count = count
            if count >= kill_after_calls:
                # Snapshot the workspace before we kill; this freezes our
                # ground-truth "state at kill time."
                snapshot_dir = workspace_dir.parent / (workspace_dir.name + "_snapshot")
                if snapshot_dir.exists():
                    # Stale snapshot from a prior run; blow it away.
                    import shutil

                    shutil.rmtree(snapshot_dir)
                _copy_tree(workspace_dir, snapshot_dir)
                print(f"reached {count} tool calls; SIGKILLing pid {proc.pid}")
                proc.send_signal(signal.SIGKILL)
                break
            # Detect premature exit (e.g., already finished before we could kill).
            if proc.poll() is not None:
                print(
                    f"subprocess exited before reaching {kill_after_calls} tool calls "
                    f"(observed {count}, exit code {proc.returncode})"
                )
                snapshot_dir = workspace_dir.parent / (workspace_dir.name + "_snapshot")
                if snapshot_dir.exists():
                    import shutil

                    shutil.rmtree(snapshot_dir)
                _copy_tree(workspace_dir, snapshot_dir)
                return session_dir, count, snapshot_dir
            await trio.sleep(0.2)
        else:
            print("deadline exceeded waiting for tool calls; killing subprocess")
            proc.send_signal(signal.SIGKILL)
            snapshot_dir = workspace_dir.parent / (workspace_dir.name + "_snapshot")
            if snapshot_dir.exists():
                import shutil

                shutil.rmtree(snapshot_dir)
            _copy_tree(workspace_dir, snapshot_dir)
            return session_dir or session_base_dir, last_count, snapshot_dir

        # Drain and wait for actual exit (should be immediate after SIGKILL).
        proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    assert session_dir is not None
    return session_dir, last_count, snapshot_dir


def _copy_tree(src: Path, dst: Path) -> None:
    import shutil

    if src.exists():
        shutil.copytree(src, dst)
    else:
        dst.mkdir(parents=True, exist_ok=True)


async def verify_post_kill(
    session_base_dir: Path,
    session_dir: Path,
    workspace_snapshot_dir: Path,
) -> int:
    """Post-kill assertions. Returns number of violations."""
    session_id = session_dir.name
    print()
    print("=" * 60)
    print("PASS 3 POST-KILL VERIFICATION")
    print("=" * 60)

    all_violations: list[tuple[str, list[str]]] = []

    print("▸ messages.jsonl parses (tolerant of trailing partial line)")
    messages, had_partial = load_messages_tolerant(session_dir / "messages.jsonl")
    v = []
    if had_partial:
        print(f"  note: dropped one trailing partial line (len={len(messages)} parsed)")
    if not messages:
        v.append("zero messages parsed")
    all_violations.append(("parse", v))
    print(f"  violations: {len(v)}")

    print("▸ FileSessionStore.get_trajectory loads")
    v = []
    store = FileSessionStore(base_dir=session_base_dir)
    trajectory, err = await store.get_trajectory(session_id)
    if err is not None:
        # Tolerant: a trailing partial might make get_trajectory complain.
        # If it does, that's a finding worth recording but not necessarily a
        # failure of the kill test itself — it's information about how the
        # store handles partial writes.
        v.append(f"get_trajectory error: {err}")
    elif trajectory is None or not trajectory.messages:
        v.append("get_trajectory returned empty")
    all_violations.append(("store_load", v))
    print(f"  violations: {len(v)}")

    print("▸ fold of observed effects matches workspace snapshot at kill time")
    records = extract_tool_calls(messages)
    # Drop records without a tool result — those are the "dispatched but
    # incomplete" calls at the kill boundary. The fold below assumes each
    # call we replay did complete in the real run. Replaying an unfinished
    # call is nondeterministic in general.
    completed = [r for r in records if r.result_message_idx is not None]
    print(
        f"  tool calls total={len(records)} "
        f"completed(has result)={len(completed)} "
        f"pending(no result)={len(records) - len(completed)}"
    )
    v = []
    with tempfile.TemporaryDirectory(prefix="haiku_kill_replay_") as td:
        fresh = Path(td)
        replay_violations = replay_effects_into_fresh_workspace(completed, fresh)
        tree_violations = compare_trees(fresh, workspace_snapshot_dir)
    v = replay_violations + tree_violations
    all_violations.append(("fold_matches_snapshot", v))
    print(f"  violations: {len(v)}")

    total = sum(len(vs) for _, vs in all_violations)
    print()
    print("=" * 60)
    if total == 0:
        print("✓ ALL PASS-3 CHECKS PASSED")
    else:
        print(f"✗ {total} VIOLATION(S)")
        for name, vs in all_violations:
            if vs:
                print(f"\n  [{name}]")
                for line in vs:
                    print(f"    {line}")
    print("=" * 60)
    return total


async def run_main(args: argparse.Namespace) -> int:
    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    cleanup_dirs: list[tempfile.TemporaryDirectory] = []

    if args.session_dir is None:
        d = tempfile.TemporaryDirectory(prefix="haiku_kill_sessions_")
        session_base = Path(d.name)
        cleanup_dirs.append(d)
    else:
        args.session_dir.mkdir(parents=True, exist_ok=True)
        session_base = args.session_dir

    if args.workspace_dir is None:
        d = tempfile.TemporaryDirectory(prefix="haiku_kill_workspace_")
        workspace_dir = Path(d.name)
        cleanup_dirs.append(d)
    else:
        args.workspace_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir = args.workspace_dir

    try:
        session_dir, n_observed, snapshot_dir = await kill_at_boundary(
            session_base, workspace_dir, args.model, args.kill_after_calls
        )
        print()
        print(f"observed {n_observed} tool calls before/at kill")
        print(f"snapshot taken at kill: {snapshot_dir}")

        n_violations = await verify_post_kill(session_base, session_dir, snapshot_dir)
        return 0 if n_violations == 0 else 1
    finally:
        if args.keep:
            print()
            print(f"keeping session dir: {session_base}")
            print(f"keeping workspace dir: {workspace_dir}")
            for d in cleanup_dirs:
                d._finalizer.detach()  # type: ignore[attr-defined]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument("--session-dir", type=Path, default=None)
    parser.add_argument("--workspace-dir", type=Path, default=None)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--kill-after-calls",
        type=int,
        default=3,
        help="SIGKILL the subprocess after the Nth ToolCallContent lands in messages.jsonl.",
    )
    args = parser.parse_args()
    sys.exit(trio.run(run_main, args))


if __name__ == "__main__":
    main()
