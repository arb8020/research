"""Haiku session test — pass 1: runs the agent end-to-end and writes a session.

Used as a baseline/gate for the runtime session refactor (see
rollouts/rollouts/agents/runtime_refactor.md).

What this script does right now:
  1. Spin up a fresh tempdir as a coding workspace.
  2. Hand the agent a prompt that exercises mkdir/write/edit/glob/grep/rm/verify.
  3. Run the agent loop against a real LLM endpoint, persisting to a session store.
  4. Print the session path and the final workspace state.

This pass intentionally has no assertions. We're verifying end-to-end plumbing
(endpoint + environment + store) before layering test logic on top.

Passes 2 and 3 (structural assertions + serialize-at-boundary resume,
SIGKILL harness) will follow.

Usage:
  export ANTHROPIC_API_KEY=...
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_session_test.py
  /Users/chiraagbalu/research/.venv/bin/python dev/session_refactor_tests/haiku_session_test.py --model claude-sonnet-4-5-20250929
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import trio

# Workspace root on sys.path so imports resolve when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "rollouts"))

from rollouts.agents import RunConfig, stdout_handler  # noqa: E402
from rollouts.agents.runtime import run_agent  # noqa: E402
from rollouts.agents.types import Actor, AgentState  # noqa: E402
from rollouts.dtypes import Endpoint, Trajectory  # noqa: E402
from rollouts.core import Message  # noqa: E402
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


def build_initial_state(
    workspace_dir: Path,
    endpoint: Endpoint,
) -> AgentState:
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


def walk_tree(root: Path) -> list[str]:
    """Return sorted list of paths under root, relative to root."""
    if not root.exists():
        return []
    out: list[str] = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        suffix = "/" if p.is_dir() else ""
        out.append(f"{rel}{suffix}")
    return out


async def run(model: str, session_base_dir: Path, workspace_dir: Path) -> None:
    # Span-persistence code path in providers/base.py constructs a default-dir
    # FileSessionStore for observability (cost/latency spans), ignoring our
    # custom store. Pre-create that path to avoid noisy warnings.
    # See runtime_refactor.md — this is another ownership smear worth fixing.
    default_store = Path.home() / ".rollouts" / "sessions"
    default_store.mkdir(parents=True, exist_ok=True)

    # Endpoint.model uses "provider/model-id" format.
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
    print()
    print("=" * 60)
    print(f"stop_reason: {final.stop}")
    print(f"turns: {final.turn_idx}")
    print(f"session_id: {final.session_id}")
    if final.session_id:
        session_dir = session_base_dir / final.session_id
        print(f"session dir: {session_dir}")
        for f in sorted(session_dir.iterdir()) if session_dir.exists() else []:
            print(f"  - {f.name} ({f.stat().st_size} bytes)")
    print()
    print("final workspace tree:")
    for entry in walk_tree(workspace_dir):
        print(f"  {entry}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929")
    parser.add_argument(
        "--session-dir",
        type=Path,
        default=None,
        help="Base dir for FileSessionStore. Default: a fresh tempdir.",
    )
    parser.add_argument(
        "--workspace-dir",
        type=Path,
        default=None,
        help="Workspace dir for the agent. Default: a fresh tempdir.",
    )
    parser.add_argument(
        "--keep",
        action="store_true",
        help="Do not delete tempdirs on exit.",
    )
    args = parser.parse_args()

    if "ANTHROPIC_API_KEY" not in os.environ:
        print("ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

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

    try:
        trio.run(run, args.model, session_base, workspace_dir)
    finally:
        if args.keep:
            print()
            print(f"keeping session dir: {session_base}")
            print(f"keeping workspace dir: {workspace_dir}")
            for d in cleanup_dirs:
                # Prevent TemporaryDirectory's __del__ cleanup by leaking the handle
                d._finalizer.detach()  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
