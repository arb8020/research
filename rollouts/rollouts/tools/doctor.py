"""rollouts doctor -- reparse external agent session files into trajectories.

Re-exports trajectory files from raw session data stored by external agent CLIs.
Useful when:
  - A parser bug is fixed (e.g. Codex response_item format)
  - Dtype semantics change (e.g. new content block types)
  - You want to normalize old results to the current format

Usage:
    rollouts doctor reparse <results-dir>
    rollouts doctor reparse <results-dir> --driver codex
    rollouts doctor reparse <results-dir> --driver claude
    rollouts doctor reparse <results-dir> --dry-run

Supported drivers:
    codex       ~/.codex/sessions/YYYY/MM/DD/<session_id>.jsonl
    claude      ~/.claude/projects/<cwd-as-path>/<session_id>.jsonl
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ── Session file locators ─────────────────────────────────────────────────────


def _codex_session_path(session_id: str) -> Path | None:
    """Find a Codex session file by session ID.

    Codex stores sessions at:
        ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<session_id>.jsonl

    The file name embeds the session ID as a suffix, so we search by glob.
    """
    base = Path.home() / ".codex" / "sessions"
    if not base.exists():
        return None
    # Files are named rollout-<timestamp>-<session_id>.jsonl
    matches = list(base.rglob(f"*{session_id}.jsonl"))
    return matches[0] if matches else None


def _claude_session_path(session_id: str, cwd: str) -> Path | None:
    """Find a Claude Code session file by session ID and cwd.

    Claude Code stores sessions at:
        ~/.claude/projects/<cwd-with-slashes-as-dashes>/<session_id>.jsonl

    On macOS, /var is a symlink to /private/var. Claude Code resolves the real
    path, so we try both the raw cwd and the resolved path.
    """
    base = Path.home() / ".claude" / "projects"
    # Claude Code encodes the cwd as a project directory name by replacing
    # both / and _ with -. Try both raw cwd and the resolved path (macOS:
    # /var → /private/var symlink).
    candidates = [cwd, str(Path(cwd).resolve())]
    for c in candidates:
        project_name = c.replace("/", "-").replace("_", "-")
        candidate = base / project_name / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


def _find_session_file(metadata: dict[str, Any]) -> tuple[str, Path] | tuple[None, None]:
    """Locate the session file for a sample given its metadata.

    Returns (driver_name, session_path) or (None, None) if not found.
    """
    driver = metadata.get("driver")
    session_id = metadata.get("session_id")
    cwd = metadata.get("cwd", "")

    if not session_id:
        return None, None

    if driver == "codex":
        path = _codex_session_path(session_id)
        return ("codex", path) if path else (None, None)

    if driver == "claude":
        path = _claude_session_path(session_id, cwd)
        return ("claude", path) if path else (None, None)

    return None, None


# ── Trajectory reconstruction ─────────────────────────────────────────────────


def _reparse_codex_session(session_path: Path) -> list[dict[str, Any]]:
    """Parse a Codex session file into a list of message dicts."""
    from dataclasses import asdict, is_dataclass

    from rollouts.drivers.codex import _CodexEventParser
    from rollouts.drivers.runner import _EventAccumulator

    parser = _CodexEventParser()
    acc = _EventAccumulator()

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            for event in parser.parse(msg):
                acc.handle(event)

    messages = acc.finalize()
    result = []
    for m in messages:
        content = m.content
        if isinstance(content, list):
            blocks = []
            for b in content:
                blocks.append(asdict(b) if is_dataclass(b) else {"text": str(b)})
            result.append({"role": m.role, "content": blocks})
        else:
            result.append({
                "role": m.role,
                "content": content,
                "tool_call_id": getattr(m, "tool_call_id", None),
            })
    return result


def _reparse_claude_session(session_path: Path) -> list[dict[str, Any]]:
    """Parse a Claude Code session file into a list of message dicts.

    Claude Code session files store messages directly as
    {"type": "user"|"assistant", "message": {role, content, ...}}.
    We filter out queue-operation entries and extract the message objects.
    """
    messages = []
    seen_ids: set[str] = set()

    with open(session_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")
            if entry_type not in ("user", "assistant"):
                continue

            # Claude Code session files may contain sidechain messages
            # (sub-agent calls). Skip those — they're not the main trajectory.
            if entry.get("isSidechain"):
                continue

            msg = entry.get("message", {})
            if not msg:
                continue

            # Deduplicate by message ID if present
            msg_id = msg.get("id")
            if msg_id:
                if msg_id in seen_ids:
                    continue
                seen_ids.add(msg_id)

            messages.append(msg)

    return messages


# ── Trajectory file writer ────────────────────────────────────────────────────


def _write_trajectory(trajectory_path: Path, messages: list[dict[str, Any]]) -> None:
    """Write messages as a trajectory JSONL file (one message per line)."""
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trajectory_path, "w") as f:
        for msg in messages:
            f.write(json.dumps(msg) + "\n")


# ── Main reparse command ──────────────────────────────────────────────────────


def reparse(
    results_dir: str | Path,
    *,
    driver_filter: str | None = None,
    dry_run: bool = False,
) -> None:
    """Reparse all external agent session files in a results directory.

    Walks results_dir looking for sample metadata files, locates the
    corresponding CLI session files, re-runs them through the current
    parser, and overwrites the trajectory files.

    Args:
        results_dir: Path to a rollouts results directory (contains samples/).
        driver_filter: If set, only reparse samples from this driver ("codex" or "claude").
        dry_run: If True, print what would be done without writing anything.
    """
    results_dir = Path(results_dir)

    # Support both a single run directory and a parent containing multiple runs
    run_dirs: list[Path] = []
    if (results_dir / "samples").exists():
        run_dirs = [results_dir]
    else:
        run_dirs = [d for d in sorted(results_dir.iterdir()) if (d / "samples").exists()]

    if not run_dirs:
        print(f"No eval run directories found in {results_dir}")
        return

    total_reparsed = 0
    total_skipped = 0
    total_missing = 0

    for run_dir in run_dirs:
        samples_dir = run_dir / "samples"
        trajectories_dir = run_dir / "trajectories"

        sample_files = sorted(samples_dir.glob("*.json"))
        if not sample_files:
            continue

        print(f"\n{run_dir.name}")

        for sample_file in sample_files:
            sample_id = sample_file.stem
            trajectory_path = trajectories_dir / f"{sample_id}.jsonl"

            try:
                sample = json.loads(sample_file.read_text())
            except (json.JSONDecodeError, OSError) as e:
                print(f"  {sample_id}: cannot read sample file: {e}")
                total_skipped += 1
                continue

            metadata = sample.get("metadata", {})
            driver, session_path = _find_session_file(metadata)

            if driver is None:
                total_skipped += 1
                continue

            if driver_filter and driver != driver_filter:
                total_skipped += 1
                continue

            if session_path is None:
                print(
                    f"  {sample_id}: session file not found (driver={metadata.get('driver')}, session_id={metadata.get('session_id')})"
                )
                total_missing += 1
                continue

            try:
                if driver == "codex":
                    messages = _reparse_codex_session(session_path)
                else:
                    messages = _reparse_claude_session(session_path)
            except Exception as e:
                print(f"  {sample_id}: parse failed: {e}")
                total_skipped += 1
                continue

            if dry_run:
                print(
                    f"  {sample_id}: would write {len(messages)} messages from {session_path.name} [{driver}]"
                )
            else:
                _write_trajectory(trajectory_path, messages)
                print(
                    f"  {sample_id}: wrote {len(messages)} messages from {session_path.name} [{driver}]"
                )

            total_reparsed += 1

    print(
        f"\nDone: {total_reparsed} reparsed, {total_skipped} skipped, {total_missing} session files missing"
    )


# ── CLI entry point ───────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="rollouts doctor",
        description="Reparse external agent session files into trajectories.",
    )
    subparsers = parser.add_subparsers(dest="command")

    reparse_parser = subparsers.add_parser(
        "reparse",
        help="Re-export trajectory files from raw CLI session data.",
    )
    reparse_parser.add_argument("results_dir", help="Path to results directory")
    reparse_parser.add_argument(
        "--driver",
        choices=["codex", "claude"],
        default=None,
        help="Only reparse samples from this driver",
    )
    reparse_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing",
    )

    args = parser.parse_args(argv)

    if args.command == "reparse":
        reparse(
            args.results_dir,
            driver_filter=args.driver,
            dry_run=args.dry_run,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
