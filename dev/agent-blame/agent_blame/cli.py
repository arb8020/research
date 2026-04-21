"""CLI: `agent-blame <repo>` -> coverage stats + sample attribution.

This is the "oh cool it worked" surface. Given a repo, it:

    1. Finds Claude Code sessions whose cwd is inside the repo (later: Codex).
    2. Parses them into FileEdits.
    3. Folds into per-file virtual state with line-level attribution.
    4. Reconciles against the current repo via content match.
    5. Prints coverage stats, top sessions, and a sample file breakdown.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .adapters import claude_code, codex
from .fold import fold_edits
from .reconcile import (
    FileAttribution,
    reconcile_repo,
    summary_stats,
    top_sessions_by_lines,
)


def _tracked_text_files(git_root: Path, scope: Path) -> list[Path]:
    """Return git-tracked files under `git_root/scope`.

    `scope` is relative to `git_root`; pass `Path(".")` for the whole repo.
    Binary files are filtered later by read-text failure.
    """
    cmd = ["git", "-C", str(git_root), "ls-files"]
    if str(scope) != ".":
        cmd.append(str(scope))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [git_root / line for line in result.stdout.splitlines() if line]


def _format_session(source: str, sid: str) -> str:
    return f"{source:<11} {sid[:8]}"


def _sample_file_breakdown(fa: FileAttribution, max_runs: int = 12) -> list[str]:
    """Condense consecutive-same-attribution lines into runs for display."""
    if not fa.lines:
        return []
    runs: list[tuple[int, int, str]] = []  # (start, end, label)
    cur_start = fa.lines[0].line_number
    cur_end = cur_start
    cur_label = _run_label(fa.lines[0])
    for line in fa.lines[1:]:
        label = _run_label(line)
        if label == cur_label:
            cur_end = line.line_number
        else:
            runs.append((cur_start, cur_end, cur_label))
            cur_start = cur_end = line.line_number
            cur_label = label
    runs.append((cur_start, cur_end, cur_label))
    out = []
    for start, end, label in runs[:max_runs]:
        span = f"{start:>4}-{end:>4}" if start != end else f"{start:>4}     "
        out.append(f"  lines {span}  {label}")
    if len(runs) > max_runs:
        out.append(f"  ... ({len(runs) - max_runs} more runs)")
    return out


def _run_label(line) -> str:
    if line.edit is None:
        return "unknown"
    label = _format_session(line.edit.source, line.edit.session_id)
    if line.ambiguous:
        label += "  (ambiguous)"
    return label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="git blame for coding agents")
    parser.add_argument("repo", type=Path, help="repo root (must be a git repo)")
    parser.add_argument("--sample-file", type=str, default=None,
                        help="show per-line run breakdown for this path (relative to repo)")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    repo_root = args.repo.resolve()
    # Resolve to the actual repo top-level so `git ls-files` and relative
    # paths behave. Accepts subdirectories, submodules, worktrees.
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"error: {repo_root} is not inside a git repo", file=sys.stderr)
        return 2
    git_root = Path(toplevel)
    if git_root != repo_root:
        print(f"note: using git toplevel {git_root} (argument was {repo_root})")
    # But we want session filtering scoped to the argument the user gave,
    # and ls-files scoped to repo_root (not git_root, to avoid pulling in
    # unrelated parts of a monorepo). Use `git ls-files` with a pathspec.
    scope_rel = repo_root.relative_to(git_root) if repo_root != git_root else Path(".")

    # 1. Find candidate sessions across all supported agents.
    cc_sessions = claude_code.list_sessions_for_cwd(repo_root)
    cx_sessions = codex.list_sessions_for_cwd(repo_root)
    print(f"Claude Code sessions with cwd in {repo_root}: {len(cc_sessions)}")
    print(f"Codex       sessions with cwd in {repo_root}: {len(cx_sessions)}")
    if not cc_sessions and not cx_sessions:
        print("No sessions found. Nothing to attribute.", file=sys.stderr)
        return 0

    # 2. Parse -> FileEdits from each adapter.
    all_edits = []
    for session_path in cc_sessions:
        all_edits.extend(claude_code.iter_file_edits(session_path))
    cc_edit_count = len(all_edits)
    for session_path in cx_sessions:
        all_edits.extend(codex.iter_file_edits(session_path))
    cx_edit_count = len(all_edits) - cc_edit_count
    print(f"Parsed {cc_edit_count} Claude Code edits + {cx_edit_count} Codex edits "
          f"= {len(all_edits)} total")

    if not all_edits:
        return 0

    # 3. Fold, seeded from current repo contents for files we didn't observe
    # a Write for. Without seeding, all edits against pre-existing files
    # (most of them) show up as stale.
    def seed_reader(path: str) -> str | None:
        p = Path(path)
        try:
            return p.read_text()
        except (OSError, UnicodeDecodeError):
            return None
    virtual_states = fold_edits(all_edits, seed_reader=seed_reader)
    stale_total = sum(len(s.stale_edits) for s in virtual_states.values())
    print(f"Touched {len(virtual_states)} distinct file paths "
          f"({stale_total} stale edits — old_content missing from virtual state)")

    # 4. Reconcile against tracked files in scope.
    tracked = _tracked_text_files(git_root, scope_rel)
    print(f"Reconciling against {len(tracked)} git-tracked files under {scope_rel}...")
    attributions = reconcile_repo(git_root, virtual_states, tracked)

    # 5. Print stats.
    stats = summary_stats(attributions)
    total = stats["total_lines"] or 1
    pct_attr = 100.0 * stats["attributed"] / total
    pct_unk = 100.0 * stats["unknown"] / total
    print()
    print("Coverage:")
    print(f"  attributed   {stats['attributed']:>8}  ({pct_attr:5.1f}%)")
    print(f"  unknown      {stats['unknown']:>8}  ({pct_unk:5.1f}%)")

    print()
    print("Top sessions by lines:")
    for source, sid, count in top_sessions_by_lines(attributions, n=10):
        print(f"  {_format_session(source, sid)}  {count:>6} lines")

    # Per-session per-file breakdown for context
    print()
    print("Per-session top files:")
    per_session_file: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for fa in attributions:
        for line in fa.lines:
            if line.edit is not None:
                per_session_file[(line.edit.source, line.edit.session_id)][str(fa.repo_path)] += 1
    top = top_sessions_by_lines(attributions, n=5)
    for source, sid, _ in top:
        files = per_session_file[(source, sid)].most_common(3)
        print(f"  {_format_session(source, sid)}")
        for f, n in files:
            print(f"      {n:>5}  {f}")

    if args.sample_file:
        rel = Path(args.sample_file)
        match = next((fa for fa in attributions if fa.repo_path == rel), None)
        print()
        if match is None:
            print(f"(no attribution for {rel} — not tracked or unreadable)")
        else:
            print(f"Sample: {rel} ({len(match.lines)} lines, "
                  f"{match.attributed_count} attributed)")
            for line in _sample_file_breakdown(match):
                print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
