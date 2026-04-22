"""Source readers: how reconcile/fold fetch file contents.

A `SourceReader` is any callable `(abs_path: str) -> str | None`. Returning
None means "file absent / unreadable / binary" — caller handles it.

Readers in this module:

    working_tree_reader()          — reads files from disk as-is
    git_sha_reader(sha, git_root)  — reads files at a specific commit SHA
    git_timestamp_seeder(git_root) — reads each file as of the commit in
                                     effect at a given timestamp. Used by
                                     fold when replaying across drift:
                                     each stale edit gets a fresh seed
                                     from the file-as-of-that-moment.

## Why a callable, not a class

The contract is one function. Classes should earn themselves; these don't.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

SourceReader = Callable[[str], "str | None"]

# A "timestamped seeder" reads a file as of a specific moment. Used by
# fold to recover from stale edits: if our simulated state drifted, reset
# to what git says the file looked like when the edit fired.
TimestampedSeeder = Callable[[str, datetime], "str | None"]


def working_tree_reader() -> SourceReader:
    """Reads files from disk. Best choice for 'blame the code as it sits today'."""
    def read(abs_path: str) -> str | None:
        try:
            return Path(abs_path).read_text()
        except (OSError, UnicodeDecodeError):
            return None
    return read


def git_sha_reader(sha: str, git_root: Path) -> SourceReader:
    """Reads files at `sha` via a long-running `git cat-file --batch` process.

    `git show <sha>:<path>` per file spawns a new git process each call,
    which dominates runtime when reconciling ~1000+ files. `cat-file
    --batch` keeps one process alive and streams `<sha>:<path>` queries.

    `abs_path` is converted to a repo-relative path (git cannot address
    things outside the repo). Returns None for paths that do not exist at
    this sha, that resolve outside git_root, or that are not blobs (trees
    etc. shouldn't happen for tracked file paths but we guard anyway).
    """
    # Validate the SHA up front.
    try:
        subprocess.run(
            ["git", "-C", str(git_root), "rev-parse", "--verify", f"{sha}^{{commit}}"],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(
            f"git sha {sha!r} does not resolve to a commit in {git_root}: "
            f"{e.stderr.strip()}"
        ) from e

    git_root_resolved = git_root.resolve()

    proc = subprocess.Popen(
        ["git", "-C", str(git_root_resolved), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        bufsize=0,  # unbuffered: we manage boundaries ourselves
    )
    assert proc.stdin is not None and proc.stdout is not None

    def read(abs_path: str) -> str | None:
        try:
            p = Path(abs_path).resolve()
            rel = p.relative_to(git_root_resolved)
        except (ValueError, OSError):
            return None
        query = f"{sha}:{rel.as_posix()}\n".encode()
        proc.stdin.write(query)
        proc.stdin.flush()
        header = proc.stdout.readline()
        if not header:
            return None
        header_str = header.decode().rstrip("\n")
        # Missing objects: "<query> missing"
        if header_str.endswith(" missing"):
            return None
        # Valid: "<sha> <type> <size>"
        parts = header_str.split()
        if len(parts) != 3 or parts[1] != "blob":
            # e.g. tree/commit — skip. Drain nothing; missing had no body.
            return None
        size = int(parts[2])
        # Read exactly `size` bytes of content, then the trailing newline.
        body = b""
        remaining = size
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        # Trailing LF after body
        proc.stdout.read(1)
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return None
    return read


def git_timestamp_seeder(git_root: Path) -> TimestampedSeeder:
    """Return a `(abs_path, timestamp) -> str | None` reader backed by git history.

    Given a timestamp, find the last commit touching the file at or before
    that moment, and return the file contents at that commit. If no such
    commit exists (file predates the repo's earliest commit touching it,
    or is untracked), return None.

    Implementation notes:
    - We use `git log --before=<ts> -n1 --format=%H -- <path>` to find
      the most recent relevant commit. This is the "pre-edit" snapshot
      from the agent's perspective — what the file looked like when the
      agent opened it.
    - Commit lookup is cached per (path, ts) to avoid repeated log calls
      during fold replay. Many edits in a session share timestamps and
      paths.
    - Blob reads reuse a single `git cat-file --batch` process, like
      `git_sha_reader` — spawning git per read dominates runtime.

    Returns None if:
    - path resolves outside git_root
    - no commit exists in history touching this file before ts
    - the resulting blob is not text

    Timestamps are treated as UTC wall-clock by git (--before expects an
    ISO string). The caller is responsible for passing timezone-aware
    datetimes; FileEdit.timestamp is always UTC-aware by FileEdit's
    post_init invariant.
    """
    git_root_resolved = git_root.resolve()

    # Long-running cat-file process for blob reads.
    proc = subprocess.Popen(
        ["git", "-C", str(git_root_resolved), "cat-file", "--batch"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        bufsize=0,
    )
    assert proc.stdin is not None and proc.stdout is not None

    # (rel_path, iso_ts) -> commit_sha | None. None means "no commit found."
    commit_cache: dict[tuple[str, str], str | None] = {}

    def _find_commit(rel: str, ts_iso: str) -> str | None:
        key = (rel, ts_iso)
        if key in commit_cache:
            return commit_cache[key]
        # git log --before excludes commits at exactly the given moment;
        # we want the last commit at-or-before. --until is the inclusive
        # variant.
        try:
            result = subprocess.run(
                [
                    "git", "-C", str(git_root_resolved),
                    "log", "--until", ts_iso, "-n", "1",
                    "--format=%H", "--", rel,
                ],
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.debug("git log failed for %s at %s: %s", rel, ts_iso, e.stderr)
            commit_cache[key] = None
            return None
        sha = result.stdout.strip()
        commit_cache[key] = sha or None
        return sha or None

    def _read_blob(sha: str, rel: str) -> str | None:
        query = f"{sha}:{rel}\n".encode()
        proc.stdin.write(query)
        proc.stdin.flush()
        header = proc.stdout.readline()
        if not header:
            return None
        header_str = header.decode().rstrip("\n")
        if header_str.endswith(" missing"):
            return None
        parts = header_str.split()
        if len(parts) != 3 or parts[1] != "blob":
            return None
        size = int(parts[2])
        body = b""
        remaining = size
        while remaining > 0:
            chunk = proc.stdout.read(remaining)
            if not chunk:
                break
            body += chunk
            remaining -= len(chunk)
        proc.stdout.read(1)  # trailing newline
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def seed(abs_path: str, ts: datetime) -> str | None:
        try:
            p = Path(abs_path).resolve()
            rel = p.relative_to(git_root_resolved).as_posix()
        except (ValueError, OSError):
            return None
        # Normalize ts to an ISO string git understands. UTC assumed.
        ts_iso = ts.isoformat()
        sha = _find_commit(rel, ts_iso)
        if sha is None:
            return None
        return _read_blob(sha, rel)

    return seed
