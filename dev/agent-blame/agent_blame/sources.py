"""Source readers: how reconcile fetches the "current" file contents.

A `SourceReader` is any callable `(abs_path: str) -> str | None`. Returning
None means "file absent / unreadable / binary" and reconcile skips it.

Two built-in readers today:

    working_tree_reader()  — reads files from disk as-is (working tree)
    git_sha_reader(sha)    — reads files from `git show <sha>:<path>`

Both return None when the file does not exist in that source, which is the
honest failure for reconcile's purposes — no lines to attribute.

## Why a callable, not a class

The contract is one function. A class would only hold the SHA (for the
git variant) — the other reader holds nothing at all. A closure captures
the SHA cleanly without inventing a `SourceReader` abstract base class.
Classes should earn themselves; this one doesn't.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)

SourceReader = Callable[[str], "str | None"]


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
