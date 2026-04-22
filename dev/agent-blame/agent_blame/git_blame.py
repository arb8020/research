"""Fallback attribution via `git blame`.

Agent-level attribution tells us "which indexed session added this line."
Git-level attribution tells us "which commit introduced this line, by
whom, when." They're complementary channels — an `unknown` line in the
agent channel can still have a git blame answer.

This module produces per-line commit info from `git blame --line-porcelain`,
cached per file. We also inspect the commit message for agent markers
(`Co-Authored-By: Claude`, `Generated with [Claude Code]`, etc.) so the
UI can distinguish "human-authored commit" from "commit landed via an
agent whose transcript we don't have."

## Why not use this as the primary channel?

`git blame` tracks commits, not agent sessions. A single commit often
bundles many edits from one or more agent sessions, plus human tweaks.
If we used `git blame` as the primary, we'd lose per-edit granularity
and we'd conflate co-authors. The agent-level fold is the authoritative
channel; git blame is the fallback.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommitInfo:
    """Per-commit metadata we care about for attribution display."""

    sha: str  # full 40-char sha
    short_sha: str  # first 8 chars
    author_name: str
    author_email: str
    author_time_iso: str  # ISO8601 UTC
    summary: str  # first line of commit message
    agent_marker: str | None
    """If the commit's full message contains a recognized agent marker,
    a short tag for it ('claude_code', 'codex', 'copilot', or 'generic').
    None means no marker — i.e. authored by the human committer."""


_CLAUDE_MARKER = re.compile(
    r"(Generated with \[Claude Code\]|Co-Authored-By:\s*Claude)",
    re.IGNORECASE,
)
_CODEX_MARKER = re.compile(
    r"(Generated with \[Codex\]|Co-Authored-By:\s*Codex)",
    re.IGNORECASE,
)
_COPILOT_MARKER = re.compile(
    r"(Co-Authored-By:\s*github-copilot|Co-Authored-By:\s*Copilot)",
    re.IGNORECASE,
)
_GENERIC_AGENT_MARKER = re.compile(
    r"Co-Authored-By:\s*(Cursor|Aider|Devin)",
    re.IGNORECASE,
)


def _detect_agent_marker(full_message: str) -> str | None:
    if _CLAUDE_MARKER.search(full_message):
        return "claude_code"
    if _CODEX_MARKER.search(full_message):
        return "codex"
    if _COPILOT_MARKER.search(full_message):
        return "copilot"
    if _GENERIC_AGENT_MARKER.search(full_message):
        return "generic"
    return None


def blame_file(
    git_root: Path,
    rel_path: str,
    ref: str | None = None,
) -> list[CommitInfo | None]:
    """Return one CommitInfo per line of the file at `ref` (or HEAD if None).

    The returned list is 0-indexed; caller looks up by (line_number - 1).
    Entries are None when git blame can't attribute (empty file, shouldn't
    happen for normal files).

    Implementation: one `git blame --line-porcelain` call per file gets
    us everything in a single subprocess. We parse the porcelain output
    into (sha, author_name, author_mail, author_time, summary) for each
    line. Commit details are deduped — the porcelain format only emits
    the full metadata the first time a commit appears in the output.
    """
    cmd = ["git", "-C", str(git_root), "blame", "--line-porcelain"]
    if ref is not None:
        cmd.append(ref)
    cmd.extend(["--", rel_path])

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        logger.debug("git blame failed for %s: %s", rel_path, e.stderr.strip())
        return []

    # Also pull full commit messages for the unique SHAs we see, so we
    # can detect agent markers. `git log --format='%H\n%B\n\x00'` over
    # the SHAs gives us {sha: full_message}.
    lines_raw = result.stdout.splitlines()

    per_line: list[dict] = []  # one dict per line of the file
    seen_commits: dict[str, dict] = {}  # sha -> header dict
    cur: dict | None = None

    # Porcelain structure per line:
    #   <sha> <orig-line> <final-line> <num-lines-in-group>
    #   author <name>
    #   author-mail <email>
    #   author-time <unix>
    #   author-tz <offset>
    #   committer ...
    #   summary <first-line>
    #   <tab>actual line content
    # Second and subsequent lines of a multi-line group may omit the
    # author/... headers (porcelain convention) — we reuse the last
    # commit dict keyed by sha.
    i = 0
    while i < len(lines_raw):
        line = lines_raw[i]
        if not line:
            i += 1
            continue
        # Commit header line starts with a 40-char hex sha followed by space.
        parts = line.split(" ", 3)
        if len(parts) >= 3 and len(parts[0]) == 40 and all(c in "0123456789abcdef" for c in parts[0]):
            sha = parts[0]
            header = seen_commits.get(sha)
            if header is None:
                header = {"sha": sha}
                seen_commits[sha] = header
            cur = header
            i += 1
            # Read subsequent header lines until we hit the \t<content> line.
            while i < len(lines_raw):
                hl = lines_raw[i]
                if hl.startswith("\t"):
                    per_line.append({"sha": sha, **seen_commits[sha]})
                    i += 1
                    break
                if " " in hl:
                    key, _, val = hl.partition(" ")
                    seen_commits[sha][key] = val
                i += 1
        else:
            i += 1

    # Pull full message / agent markers for unique SHAs.
    unique_shas = list(seen_commits.keys())
    if unique_shas:
        try:
            log_result = subprocess.run(
                ["git", "-C", str(git_root), "show", "-s", "--format=%H%x00%B%x01"]
                + unique_shas,
                capture_output=True, text=True, check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.debug("git show failed: %s", e.stderr.strip())
            log_result = None
        messages: dict[str, str] = {}
        if log_result is not None:
            for rec in log_result.stdout.split("\x01"):
                rec = rec.strip("\n")
                if not rec:
                    continue
                sha_hdr, _, body = rec.partition("\x00")
                if len(sha_hdr) == 40:
                    messages[sha_hdr] = body
        for sha, header in seen_commits.items():
            msg = messages.get(sha, "")
            header["_agent_marker"] = _detect_agent_marker(msg)

    # Build CommitInfo per line.
    out: list[CommitInfo | None] = []
    for entry in per_line:
        sha = entry["sha"]
        hdr = seen_commits[sha]
        author_name = hdr.get("author", "")
        mail = hdr.get("author-mail", "").strip("<>")
        ts_unix = hdr.get("author-time", "0")
        try:
            from datetime import datetime, timezone
            ts_iso = datetime.fromtimestamp(int(ts_unix), tz=timezone.utc).isoformat()
        except (ValueError, OSError):
            ts_iso = ""
        out.append(CommitInfo(
            sha=sha,
            short_sha=sha[:8],
            author_name=author_name,
            author_email=mail,
            author_time_iso=ts_iso,
            summary=hdr.get("summary", ""),
            agent_marker=hdr.get("_agent_marker"),
        ))
    return out
