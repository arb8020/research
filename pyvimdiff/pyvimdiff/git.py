"""
Git integration - parse diffs, list changed files.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field


@dataclass
class HunkLine:
    """A single line in a diff hunk."""

    type: str  # "add", "remove", "context"
    content: str
    old_line: int | None = None
    new_line: int | None = None


@dataclass
class Hunk:
    """A diff hunk (@@...@@)."""

    header: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[HunkLine] = field(default_factory=list)


@dataclass
class FileDiff:
    """Diff for a single file."""

    path: str
    old_path: str | None = None  # For renames
    status: str = "modified"  # modified, added, deleted, renamed
    hunks: list[Hunk] = field(default_factory=list)
    binary: bool = False


def run_git(*args: str, cwd: str | None = None) -> tuple[str, int]:
    """Run a git command and return (output, returncode)."""
    result = subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout + result.stderr, result.returncode


def get_changed_files(ref: str | None = None, staged: bool = False, cwd: str | None = None) -> list[str]:
    """Get list of changed file paths."""
    if staged:
        args = ["diff", "--cached", "--name-only"]
    elif ref:
        args = ["diff", ref, "--name-only"]
    else:
        args = ["diff", "--name-only"]

    output, code = run_git(*args, cwd=cwd)
    if code != 0:
        return []

    return [f.strip() for f in output.strip().split("\n") if f.strip()]


def parse_diff_output(diff_text: str) -> list[FileDiff]:
    """Parse unified diff output into structured data."""
    files: list[FileDiff] = []
    current_file: FileDiff | None = None
    current_hunk: Hunk | None = None

    lines = diff_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # New file header
        if line.startswith("diff --git"):
            if current_file:
                files.append(current_file)

            # Extract paths from "diff --git a/path b/path"
            parts = line.split(" ")
            if len(parts) >= 4:
                path = parts[3][2:]  # Remove "b/" prefix
            else:
                path = "unknown"

            current_file = FileDiff(path=path)
            current_hunk = None
            i += 1
            continue

        # File status indicators
        if line.startswith("new file"):
            if current_file:
                current_file.status = "added"
            i += 1
            continue

        if line.startswith("deleted file"):
            if current_file:
                current_file.status = "deleted"
            i += 1
            continue

        if line.startswith("rename from"):
            if current_file:
                current_file.status = "renamed"
                current_file.old_path = line[12:]
            i += 1
            continue

        if line.startswith("Binary files"):
            if current_file:
                current_file.binary = True
            i += 1
            continue

        # Hunk header
        if line.startswith("@@"):
            if current_file is None:
                i += 1
                continue

            # Parse "@@ -old_start,old_count +new_start,new_count @@"
            try:
                header_end = line.find("@@", 2)
                header_part = line[3:header_end].strip()
                parts = header_part.split()

                old_part = parts[0]  # -X,Y
                new_part = parts[1]  # +X,Y

                if "," in old_part:
                    old_start, old_count = map(int, old_part[1:].split(","))
                else:
                    old_start = int(old_part[1:])
                    old_count = 1

                if "," in new_part:
                    new_start, new_count = map(int, new_part[1:].split(","))
                else:
                    new_start = int(new_part[1:])
                    new_count = 1

                current_hunk = Hunk(
                    header=line,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                current_file.hunks.append(current_hunk)

            except (ValueError, IndexError):
                pass

            i += 1
            continue

        # Diff content lines
        if current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.lines.append(HunkLine(type="add", content=line[1:]))
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.lines.append(HunkLine(type="remove", content=line[1:]))
            elif line.startswith(" "):
                current_hunk.lines.append(HunkLine(type="context", content=line[1:]))
            elif line == "\\ No newline at end of file":
                pass  # Skip this marker

        i += 1

    if current_file:
        files.append(current_file)

    return files


def get_diff(ref: str | None = None, staged: bool = False, path: str | None = None, cwd: str | None = None) -> list[FileDiff]:
    """Get parsed diff for given ref/staged state."""
    args = ["diff", "--no-color"]

    if staged:
        args.append("--cached")
    elif ref:
        args.append(ref)

    if path:
        args.extend(["--", path])

    output, code = run_git(*args, cwd=cwd)
    if code != 0:
        return []

    return parse_diff_output(output)


def diff_files(local_path: str, remote_path: str, name: str | None = None) -> FileDiff:
    """Generate diff between two files (for git difftool mode)."""
    result = subprocess.run(
        ["diff", "-u", local_path, remote_path],
        capture_output=True,
        text=True,
    )

    # diff returns 1 if files differ, 0 if same, 2 on error
    if result.returncode == 2:
        return FileDiff(path=name or remote_path)

    diff_text = result.stdout
    if not diff_text:
        return FileDiff(path=name or remote_path)

    # Parse the diff output
    files = parse_diff_output(diff_text)
    if files:
        file_diff = files[0]
        file_diff.path = name or remote_path
        return file_diff

    # Manual parse for non-git diff format
    file_diff = FileDiff(path=name or remote_path)
    current_hunk: Hunk | None = None

    for line in diff_text.split("\n"):
        if line.startswith("@@"):
            # Parse hunk header
            try:
                header_end = line.find("@@", 2)
                header_part = line[3:header_end].strip()
                parts = header_part.split()

                old_part = parts[0]
                new_part = parts[1]

                if "," in old_part:
                    old_start, old_count = map(int, old_part[1:].split(","))
                else:
                    old_start = int(old_part[1:])
                    old_count = 1

                if "," in new_part:
                    new_start, new_count = map(int, new_part[1:].split(","))
                else:
                    new_start = int(new_part[1:])
                    new_count = 1

                current_hunk = Hunk(
                    header=line,
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                )
                file_diff.hunks.append(current_hunk)
            except (ValueError, IndexError):
                pass
        elif current_hunk is not None:
            if line.startswith("+") and not line.startswith("+++"):
                current_hunk.lines.append(HunkLine(type="add", content=line[1:]))
            elif line.startswith("-") and not line.startswith("---"):
                current_hunk.lines.append(HunkLine(type="remove", content=line[1:]))
            elif line.startswith(" "):
                current_hunk.lines.append(HunkLine(type="context", content=line[1:]))

    return file_diff
