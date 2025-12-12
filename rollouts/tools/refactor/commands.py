"""
Command parsing and application.

Parses <write>, <patch>, and <delete> commands from AI responses
and applies them to the filesystem.
"""

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class WriteCommand:
    """Create or overwrite a file."""
    file: str
    content: str


@dataclass
class PatchCommand:
    """Replace a specific block or search/replace in a file."""
    file: str
    block_id: int | None  # If using block-based patching
    search: str | None    # If using search/replace
    replace: str


@dataclass
class DeleteCommand:
    """Delete a file."""
    file: str


Command = WriteCommand | PatchCommand | DeleteCommand


def parse_commands(response: str) -> list[Command]:
    """Parse all commands from AI response."""
    commands: list[Command] = []

    # Parse <write file="path">content</write>
    write_pattern = re.compile(
        r'<write\s+file=["\']([^"\']+)["\']>(.*?)</write>',
        re.DOTALL
    )
    for match in write_pattern.finditer(response):
        file_path = match.group(1).strip()
        content = match.group(2)
        # Strip leading/trailing newline from content
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        commands.append(WriteCommand(file=file_path, content=content))

    # Parse <patch file="path">search/replace</patch>
    patch_pattern = re.compile(
        r'<patch\s+file=["\']([^"\']+)["\']>(.*?)</patch>',
        re.DOTALL
    )
    for match in patch_pattern.finditer(response):
        file_path = match.group(1).strip()
        inner = match.group(2)

        # Try to parse SEARCH/REPLACE blocks
        search_replace = re.search(
            r'<{7}\s*SEARCH\s*\n(.*?)\n={7}\n(.*?)\n>{7}',
            inner,
            re.DOTALL
        )
        if search_replace:
            search = search_replace.group(1)
            replace = search_replace.group(2)
            commands.append(PatchCommand(
                file=file_path,
                block_id=None,
                search=search,
                replace=replace
            ))

    # Parse <patch id=N>content</patch> (block-based)
    block_patch_pattern = re.compile(
        r'<patch\s+id=!?(\d+)>(.*?)</patch>',
        re.DOTALL
    )
    for match in block_patch_pattern.finditer(response):
        block_id = int(match.group(1))
        content = match.group(2)
        if content.startswith('\n'):
            content = content[1:]
        if content.endswith('\n'):
            content = content[:-1]
        commands.append(PatchCommand(
            file="",  # Will be resolved from block map
            block_id=block_id,
            search=None,
            replace=content
        ))

    # Parse <delete file="path"/>
    delete_pattern = re.compile(r'<delete\s+file=["\']([^"\']+)["\']/?>')
    for match in delete_pattern.finditer(response):
        file_path = match.group(1).strip()
        commands.append(DeleteCommand(file=file_path))

    return commands


@dataclass
class ApplyResult:
    """Result of applying a command."""
    success: bool
    message: str


def apply_commands(
    commands: list[Command],
    workspace_root: Path,
    block_map: dict[int, tuple[str, int, int]] | None = None,
) -> list[ApplyResult]:
    """
    Apply commands to filesystem.

    block_map: Maps block_id -> (file_path, start_line, end_line)
               Used for block-based patching.
    """
    results: list[ApplyResult] = []
    workspace_root = Path(workspace_root).resolve()

    for cmd in commands:
        if isinstance(cmd, WriteCommand):
            result = _apply_write(cmd, workspace_root)
        elif isinstance(cmd, PatchCommand):
            result = _apply_patch(cmd, workspace_root, block_map)
        elif isinstance(cmd, DeleteCommand):
            result = _apply_delete(cmd, workspace_root)
        else:
            result = ApplyResult(success=False, message=f"Unknown command type: {type(cmd)}")

        results.append(result)

    return results


def _apply_write(cmd: WriteCommand, workspace_root: Path) -> ApplyResult:
    """Apply a write command."""
    file_path = workspace_root / cmd.file

    # Security: ensure path is within workspace
    try:
        file_path.resolve().relative_to(workspace_root)
    except ValueError:
        return ApplyResult(
            success=False,
            message=f"Path {cmd.file} is outside workspace"
        )

    # Create parent directories
    file_path.parent.mkdir(parents=True, exist_ok=True)

    # Write file
    try:
        file_path.write_text(cmd.content)
        return ApplyResult(success=True, message=f"Wrote {cmd.file}")
    except OSError as e:
        return ApplyResult(success=False, message=f"Failed to write {cmd.file}: {e}")


def _apply_patch(
    cmd: PatchCommand,
    workspace_root: Path,
    block_map: dict[int, tuple[str, int, int]] | None,
) -> ApplyResult:
    """Apply a patch command."""

    # Block-based patching
    if cmd.block_id is not None:
        if block_map is None:
            return ApplyResult(
                success=False,
                message=f"Block {cmd.block_id} patch requires block_map"
            )
        if cmd.block_id not in block_map:
            return ApplyResult(
                success=False,
                message=f"Block {cmd.block_id} not found in block_map"
            )

        file_path_str, start_line, end_line = block_map[cmd.block_id]
        file_path = workspace_root / file_path_str

        try:
            lines = file_path.read_text().splitlines(keepends=True)
            new_lines = lines[:start_line] + [cmd.replace + '\n'] + lines[end_line:]
            file_path.write_text(''.join(new_lines))
            return ApplyResult(success=True, message=f"Patched block {cmd.block_id} in {file_path_str}")
        except OSError as e:
            return ApplyResult(success=False, message=f"Failed to patch {file_path_str}: {e}")

    # Search/replace patching
    if cmd.search is not None:
        file_path = workspace_root / cmd.file

        # Security check
        try:
            file_path.resolve().relative_to(workspace_root)
        except ValueError:
            return ApplyResult(
                success=False,
                message=f"Path {cmd.file} is outside workspace"
            )

        try:
            content = file_path.read_text()
        except OSError as e:
            return ApplyResult(success=False, message=f"Failed to read {cmd.file}: {e}")

        # Find and replace
        if cmd.search not in content:
            return ApplyResult(
                success=False,
                message=f"Search pattern not found in {cmd.file}"
            )

        new_content = content.replace(cmd.search, cmd.replace, 1)

        try:
            file_path.write_text(new_content)
            return ApplyResult(success=True, message=f"Patched {cmd.file}")
        except OSError as e:
            return ApplyResult(success=False, message=f"Failed to write {cmd.file}: {e}")

    return ApplyResult(success=False, message="Patch command has neither block_id nor search")


def _apply_delete(cmd: DeleteCommand, workspace_root: Path) -> ApplyResult:
    """Apply a delete command."""
    file_path = workspace_root / cmd.file

    # Security check
    try:
        file_path.resolve().relative_to(workspace_root)
    except ValueError:
        return ApplyResult(
            success=False,
            message=f"Path {cmd.file} is outside workspace"
        )

    try:
        file_path.unlink()
        return ApplyResult(success=True, message=f"Deleted {cmd.file}")
    except FileNotFoundError:
        return ApplyResult(success=False, message=f"File not found: {cmd.file}")
    except OSError as e:
        return ApplyResult(success=False, message=f"Failed to delete {cmd.file}: {e}")
