"""
Block-based file representation.

Splits files into "blocks" (chunks separated by blank lines).
Each block gets a unique ID for precise patching.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Block:
    """A block of non-empty lines."""

    id: int
    file: str
    content: str
    start_line: int  # 0-indexed
    end_line: int  # exclusive


@dataclass
class BlockState:
    """State of all blocks across files."""

    blocks: list[Block]
    block_map: dict[int, Block]  # id -> Block

    def get_block(self, block_id: int) -> Block | None:
        return self.block_map.get(block_id)

    def get_file_blocks(self, file: str) -> list[Block]:
        return [b for b in self.blocks if b.file == file]


def split_into_blocks(content: str) -> list[tuple[str, int, int]]:
    """
    Split content into blocks (non-empty line sequences separated by blank lines).

    Returns list of (block_content, start_line, end_line).
    """
    lines = content.split("\n")
    blocks = []

    current_block_lines = []
    current_start = 0

    for i, line in enumerate(lines):
        if line.strip() == "":
            # Blank line - end current block if any
            if current_block_lines:
                block_content = "\n".join(current_block_lines)
                blocks.append((block_content, current_start, i))
                current_block_lines = []
        else:
            # Non-blank line
            if not current_block_lines:
                current_start = i
            current_block_lines.append(line)

    # Don't forget last block
    if current_block_lines:
        block_content = "\n".join(current_block_lines)
        blocks.append((block_content, current_start, len(lines)))

    return blocks


def build_block_state(context: dict[str, str]) -> BlockState:
    """
    Build BlockState from context map.

    Each file is split into blocks, and each block gets a unique ID.
    """
    blocks = []
    block_id = 0

    for file_path, content in sorted(context.items()):
        file_blocks = split_into_blocks(content)

        for block_content, start_line, end_line in file_blocks:
            blocks.append(
                Block(
                    id=block_id,
                    file=file_path,
                    content=block_content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
            block_id += 1

    block_map = {b.id: b for b in blocks}

    return BlockState(blocks=blocks, block_map=block_map)


def format_blocks(block_state: BlockState, omit_ids: set[int] | None = None) -> str:
    """
    Format blocks for the prompt.

    Output format:
        ./file.py:

        !0
        first block content

        !1
        second block content

        ./other.py:

        !2
        ...
    """
    if omit_ids is None:
        omit_ids = set()

    lines = []
    current_file = None

    for block in block_state.blocks:
        if block.id in omit_ids:
            continue

        # File header
        if block.file != current_file:
            if current_file is not None:
                lines.append("")  # Blank line between files
            lines.append(f"./{block.file}:")
            lines.append("")
            current_file = block.file

        # Block with ID marker
        lines.append(f"!{block.id}")
        lines.append(block.content)
        lines.append("")

    return "\n".join(lines)


def apply_block_patch(
    block_state: BlockState,
    block_id: int,
    new_content: str,
    workspace_root: Path,
) -> tuple[bool, str]:
    """
    Apply a patch to a specific block.

    Returns (success, message).
    """
    block = block_state.get_block(block_id)
    if block is None:
        return False, f"Block {block_id} not found"

    file_path = workspace_root / block.file

    try:
        full_content = file_path.read_text()
    except OSError as e:
        return False, f"Failed to read {block.file}: {e}"

    lines = full_content.split("\n")

    # Replace the block's lines with new content
    new_lines = new_content.split("\n")
    result_lines = lines[: block.start_line] + new_lines + lines[block.end_line :]

    try:
        file_path.write_text("\n".join(result_lines))
    except OSError as e:
        return False, f"Failed to write {block.file}: {e}"

    return True, f"Patched block !{block_id} in {block.file}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars / 4)."""
    return len(text) // 4
