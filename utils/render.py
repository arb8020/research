"""
https://blog.ezyang.com/2025/10/draw-high-dimensional-tensors-as-a-matrix-of-matrices/
"""


def render(tensor, cell_width=None):
    """
    Print a tensor following the matrix-of-matrices algorithm.

    Args:
        tensor: A tensor-like object with .shape attribute and indexing
        cell_width: Width for each cell (calculated globally if None)

    Returns:
        String representation of the tensor
    """
    # Calculate global cell width if not provided (top-level call)
    if cell_width is None:
        cell_width = calculate_cell_width(tensor)

    ndim = len(tensor.shape)

    if ndim == 0:
        # Scalar - just return the value, right-justified to cell_width
        return str(tensor.item()).rjust(cell_width)
    elif ndim == 1:
        # Vector - print as a row with spacing
        return "  ".join(str(x.item()).rjust(cell_width) for x in tensor)
    elif ndim == 2:
        # Matrix - print rows
        lines = []
        for row in tensor:
            formatted_row = "  ".join(str(x.item()).rjust(cell_width) for x in row)
            lines.append(formatted_row)
        return "\n".join(lines)
    else:
        # Higher dimensions - recursively print sub-tensors
        sub_prints = [render(sub, cell_width) for sub in tensor]

        # Determine if we stack horizontally or vertically
        # 3D, 5D, 7D... -> horizontal (odd offset from 2D)
        # 4D, 6D, 8D... -> vertical (even offset from 2D)
        dim_offset = ndim - 2
        stack_horizontally = dim_offset % 2 == 1

        if stack_horizontally:
            # Stack horizontally
            if ndim >= 5:
                # Use ':' separator for 5D+
                # Calculate number of colons: 5D -> 1 colon, 7D -> 2 colons, etc.
                num_colons = (ndim - 3) // 2
                separator = ":" * num_colons
                return join_horizontal_with_separator(sub_prints, separator)
            else:
                # 3D case - simple horizontal join
                return join_horizontal(sub_prints)
        else:
            # Stack vertically (4D, 6D, etc.)
            if ndim >= 6:
                # Use '--' separator for 6D+
                # Calculate number of separator lines: 6D -> 1 line, 8D -> 2 lines, etc.
                num_separator_lines = (ndim - 4) // 2
                return join_vertical(sub_prints, num_separator_lines)
            else:
                # 4D case - just whitespace
                return join_vertical(sub_prints, 0)


def calculate_cell_width(tensor):
    """
    Calculate the maximum width needed for any element in the tensor.

    Args:
        tensor: A tensor-like object

    Returns:
        Maximum width (number of characters) needed
    """
    # Flatten the tensor and find the maximum string length
    flat = tensor.flatten()
    max_width = max(len(str(x.item())) for x in flat)
    return max_width


def join_horizontal(blocks):
    """Join multiple text blocks horizontally with spacing."""
    if not blocks:
        return ""

    # Split each block into lines
    block_lines = [block.split("\n") for block in blocks]

    # Find the height of each block
    heights = [len(lines) for lines in block_lines]
    max_height = max(heights)

    # Find the width of each block
    widths = [max(len(line) for line in lines) if lines else 0 for lines in block_lines]

    # Pad all blocks to the same height and width
    padded_blocks = []
    for lines, width in zip(block_lines, widths, strict=False):
        padded = []
        for i in range(max_height):
            if i < len(lines):
                padded.append(lines[i].ljust(width))
            else:
                padded.append(" " * width)
        padded_blocks.append(padded)

    # Join horizontally with 4 spaces between blocks
    result_lines = []
    for i in range(max_height):
        line = "    ".join(block[i] for block in padded_blocks)
        result_lines.append(line)

    return "\n".join(result_lines)


def join_horizontal_with_separator(blocks, separator=":"):
    """Join multiple text blocks horizontally with separator."""
    if not blocks:
        return ""

    # Split each block into lines
    block_lines = [block.split("\n") for block in blocks]

    # Find the height of each block
    heights = [len(lines) for lines in block_lines]
    max_height = max(heights)

    # Find the width of each block
    widths = [max(len(line) for line in lines) if lines else 0 for lines in block_lines]

    # Pad all blocks to the same height and width
    padded_blocks = []
    for lines, width in zip(block_lines, widths, strict=False):
        padded = []
        for i in range(max_height):
            if i < len(lines):
                padded.append(lines[i].ljust(width))
            else:
                padded.append(" " * width)
        padded_blocks.append(padded)

    # Join horizontally with separator
    result_lines = []
    for i in range(max_height):
        parts = []
        for j, block in enumerate(padded_blocks):
            parts.append(block[i])
            if j < len(padded_blocks) - 1:
                # Add separator between blocks (not after the last one)
                parts.append(separator)
        line = "  ".join(parts)
        result_lines.append(line)

    return "\n".join(result_lines)


def join_vertical(blocks, num_separator_lines=0):
    """Join multiple text blocks vertically with separator lines.

    Args:
        blocks: List of text blocks to join
        num_separator_lines: Number of '--' separator lines to insert between blocks
    """
    if not blocks:
        return ""

    if num_separator_lines == 0:
        # No separator, just a blank line
        return "\n\n".join(blocks)

    # Calculate the width of the blocks for the separator
    all_lines = []
    for block in blocks:
        all_lines.extend(block.split("\n"))

    # Find the maximum line width
    max_width = max(len(line) for line in all_lines) if all_lines else 0

    # Create separator: multiple lines of '--' repeated to fill width
    separator_line = "--" * ((max_width + 1) // 2)  # Repeat '--' to fill width
    separator_lines = [separator_line] * num_separator_lines
    separator = "\n".join(separator_lines)

    # Join blocks with separator
    return ("\n" + separator + "\n").join(blocks)
