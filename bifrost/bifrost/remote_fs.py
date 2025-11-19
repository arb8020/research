"""Remote filesystem operations for Bifrost.

Casey Muratori philosophy:
- Write usage code first
- Continuous granularity (low-level still available)
- Explicit over implicit

Tiger Style:
- Functions < 70 lines
- Assert all preconditions
- Explicit control flow
- Tuple returns for errors
"""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient


def write_file_safe(
    client: "BifrostClient",
    file_path: str,
    content: str,
) -> str | None:
    """Write file to remote, creating parent dirs if needed.

    This eliminates the common pattern:
        exec(client, f"mkdir -p '{parent_dir}'")
        exec(client, f"cat > '{file_path}' << 'EOF'\\n{content}\\nEOF")

    Args:
        client: BifrostClient instance
        file_path: Absolute path to file (will be quoted properly)
        content: File contents to write

    Returns:
        None on success, error message on failure

    Example:
        # Safe file writing (creates parent dirs automatically)
        err = write_file_safe(
            client,
            "/path/to/new/dir/kernel.cu",
            kernel_code
        )
        if err:
            logger.error(f"Failed to write kernel: {err}")

        # Handles paths with spaces
        err = write_file_safe(
            client,
            "/path with spaces/file.txt",
            "content"
        )
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert file_path, "file_path must be non-empty string"
    assert content is not None, "content cannot be None (use empty string for empty file)"

    # Create parent directory
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        err = ensure_dir(client, parent_dir)
        if err:
            return f"Failed to create parent directory: {err}"

    # Write file with proper quoting and heredoc
    # Use heredoc to avoid escaping issues in content
    write_cmd = f"cat > '{file_path}' << 'EOF'\n{content}\nEOF"
    result = client.exec(write_cmd)

    if result.exit_code != 0:
        return f"Failed to write file {file_path}: {result.stderr}"

    return None


def ensure_dir(
    client: "BifrostClient",
    dir_path: str,
) -> str | None:
    """Ensure directory exists (mkdir -p wrapper).

    Idempotent - safe to call multiple times.
    Creates all parent directories as needed.

    Args:
        client: BifrostClient instance
        dir_path: Absolute path to directory

    Returns:
        None on success, error message on failure

    Example:
        # Create nested directories
        err = ensure_dir(client, "/path/to/new/dir")
        if err:
            logger.error(f"Failed to create directory: {err}")

        # Idempotent - safe to call multiple times
        ensure_dir(client, "/tmp/foo")
        ensure_dir(client, "/tmp/foo")  # No error, already exists
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert dir_path, "dir_path must be non-empty string"

    # Create directory with proper quoting
    result = client.exec(f"mkdir -p '{dir_path}'")

    if result.exit_code != 0:
        return f"Failed to create directory {dir_path}: {result.stderr}"

    return None


def path_exists(
    client: "BifrostClient",
    path: str,
) -> bool:
    """Check if path exists on remote (file or directory).

    Args:
        client: BifrostClient instance
        path: Absolute path to check

    Returns:
        True if path exists, False otherwise

    Example:
        if path_exists(client, "/path/to/file.txt"):
            logger.info("File exists")
        else:
            logger.info("File does not exist")
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert path, "path must be non-empty string"

    result = client.exec(f"test -e '{path}'")
    return result.exit_code == 0


def read_file(
    client: "BifrostClient",
    file_path: str,
) -> tuple[str | None, str | None]:
    """Read file contents from remote.

    Args:
        client: BifrostClient instance
        file_path: Absolute path to file

    Returns:
        (content, error): File contents or None and error message

    Example:
        content, err = read_file(client, "/path/to/config.json")
        if err:
            logger.error(f"Failed to read file: {err}")
        else:
            config = json.loads(content)
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert file_path, "file_path must be non-empty string"

    result = client.exec(f"cat '{file_path}'")

    if result.exit_code != 0:
        return None, f"Failed to read file {file_path}: {result.stderr}"

    return result.stdout, None


def remove_file(
    client: "BifrostClient",
    file_path: str,
    force: bool = False,
) -> str | None:
    """Remove file from remote.

    Args:
        client: BifrostClient instance
        file_path: Absolute path to file
        force: If True, don't error if file doesn't exist

    Returns:
        None on success, error message on failure

    Example:
        # Remove file
        err = remove_file(client, "/tmp/tempfile.txt")
        if err:
            logger.error(f"Failed to remove file: {err}")

        # Don't error if file doesn't exist
        err = remove_file(client, "/tmp/maybe.txt", force=True)
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert file_path, "file_path must be non-empty string"

    flag = "-f" if force else ""
    result = client.exec(f"rm {flag} '{file_path}'")

    if result.exit_code != 0:
        return f"Failed to remove file {file_path}: {result.stderr}"

    return None
