"""Remote path normalization helpers."""

from __future__ import annotations


def normalize_remote_workspace_root(workspace_root: str, remote_home: str) -> str:
    """Normalize `~`-prefixed remote workspace roots into absolute paths."""
    assert workspace_root, "workspace_root cannot be empty"
    assert remote_home.startswith("/"), "remote_home must be absolute"

    if workspace_root == "~":
        return remote_home
    if workspace_root.startswith("~/"):
        return f"{remote_home}/{workspace_root[2:]}"
    return workspace_root
