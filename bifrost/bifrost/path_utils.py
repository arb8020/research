"""Remote path normalization helpers."""

from __future__ import annotations

from pathlib import Path


def normalize_remote_workspace_root(workspace_root: str, remote_home: str) -> str:
    """Normalize `~`-prefixed remote workspace roots into absolute paths."""
    assert workspace_root, "workspace_root cannot be empty"
    assert remote_home.startswith("/"), "remote_home must be absolute"

    if workspace_root == "~":
        return remote_home
    if workspace_root.startswith("~/"):
        return f"{remote_home}/{workspace_root[2:]}"
    return workspace_root


def build_path_rewrite_map(
    *,
    primary_workspace_local_root: str | None,
    remote_workspace_root: str,
    extra_project_roots: tuple[tuple[str, str], ...] = (),
) -> dict[str, str]:
    """Build local->remote path rewrites for staged extra project metadata."""
    rewrite_map: dict[str, str] = {}
    if primary_workspace_local_root is not None:
        rewrite_map[str(Path(primary_workspace_local_root).expanduser().resolve())] = (
            remote_workspace_root
        )
    for local_root, remote_root in extra_project_roots:
        rewrite_map[str(Path(local_root).expanduser().resolve())] = remote_root
    return rewrite_map
