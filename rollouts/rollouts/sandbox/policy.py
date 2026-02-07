"""Sandbox policy configuration.

Defines what the sandboxed process is allowed to do.
Modeled after Codex's SandboxPolicy.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class SandboxMode(Enum):
    """Sandbox enforcement mode.

    - ENABLED: Sandbox is required. Fail if unavailable on this platform.
    - DISABLED: No sandboxing. User accepts liability for agent actions.
    """

    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass(frozen=True)
class SandboxPolicy:
    """Policy defining sandbox restrictions.

    Attributes:
        writable_roots: Directories where writes are allowed.
        read_only_paths: Paths that should be read-only even within writable_roots
                        (e.g., .git directories).
        network_access: Whether network access is allowed.
        working_dir: The primary working directory (added to writable_roots).
    """

    working_dir: Path
    writable_roots: tuple[Path, ...] = field(default_factory=tuple)
    read_only_paths: tuple[Path, ...] = field(default_factory=tuple)
    network_access: bool = False

    def __post_init__(self) -> None:
        # Ensure working_dir is absolute
        if not self.working_dir.is_absolute():
            object.__setattr__(self, "working_dir", self.working_dir.resolve())

    @classmethod
    def workspace_write(
        cls,
        working_dir: Path,
        extra_writable: list[Path] | None = None,
        network_access: bool = False,
    ) -> "SandboxPolicy":
        """Create a policy that allows writes only to the workspace.

        This is the recommended default for agent execution:
        - Write access to working_dir and extra_writable paths
        - .git and .codex directories are protected as read-only
        - Network access disabled by default

        Args:
            working_dir: The primary working directory.
            extra_writable: Additional directories to allow writes to.
            network_access: Whether to allow network access.
        """
        working_dir = working_dir.resolve()
        writable = [working_dir]
        if extra_writable:
            writable.extend(p.resolve() for p in extra_writable)

        # Auto-protect .git and .codex directories
        read_only = []
        for root in writable:
            git_dir = root / ".git"
            codex_dir = root / ".codex"
            if git_dir.exists():
                read_only.append(git_dir)
            if codex_dir.exists():
                read_only.append(codex_dir)

        return cls(
            working_dir=working_dir,
            writable_roots=tuple(writable),
            read_only_paths=tuple(read_only),
            network_access=network_access,
        )

    @classmethod
    def read_only(cls, working_dir: Path) -> "SandboxPolicy":
        """Create a read-only policy (no writes allowed anywhere)."""
        return cls(
            working_dir=working_dir.resolve(),
            writable_roots=(),
            read_only_paths=(),
            network_access=False,
        )

    def get_all_writable_roots(self) -> list[Path]:
        """Get all writable roots including working_dir."""
        roots = list(self.writable_roots)
        if self.working_dir not in roots:
            roots.insert(0, self.working_dir)
        return roots
