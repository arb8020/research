"""Structured path management for remote deployments.

Casey Muratori philosophy:
- Transparent data structures (frozen dataclasses)
- Explicit over implicit
- Write usage code first

Tiger Style:
- Assert all preconditions
- Frozen dataclass for immutable data
- Clear documentation
"""

from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient


@dataclass(frozen=True)
class DeploymentPaths:
    """Structured paths for remote deployment.

    All paths are absolute. This structure eliminates path confusion
    by making the path hierarchy explicit and type-safe.

    Fields:
        workspace_root: Base workspace directory (e.g., ~/.bifrost/workspaces/wafer)
        project_dir: Project directory within workspace (e.g., workspace_root/research/benchmarks/gpumode)
        venv_python: Path to venv Python binary (e.g., project_dir/.venv/bin/python)

    Example:
        # Create paths structure
        paths = create_deployment_paths(
            client,
            workspace_path="~/.bifrost/workspaces/wafer",
            project_subdir="research/benchmarks/gpumode"
        )

        # Clear, type-safe access
        kernel_path = f"{paths.project_dir}/optimized/kernel.cu"
        run_python_script(
            client=client,
            script=f"{paths.project_dir}/eval.py",
            cwd=paths.project_dir,
            pythonpath=[paths.shared_dir()],
            venv_python=paths.venv_python
        )
    """
    workspace_root: str
    project_dir: str
    venv_python: str

    def shared_dir(self, shared_subdir: str = "research/benchmarks/shared") -> str:
        """Get path to shared utilities directory.

        Args:
            shared_subdir: Relative path from workspace_root to shared dir
                          (default: research/benchmarks/shared)

        Returns:
            Absolute path to shared directory
        """
        return f"{self.workspace_root}/{shared_subdir}"

    def to_dict(self) -> dict:
        """Serialize to dict for debugging/logging.

        Returns:
            Dict with all path fields
        """
        return asdict(self)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return (
            f"DeploymentPaths(\n"
            f"  workspace_root={self.workspace_root}\n"
            f"  project_dir={self.project_dir}\n"
            f"  venv_python={self.venv_python}\n"
            f")"
        )


def create_deployment_paths(
    client: "BifrostClient",
    workspace_path: str,
    project_subdir: str = "",
    venv_path: str = ".venv",
) -> DeploymentPaths:
    """Create structured deployment paths.

    This helper eliminates manual path construction and expansion bugs.

    Args:
        client: Connected BifrostClient
        workspace_path: Workspace path (may contain ~, will be expanded)
        project_subdir: Relative path from workspace to project
                       (e.g., "research/benchmarks/gpumode")
                       Leave empty if workspace IS the project directory
        venv_path: Venv location relative to project_dir (default: .venv)

    Returns:
        DeploymentPaths with all paths expanded and absolute

    Example:
        # With project subdirectory
        paths = create_deployment_paths(
            client,
            workspace_path="~/.bifrost/workspaces/wafer",
            project_subdir="research/benchmarks/gpumode"
        )

        # Workspace IS the project
        paths = create_deployment_paths(
            client,
            workspace_path="~/.bifrost/workspaces/my-project"
        )
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert workspace_path, "workspace_path must be non-empty string"
    assert venv_path, "venv_path must be non-empty string"

    # Expand workspace path (handle ~)
    workspace_root = client.expand_path(workspace_path)

    # Construct project directory
    if project_subdir:
        project_dir = f"{workspace_root}/{project_subdir}"
    else:
        project_dir = workspace_root

    # Construct venv python path
    venv_python = f"{project_dir}/{venv_path}/bin/python"

    return DeploymentPaths(
        workspace_root=workspace_root,
        project_dir=project_dir,
        venv_python=venv_python,
    )
