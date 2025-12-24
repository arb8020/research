"""Package configuration and asset discovery.

Provides paths to package assets (README, docs, presets) that work with:
- `uv tool install -e ~/research/rollouts` (editable/dev install)
- `uv tool install rollouts` (from PyPI)
- `python -m rollouts` (direct execution)

For editable installs, assets are found relative to the source tree.
For installed packages, key docs should be bundled in rollouts/_docs/.
"""

from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

# =============================================================================
# Version
# =============================================================================


def get_version() -> str:
    """Get rollouts package version."""
    try:
        return version("rollouts")
    except Exception:
        return "dev"


# =============================================================================
# Package Paths
# =============================================================================


def get_package_dir() -> Path:
    """Get the rollouts package directory.

    Returns the directory containing the rollouts Python modules.
    Works with both editable installs and installed packages.
    """
    return Path(str(files("rollouts"))).resolve()


def get_repo_root() -> Path:
    """Get the repository root (parent of package dir).

    For editable installs, this is the repo root with README.md, docs/, etc.
    For installed packages, assets should be included in the wheel.
    """
    return get_package_dir().parent


def get_readme_path() -> Path:
    """Get path to README.md."""
    return get_repo_root() / "README.md"


def get_docs_dir() -> Path:
    """Get path to docs/ directory."""
    return get_repo_root() / "docs"


def get_presets_dir() -> Path:
    """Get path to agent_presets/ directory."""
    return get_package_dir() / "agent_presets"


# =============================================================================
# Asset Loading
# =============================================================================


def get_readme_content() -> str | None:
    """Read README.md content, returns None if not found."""
    path = get_readme_path()
    if path.exists():
        return path.read_text()

    # Fallback: check for bundled docs inside package
    bundled = get_package_dir() / "_docs" / "README.md"
    if bundled.exists():
        return bundled.read_text()

    return None


def get_doc_content(name: str) -> str | None:
    """Read a doc file from docs/ directory, returns None if not found.

    Args:
        name: Filename (e.g., "SESSION_DESIGN.md") or relative path
    """
    path = get_docs_dir() / name
    if path.exists():
        return path.read_text()

    # Fallback: check for bundled docs inside package
    bundled = get_package_dir() / "_docs" / name
    if bundled.exists():
        return bundled.read_text()

    return None
