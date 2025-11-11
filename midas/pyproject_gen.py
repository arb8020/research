"""Generate pyproject.toml from DependencyConfig.

This module provides the logic to generate a valid pyproject.toml file
from a DependencyConfig specification.

Tiger Style:
- Functions < 70 lines
- Explicit operations
- Assert preconditions
"""

from midas.protocol import DependencyConfig


def generate_pyproject_toml(deps: DependencyConfig) -> str:
    """Generate pyproject.toml content from DependencyConfig.

    Casey: Explicit data transformation, no hidden magic.
    Tiger Style: < 70 lines, asserts preconditions.

    Args:
        deps: DependencyConfig specifying project dependencies

    Returns:
        Complete pyproject.toml file content as a string

    Example:
        >>> deps = DependencyConfig(
        ...     project_name="my-training",
        ...     dependencies=["torch>=2.0", "transformers"],
        ...     optional_dependencies={"dev": ["pytest"]},
        ... )
        >>> toml_content = generate_pyproject_toml(deps)
        >>> assert "[project]" in toml_content
    """
    assert deps.project_name, "project_name required"
    assert deps.python_version, "python_version required"

    lines = []

    # Header
    lines.append("[project]")
    lines.append(f'name = "{deps.project_name}"')
    lines.append('version = "0.1.0"')
    lines.append(f'requires-python = "{deps.python_version}"')
    lines.append("")

    # Core dependencies
    if deps.dependencies:
        lines.append("dependencies = [")
        for dep in deps.dependencies:
            lines.append(f'    "{dep}",')
        lines.append("]")
        lines.append("")

    # Optional dependencies (extras)
    if deps.optional_dependencies:
        lines.append("[project.optional-dependencies]")
        for extra_name, extra_deps in deps.optional_dependencies.items():
            lines.append(f"{extra_name} = [")
            for dep in extra_deps:
                lines.append(f'    "{dep}",')
            lines.append("]")
        lines.append("")

    # Build system (required for uv)
    lines.append("[build-system]")
    lines.append('requires = ["setuptools>=61.0"]')
    lines.append('build-backend = "setuptools.build_meta"')

    return "\n".join(lines)
