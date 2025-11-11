"""Data structures for remote script execution.

Tiger Style: Data is data, not behavior.
"""

from dataclasses import dataclass, field


@dataclass
class CommandResult:
    """Result of executing a command remotely.

    Transparent data structure (Casey: never use opaque types).
    """
    exit_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0


@dataclass
class DependencyConfig:
    """Dependency specification for Python environment.

    This replaces brittle pyproject.toml extras with explicit dependency management.
    Kerbal's python_env module generates pyproject.toml from this specification.

    Tiger Style: Explicit is better than implicit.

    Example:
        deps = DependencyConfig(
            project_name="my-training",
            python_version=">=3.10",
            dependencies=[
                "torch>=2.0",
                "transformers>=4.30",
                "wandb",
            ],
            extras={
                "dev": ["pytest", "black"],
                "training": ["accelerate", "datasets"],
            }
        )
    """
    project_name: str  # Name for generated pyproject.toml
    dependencies: list[str] = field(default_factory=list)  # Core dependencies
    extras: dict[str, list[str]] = field(default_factory=dict)  # Optional dependency groups
    python_version: str = ">=3.10"  # Python version requirement
