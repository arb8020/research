"""Environment backend protocol for remote execution.

This module defines the protocol that all environment backends must implement.
Backends handle the complexity of "SSH connection -> working Python environment".

Casey Muratori principles:
- Protocol-based (decoupled, swappable implementations)
- Explicit operations (no hidden magic)
- Granular control (separate bootstrap from execution)

Available backends:
- UvBackend: UV-based Python environment (current, production-ready)
- NixBackend: Nix-based reproducible environment (future)
- DockerBackend: Docker-based environment (future)
"""

from typing import Protocol, runtime_checkable, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from bifrost import BifrostClient


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
    The backend generates pyproject.toml from this specification.

    Tiger Style: Explicit is better than implicit.
    Casey: No hidden magic - dependencies are data, not configuration files.

    Example:
        deps = DependencyConfig(
            project_name="my-training",
            python_version=">=3.10",
            dependencies=[
                "torch>=2.0",
                "transformers>=4.30",
                "wandb",
            ],
            optional_dependencies={
                "dev": ["pytest", "black"],
                "training": ["accelerate", "datasets"],
            }
        )
    """
    project_name: str  # Name for generated pyproject.toml
    dependencies: list[str] = field(default_factory=list)  # Core dependencies
    optional_dependencies: dict[str, list[str]] = field(default_factory=dict)  # Extras
    python_version: str = ">=3.10"  # Python version requirement


@runtime_checkable
class EnvBackend(Protocol):
    """Protocol for environment setup backends.

    This protocol defines the contract that all backends must implement.
    Each backend handles getting from "SSH connection" to "can run Python code".

    Ray-Ready design principle: Use Protocols, Not Concrete Classes.
    """

    def bootstrap(
        self,
        client: "BifrostClient",
        workspace: str,
        dependencies: DependencyConfig,
        extra: str | None = None,
    ) -> None:
        """Bootstrap environment from unknown state to working Python.

        After this returns, Python code should be runnable via run_in_env().

        Tiger Style: Assert preconditions, fail fast if something is wrong.

        Args:
            client: BifrostClient instance for SSH operations
            workspace: Remote workspace path (absolute)
            dependencies: DependencyConfig specifying project dependencies
            extra: Optional extra group to install from optional_dependencies

        Raises:
            AssertionError: If preconditions fail or bootstrap fails
            RuntimeError: If environment setup fails

        Example:
            from bifrost import BifrostClient
            from midas import UvBackend, DependencyConfig

            client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")
            deps = DependencyConfig(
                project_name="training",
                dependencies=["torch>=2.0", "transformers"],
                optional_dependencies={"dev": ["pytest"]},
            )
            backend = UvBackend()
            backend.bootstrap(client, "/root/workspace", deps, extra="dev")
        """
        ...

    def run_in_env(
        self,
        client: "BifrostClient",
        workspace: str,
        command: str,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run command inside the bootstrapped environment.

        The command runs with all environment setup (PATH, venv, etc.) applied.

        Casey: Immediate mode - just execute, no retained state.

        Args:
            client: BifrostClient instance for SSH operations
            workspace: Remote workspace path (absolute)
            command: Command to execute (e.g., "python train.py")
            env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

        Returns:
            CommandResult with exit_code, stdout, stderr

        Example:
            result = backend.run_in_env(
                client, "/root/workspace", "python --version",
                env_vars={"CUDA_VISIBLE_DEVICES": "0"}
            )
            assert result.success
            print(result.stdout)  # "Python 3.11.5"
        """
        ...

    def verify_env(
        self,
        client: "BifrostClient",
        workspace: str,
    ) -> bool:
        """Verify that environment is working.

        Optional method for health checking. Backends can override this
        to provide custom verification logic.

        Args:
            client: BifrostClient instance for SSH operations
            workspace: Remote workspace path

        Returns:
            True if environment is working, False otherwise

        Example:
            if not backend.verify_env(client, workspace):
                print("Environment broken, re-bootstrapping...")
                backend.bootstrap(client, workspace, dependencies, extra)
        """
        ...
