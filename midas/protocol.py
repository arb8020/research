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

from typing import Protocol, runtime_checkable
from dataclasses import dataclass


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


@runtime_checkable
class EnvBackend(Protocol):
    """Protocol for environment setup backends.

    This protocol defines the contract that all backends must implement.
    Each backend handles getting from "SSH connection" to "can run Python code".

    Ray-Ready design principle: Use Protocols, Not Concrete Classes.
    """

    def bootstrap(
        self,
        bifrost: "BifrostClient",
        workspace: str,
        extra: str,
    ) -> None:
        """Bootstrap environment from unknown state to working Python.

        After this returns, Python code should be runnable via run_in_env().

        Tiger Style: Assert preconditions, fail fast if something is wrong.

        Args:
            bifrost: Connected bifrost client for SSH operations
            workspace: Remote workspace path (absolute)
            extra: Python extra group to install (e.g., "dev-speedrun")

        Raises:
            AssertionError: If preconditions fail or bootstrap fails
            RuntimeError: If environment setup fails

        Example:
            backend = UvBackend()
            backend.bootstrap(bifrost, "/root/workspace", "dev-training")
        """
        ...

    def run_in_env(
        self,
        bifrost: "BifrostClient",
        workspace: str,
        command: str,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run command inside the bootstrapped environment.

        The command runs with all environment setup (PATH, venv, etc.) applied.

        Casey: Immediate mode - just execute, no retained state.

        Args:
            bifrost: Connected bifrost client
            workspace: Remote workspace path (absolute)
            command: Command to execute (e.g., "python train.py")
            env_vars: Environment variables to export (e.g., {"HF_TOKEN": "...", "CUDA_VISIBLE_DEVICES": "0,1"})

        Returns:
            CommandResult with exit_code, stdout, stderr

        Example:
            result = backend.run_in_env(
                bifrost, "/root/workspace", "python --version",
                env_vars={"CUDA_VISIBLE_DEVICES": "0"}
            )
            assert result.success
            print(result.stdout)  # "Python 3.11.5"
        """
        ...

    def verify_env(
        self,
        bifrost: "BifrostClient",
        workspace: str,
    ) -> bool:
        """Verify that environment is working.

        Optional method for health checking. Backends can override this
        to provide custom verification logic.

        Args:
            bifrost: Connected bifrost client
            workspace: Remote workspace path

        Returns:
            True if environment is working, False otherwise

        Example:
            if not backend.verify_env(bifrost, workspace):
                print("Environment broken, re-bootstrapping...")
                backend.bootstrap(bifrost, workspace, extra)
        """
        ...
