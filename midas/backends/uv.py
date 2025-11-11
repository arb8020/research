"""UV-based Python environment backend.

This backend uses UV (https://github.com/astral-sh/uv) for Python environment
management. It's the current production backend, extracted and cleaned up from
the original deploy.py files.

Tiger Style principles:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow

Casey principles:
- Granular operations (separate install, sync, verify)
- No coupling (each function does one thing)
"""

import logging
from typing import TYPE_CHECKING

from midas.protocol import CommandResult, EnvBackend, DependencyConfig
from midas.pyproject_gen import generate_pyproject_toml

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


class UvBackend:
    """UV-based Python environment setup.

    This is the current production backend. It handles:
    1. Installing UV if not present
    2. Ensuring UV is in PATH
    3. Syncing Python dependencies via uv sync
    4. Running commands in the UV venv

    Example:
        backend = UvBackend()
        backend.bootstrap(bifrost, "/root/workspace", "dev-speedrun")
        result = backend.run_in_env(bifrost, "/root/workspace", "python train.py")
    """

    def bootstrap(
        self,
        client: "BifrostClient",
        workspace: str,
        dependencies: DependencyConfig,
        extra: str | None = None,
    ) -> None:
        """Bootstrap UV environment from unknown state.

        Steps:
        1. Check if UV exists, install if not
        2. Ensure UV is in PATH
        3. Generate pyproject.toml from dependencies
        4. Run uv sync to install dependencies
        5. Verify Python venv is working

        Tiger Style: < 70 lines, explicit steps.
        """
        assert client is not None, "BifrostClient instance required"
        assert workspace, "workspace path required"
        assert dependencies is not None, "DependencyConfig required"

        logger.info("🔧 Bootstrapping UV environment...")

        # Step 1: Ensure UV is installed
        result = client.exec("command -v uv")
        if result.exit_code != 0:
            logger.info("📦 UV not found, installing...")
            self._install_uv(client)
        else:
            logger.info("✅ UV already installed")

        # Step 2: Ensure UV is in PATH
        self._ensure_uv_in_path(client)

        # Step 3: Generate and upload pyproject.toml
        logger.info(f"📝 Generating pyproject.toml for {dependencies.project_name}...")
        self._generate_pyproject(client, workspace, dependencies)

        # Step 4: Sync dependencies
        extra_msg = f" with extra '{extra}'" if extra else ""
        logger.info(f"📚 Syncing dependencies{extra_msg}...")
        self._sync_dependencies(client, workspace, extra)

        # Step 5: Verify environment works
        logger.info("🔍 Verifying Python environment...")
        self._verify_python_env(client, workspace)

        logger.info("✅ UV environment ready")

    def run_in_env(
        self,
        client: "BifrostClient",
        workspace: str,
        command: str,
        env_vars: dict[str, str] | None = None,
    ) -> CommandResult:
        """Run command with UV venv activated.

        Uses the venv's Python directly instead of shell activation.
        This is more reliable and avoids shell-specific issues.

        Tiger Style: < 70 lines.
        """
        assert client is not None, "BifrostClient instance required"
        assert workspace, "workspace path required"
        assert command, "command required"

        # Build env prefix if provided
        env_prefix = ""
        if env_vars:
            exports = []
            for key, value in env_vars.items():
                assert key, "env var key cannot be empty"
                assert value is not None, f"env var {key} value cannot be None"
                # Escape single quotes in value
                escaped_value = value.replace("'", "'\\''")
                exports.append(f"{key}='{escaped_value}'")
            env_prefix = "export " + " ".join(exports) + " && "

        # Use venv's python directly (no need to activate)
        venv_python = f"{workspace}/.venv/bin/python"

        # Execute in workspace with venv python
        full_command = f"cd {workspace} && {env_prefix}{venv_python} -c 'import sys; sys.exit(0)' && {env_prefix}{venv_python} -m {command}" if command.startswith("-m ") else f"cd {workspace} && {env_prefix}{venv_python} {command}"

        result = client.exec(full_command)

        return CommandResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
        )

    def verify_env(
        self,
        client: "BifrostClient",
        workspace: str,
    ) -> bool:
        """Verify UV environment is working.

        Checks that Python venv exists and is runnable.
        """
        assert client is not None, "BifrostClient instance required"
        assert workspace, "workspace path required"

        venv_python = f"{workspace}/.venv/bin/python"
        result = client.exec(f"{venv_python} --version")

        return result.exit_code == 0

    # === Private helper functions (< 70 lines each) ===

    def _install_uv(self, client: "BifrostClient") -> None:
        """Install UV via official installer.

        Uses the standalone installer from astral.sh.
        Tiger Style: Assert success.
        """
        install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
        result = client.exec(install_cmd)

        assert result.exit_code == 0, f"UV installation failed: {result.stderr}"
        logger.info("✅ UV installed successfully")

    def _ensure_uv_in_path(self, client: "BifrostClient") -> None:
        """Ensure UV is in PATH.

        UV installs to ~/.local/bin or ~/.cargo/bin depending on the system.
        Some shells don't automatically add these to PATH.

        Casey principle: Make the state explicit, don't hide it.
        """
        # Check common UV install locations
        check_cmd = """
        if command -v uv >/dev/null 2>&1; then
            echo "FOUND"
        elif [ -f "$HOME/.local/bin/uv" ]; then
            export PATH="$HOME/.local/bin:$PATH"
            echo "ADDED_LOCAL"
        elif [ -f "$HOME/.cargo/bin/uv" ]; then
            export PATH="$HOME/.cargo/bin:$PATH"
            echo "ADDED_CARGO"
        else
            echo "NOT_FOUND"
        fi
        """

        result = client.exec(check_cmd)
        status = result.stdout.strip() if result.stdout else "NOT_FOUND"

        assert status != "NOT_FOUND", "UV not found after installation"

        if status == "FOUND":
            logger.info("✅ UV already in PATH")
        else:
            logger.info(f"✅ UV added to PATH ({status})")

    def _generate_pyproject(
        self,
        client: "BifrostClient",
        workspace: str,
        dependencies: DependencyConfig,
    ) -> None:
        """Generate pyproject.toml on remote from DependencyConfig.

        Tiger Style: Explicit generation, assert success.
        """
        # Generate pyproject.toml content
        toml_content = generate_pyproject_toml(dependencies)

        # Escape single quotes for shell
        escaped_content = toml_content.replace("'", "'\\''")

        # Write to remote
        write_cmd = f"cat > {workspace}/pyproject.toml << 'EOF'\n{toml_content}\nEOF"
        result = client.exec(write_cmd)

        assert result.exit_code == 0, f"Failed to write pyproject.toml: {result.stderr}"
        logger.info(f"✅ Generated pyproject.toml ({dependencies.project_name})")

    def _sync_dependencies(
        self,
        client: "BifrostClient",
        workspace: str,
        extra: str | None,
    ) -> None:
        """Sync Python dependencies using uv.

        Tiger Style: Explicit command, assert success.
        """
        # Build sync command
        extra_flag = f" --extra {extra}" if extra else ""
        sync_cmd = f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        cd {workspace}
        uv sync{extra_flag}
        """

        result = client.exec(sync_cmd)

        assert result.exit_code == 0, f"uv sync failed: {result.stderr}"

        if extra:
            logger.info(f"✅ Dependencies synced with extra '{extra}'")
        else:
            logger.info("✅ Dependencies synced")

    def _verify_python_env(
        self,
        client: "BifrostClient",
        workspace: str,
    ) -> None:
        """Verify Python venv is working.

        Tiger Style: Assert the postcondition.
        """
        venv_python = f"{workspace}/.venv/bin/python"
        result = client.exec(f"{venv_python} --version")

        assert result.exit_code == 0, f"Python venv verification failed: {result.stderr}"

        version = result.stdout.strip() if result.stdout else "unknown"
        logger.info(f"✅ Python venv verified: {version}")
