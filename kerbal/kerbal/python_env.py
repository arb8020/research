"""Python environment orchestration for remote script execution.

This module handles setting up Python dependencies and running scripts remotely.
It uses uv for dependency management and venv creation.

Tiger Style:
- Functions < 70 lines
- Assert preconditions
- Explicit control flow
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient
    from kerbal.protocol import DependencyConfig, CommandResult

logger = logging.getLogger(__name__)


def setup_script_deps(
    client: "BifrostClient",
    workspace: str,
    dependencies: "DependencyConfig",
    install_extras: list[str] | None = None,
) -> None:
    """Setup Python dependencies for script execution.

    Ensures uv is installed, generates pyproject.toml, and runs uv sync.
    This is execution orchestration - "prepare to run THIS script".

    Tiger Style: Assert preconditions, explicit steps, < 70 lines.

    Args:
        client: BifrostClient instance
        workspace: Remote workspace path (absolute)
        dependencies: DependencyConfig specifying deps
        install_extras: List of extra groups to install (e.g., ["training", "dev"])

    Example:
        from bifrost import BifrostClient
        from kerbal import DependencyConfig, setup_script_deps

        client = BifrostClient("root@gpu:22", ssh_key_path="~/.ssh/id_rsa")

        deps = DependencyConfig(
            project_name="training",
            dependencies=["torch>=2.0"],
            extras={"training": ["wandb"], "inference": ["vllm"]},
        )
        setup_script_deps(client, workspace, deps, install_extras=["training"])
    """
    assert client is not None, "BifrostClient instance required"
    assert workspace, "workspace path required"
    assert dependencies is not None, "DependencyConfig required"

    logger.info(f"📝 Setting up Python environment for {dependencies.project_name}...")

    # Step 0: Ensure uv is installed
    _ensure_uv(client)

    # Step 1: Generate and upload pyproject.toml
    _generate_pyproject(client, workspace, dependencies)

    # Step 2: Sync dependencies with uv
    _sync_dependencies(client, workspace, install_extras)

    # Step 3: Verify Python environment works and packages are importable
    _verify_python_env(client, workspace, dependencies)

    logger.info("✅ Python environment ready")


def run_script(
    client: "BifrostClient",
    workspace: str,
    script: str,
    env_vars: dict[str, str] | None = None,
) -> "CommandResult":
    """Run a Python script in the uv venv.

    Uses the venv's Python directly instead of shell activation.
    This is more reliable and avoids shell-specific issues.

    Tiger Style: < 70 lines, explicit execution.

    Args:
        client: BifrostClient instance
        workspace: Remote workspace path (absolute)
        script: Python script to run (e.g., "train.py --epochs 100")
        env_vars: Environment variables to export

    Returns:
        CommandResult with exit_code, stdout, stderr

    Example:
        from kerbal import run_script

        result = run_script(
            client, workspace, "train.py --epochs 100",
            env_vars={"CUDA_VISIBLE_DEVICES": "0,1"}
        )
        if result.success:
            print("Training completed!")
    """
    assert client is not None, "BifrostClient instance required"
    assert workspace, "workspace path required"
    assert script, "script required"

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

    # Use venv's python directly
    venv_python = f"{workspace}/.venv/bin/python"

    # Execute in workspace with venv python
    full_command = f"cd {workspace} && {env_prefix}{venv_python} {script}"

    result = client.exec(full_command)

    # Import here to avoid circular dependency
    from kerbal.protocol import CommandResult
    return CommandResult(
        exit_code=result.exit_code,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


# === Private helper functions (< 70 lines each) ===


def _generate_pyproject_toml(deps: "DependencyConfig") -> str:
    """Generate pyproject.toml content from DependencyConfig.

    Tiger Style: < 70 lines, asserts preconditions.

    Args:
        deps: DependencyConfig specifying project dependencies

    Returns:
        Complete pyproject.toml file content as a string
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
    if deps.extras:
        lines.append("[project.optional-dependencies]")
        for extra_name, extra_deps in deps.extras.items():
            lines.append(f"{extra_name} = [")
            for dep in extra_deps:
                lines.append(f'    "{dep}",')
            lines.append("]")
        lines.append("")

    # Build system (required for uv)
    lines.append("[build-system]")
    lines.append('requires = ["setuptools>=61.0"]')
    lines.append('build-backend = "setuptools.build_meta"')
    lines.append("")

    # Setuptools config - exclude subdirectories from package discovery
    # This prevents "Multiple top-level packages discovered" errors
    # when workspace has subdirs like configs/, results/, etc.
    lines.append("[tool.setuptools]")
    lines.append('py-modules = []')

    return "\n".join(lines)


def _generate_pyproject(
    client: "BifrostClient",
    workspace: str,
    dependencies: "DependencyConfig",
) -> None:
    """Generate pyproject.toml on remote from DependencyConfig.

    Tiger Style: Explicit generation, assert success.
    """
    # Generate pyproject.toml content
    toml_content = _generate_pyproject_toml(dependencies)

    # Write to remote
    write_cmd = f"cat > {workspace}/pyproject.toml << 'EOF'\n{toml_content}\nEOF"
    result = client.exec(write_cmd)

    assert result.exit_code == 0, f"Failed to write pyproject.toml: {result.stderr}"
    logger.info(f"✅ Generated pyproject.toml ({dependencies.project_name})")


def _sync_dependencies(
    client: "BifrostClient",
    workspace: str,
    install_extras: list[str] | None,
    refresh: bool = True,
) -> None:
    """Sync Python dependencies using uv.

    Tiger Style: Explicit command, assert success, < 70 lines.

    Args:
        refresh: If True, add --refresh to force uv to check for latest
                 git commits. Default True because deployments should
                 always get latest code (not cached versions).
    """
    # Build sync command with extras
    extra_flags = ""
    if install_extras:
        for extra in install_extras:
            assert extra, "extra name cannot be empty"
            extra_flags += f" --extra {extra}"

    # Add --refresh to get latest git commits (not cached versions)
    # This is critical for git dependencies without pinned commits
    if refresh:
        extra_flags += " --refresh"

    sync_cmd = f"""
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    cd {workspace}
    uv sync{extra_flags}
    """

    # DEBUG: Log the exact sync command being run
    logger.info(f"🔍 DEBUG: Running uv sync with flags: {extra_flags if extra_flags else '(none)'}")

    result = client.exec(sync_cmd)

    assert result.exit_code == 0, f"uv sync failed: {result.stderr}"

    if install_extras:
        extras_str = ", ".join(install_extras)
        logger.info(f"✅ Dependencies synced with extras: {extras_str}")
    else:
        logger.info("✅ Dependencies synced")


def _verify_python_env(
    client: "BifrostClient",
    workspace: str,
    dependencies: "DependencyConfig | None" = None,
) -> None:
    """Verify Python venv is working and packages are importable.

    Tiger Style: Assert the postcondition.

    NOTE: The import verification is experimental and may be removed.
    It catches installation issues early but might be too aggressive
    for complex dependency scenarios (e.g., optional dependencies,
    platform-specific packages, etc.)
    """
    venv_python = f"{workspace}/.venv/bin/python"
    result = client.exec(f"{venv_python} --version")

    assert result.exit_code == 0, f"Python venv verification failed: {result.stderr}"

    version = result.stdout.strip() if result.stdout else "unknown"
    logger.info(f"✅ Python venv verified: {version}")

    # DEBUG: Log what version of kerbal is running
    try:
        import kerbal
        kerbal_file = kerbal.__file__ if hasattr(kerbal, '__file__') else 'unknown'
        logger.info(f"🔍 DEBUG: kerbal running from: {kerbal_file}")
    except Exception as e:
        logger.info(f"🔍 DEBUG: Could not inspect kerbal: {e}")

    # EXPERIMENTAL: Verify declared packages are importable
    # TODO: Consider removing this if it causes false positives
    if dependencies and dependencies.dependencies:
        logger.info("🔍 Verifying installed packages are importable...")
        for dep in dependencies.dependencies:
            # Extract package name from dependency string
            # Examples:
            #   "rollouts @ git+..." -> "rollouts"
            #   "torch>=2.0" -> "torch"
            #   "python-dotenv>=1.0.0" -> "python-dotenv"
            package_name = _extract_package_name(dep)

            # Skip if we couldn't extract a valid name
            if not package_name:
                continue

            # Python package names use underscores, but pip uses hyphens
            # Try the hyphenated name first, then try with underscores
            import_name = package_name.replace("-", "_")

            import_cmd = f"{venv_python} -c 'import {import_name}'"
            result = client.exec(import_cmd)

            if result.exit_code != 0:
                logger.warning(f"⚠️  Package '{package_name}' installed but import '{import_name}' failed")
                logger.warning(f"   Error: {result.stderr.strip() if result.stderr else 'unknown'}")
                # Don't assert - this is a soft check for now
            else:
                logger.info(f"✅ {import_name} importable")

                # DEBUG: For rollouts, show what files are actually installed
                if import_name == "rollouts":
                    list_cmd = f"{venv_python} -c \"import {import_name}, os; print(os.listdir(os.path.dirname({import_name}.__file__)))\""
                    list_result = client.exec(list_cmd)
                    if list_result.exit_code == 0:
                        logger.info(f"🔍 DEBUG: rollouts package contains: {list_result.stdout.strip()}")

                    # Check specifically for run_eval.py
                    check_cmd = f"{venv_python} -c \"import {import_name}.run_eval; print('run_eval.py found at:', {import_name}.run_eval.__file__)\""
                    check_result = client.exec(check_cmd)
                    if check_result.exit_code == 0:
                        logger.info(f"🔍 DEBUG: ✅ {check_result.stdout.strip()}")
                    else:
                        logger.warning(f"🔍 DEBUG: ❌ rollouts.run_eval NOT importable: {check_result.stderr.strip() if check_result.stderr else 'unknown'}")


def _extract_package_name(dep_string: str) -> str:
    """Extract package name from dependency string.

    Examples:
        "rollouts @ git+https://..." -> "rollouts"
        "torch>=2.0.0" -> "torch"
        "python-dotenv>=1.0" -> "python-dotenv"

    Returns:
        Package name or empty string if can't parse
    """
    # Handle git URLs: "package @ git+..."
    if " @ " in dep_string:
        return dep_string.split(" @ ")[0].strip()

    # Handle version specifiers: "package>=1.0"
    for op in [">=", "<=", "==", "!=", ">", "<", "~="]:
        if op in dep_string:
            return dep_string.split(op)[0].strip()

    # Plain package name
    return dep_string.strip()


def _ensure_uv(client: "BifrostClient") -> None:
    """Ensure uv is installed and available.

    Checks if uv exists, installs if missing, ensures it's in PATH.
    Tiger Style: < 70 lines, explicit steps.
    """
    # Check if uv is already available
    result = client.exec("command -v uv")
    if result.exit_code == 0:
        logger.info("✅ uv already installed")
        return

    # Install uv via official installer
    logger.info("📦 Installing uv...")
    install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    result = client.exec(install_cmd)

    assert result.exit_code == 0, f"uv installation failed: {result.stderr}"
    logger.info("✅ uv installed successfully")
