"""Python environment setup for remote machines.

Casey Muratori philosophy:
- Write usage code first
- Semantic compression (no premature abstraction)
- Continuous granularity (layers without holes)

Tiger Style:
- Functions < 70 lines
- Assert all preconditions and postconditions
- Explicit control flow
- Fail fast with clear errors
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PythonEnvState:
    """Immutable state of a configured Python environment.

    Returned by setup_python_env() to provide all necessary paths and env vars.
    Makes explicit what was previously implicit (no more manual path reconstruction).

    Attributes:
        venv_python: Absolute path to venv Python binary
        venv_bin: Absolute path to venv bin directory (for PATH)
        workspace: Absolute path to workspace root (for PYTHONPATH)
        env_vars: Pre-computed environment variables (PATH, PYTHONPATH, etc.)

    Example:
        state = setup_python_env(client, workspace, requirements=["torch"])

        # Use venv python
        result = client.exec(f"{state.venv_python} -m my_module")

        # Or use with env vars (includes PATH, PYTHONPATH automatically)
        result = client.exec("python -m my_module", env=state.env_vars)
    """
    venv_python: str
    venv_bin: str
    workspace: str
    env_vars: dict[str, str]


def setup_script_deps(
    client: "BifrostClient",
    workspace: str,
    dependencies: "DependencyConfig",
    install_extras: list[str] | None = None,
) -> None:
    """DEPRECATED: Use setup_python_env() instead.

    This function is kept for backward compatibility.
    It wraps the old DependencyConfig API around the new setup_python_env().
    """
    from kerbal.protocol import DependencyConfig

    # Convert DependencyConfig to new API
    assert isinstance(dependencies, DependencyConfig), "dependencies must be DependencyConfig"

    # Merge extras into requirements if requested
    requirements = dependencies.dependencies.copy()
    if install_extras and dependencies.extras:
        for extra_name in install_extras:
            if extra_name in dependencies.extras:
                requirements.extend(dependencies.extras[extra_name])

    # Call new API
    setup_python_env(
        client=client,
        workspace=workspace,
        requirements=requirements,
        python_version=dependencies.python_version,
    )


def setup_python_env(
    client: "BifrostClient",
    workspace: str,
    requirements: list[str],
    git_packages: list[str] | None = None,
    cli_tools: list[str] | None = None,
    verify_imports: list[str] | None = None,
    python_version: str = ">=3.10",
    venv_path: str = ".venv",
    timeout_sec: int = 600,
) -> PythonEnvState:
    """Setup Python environment with dependencies.

    This is the main entry point. It replaces setup_script_deps() and DependencyConfig.

    Args:
        client: Connected BifrostClient
        workspace: Absolute path to workspace on remote
        requirements: pip packages like ["torch>=2.0", "triton"]
        git_packages: Git URLs for packages pip can't handle (installed with --no-deps)
        cli_tools: CLI tools that must be in PATH (e.g., ["ncu", "prime"])
        verify_imports: Import names to verify (e.g., ["torch", "BackendBench"])
        python_version: Python version requirement (default ">=3.10")
        venv_path: Venv location relative to workspace (default ".venv")
        timeout_sec: Max seconds for entire setup (1-3600)

    Example:
        # Simple case (kernels-gpumode)
        setup_python_env(
            client,
            workspace,
            requirements=["torch>=2.4.0", "triton", "nvidia-cutlass-dsl"],
        )

        # With git packages (integration-evaluation)
        setup_python_env(
            client,
            workspace,
            requirements=["torch>=2.0"],
            git_packages=["git+https://github.com/user/repo.git"],
            verify_imports=["torch", "BackendBench"],
            cli_tools=["prime"],
        )
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert workspace, "workspace must be non-empty string"
    assert requirements, "requirements must be non-empty list"
    assert len(requirements) <= 100, "requirements list too large (max 100)"
    assert timeout_sec > 0 and timeout_sec <= 3600, "timeout must be 1-3600 sec"

    if git_packages:
        assert all(pkg.startswith("git+http"), "git packages must start with git+http")

    logger.info(f"📝 Setting up Python environment")
    logger.debug(f"📍 Remote workspace: {workspace}")

    # Expand workspace path (handle ~)
    workspace = _expand_path(client, workspace)

    # Assert workspace exists
    result = client.exec(f"test -d {workspace}")
    assert result.exit_code == 0, f"Workspace does not exist: {workspace}"

    venv_full_path = f"{workspace}/{venv_path}"

    # Step 0: Ensure uv is installed
    _ensure_uv(client)

    # Step 1: Create venv
    _create_venv(client, workspace, venv_full_path, python_version, timeout_sec)

    # Step 2: Install requirements
    if requirements:
        _install_pip_packages(client, venv_full_path, requirements, timeout_sec)

    # Step 3: Install git packages (if any)
    if git_packages:
        _install_git_packages(client, venv_full_path, git_packages, timeout_sec)

    # Step 4: Verify CLI tools (if any)
    if cli_tools:
        _verify_cli_tools(client, cli_tools)

    # Step 5: Verify venv works
    _verify_venv(client, venv_full_path)

    # Step 6: Verify imports (if requested)
    if verify_imports:
        _verify_imports(client, venv_full_path, verify_imports)

    logger.info("✅ Python environment ready")

    # Construct PythonEnvState with all necessary paths and env vars
    venv_python = f"{venv_full_path}/bin/python"
    venv_bin = f"{venv_full_path}/bin"

    # Build environment variables dict
    env_vars = {
        "PATH": f"{venv_bin}:$PATH",  # Include venv bin for executables like ninja
        "PYTHONUNBUFFERED": "1",       # Always flush Python output immediately
    }

    # Return immutable state
    state = PythonEnvState(
        venv_python=venv_python,
        venv_bin=venv_bin,
        workspace=workspace,  # Already expanded at line 114
        env_vars=env_vars,
    )

    return state


# === Layer 2: Lower-level functions (continuous granularity) ===


def create_venv(
    client: "BifrostClient",
    workspace: str,
    python_version: str = ">=3.10",
    venv_path: str = ".venv",
    timeout_sec: int = 600,
) -> None:
    """Create virtualenv at workspace.

    Layer 2 function - more control than setup_python_env().
    """
    assert client is not None
    assert workspace

    workspace = _expand_path(client, workspace)
    venv_full_path = f"{workspace}/{venv_path}"

    _ensure_uv(client)
    _create_venv(client, workspace, venv_full_path, python_version, timeout_sec)
    _verify_venv(client, venv_full_path)


def install_packages(
    client: "BifrostClient",
    workspace: str,
    requirements: list[str],
    venv_path: str = ".venv",
    timeout_sec: int = 600,
) -> None:
    """Install packages into existing venv.

    Layer 2 function - assumes venv already exists.
    """
    assert client is not None
    assert workspace
    assert requirements

    workspace = _expand_path(client, workspace)
    venv_full_path = f"{workspace}/{venv_path}"

    _install_pip_packages(client, venv_full_path, requirements, timeout_sec)


def ensure_packages_installed(
    client: "BifrostClient",
    venv_python: str,
    packages: list[str],
    force: bool = False,
) -> tuple[list[str], str | None]:
    """Ensure packages are installed in venv (idempotent).

    Layer 2 function - more control than setup_python_env().
    Checks which packages are missing and only installs those.

    Args:
        client: BifrostClient instance
        venv_python: Path to venv python binary
        packages: List of package names to verify/install
        force: If True, reinstall even if present

    Returns:
        (newly_installed, error): List of packages that were installed,
                                  or None and error message if failed

    Example:
        # Idempotent package installation
        installed, err = ensure_packages_installed(
            client,
            venv_python="/path/.venv/bin/python",
            packages=["torch", "triton", "numpy"]
        )
        if err:
            logger.error(f"Installation failed: {err}")
        elif installed:
            logger.info(f"Installed: {', '.join(installed)}")
        else:
            logger.info("All packages already satisfied")

        # Force reinstall
        installed, err = ensure_packages_installed(
            client, venv_python, packages, force=True
        )
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert venv_python, "venv_python must be non-empty string"
    assert packages, "packages must be non-empty list"
    assert len(packages) <= 100, "packages list too large (max 100)"

    if not force:
        # Check which packages are missing
        missing = []
        for pkg in packages:
            # Extract package name (remove version specifiers)
            # torch>=2.0 -> torch
            # git+https://... -> skip verification (too complex)
            if pkg.startswith("git+"):
                missing.append(pkg)
                continue

            pkg_name = pkg.split(">=")[0].split("<=")[0].split("==")[0].split("[")[0].strip()

            # Try importing to verify it works
            import_cmd = f"{venv_python} -c 'import {pkg_name}'"
            result = client.exec(import_cmd)

            if result.exit_code != 0:
                missing.append(pkg)

        if not missing:
            logger.info(f"✓ Already satisfied: {', '.join(packages)}")
            return [], None  # All satisfied

        packages = missing
        logger.info(f"Installing: {', '.join(missing)}")

    # Install packages using uv
    packages_str = " ".join(f'"{pkg}"' for pkg in packages)
    cmd = f"""
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    uv pip install --python {venv_python} {packages_str}
    """
    result = client.exec(cmd)

    if result.exit_code != 0:
        return [], f"Installation failed: {result.stderr}"

    logger.info(f"✅ Installed: {', '.join(packages)}")
    return packages, None


def verify_package(
    client: "BifrostClient",
    venv_python: str,
    package_name: str,
) -> tuple[bool, str | None]:
    """Verify package is importable.

    Actually tries importing to confirm it works, not just checking
    if it's listed in pip freeze. This is the most reliable way to
    verify a package is truly usable.

    Args:
        client: BifrostClient instance
        venv_python: Path to venv python binary
        package_name: Import name to verify (e.g., "torch", "triton")

    Returns:
        (success, error): True if package imports successfully,
                          or False and error message if failed

    Example:
        # Verify torch is importable
        success, err = verify_package(
            client,
            "/path/.venv/bin/python",
            "torch"
        )
        if not success:
            logger.error(f"torch not available: {err}")

        # Verify multiple packages
        for pkg in ["torch", "triton", "numpy"]:
            success, err = verify_package(client, venv_python, pkg)
            if not success:
                logger.error(f"{pkg} verification failed: {err}")
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "client cannot be None"
    assert venv_python, "venv_python must be non-empty string"
    assert package_name, "package_name must be non-empty string"

    # Try importing
    import_cmd = f"{venv_python} -c 'import {package_name}'"
    result = client.exec(import_cmd)

    if result.exit_code != 0:
        return False, f"Import failed: {result.stderr.strip()}"

    return True, None


# === Layer 1: Private helpers (lowest level) ===


def _expand_path(client: "BifrostClient", path: str) -> str:
    """Expand ~ in path to absolute path."""
    result = client.exec(f"echo {path}")
    assert result.exit_code == 0, f"Failed to expand path: {result.stderr}"
    return result.stdout.strip()


def _ensure_uv(client: "BifrostClient") -> None:
    """Ensure uv is installed and available.

    Tiger Style: < 70 lines, explicit steps.
    """
    # Check if uv is already available
    result = client.exec("command -v uv")
    if result.exit_code == 0:
        logger.debug("✅ uv already installed")
        return

    # Install uv via official installer
    logger.debug("📦 Installing uv...")
    install_cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    result = client.exec(install_cmd)

    assert result.exit_code == 0, f"uv installation failed: {result.stderr}"
    logger.debug("✅ uv installed successfully")


def _create_venv(
    client: "BifrostClient",
    workspace: str,
    venv_full_path: str,
    python_version: str,
    timeout_sec: int,
) -> None:
    """Create venv using uv.

    Tiger Style: Explicit command, assert success.
    """
    logger.debug("🔧 Creating virtual environment...")

    # Generate minimal pyproject.toml for uv
    # uv needs this to create venv with correct Python version
    pyproject_toml = f"""[project]
name = "remote-env"
version = "0.1.0"
requires-python = "{python_version}"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
"""

    # Write pyproject.toml
    write_cmd = f"cat > {workspace}/pyproject.toml << 'EOF'\n{pyproject_toml}\nEOF"
    result = client.exec(write_cmd)
    assert result.exit_code == 0, f"Failed to write pyproject.toml: {result.stderr}"

    # Create venv with uv
    cmd = f"""
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    cd {workspace}
    uv venv {venv_full_path}
    """
    result = client.exec(cmd)
    assert result.exit_code == 0, f"venv creation failed: {result.stderr}"

    logger.debug(f"✅ Virtual environment created at {venv_full_path}")


def _install_pip_packages(
    client: "BifrostClient",
    venv_full_path: str,
    requirements: list[str],
    timeout_sec: int,
) -> None:
    """Install pip packages into venv.

    Tiger Style: Explicit installation, stream output for visibility.
    """
    logger.debug(f"🔧 Installing {len(requirements)} package(s)...")
    logger.debug("=" * 60)

    # Build pip install command
    # Use 'uv pip install' instead of venv's pip (uv venv doesn't include pip)
    packages = " ".join(f'"{pkg}"' for pkg in requirements)
    cmd = f"""
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    uv pip install --python {venv_full_path}/bin/python {packages}
    EXIT=$?
    echo '::EXIT_CODE::'$EXIT
    exit $EXIT
    """

    exit_code = None

    try:
        for line in client.exec_stream(cmd):
            # Capture exit code from marker
            if line.startswith("::EXIT_CODE::"):
                exit_code_str = line.replace("::EXIT_CODE::", "").strip()
                if exit_code_str.isdigit():
                    exit_code = int(exit_code_str)
            else:
                # Print output in real-time
                print(line, end='', flush=True)
    except Exception as e:
        raise RuntimeError(f"pip install execution failed: {e}")

    logger.debug("=" * 60)

    if exit_code is None:
        raise RuntimeError("pip install failed - could not determine exit code")

    assert exit_code == 0, f"pip install failed with exit code {exit_code}"
    logger.debug("✅ Packages installed")


def _install_git_packages(
    client: "BifrostClient",
    venv_full_path: str,
    git_packages: list[str],
    timeout_sec: int,
) -> None:
    """Install git packages that pip can't handle normally.

    Strategy: pip install directly from git URL.
    Falls back to --no-deps if standard install fails.

    Tiger Style: < 70 lines, explicit strategy.
    """
    logger.info(f"🔧 Installing {len(git_packages)} git package(s)...")

    for pkg_url in git_packages:
        logger.info(f"   Installing: {pkg_url}")

        # Try standard install first
        # Use 'uv pip install' instead of venv's pip
        packages = f'"{pkg_url}"'
        cmd = f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        uv pip install --python {venv_full_path}/bin/python {packages}
        """
        result = client.exec(cmd)

        if result.exit_code == 0:
            logger.info(f"   ✅ Installed successfully")
            continue

        # If failed, try with --no-deps (works around URL dependency issues)
        logger.warning(f"   Standard install failed, trying --no-deps...")
        cmd = f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        uv pip install --python {venv_full_path}/bin/python --no-deps {packages}
        """
        result = client.exec(cmd)

        assert result.exit_code == 0, (
            f"Git package install failed: {pkg_url}\n"
            f"Error: {result.stderr}"
        )

        logger.info(f"   ✅ Installed with --no-deps")

    logger.info("✅ Git packages installed")


def _verify_cli_tools(
    client: "BifrostClient",
    cli_tools: list[str],
) -> None:
    """Verify CLI tools exist and are in PATH.

    If not found, search common locations and fail with actionable error.

    Tiger Style: Fail fast, clear errors.
    """
    search_paths = [
        "~/.local/bin",
        "~/.cargo/bin",
        "/usr/local/bin",
        "/usr/local/cuda/bin",
    ]

    for tool in cli_tools:
        # Try to find it in PATH
        result = client.exec(f"which {tool}")

        if result.exit_code == 0:
            tool_path = result.stdout.strip()
            logger.info(f"✅ Found {tool} at {tool_path}")
            continue

        # Not in PATH - search common locations
        found_at = None
        for path in search_paths:
            result = client.exec(f"test -f {path}/{tool}")
            if result.exit_code == 0:
                found_at = f"{path}/{tool}"
                break

        if found_at:
            # Found it, but not in PATH
            assert False, (
                f"Tool '{tool}' found at {found_at} but not in PATH.\n"
                f"Fix: Add {path} to PATH or use absolute path."
            )

        # Not found anywhere
        assert False, (
            f"Tool '{tool}' not found.\n"
            f"Searched: {', '.join(search_paths)}\n"
            f"Install it on the remote machine first."
        )


def _verify_venv(client: "BifrostClient", venv_full_path: str) -> None:
    """Verify venv works.

    Tiger Style: Assert postcondition.
    """
    venv_python = f"{venv_full_path}/bin/python"
    result = client.exec(f"{venv_python} --version")

    assert result.exit_code == 0, f"Python venv verification failed: {result.stderr}"

    version = result.stdout.strip() if result.stdout else "unknown"
    logger.debug(f"✅ Python venv verified: {version}")


def _verify_imports(
    client: "BifrostClient",
    venv_full_path: str,
    imports: list[str],
) -> None:
    """Verify imports work, suggest fixes if not.

    Tiger Style: Fail fast with diagnostic info.
    """
    logger.info("🔍 Verifying imports...")

    venv_python = f"{venv_full_path}/bin/python"

    for import_name in imports:
        import_cmd = f"{venv_python} -c 'import {import_name}'"
        result = client.exec(import_cmd)

        if result.exit_code == 0:
            logger.info(f"   ✅ {import_name}")
            continue

        # Import failed - provide diagnostic info
        logger.error(f"   ❌ Import verification failed: {import_name}")
        logger.error(f"      Error: {result.stderr.strip() if result.stderr else 'unknown'}")

        # Try to find the actual importable name
        # List installed packages using uv pip list
        list_cmd = f"""
        export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
        uv pip list --python {venv_full_path}/bin/python | grep -i {import_name}
        """
        list_result = client.exec(list_cmd)

        if list_result.exit_code == 0 and list_result.stdout:
            logger.error(f"      Found installed packages:")
            for line in list_result.stdout.strip().split('\n'):
                logger.error(f"        {line}")
            logger.error(f"      The package may be installed with a different name.")
            logger.error(f"      Check the actual import name and update verify_imports.")
        else:
            logger.error(f"      Package not found in pip list.")
            logger.error(f"      Make sure it's in requirements or git_packages.")

        assert False, f"Import verification failed: {import_name}"


# === Script execution (kept for backward compatibility) ===


def run_script(
    client: "BifrostClient",
    workspace: str,
    script: str,
    env_vars: dict[str, str] | None = None,
) -> "CommandResult":
    """Run a Python script in the venv.

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
    from kerbal.protocol import CommandResult

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

    return CommandResult(
        exit_code=result.exit_code,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def run_python_script(
    client: "BifrostClient",
    script: str,
    cwd: str,
    pythonpath: list[str] | None = None,
    env: dict[str, str] | None = None,
    venv_python: str | None = None,
    check: bool = True,
) -> tuple[int, str, str]:
    """Run Python script with proper environment setup.

    Handles PYTHONPATH, working directory, and env vars correctly.
    Current directory ('.') is always added to PYTHONPATH automatically.

    This eliminates the common manual pattern of building commands like:
        cd {workspace} && PYTHONPATH=.:{shared_dir} {venv_python} {script}

    Args:
        client: BifrostClient instance
        script: Absolute path to Python script
        cwd: Working directory to run from
        pythonpath: Additional PYTHONPATH entries (. is always included automatically)
        env: Additional environment variables
        venv_python: Path to venv python (uses system python3 if None)
        check: If True, raise error on non-zero exit (default: True)

    Returns:
        (exit_code, stdout, stderr)

    Raises:
        RuntimeError: If check=True and script fails

    Examples:
        # Simple script execution
        exit_code, stdout, stderr = run_python_script(
            client,
            "/path/to/eval.py",
            cwd="/path/to"
        )

        # With shared libraries and GPU
        exit_code, stdout, stderr = run_python_script(
            client,
            "/path/to/challenge_1/eval.py",
            cwd="/path/to/challenge_1",
            pythonpath=["/path/to/shared"],
            env={"CUDA_VISIBLE_DEVICES": "0"},
            venv_python="/path/to/.venv/bin/python"
        )

        # Don't raise on failure (capture error)
        exit_code, stdout, stderr = run_python_script(
            client, script, cwd, check=False
        )
        if exit_code != 0:
            logger.error(f"Script failed: {stderr}")
    """
    # Assert preconditions (Tiger Style)
    assert client is not None, "BifrostClient instance required"
    assert script, "script path required"
    assert cwd, "working directory required"

    # Use venv python or system python
    python_bin = venv_python or "python3"

    # Build PYTHONPATH: current directory always first
    pythonpath_parts = ["."]
    if pythonpath:
        pythonpath_parts.extend(pythonpath)
    pythonpath_str = ":".join(pythonpath_parts)

    # Build environment variables
    env_parts = [f"PYTHONPATH={pythonpath_str}"]
    if env:
        for key, value in env.items():
            # Escape single quotes in value
            escaped_value = value.replace("'", "'\\''")
            env_parts.append(f"{key}='{escaped_value}'")
    env_str = " ".join(env_parts)

    # Build command with proper quoting
    # cd to cwd, set env, run script
    cmd = f"cd '{cwd}' && {env_str} '{python_bin}' '{script}'"

    # Execute
    result = client.exec(cmd)

    if check and result.exit_code != 0:
        raise RuntimeError(
            f"Script failed with exit code {result.exit_code}\n"
            f"Command: {cmd}\n"
            f"Stderr: {result.stderr}"
        )

    return result.exit_code, result.stdout or "", result.stderr or ""
