"""
Configuration using .env files - no hidden config directories.

Precedence:
1. CLI flags (highest priority)
2. Environment variables
3. .env file in current directory (via python-dotenv)
4. Error with helpful message
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env on import
load_dotenv()


def get_runpod_key() -> str | None:
    """Get RunPod API key from environment"""
    return os.getenv("RUNPOD_API_KEY")


def get_prime_key() -> str | None:
    """Get Prime Intellect API key from environment"""
    return os.getenv("PRIME_API_KEY")


def get_lambda_key() -> str | None:
    """Get Lambda Labs API key from environment"""
    return os.getenv("LAMBDA_API_KEY")


def get_vast_key() -> str | None:
    """Get Vast.ai API key from environment"""
    return os.getenv("VAST_API_KEY")


def get_modal_token() -> str | None:
    """Get Modal token from environment or ~/.modal.toml

    Precedence:
    1. MODAL_TOKEN_ID from environment
    2. Active workspace from ~/.modal.toml

    Returns token_id if found, None otherwise.
    """
    # Try environment variable first
    if token_id := os.getenv("MODAL_TOKEN_ID"):
        return token_id

    # Fall back to ~/.modal.toml
    modal_config = Path.home() / ".modal.toml"
    if modal_config.exists():
        try:
            # Parse TOML manually (avoid dependency)
            content = modal_config.read_text()
            current_section = None
            sections = {}

            for line in content.split("\n"):
                line = line.strip()
                # Section header
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1]
                    sections[current_section] = {}
                # Key-value pair
                elif "=" in line and current_section:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    sections[current_section][key] = value

            # Find active workspace
            for workspace, config in sections.items():
                if config.get("active") == "true":
                    return config.get("token_id")
        except Exception:
            pass  # If parsing fails, return None

    return None


def get_ssh_key_path() -> str | None:
    """Get SSH key path from environment"""
    return os.getenv("SSH_KEY_PATH")


def discover_ssh_keys() -> list[str]:
    """Find SSH keys in common locations"""
    common_paths = [
        Path.home() / ".ssh" / "id_ed25519",
        Path.home() / ".ssh" / "id_rsa",
        Path.home() / ".ssh" / "id_ecdsa",
    ]
    return [str(p) for p in common_paths if p.exists()]


def create_env_template(tool: str):
    """Create .env template for broker or bifrost

    Args:
        tool: "broker" or "bifrost"

    Raises:
        FileExistsError: If .env already exists
    """
    if Path(".env").exists():
        raise FileExistsError(".env already exists")

    if tool == "broker":
        template = """# GPU Broker Credentials
RUNPOD_API_KEY=
PRIME_API_KEY=
LAMBDA_API_KEY=
VAST_API_KEY=
SSH_KEY_PATH=~/.ssh/id_ed25519
"""
    else:  # bifrost
        template = """# Bifrost SSH Configuration
SSH_KEY_PATH=~/.ssh/id_ed25519
"""

    Path(".env").write_text(template)
    Path(".env").chmod(0o600)  # Secure permissions
