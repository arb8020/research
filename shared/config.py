"""
Configuration using .env files - no hidden config directories.

Precedence:
1. CLI flags (highest priority)
2. Environment variables
3. .env file in current directory (via python-dotenv)
4. Error with helpful message
"""

from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv
import os

# Load .env on import
load_dotenv()


def get_runpod_key() -> Optional[str]:
    """Get RunPod API key from environment"""
    return os.getenv("RUNPOD_API_KEY")


def get_vast_key() -> Optional[str]:
    """Get Vast API key from environment"""
    return os.getenv("VAST_API_KEY")


def get_prime_key() -> Optional[str]:
    """Get Prime Intellect API key from environment"""
    return os.getenv("PRIME_API_KEY")


def get_ssh_key_path() -> Optional[str]:
    """Get SSH key path from environment"""
    return os.getenv("SSH_KEY_PATH")


def discover_ssh_keys() -> List[str]:
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
VAST_API_KEY=
PRIME_API_KEY=
SSH_KEY_PATH=~/.ssh/id_ed25519
"""
    else:  # bifrost
        template = """# Bifrost SSH Configuration
SSH_KEY_PATH=~/.ssh/id_ed25519
"""

    Path(".env").write_text(template)
    Path(".env").chmod(0o600)  # Secure permissions
