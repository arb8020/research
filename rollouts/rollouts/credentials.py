"""Rollouts credentials management.

Reads and writes ~/.rollouts/credentials.toml with provider API keys.

File format:

    [default]
    active = true
    anthropic = "sk-ant-..."
    cerebras = "csk-..."
    openai = "sk-..."

    [work]
    anthropic = "sk-ant-..."

Precedence when loading credentials:
    1. Environment variables (explicit override)
    2. Active profile from ~/.rollouts/credentials.toml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path.home() / ".rollouts" / "credentials.toml"

# Maps provider names to their env var names
PROVIDER_ENV_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "cerebras": "CEREBRAS_API_KEY",
    "groq": "GROQ_API_KEY",
    "xai": "XAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

# Known provider names for validation
KNOWN_PROVIDERS = set(PROVIDER_ENV_MAP.keys())


def _load_toml(path: Path) -> dict:
    """Load a TOML file, returning {} on any failure."""
    if not path.exists():
        return {}

    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[no-redef]

    try:
        return tomllib.loads(path.read_text())
    except Exception:
        logger.warning("Failed to parse %s", path)
        return {}


def _write_toml(path: Path, data: dict) -> None:
    """Write a dict as TOML. Minimal writer — no dependency on tomli_w."""
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    for section_name, section in data.items():
        if not isinstance(section, dict):
            continue
        lines.append(f"[{section_name}]")
        for key, value in section.items():
            if isinstance(value, bool):
                lines.append(f"{key} = {'true' if value else 'false'}")
            elif isinstance(value, str):
                lines.append(f'{key} = "{value}"')
            elif isinstance(value, list):
                items = ", ".join(f'"{v}"' for v in value)
                lines.append(f"{key} = [{items}]")
            else:
                lines.append(f"{key} = {value}")
        lines.append("")

    path.write_text("\n".join(lines))
    path.chmod(0o600)


def load_profiles() -> dict[str, dict]:
    """Load all profiles from credentials file."""
    return _load_toml(CREDENTIALS_FILE)


# Keys that are profile metadata, not provider credentials
_PROFILE_META_KEYS = {"active"}


def get_active_profile() -> tuple[str | None, dict]:
    """Get the active profile name and its credentials.

    Returns (profile_name, credentials_dict) or (None, {}) if no active profile.
    """
    config = load_profiles()
    for name, profile in config.items():
        if not isinstance(profile, dict):
            continue
        if profile.get("active") is True:
            creds = {}
            for k, v in profile.items():
                if k in _PROFILE_META_KEYS or not isinstance(v, str):
                    continue
                creds[k] = v
            return name, creds
    return None, {}


def get_api_key(provider: str) -> str | None:
    """Get API key for a provider.

    Precedence:
        1. Environment variable (explicit override)
        2. Active profile from ~/.rollouts/credentials.toml

    Returns None if no key found.
    """
    # Check env var first
    env_var = PROVIDER_ENV_MAP.get(provider)
    if env_var:
        env_key = os.environ.get(env_var)
        if env_key:
            return env_key

    # Check active profile
    _, profile_creds = get_active_profile()
    return profile_creds.get(provider)


def get_all_credentials() -> dict[str, str]:
    """Get all credentials, merging env vars with profile.

    Env vars take precedence over profile values.
    """
    # Start with profile credentials
    _, profile_creds = get_active_profile()
    credentials = dict(profile_creds)

    # Env vars override
    for provider, env_var in PROVIDER_ENV_MAP.items():
        if key := os.environ.get(env_var):
            credentials[provider] = key

    return credentials


def set_profile_key(profile_name: str, provider: str, api_key: str) -> None:
    """Set a provider API key in a profile. Creates profile if needed."""
    config = load_profiles()

    if profile_name not in config:
        config[profile_name] = {}

    config[profile_name][provider] = api_key

    # If no active profile exists, make this one active
    has_active = any(
        isinstance(p, dict) and p.get("active") is True for p in config.values()
    )
    if not has_active:
        config[profile_name]["active"] = True

    _write_toml(CREDENTIALS_FILE, config)


def set_active_profile(profile_name: str) -> None:
    """Switch the active profile."""
    config = load_profiles()
    if profile_name not in config:
        raise ValueError(f"Profile '{profile_name}' not found in {CREDENTIALS_FILE}")

    # Deactivate all, activate the target
    for _name, profile in config.items():
        if isinstance(profile, dict):
            profile.pop("active", None)

    config[profile_name]["active"] = True
    _write_toml(CREDENTIALS_FILE, config)


def key_preview(key: str) -> str:
    """Mask an API key for display: sk-ant...xyz"""
    if len(key) <= 8:
        return "***"
    return f"{key[:6]}...{key[-4:]}"
