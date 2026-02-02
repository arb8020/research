"""Broker credentials management.

Reads and writes ~/.broker/credentials.toml with Modal-style profiles.

File format:

    [default]
    active = true
    runpod = "rp_..."
    vast = "vast_..."
    lambdalabs = "lambda_..."

    [team-account]
    runpod = "rp_..."

Precedence when loading credentials:
    1. Environment variables (explicit override)
    2. Active profile from ~/.broker/credentials.toml
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

CREDENTIALS_FILE = Path.home() / ".broker" / "credentials.toml"

# Maps env var names to provider keys
ENV_VAR_MAP = {
    "RUNPOD_API_KEY": "runpod",
    "VAST_API_KEY": "vast",
    "PRIME_API_KEY": "primeintellect",
    "LAMBDA_API_KEY": "lambdalabs",
}

# Known provider names for validation
KNOWN_PROVIDERS = {"runpod", "vast", "primeintellect", "lambdalabs", "digitalocean"}


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
_PROFILE_META_KEYS = {"active", "providers"}


def get_active_profile() -> tuple[str | None, dict]:
    """Get the active profile name and its credentials.

    If the profile has a `providers` list, only those providers' keys are returned.
    Otherwise all provider keys in the profile are returned.

    Returns (profile_name, credentials_dict) or (None, {}) if no active profile.
    """
    config = load_profiles()
    for name, profile in config.items():
        if not isinstance(profile, dict):
            continue
        if profile.get("active") is True:
            allowed = profile.get("providers")  # list[str] or None
            creds = {}
            for k, v in profile.items():
                if k in _PROFILE_META_KEYS or not isinstance(v, str):
                    continue
                if allowed is None or k in allowed:
                    creds[k] = v
            return name, creds
    return None, {}


def get_credentials() -> dict[str, str]:
    """Load credentials from TOML profile, falling back to env vars.

    Precedence per provider:
        1. Active profile from ~/.broker/credentials.toml (set via `broker auth login`)
        2. Environment variable (fallback if provider not in profile)

    If the active profile has a `providers` list, only those providers are returned.
    """
    config = load_profiles()
    allowed_providers = None
    profile_creds: dict[str, str] = {}

    for _name, profile in config.items():
        if not isinstance(profile, dict):
            continue
        if profile.get("active") is True:
            allowed_providers = profile.get("providers")  # list[str] or None
            for k, v in profile.items():
                if k in _PROFILE_META_KEYS or not isinstance(v, str):
                    continue
                profile_creds[k] = v
            break

    # Start with env vars as base, then TOML wins
    credentials: dict[str, str] = {}
    for env_var, provider in ENV_VAR_MAP.items():
        if key := os.getenv(env_var):
            credentials[provider] = key

    # TOML profile overrides env vars
    credentials.update(profile_creds)

    # If profile specifies allowed providers, filter everything
    if allowed_providers is not None:
        credentials = {k: v for k, v in credentials.items() if k in allowed_providers}

    return credentials


def set_profile_key(profile_name: str, provider: str, api_key: str) -> None:
    """Set a provider API key in a profile. Creates profile if needed."""
    config = load_profiles()

    if profile_name not in config:
        config[profile_name] = {}

    config[profile_name][provider] = api_key

    # If no active profile exists, make this one active
    has_active = any(isinstance(p, dict) and p.get("active") is True for p in config.values())
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
    """Mask an API key for display: rp_abc...xyz"""
    if len(key) <= 8:
        return "***"
    return f"{key[:6]}...{key[-4:]}"
