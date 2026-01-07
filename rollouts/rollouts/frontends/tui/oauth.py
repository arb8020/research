"""
OAuth module for Claude Pro/Max authentication using PKCE flow.

Based on opencode-anthropic-auth implementation.
Uses "code" flow where user copies authorization code from browser.

Token format:
- Access token: sk-ant-oat01-...  (8 hour expiry)
- Refresh token: sk-ant-ort01-...
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# OAuth configuration
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
REDIRECT_URI = "https://console.anthropic.com/oauth/code/callback"
API_KEY_URL = "https://api.anthropic.com/api/oauth/claude_cli/create_api_key"

# Scopes - match Claude Code's scope set
# Console scopes: org:create_api_key, user:profile
# Claude.ai scopes: user:profile, user:inference, user:sessions:claude_code
SCOPES = "org:create_api_key user:profile user:inference user:sessions:claude_code"

# The scope that indicates direct OAuth inference is available
INFERENCE_SCOPE = "user:inference"

# Token storage
OAUTH_DIR = Path.home() / ".rollouts" / "oauth"

# Refresh tokens 3 minutes before expiry to avoid mid-request failures
EXPIRY_BUFFER_MS = 3 * 60 * 1000


@dataclass
class OAuthTokens:
    """OAuth token pair with metadata."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp in milliseconds
    scopes: str | None = None  # Space-separated scopes granted by the token
    api_key: str | None = None  # API key created via OAuth (for console accounts)

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() * 1000 >= self.expires_at

    def has_inference_scope(self) -> bool:
        """Check if token has user:inference scope for direct OAuth inference."""
        if self.scopes is None:
            return False
        return INFERENCE_SCOPE in self.scopes.split()

    def to_dict(self) -> dict[str, Any]:
        result = {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }
        if self.scopes:
            result["scopes"] = self.scopes
        if self.api_key:
            result["api_key"] = self.api_key
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthTokens:
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            scopes=data.get("scopes"),
            api_key=data.get("api_key"),
        )


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    # Generate 32 bytes of random data for verifier
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    # Create S256 challenge
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


def validate_profile_name(profile: str) -> tuple[None, str | None]:
    """Validate profile name is safe for filesystem.

    Returns (None, error) where error is None on success.
    """
    if not profile:
        return None, "Profile name cannot be empty"

    # Allow alphanumeric, underscore, dash
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+$", profile):
        return (
            None,
            f"Profile name '{profile}' must contain only letters, numbers, underscore, and dash",
        )

    if len(profile) > 64:
        return None, f"Profile name '{profile}' too long (max 64 characters)"

    return None, None


def _get_profile_path(profile: str) -> Path:
    """Get path to profile token file."""
    return OAUTH_DIR / f"{profile}.json"


def save_tokens(tokens: OAuthTokens, profile: str) -> tuple[None, str | None]:
    """Save tokens to profile file.

    Returns (None, error) where error is None on success.
    """
    _, err = validate_profile_name(profile)
    if err:
        return None, err

    path = _get_profile_path(profile)

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(tokens.to_dict(), f, indent=2)
        os.chmod(path, 0o600)
        return None, None
    except OSError as e:
        return None, f"Failed to save tokens: {e}"


def load_tokens(profile: str) -> tuple[OAuthTokens | None, str | None]:
    """Load tokens from profile file.

    Returns (tokens, error) where both are None if file doesn't exist.
    """
    _, err = validate_profile_name(profile)
    if err:
        return None, err

    path = _get_profile_path(profile)

    if not path.exists():
        return None, None

    try:
        with open(path) as f:
            data = json.load(f)
        tokens = OAuthTokens.from_dict(data)
        return tokens, None
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        return None, f"Failed to load tokens: {e}"
    except OSError as e:
        return None, f"Failed to read token file: {e}"


def delete_tokens(profile: str) -> tuple[None, str | None]:
    """Delete profile tokens file.

    Returns (None, error) where error is None on success.
    """
    _, err = validate_profile_name(profile)
    if err:
        return None, err

    path = _get_profile_path(profile)

    try:
        if path.exists():
            path.unlink()
        return None, None
    except OSError as e:
        return None, f"Failed to delete tokens: {e}"


def list_profiles() -> list[str]:
    """List available profile names.

    Returns list of profile names (without .json extension).
    """
    if not OAUTH_DIR.exists():
        return []

    profiles = []
    for path in OAUTH_DIR.glob("*.json"):
        profiles.append(path.stem)

    return sorted(profiles)


def set_default_profile(profile: str) -> tuple[None, str | None]:
    """Set a profile as the default by copying it to default.json.

    Returns (None, error) where error is None on success.
    """
    import shutil

    _, err = validate_profile_name(profile)
    if err:
        return None, err

    source_path = _get_profile_path(profile)
    default_path = _get_profile_path("default")

    if not source_path.exists():
        return (
            None,
            f"Profile '{profile}' not found. Run: rollouts --login-claude --profile {profile}",
        )

    try:
        shutil.copy2(source_path, default_path)
        return None, None
    except OSError as e:
        return None, f"Failed to set default profile: {e}"


class OAuthError(Exception):
    """OAuth-related error."""

    pass


class OAuthExpiredError(OAuthError):
    """OAuth token expired and refresh failed - re-login required."""

    pass


class OAuthClient:
    """OAuth client for Claude authentication with profile support."""

    def __init__(self, profile: str = "default") -> None:
        self.profile = profile
        self._tokens: OAuthTokens | None = None
        self._verifier: str | None = None

    @property
    def tokens(self) -> OAuthTokens | None:
        if self._tokens is None:
            self._tokens, err = load_tokens(self.profile)
            if err:
                logger.warning(f"Failed to load tokens for profile '{self.profile}': {err}")
        return self._tokens

    def is_logged_in(self) -> bool:
        tokens = self.tokens
        return tokens is not None

    def get_authorize_url(self, mode: str = "max") -> str:
        """
        Generate authorization URL for user to visit.

        Args:
            mode: "max" for Claude Pro/Max (claude.ai), "console" for API key creation

        Returns:
            URL string and stores verifier internally
        """
        self._verifier, challenge = _generate_pkce()

        if mode == "console":
            base_url = "https://console.anthropic.com/oauth/authorize"
        else:
            base_url = "https://claude.ai/oauth/authorize"

        from urllib.parse import urlencode

        params = {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": self._verifier,
        }

        query = urlencode(params)
        return f"{base_url}?{query}"

    async def exchange_code(self, code: str) -> OAuthTokens:
        """
        Exchange authorization code for tokens.

        Args:
            code: The code from the callback URL (may contain #state suffix)
        """
        if self._verifier is None:
            raise OAuthError("No verifier - call get_authorize_url first")

        # Code may be in format "code#state"
        parts = code.split("#")
        auth_code = parts[0]

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TOKEN_URL,
                json={
                    "code": auth_code,
                    "state": parts[1] if len(parts) > 1 else self._verifier,
                    "grant_type": "authorization_code",
                    "client_id": CLIENT_ID,
                    "redirect_uri": REDIRECT_URI,
                    "code_verifier": self._verifier,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                raise OAuthError(f"Token exchange failed: {response.status_code} {response.text}")

            data = response.json()

            # Capture scopes from response (space-separated string)
            scopes = data.get("scope", "")

            tokens = OAuthTokens(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=time.time() * 1000 + data["expires_in"] * 1000 - EXPIRY_BUFFER_MS,
                scopes=scopes,
            )

            self._tokens = tokens
            _, err = save_tokens(tokens, self.profile)
            if err:
                raise OAuthError(f"Failed to save tokens: {err}")
            self._verifier = None

            return tokens

    async def refresh_tokens(self) -> OAuthTokens:
        """Refresh access token using refresh token."""
        tokens = self.tokens
        if tokens is None:
            raise OAuthError("No tokens to refresh")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                TOKEN_URL,
                json={
                    "grant_type": "refresh_token",
                    "refresh_token": tokens.refresh_token,
                    "client_id": CLIENT_ID,
                },
                headers={"Content-Type": "application/json"},
            )

            if response.status_code != 200:
                # Token likely revoked
                _, err = delete_tokens(self.profile)
                if err:
                    logger.warning(f"Failed to delete tokens after refresh failure: {err}")
                self._tokens = None
                raise OAuthError(f"Token refresh failed: {response.status_code}")

            data = response.json()

            # Preserve scopes and api_key from original tokens
            new_tokens = OAuthTokens(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", tokens.refresh_token),
                expires_at=time.time() * 1000 + data["expires_in"] * 1000 - EXPIRY_BUFFER_MS,
                scopes=data.get(
                    "scope", tokens.scopes
                ),  # Use new scopes if provided, else keep old
                api_key=tokens.api_key,  # Preserve API key
            )

            self._tokens = new_tokens
            _, err = save_tokens(new_tokens, self.profile)
            if err:
                raise OAuthError(f"Failed to save refreshed tokens: {err}")

            return new_tokens

    async def get_valid_access_token(self) -> str | None:
        """Get a valid access token, refreshing if needed.

        Returns:
            Valid access token, or None if not logged in.

        Raises:
            OAuthExpiredError: If token refresh fails (re-login required).
        """
        tokens = self.tokens
        if tokens is None:
            return None

        if tokens.is_expired():
            try:
                tokens = await self.refresh_tokens()
            except OAuthError as e:
                logger.warning(f"OAuth token refresh failed: {e}")
                raise OAuthExpiredError(
                    "OAuth token expired and refresh failed. Please run /login to re-authenticate."
                ) from e

        return tokens.access_token

    async def create_api_key(self) -> str:
        """Create an API key using the OAuth token.

        This is used for accounts that have org:create_api_key scope but not user:inference.
        (e.g., enterprise/developer console accounts)

        Returns:
            The created API key.

        Raises:
            OAuthError: If API key creation fails.
        """
        tokens = self.tokens
        if tokens is None:
            raise OAuthError("No tokens - login first")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                API_KEY_URL,
                json=None,
                headers={
                    "Authorization": f"Bearer {tokens.access_token}",
                    "Content-Type": "application/json",
                },
            )

            if response.status_code != 200:
                raise OAuthError(f"API key creation failed: {response.status_code} {response.text}")

            data = response.json()
            api_key = data.get("raw_key")

            if not api_key:
                raise OAuthError("API key creation response missing raw_key")

            # Update tokens with API key and save
            tokens.api_key = api_key
            self._tokens = tokens
            _, err = save_tokens(tokens, self.profile)
            if err:
                raise OAuthError(f"Failed to save API key: {err}")

            return api_key

    def logout(self) -> None:
        """Clear stored tokens."""
        _, err = delete_tokens(self.profile)
        if err:
            raise OAuthError(f"Failed to logout: {err}")
        self._tokens = None
        print(f"✅ Logged out from Claude (profile: {self.profile})")


# Global client instances per profile
_global_clients: dict[str, OAuthClient] = {}


def get_oauth_client(profile: str = "default") -> OAuthClient:
    """Get or create OAuth client for profile."""
    if profile not in _global_clients:
        _global_clients[profile] = OAuthClient(profile)
    return _global_clients[profile]


def is_logged_in(profile: str = "default") -> bool:
    """Check if profile has valid tokens."""
    return get_oauth_client(profile).is_logged_in()


async def login(profile: str = "default", mode: str | None = None) -> OAuthTokens:
    """Interactive login flow for profile.

    Args:
        profile: Profile name to save tokens under
        mode: Login mode - "claude" for Claude subscription (Pro/Max/Team/Enterprise),
              "console" for Anthropic Console (API billing). If None, prompts user.
    """
    _, err = validate_profile_name(profile)
    if err:
        raise OAuthError(err)

    client = get_oauth_client(profile)

    # Prompt for login method if not specified
    if mode is None:
        print("\n🔐 Claude Code can be used with your Claude subscription or")
        print("   billed based on API usage through your Console account.\n")
        print("   Select login method:\n")
        print("   1. Claude account with subscription · Pro, Max, Team, or Enterprise")
        print("   2. Anthropic Console account · API usage billing\n")

        while True:
            try:
                choice = input("Enter 1 or 2: ").strip()
                if choice == "1":
                    mode = "max"
                    break
                elif choice == "2":
                    mode = "console"
                    break
                else:
                    print("Please enter 1 or 2")
            except (KeyboardInterrupt, EOFError) as e:
                print("\n⚠️  Login cancelled")
                raise KeyboardInterrupt() from e

    url = client.get_authorize_url(mode)

    print(f"\n🔐 Logging in to profile: {profile}")
    print("Open this URL in your browser to log in:")
    print(f"\n   {url}\n")
    print("After authorizing, you'll see a page with a code.")
    print("Copy the ENTIRE code (including any # and text after it).\n")

    # Use readline for proper line editing support
    try:
        code = input("Paste the code here: ")
        # Clean up the input - remove CR and strip whitespace
        code = code.replace("\r", "").strip()
    except (KeyboardInterrupt, EOFError) as e:
        print("\n⚠️  Login cancelled")
        raise KeyboardInterrupt() from e

    if not code:
        raise OAuthError("No code provided")

    tokens = await client.exchange_code(code)

    # Check if we have inference scope for direct OAuth usage
    if tokens.has_inference_scope():
        print(f"✅ Successfully logged in to Claude (profile: {profile})!")
    else:
        # No inference scope - need to create an API key (console/developer accounts)
        print("🔑 Creating API key for inference...")
        try:
            api_key = await client.create_api_key()
            print(f"✅ Successfully logged in to Claude (profile: {profile})!")
            print(f"   Using API key for inference (key prefix: {api_key[:15]}...)")
        except OAuthError as e:
            print(f"⚠️  Logged in but API key creation failed: {e}")
            print("   You may need to use --api-key manually")

    return tokens


def logout(profile: str = "default") -> None:
    """Logout from profile."""
    get_oauth_client(profile).logout()
