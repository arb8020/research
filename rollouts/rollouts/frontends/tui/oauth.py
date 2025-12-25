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

# Scopes for Claude Pro/Max
SCOPES = "org:create_api_key user:profile user:inference"

# Token storage
DEFAULT_TOKEN_PATH = Path.home() / ".rollouts" / "oauth" / "anthropic.json"

# Refresh tokens 3 minutes before expiry to avoid mid-request failures
EXPIRY_BUFFER_MS = 3 * 60 * 1000


@dataclass
class OAuthTokens:
    """OAuth token pair with metadata."""

    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp in milliseconds

    def is_expired(self) -> bool:
        """Check if token is expired."""
        return time.time() * 1000 >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthTokens:
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
        )


def _generate_pkce() -> tuple[str, str]:
    """Generate PKCE code verifier and challenge."""
    # Generate 32 bytes of random data for verifier
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")
    # Create S256 challenge
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return verifier, challenge


class TokenStorage:
    """Persistent storage for OAuth tokens."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH):
        self.path = path

    def save(self, tokens: OAuthTokens) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(tokens.to_dict(), f, indent=2)
        os.chmod(self.path, 0o600)

    def load(self) -> OAuthTokens | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path) as f:
                data = json.load(f)
            return OAuthTokens.from_dict(data)
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load tokens: {e}")
            return None

    def delete(self) -> None:
        if self.path.exists():
            self.path.unlink()


class OAuthError(Exception):
    """OAuth-related error."""

    pass


class OAuthClient:
    """OAuth client for Claude authentication."""

    def __init__(self, storage: TokenStorage | None = None):
        self.storage = storage or TokenStorage()
        self._tokens: OAuthTokens | None = None
        self._verifier: str | None = None

    @property
    def tokens(self) -> OAuthTokens | None:
        if self._tokens is None:
            self._tokens = self.storage.load()
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

            tokens = OAuthTokens(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=time.time() * 1000 + data["expires_in"] * 1000 - EXPIRY_BUFFER_MS,
            )

            self._tokens = tokens
            self.storage.save(tokens)
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
                self.storage.delete()
                self._tokens = None
                raise OAuthError(f"Token refresh failed: {response.status_code}")

            data = response.json()

            new_tokens = OAuthTokens(
                access_token=data["access_token"],
                refresh_token=data.get("refresh_token", tokens.refresh_token),
                expires_at=time.time() * 1000 + data["expires_in"] * 1000 - EXPIRY_BUFFER_MS,
            )

            self._tokens = new_tokens
            self.storage.save(new_tokens)

            return new_tokens

    async def get_valid_access_token(self) -> str | None:
        """Get a valid access token, refreshing if needed."""
        tokens = self.tokens
        if tokens is None:
            return None

        if tokens.is_expired():
            try:
                tokens = await self.refresh_tokens()
            except OAuthError as e:
                logger.error(f"Failed to refresh tokens: {e}")
                return None

        return tokens.access_token

    def logout(self) -> None:
        """Clear stored tokens."""
        self.storage.delete()
        self._tokens = None
        print("✅ Logged out from Claude")


# Global client instance
_global_client: OAuthClient | None = None


def get_oauth_client() -> OAuthClient:
    global _global_client
    if _global_client is None:
        _global_client = OAuthClient()
    return _global_client


def is_logged_in() -> bool:
    return get_oauth_client().is_logged_in()


async def login() -> OAuthTokens:
    """Interactive login flow."""

    client = get_oauth_client()

    url = client.get_authorize_url("max")

    print("\n🔐 Open this URL in your browser to log in:")
    print(f"\n   {url}\n")
    print("After authorizing, you'll see a page with a code.")
    print("Copy the ENTIRE code (including any # and text after it).\n")

    # Use readline for proper line editing support
    try:
        code = input("Paste the code here: ")
        # Clean up the input - remove CR and strip whitespace
        code = code.replace("\r", "").strip()
    except (KeyboardInterrupt, EOFError):
        print("\n⚠️  Login cancelled")
        raise KeyboardInterrupt()

    if not code:
        raise OAuthError("No code provided")

    tokens = await client.exchange_code(code)
    print("✅ Successfully logged in to Claude!")

    return tokens


def logout() -> None:
    get_oauth_client().logout()
