"""
OAuth module for Claude Pro/Max authentication using PKCE flow.

Implements OAuth 2.0 PKCE (Proof Key for Code Exchange) for authenticating
with Claude using Anthropic account credentials instead of API keys.

Token format:
- Access token: sk-ant-oat01-...  (8 hour expiry)
- Refresh token: sk-ant-ort01-...

Storage:
- Tokens stored in ~/.rollouts/oauth/tokens.json
- Automatic refresh when access token expires
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import socket
import time
import webbrowser
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
import trio

logger = logging.getLogger(__name__)

# OAuth configuration for Claude (via Anthropic Console)
CLAUDE_CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
CLAUDE_AUTHORIZE_URL = "https://console.anthropic.com/oauth/authorize"
CLAUDE_TOKEN_URL = "https://console.anthropic.com/v1/oauth/token"
CLAUDE_REVOKE_URL = "https://api.anthropic.com/api/oauth/claude"  # Revoke endpoint

# Local callback server
CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 19485  # Match Claude Code's port
CALLBACK_REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}/oauth/callback"

# Token storage
DEFAULT_TOKEN_PATH = Path.home() / ".rollouts" / "oauth" / "tokens.json"


@dataclass
class OAuthTokens:
    """OAuth token pair with metadata."""
    access_token: str
    refresh_token: str
    expires_at: float  # Unix timestamp
    scope: str = "user:inference"

    def is_expired(self, buffer_seconds: int = 300) -> bool:
        """Check if token is expired or will expire soon."""
        return time.time() >= (self.expires_at - buffer_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "access_token": self.access_token,
            "refresh_token": self.refresh_token,
            "expires_at": self.expires_at,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OAuthTokens:
        """Deserialize from dictionary."""
        return cls(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=data["expires_at"],
            scope=data.get("scope", "user:inference"),
        )


def _generate_code_verifier() -> str:
    """Generate PKCE code verifier (43-128 chars, URL-safe)."""
    # 32 bytes = 43 chars base64url
    return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode("ascii").rstrip("=")


def _generate_code_challenge(verifier: str) -> str:
    """Generate PKCE code challenge from verifier (S256 method)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _generate_state() -> str:
    """Generate random state parameter for CSRF protection."""
    return secrets.token_urlsafe(32)


class TokenStorage:
    """Persistent storage for OAuth tokens."""

    def __init__(self, path: Path = DEFAULT_TOKEN_PATH):
        self.path = path

    def save(self, tokens: OAuthTokens) -> None:
        """Save tokens to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(tokens.to_dict(), f, indent=2)
        # Secure permissions (user-only read/write)
        os.chmod(self.path, 0o600)

    def load(self) -> OAuthTokens | None:
        """Load tokens from disk, or None if not found."""
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
        """Delete stored tokens."""
        if self.path.exists():
            self.path.unlink()


class OAuthError(Exception):
    """OAuth-related error."""
    pass


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback."""

    # Class variables set by the server
    expected_state: str = ""
    authorization_code: str | None = None
    error_message: str | None = None
    callback_received: bool = False

    def log_message(self, format: str, *args: Any) -> None:
        """Suppress default logging."""
        pass

    def do_GET(self) -> None:
        """Handle OAuth callback GET request."""
        parsed = urlparse(self.path)

        if parsed.path != "/oauth/callback":
            self.send_response(404)
            self.end_headers()
            return

        params = parse_qs(parsed.query)

        # Check for errors
        if "error" in params:
            OAuthCallbackHandler.error_message = params.get(
                "error_description", params.get("error", ["Unknown error"])
            )[0]
            OAuthCallbackHandler.callback_received = True
            self._send_error_page(OAuthCallbackHandler.error_message)
            return

        # Verify state
        state = params.get("state", [""])[0]
        if state != OAuthCallbackHandler.expected_state:
            OAuthCallbackHandler.error_message = "Invalid state parameter (possible CSRF attack)"
            OAuthCallbackHandler.callback_received = True
            self._send_error_page(OAuthCallbackHandler.error_message)
            return

        # Get authorization code
        code = params.get("code", [""])[0]
        if not code:
            OAuthCallbackHandler.error_message = "No authorization code received"
            OAuthCallbackHandler.callback_received = True
            self._send_error_page(OAuthCallbackHandler.error_message)
            return

        OAuthCallbackHandler.authorization_code = code
        OAuthCallbackHandler.callback_received = True
        self._send_success_page()

    def _send_success_page(self) -> None:
        """Send success HTML page."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"""
        <html><body style="font-family: system-ui; padding: 40px; text-align: center;">
        <h1 style="color: #22c55e;">Login Successful!</h1>
        <p>You can close this window and return to the terminal.</p>
        <script>window.close();</script>
        </body></html>
        """)

    def _send_error_page(self, error: str) -> None:
        """Send error HTML page."""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(f"""
        <html><body style="font-family: system-ui; padding: 40px; text-align: center;">
        <h1 style="color: #ef4444;">Login Failed</h1>
        <p>{error}</p>
        <p>You can close this window.</p>
        </body></html>
        """.encode())


class OAuthClient:
    """OAuth client for Claude authentication."""

    def __init__(self, storage: TokenStorage | None = None):
        self.storage = storage or TokenStorage()
        self._tokens: OAuthTokens | None = None
        self._http_client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> OAuthClient:
        self._http_client = httpx.AsyncClient(timeout=30.0)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._http_client:
            await self._http_client.aclose()

    @property
    def tokens(self) -> OAuthTokens | None:
        """Get current tokens, loading from storage if needed."""
        if self._tokens is None:
            self._tokens = self.storage.load()
        return self._tokens

    def is_logged_in(self) -> bool:
        """Check if we have valid (non-expired) tokens."""
        tokens = self.tokens
        return tokens is not None and not tokens.is_expired()

    async def get_valid_access_token(self) -> str | None:
        """Get a valid access token, refreshing if needed."""
        tokens = self.tokens
        if tokens is None:
            return None

        if tokens.is_expired():
            try:
                await self.refresh_tokens()
                tokens = self.tokens
                if tokens is None:
                    return None
            except OAuthError as e:
                logger.error(f"Failed to refresh tokens: {e}")
                return None

        return tokens.access_token

    async def login(self) -> OAuthTokens:
        """
        Perform OAuth login flow.

        1. Generate PKCE code verifier/challenge
        2. Open browser to authorization URL
        3. Start local server to receive callback
        4. Exchange authorization code for tokens
        """
        # Generate PKCE parameters
        code_verifier = _generate_code_verifier()
        code_challenge = _generate_code_challenge(code_verifier)
        state = _generate_state()

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": CLAUDE_CLIENT_ID,
            "redirect_uri": CALLBACK_REDIRECT_URI,
            "scope": "user:inference",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{CLAUDE_AUTHORIZE_URL}?{urlencode(params)}"

        print(f"\n🔐 Opening browser for Claude login...")
        print(f"   If browser doesn't open, visit:\n   {auth_url}\n")

        # Start callback server and open browser
        authorization_code = await self._wait_for_callback(state, auth_url)

        # Exchange code for tokens
        tokens = await self._exchange_code(authorization_code, code_verifier)

        # Save tokens
        self._tokens = tokens
        self.storage.save(tokens)

        print("✅ Successfully logged in to Claude!")
        return tokens

    async def _wait_for_callback(self, expected_state: str, auth_url: str) -> str:
        """Start callback server and wait for OAuth redirect."""
        import signal

        # Reset handler state
        OAuthCallbackHandler.expected_state = expected_state
        OAuthCallbackHandler.authorization_code = None
        OAuthCallbackHandler.error_message = None
        OAuthCallbackHandler.callback_received = False

        # Create and start server in a thread (blocking I/O)
        server = HTTPServer((CALLBACK_HOST, CALLBACK_PORT), OAuthCallbackHandler)
        server.timeout = 1.0  # 1 second timeout for handle_request

        cancelled = False

        def signal_handler(signum: int, frame: Any) -> None:
            nonlocal cancelled
            cancelled = True
            OAuthCallbackHandler.callback_received = True
            print("\n⚠️  Login cancelled")

        # Set up signal handler for Ctrl+C
        old_handler = signal.signal(signal.SIGINT, signal_handler)

        def run_server() -> None:
            while not OAuthCallbackHandler.callback_received:
                try:
                    server.handle_request()
                except Exception:
                    break

        server_thread = Thread(target=run_server, daemon=True)
        server_thread.start()

        # Open browser
        webbrowser.open(auth_url)

        # Wait for callback with timeout (5 minutes)
        start_time = time.time()
        timeout = 300

        try:
            while not OAuthCallbackHandler.callback_received:
                if time.time() - start_time > timeout:
                    raise OAuthError("Login timed out (5 minutes)")
                await trio.sleep(0.1)
        finally:
            # Restore old signal handler
            signal.signal(signal.SIGINT, old_handler)
            # Clean up server
            OAuthCallbackHandler.callback_received = True  # Signal thread to stop
            try:
                server.shutdown()
            except Exception:
                pass

        if cancelled:
            raise OAuthError("Login cancelled by user")

        if OAuthCallbackHandler.error_message:
            raise OAuthError(OAuthCallbackHandler.error_message)

        if OAuthCallbackHandler.authorization_code is None:
            raise OAuthError("No authorization code received")

        return OAuthCallbackHandler.authorization_code

    async def _exchange_code(self, code: str, code_verifier: str) -> OAuthTokens:
        """Exchange authorization code for tokens."""
        if self._http_client is None:
            raise OAuthError("HTTP client not initialized")

        data = {
            "grant_type": "authorization_code",
            "client_id": CLAUDE_CLIENT_ID,
            "code": code,
            "redirect_uri": CALLBACK_REDIRECT_URI,
            "code_verifier": code_verifier,
        }

        response = await self._http_client.post(
            CLAUDE_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            error_body = response.text
            raise OAuthError(f"Token exchange failed: {response.status_code} {error_body}")

        token_data = response.json()

        expires_in = token_data.get("expires_in", 28800)  # Default 8 hours
        expires_at = time.time() + expires_in

        return OAuthTokens(
            access_token=token_data["access_token"],
            refresh_token=token_data["refresh_token"],
            expires_at=expires_at,
            scope=token_data.get("scope", "user:inference"),
        )

    async def refresh_tokens(self) -> OAuthTokens:
        """Refresh access token using refresh token."""
        tokens = self.tokens
        if tokens is None:
            raise OAuthError("No tokens to refresh")

        if self._http_client is None:
            # Create temporary client for refresh
            async with httpx.AsyncClient(timeout=30.0) as client:
                return await self._do_refresh(client, tokens.refresh_token)
        else:
            return await self._do_refresh(self._http_client, tokens.refresh_token)

    async def _do_refresh(self, client: httpx.AsyncClient, refresh_token: str) -> OAuthTokens:
        """Perform token refresh request."""
        data = {
            "grant_type": "refresh_token",
            "client_id": CLAUDE_CLIENT_ID,
            "refresh_token": refresh_token,
        }

        response = await client.post(
            CLAUDE_TOKEN_URL,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        if response.status_code != 200:
            error_body = response.text
            # If refresh fails, tokens are likely revoked
            if response.status_code in (400, 401):
                self.storage.delete()
                self._tokens = None
            raise OAuthError(f"Token refresh failed: {response.status_code} {error_body}")

        token_data = response.json()

        expires_in = token_data.get("expires_in", 28800)
        expires_at = time.time() + expires_in

        new_tokens = OAuthTokens(
            access_token=token_data["access_token"],
            refresh_token=token_data.get("refresh_token", refresh_token),  # May not change
            expires_at=expires_at,
            scope=token_data.get("scope", "user:inference"),
        )

        self._tokens = new_tokens
        self.storage.save(new_tokens)
        logger.info("Successfully refreshed OAuth tokens")

        return new_tokens

    async def logout(self) -> None:
        """Revoke tokens and clear storage."""
        tokens = self.tokens
        if tokens is None:
            print("Not logged in.")
            return

        # Try to revoke on server (best effort)
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    CLAUDE_REVOKE_URL,
                    data={
                        "client_id": CLAUDE_CLIENT_ID,
                        "token": tokens.access_token,
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
        except Exception as e:
            logger.warning(f"Failed to revoke token: {e}")

        # Clear local storage
        self.storage.delete()
        self._tokens = None
        print("✅ Logged out from Claude")


# Convenience functions
_global_client: OAuthClient | None = None


def get_oauth_client() -> OAuthClient:
    """Get or create global OAuth client."""
    global _global_client
    if _global_client is None:
        _global_client = OAuthClient()
    return _global_client


def is_logged_in() -> bool:
    """Check if user is logged in with OAuth."""
    return get_oauth_client().is_logged_in()


async def get_access_token() -> str | None:
    """Get valid access token, refreshing if needed."""
    client = get_oauth_client()
    async with client:
        return await client.get_valid_access_token()


async def login() -> OAuthTokens:
    """Perform OAuth login."""
    client = get_oauth_client()
    async with client:
        return await client.login()


async def logout() -> None:
    """Logout and revoke tokens."""
    client = get_oauth_client()
    async with client:
        await client.logout()
