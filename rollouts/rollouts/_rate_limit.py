"""Rate limit tracking and observability.

Parses rate limit headers from LLM provider responses to track quota usage.
Logs on interesting events (high utilization, exhausted limits) for visibility.

Design notes:
- Module-level state keyed by API key (not model - limits are org-wide)
- Each request learns from its own response headers
- Logs only on interesting events (first discovery, >80% utilization, exhausted)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Current rate limit state for an API key."""

    remaining_requests: int | None = None
    total_requests: int | None = None
    reset_time: float | None = None  # Unix timestamp
    last_updated: float = 0.0

    @property
    def utilization_pct(self) -> float | None:
        """Return utilization as percentage (0-100), or None if unknown."""
        if self.remaining_requests is None or self.total_requests is None:
            return None
        if self.total_requests == 0:
            return 100.0
        return 100.0 * (1 - self.remaining_requests / self.total_requests)


# Module-level state: API key -> RateLimitState
_rate_limit_state: dict[str, RateLimitState] = {}


def _get_api_key_hash(api_key: str) -> str:
    """Return truncated key for logging (don't log full keys)."""
    if len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def update_rate_limit_from_headers(
    api_key: str,
    provider: str,
    headers: dict[str, str],
) -> RateLimitState | None:
    """Update rate limit state from response headers.

    Called after each API response. Logs only on interesting events:
    - First time we learn the limit
    - When utilization > 80%
    - When remaining drops to 0

    Args:
        api_key: The API key used for the request
        provider: Provider name ("openai", "anthropic", etc.)
        headers: Response headers dict

    Returns:
        Updated RateLimitState, or None if no rate limit info in headers
    """
    # Extract headers based on provider
    remaining: int | None = None
    total: int | None = None
    reset_time: float | None = None

    if provider == "openai":
        remaining = _parse_int(headers.get("x-ratelimit-remaining-requests"))
        total = _parse_int(headers.get("x-ratelimit-limit-requests"))
        reset_str = headers.get("x-ratelimit-reset-requests")
        if reset_str:
            reset_time = _parse_reset_time(reset_str)
    elif provider == "anthropic":
        remaining = _parse_int(headers.get("anthropic-ratelimit-requests-remaining"))
        total = _parse_int(headers.get("anthropic-ratelimit-requests-limit"))
        reset_str = headers.get("anthropic-ratelimit-requests-reset")
        if reset_str:
            reset_time = _parse_reset_time(reset_str)
    # Google: headers not consistently available, skip

    if remaining is None and total is None:
        return None  # No rate limit info in headers

    key_hash = _get_api_key_hash(api_key)
    state = _rate_limit_state.get(api_key)
    is_first_update = state is None

    if state is None:
        state = RateLimitState()
        _rate_limit_state[api_key] = state

    state.remaining_requests = remaining
    state.total_requests = total
    state.reset_time = reset_time
    state.last_updated = time.time()

    # Log on interesting events
    utilization = state.utilization_pct

    if is_first_update:
        logger.info(
            f"[rate_limit] {provider} discovered: "
            f"remaining={remaining}/{total} ({100 - (utilization or 0):.0f}% available) "
            f"key={key_hash}"
        )
    elif utilization is not None and utilization > 80:
        logger.warning(
            f"[rate_limit] {provider} high utilization: "
            f"remaining={remaining}/{total} ({100 - utilization:.0f}% available) "
            f"key={key_hash}"
        )
    elif remaining == 0:
        reset_in = ""
        if reset_time:
            reset_in = f" resets in {reset_time - time.time():.0f}s"
        logger.warning(
            f"[rate_limit] {provider} EXHAUSTED: remaining=0/{total}{reset_in} key={key_hash}"
        )

    return state


def get_rate_limit_state(api_key: str) -> RateLimitState | None:
    """Get current rate limit state for an API key."""
    return _rate_limit_state.get(api_key)


def _parse_int(value: str | None) -> int | None:
    """Parse string to int, returning None on failure."""
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _parse_reset_time(value: str) -> float | None:
    """Parse reset time string to Unix timestamp.

    Handles formats:
    - "1s", "30s" (relative seconds)
    - "1m", "5m" (relative minutes)
    - "1ms", "500ms" (relative milliseconds)
    - ISO 8601 timestamps
    """
    if not value:
        return None

    try:
        # Relative time format: "30s", "1m", "500ms", etc.
        if value.endswith("ms"):
            ms = float(value[:-2])
            return time.time() + ms / 1000
        elif value.endswith("s"):
            seconds = float(value[:-1])
            return time.time() + seconds
        elif value.endswith("m"):
            minutes = float(value[:-1])
            return time.time() + minutes * 60
        else:
            # Try ISO 8601
            from datetime import datetime

            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
    except (ValueError, TypeError):
        return None
