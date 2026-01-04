"""Rate limit tracking and observability.

Design notes:
- Module-level state keyed by (api_key, provider, model) tuple
  - OpenAI: limits may be per-org or per-model (unclear), we track per-model to be safe
  - Anthropic: limits are per-model-class (Sonnet 4.x shares pool, etc.)
  - Over-granular keying is fine for observability; Phase 3 may aggregate
- Each request learns from its own response headers (no blocking for shared state)
- Trio's cooperative multitasking means dict updates are atomic (no preemption mid-update)
- Accept initial thundering herd; headers guide subsequent requests
- Log only on interesting events (near limit, errors) per logging_sucks.md philosophy

Providers that support header extraction:
- OpenAI Chat Completions: ✅ (via with_streaming_response)
- OpenAI Responses API streaming: ❌ (headers not exposed)
- Anthropic Messages: ✅ (via stream.response.headers)
"""

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RateLimitState:
    """Current rate limit state for an API key + provider + model combo."""

    remaining_requests: int | None = None
    total_requests: int | None = None
    reset_time: float | None = None  # Unix timestamp
    last_updated: float = field(default_factory=time.time)

    @property
    def utilization_pct(self) -> float | None:
        """Return utilization as percentage (0-100), or None if unknown."""
        if self.remaining_requests is None or self.total_requests is None:
            return None
        if self.total_requests == 0:
            return 100.0
        return 100.0 * (1 - self.remaining_requests / self.total_requests)


# Module-level state: (api_key, provider, model) -> RateLimitState
_rate_limit_state: dict[tuple[str, str, str], RateLimitState] = {}


def _make_state_key(api_key: str, provider: str, model: str) -> tuple[str, str, str]:
    """Create state key. For Anthropic, normalize to model class (e.g., 'sonnet-4.x')."""
    assert api_key, "api_key must not be empty"
    assert provider, "provider must not be empty"
    assert model, "model must not be empty"

    model_key = model
    if provider == "anthropic":
        # Anthropic limits are per model-class, not per exact model
        # e.g., claude-sonnet-4-20250514 -> sonnet-4.x
        model_lower = model.lower()
        if "sonnet-4" in model_lower:
            model_key = "sonnet-4.x"
        elif "opus-4" in model_lower:
            model_key = "opus-4.x"
        elif "haiku-4" in model_lower:
            model_key = "haiku-4.x"
        elif "sonnet-3" in model_lower:
            model_key = "sonnet-3.x"
        elif "opus-3" in model_lower:
            model_key = "opus-3.x"
        elif "haiku-3" in model_lower:
            model_key = "haiku-3.x"
        # else: keep original model name

    return (api_key, provider, model_key)


def _get_api_key_display(api_key: str) -> str:
    """Return masked key for logging (don't log full keys)."""
    if len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


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
    - "12ms" (relative milliseconds)
    - ISO 8601 timestamps (e.g., "2026-01-04T02:31:23Z")
    """
    if not value:
        return None

    try:
        # Relative time format: "30s", "1m", "12ms", etc.
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
            from datetime import datetime, timezone

            # Handle Z suffix and timezone
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            dt = datetime.fromisoformat(value)
            # Ensure timezone-aware
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    except (ValueError, TypeError):
        return None


def update_rate_limit_from_headers(
    api_key: str,
    provider: str,
    model: str,
    headers: dict[str, str],
) -> None:
    """Update rate limit state from response headers.

    Called after each API response. Logs only on interesting events:
    - First time we learn the limit for this key/provider/model
    - When utilization > 80%
    - When remaining drops to 0
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
        return  # No rate limit info in headers

    state_key = _make_state_key(api_key, provider, model)
    key_display = _get_api_key_display(api_key)
    state = _rate_limit_state.get(state_key)
    is_first_update = state is None

    if state is None:
        state = RateLimitState()
        _rate_limit_state[state_key] = state

    state.remaining_requests = remaining
    state.total_requests = total
    state.reset_time = reset_time
    state.last_updated = time.time()

    # Structured logging on interesting events
    utilization = state.utilization_pct

    if is_first_update:
        logger.info(
            "rate_limit_discovered",
            extra={
                "provider": provider,
                "model": state_key[2],  # model_class for anthropic, model for openai
                "api_key": key_display,
                "remaining_requests": remaining,
                "total_requests": total,
                "utilization_pct": utilization,
            },
        )
    elif remaining == 0:
        # Check exhausted first (more specific than high utilization)
        logger.warning(
            "rate_limit_exhausted",
            extra={
                "provider": provider,
                "model": state_key[2],
                "api_key": key_display,
                "total_requests": total,
                "reset_time": reset_time,
            },
        )
    elif utilization is not None and utilization > 80:
        logger.warning(
            "rate_limit_high_utilization",
            extra={
                "provider": provider,
                "model": state_key[2],
                "api_key": key_display,
                "remaining_requests": remaining,
                "total_requests": total,
                "utilization_pct": utilization,
                "reset_time": reset_time,
            },
        )


def get_rate_limit_state(api_key: str, provider: str, model: str) -> RateLimitState | None:
    """Get current rate limit state for an API key + provider + model combo."""
    state_key = _make_state_key(api_key, provider, model)
    return _rate_limit_state.get(state_key)


def get_all_rate_limit_states() -> dict[tuple[str, str, str], RateLimitState]:
    """Get all current rate limit states (for debugging)."""
    return _rate_limit_state.copy()


def clear_rate_limit_state() -> None:
    """Clear all rate limit state (for testing)."""
    _rate_limit_state.clear()
