# Rate Limit UX: Retries and Observability

> **Status**: Draft
> **Author**: Claude + Chiraag
> **Date**: 2025-01-02
> **Depends on**: `rate_limiter.md` (Phase 1-2)

## Context

The existing `rate_limiter.md` design doc covers the backend mechanics—increasing retries, parsing headers, sample-level retry. But the *user experience* during rate limits is poor:

**Current behavior:**
```
🔄 Anthropic API error (attempt 3/11): Error code: 429
   Endpoint model: claude-sonnet-4-20250514
   Endpoint api_base: https://api.anthropic.com
   Retrying in 8s...
```

Problems:
1. **`print()` statements** — not integrated with TUI, interleaved with other output
2. **No progress indicator** — user doesn't know if retry is happening or stuck
3. **No rate limit context** — user doesn't know remaining requests or when limits reset
4. **Inconsistent across providers** — Anthropic has manual retry loop, OpenAI relies on SDK

**Goal:** User always knows what's happening during rate limits. Retries are visible, progress is clear, and the user can make informed decisions (wait vs. cancel vs. switch models).

---

## Out of Scope

- Rate limit prevention/throttling (Phase 3 of `rate_limiter.md`)
- Per-key or per-org dashboards
- Historical rate limit analytics

---

## Usage Code First

### What the User Sees (TUI)

During normal operation, status line shows model/tokens/cost as usual.

When rate limited, a **spinner with message** appears (like Claude Code):
```
⠹ Retrying (2/3) in 4s... (esc to cancel)
```

User can press **Escape** to cancel the retry and see the error immediately.

On success, spinner disappears and agent continues. On final failure:
```
Error: Failed after 3 retries: Rate limit exceeded
```

**Future (deferred):** Warning when approaching limits (e.g., "⚠️ 12/500 requests remaining"). Requires header parsing on every response.

### What the Developer Sees (Logs)

Structured logging per `logging_sucks.md` philosophy—log only interesting events, with full context:

```python
# First discovery of rate limit info
logger.info("rate_limit_discovered", extra={
    "provider": "anthropic",
    "api_key_hash": "sk-a...xyz",
    "remaining_requests": 500,
    "total_requests": 500,
})

# When approaching limits (>80% utilization)
logger.warning("rate_limit_high_utilization", extra={
    "provider": "anthropic",
    "remaining_requests": 50,
    "total_requests": 500,
    "utilization_pct": 90.0,
    "reset_time": 1704234567.0,
})

# On retry attempt
logger.info("rate_limit_retry", extra={
    "provider": "anthropic",
    "attempt": 3,
    "max_attempts": 10,
    "delay_seconds": 8.0,
    "error_type": "RateLimitError",
})
```

---

## Design Principles

From `code_philosophy_manual.md`:

1. **Parsing at the edges** — Rate limit headers parsed once in provider layer, stored as typed `RateLimitState`. Provider throws typed errors, doesn't decide retry policy.
2. **Visible, minimal state** — One module owns rate limit state (`_rate_limit.py`), TUI queries it. Session owns retry state.
3. **Errors crash or compose** — Rate limits are transient failures at external boundary. Provider throws; session/agent decides whether to retry based on error type and user settings.
4. **Assertions document invariants** — Assert retry count is positive, delay is non-negative, max_attempts > 0, etc.
5. **Write usage code first** — Design doc starts with "What the User Sees" before implementation details.

From `logging_sucks.md`:

6. **Wide events, not chatty logs** — One log per retry attempt with full context (provider, attempt, delay, error_type, reset_time), not separate logs for "starting retry", "waiting", "retrying now"
7. **Log only interesting events** — Don't log every successful request. Log: first rate limit discovery, high utilization (>80%), retry attempts, final failures.

From `tiger_style_safety.md`:

8. **Push ifs up, fors down** — Retry decision logic centralized in session layer (one place decides "is this retryable?"). Provider layer has no branching on retry—just throws errors.
9. **Explicit control flow** — We control the retry loop with our own sleep + abort, not hidden SDK retries. SDK `max_retries` set to 0 or 1; we handle backoff ourselves.

---

## Solution

### Component 1: Rate Limit Types (`_rate_limit.py`)

Typed error for rate limits, plus optional header parsing for future observability:

```python
@dataclass(frozen=True)
class RateLimitError(Exception):
    """Raised by providers when rate limited. Caller decides retry policy."""
    message: str
    provider: str
    reset_time: float | None = None  # Unix timestamp when limits reset

    def __str__(self) -> str:
        return self.message


@dataclass
class RateLimitState:
    """Parsed from response headers. Optional—providers can skip this initially."""
    remaining_requests: int | None = None
    total_requests: int | None = None
    reset_time: float | None = None


def parse_anthropic_headers(headers: dict[str, str]) -> RateLimitState:
    """Parse Anthropic rate limit headers."""
    return RateLimitState(
        remaining_requests=_parse_int(headers.get("anthropic-ratelimit-requests-remaining")),
        total_requests=_parse_int(headers.get("anthropic-ratelimit-requests-limit")),
        reset_time=_parse_reset_time(headers.get("anthropic-ratelimit-requests-reset")),
    )


def parse_openai_headers(headers: dict[str, str]) -> RateLimitState:
    """Parse OpenAI rate limit headers."""
    return RateLimitState(
        remaining_requests=_parse_int(headers.get("x-ratelimit-remaining-requests")),
        total_requests=_parse_int(headers.get("x-ratelimit-limit-requests")),
        reset_time=_parse_reset_time(headers.get("x-ratelimit-reset-requests")),
    )
```

### Component 2: Provider Changes (Minimal)

Providers become simpler—they throw errors, don't retry. Currently:
- Anthropic: manual loop with `print()` statements (remove this)
- OpenAI: relies on SDK `max_retries` (set to 1, let session handle retry)

**Change:** Providers just:
1. Make API call (single attempt, or SDK's minimal retry for connection errors)
2. Parse rate limit headers from response, update `RateLimitState`
3. On rate limit error, throw typed `RateLimitError` with provider info
4. Let session layer decide retry policy

```python
# In providers/anthropic.py - simplified, no retry loop:

async def stream_anthropic(actor: Actor, on_chunk: OnChunk) -> tuple[Message, Completion]:
    """Make single API call. Throws RateLimitError on 429, let caller retry."""
    client = _create_anthropic_client(
        api_key=actor.endpoint.api_key,
        max_retries=1,  # Only retry connection errors, not 429s
    )

    try:
        async with client.messages.stream(**params) as stream:
            # Parse headers for rate limit state
            headers = dict(stream.response.headers)
            update_rate_limit_from_headers(
                api_key=actor.endpoint.api_key,
                provider="anthropic",
                headers=headers,
            )
            completion = await aggregate_stream(stream, on_chunk)
            return message, completion

    except anthropic.RateLimitError as e:
        # Wrap with our typed error, include reset time from headers if available
        raise RateLimitError(
            message=str(e),
            provider="anthropic",
            reset_time=_parse_reset_from_error(e),
        ) from e
```

### Component 2b: TUI Retry Logic (Inline)

Per semantic compression principle: don't abstract until you have two instances. Implement retry inline in `InteractiveAgentRunner` first. Eval/training can implement their own approach (or skip retry entirely). Extract shared utility later if patterns emerge.

```python
# In frontends/tui/interactive_agent.py - inline retry logic, no abstraction:

class InteractiveAgentRunner:
    def __init__(self, ...):
        # ... existing init ...
        self._retry_attempt = 0
        self._retry_cancel_scope: trio.CancelScope | None = None
        self._saved_escape_handler: Callable | None = None

    def _is_retryable_error(self, error: Exception) -> bool:
        """Check if error should trigger retry."""
        if isinstance(error, RateLimitError):
            return True
        msg = str(error).lower()
        return any(p in msg for p in [
            "overloaded", "rate limit", "429", "500", "502", "503", "504",
            "service unavailable", "connection error"
        ])

    async def _handle_retryable_error(self, error: Exception) -> bool:
        """Handle error with retry. Returns True if retry initiated, False if giving up."""
        max_retries = 3
        base_delay_ms = 2000

        self._retry_attempt += 1
        if self._retry_attempt > max_retries:
            self._show_error(f"Failed after {max_retries} retries: {error}")
            self._retry_attempt = 0
            return False

        delay_ms = base_delay_ms * (2 ** (self._retry_attempt - 1))
        delay_s = delay_ms // 1000

        # Show loader, hijack escape key
        self._saved_escape_handler = self._on_escape
        self._on_escape = self._abort_retry

        self.retry_loader = Loader(
            text=f"Retrying ({self._retry_attempt}/{max_retries}) in {delay_s}s... (esc to cancel)",
            spinner_color_fn=self.theme.warning,
            text_color_fn=self.theme.muted,
        )
        self.loader_container.add_child(self.retry_loader)
        self.tui.render()

        # Cancellable sleep
        with trio.CancelScope() as self._retry_cancel_scope:
            await trio.sleep(delay_ms / 1000)

        # Clean up UI
        self._cleanup_retry_ui()

        if self._retry_cancel_scope.cancelled_caught:
            self._retry_attempt = 0
            return False

        return True  # Caller should retry

    def _abort_retry(self) -> None:
        """Called when user presses escape during retry."""
        if self._retry_cancel_scope:
            self._retry_cancel_scope.cancel()

    def _cleanup_retry_ui(self) -> None:
        """Restore UI state after retry completes or is cancelled."""
        if self._saved_escape_handler:
            self._on_escape = self._saved_escape_handler
            self._saved_escape_handler = None
        if self.retry_loader:
            self.retry_loader.stop()
            self.loader_container.clear()
            self.retry_loader = None
```

### Component 3: Rate Limit Warning in Status Line (Future)

**Deferred:** Showing "12/500 requests remaining" warning requires polling rate limit state or updating on every response. Add this later if header parsing (Component 1) proves useful.

For now, we only show the Loader during active retry (Component 2b).

---

## Files Modified

| File | Change |
|------|--------|
| `rollouts/_rate_limit.py` | New file: `RateLimitState`, `RateLimitError`, header parsing |
| `rollouts/providers/anthropic.py` | Remove retry loop + `print()`, throw `RateLimitError`, parse headers |
| `rollouts/providers/openai_responses.py` | Add header parsing, throw `RateLimitError` |
| `rollouts/providers/openai_completions.py` | Same as above |
| `rollouts/frontends/tui/components/loader.py` | Already exists, use for retry spinner |
| `rollouts/frontends/tui/interactive_agent.py` | Add inline retry logic, escape key handling, show Loader |

**Not creating (yet):** `_retry.py` abstraction — wait until eval/training also need retry, then extract common patterns.

---

## Testing

Per `grugbrain_testing.md`: integration tests at cut points, manual testing first.

### Manual Testing (Do This First)

```bash
# 1. Run interactive session, hit rate limits manually by making many requests
rollouts chat --model claude-sonnet-4-20250514

# Observe:
# - Status line updates during retry
# - Warning appears when approaching limits
# - Retry countdown is accurate

# 2. Run eval with high concurrency to stress test
rollouts eval dataset.jsonl --max-concurrent=50 --samples=100

# Observe logs for structured rate_limit_retry events
```

### Integration Test

One test that verifies the retry event is emitted:

```python
# tests/test_rate_limit_events.py

async def test_retry_event_emitted_on_rate_limit():
    """Verify RetryEvent is emitted when rate limit hit."""
    from rollouts._rate_limit import add_retry_listener, remove_retry_listener, RetryEvent

    events: list[RetryEvent] = []
    add_retry_listener(events.append)

    try:
        # Mock provider that returns 429 twice then succeeds
        # ... (inject mock, call provider)
        pass
    finally:
        remove_retry_listener(events.append)

    assert len(events) == 2
    assert events[0].attempt == 1
    assert events[1].attempt == 2
```

---

## Migration

1. **Phase 1 (this PR)**: Implement `_rate_limit.py`, update providers, update status line
2. **Phase 2 (follow-up)**: Add rate limit warning when approaching limits (requires polling/checking state periodically)
3. **Phase 3 (future)**: Proactive concurrency control per `rate_limiter.md`

---

## Answers from Claude Code (pi-mono)

Reviewed `/tmp/pi-mono/packages/coding-agent/src/core/agent-session.ts` and `/tmp/pi-mono/packages/coding-agent/src/modes/interactive/interactive-mode.ts` to answer the open questions:

### 1. Should retry state persist across messages?

**Answer: No, clear on success or final failure.**

Claude Code emits `auto_retry_end` with `success: true` when retry succeeds, then clears the retry counter. The UI clears the loader on any `auto_retry_end`. State does NOT persist—retry indicator disappears as soon as agent continues.

### 2. Animation during retry wait?

**Answer: Yes, use a spinner with static countdown.**

Claude Code shows a `Loader` component (spinning animation) with message:
```
Retrying (1/3) in 4s... (esc to cancel)
```

The countdown is static (set once when retry starts), NOT live-updating. This is simpler and sufficient—user knows roughly how long to wait.

### 3. Keyboard interrupt during retry?

**Answer: Yes, Escape cancels retry.**

Claude Code:
1. On `auto_retry_start`: Temporarily replaces editor's `onEscape` handler with `session.abortRetry()`
2. The `abortRetry()` method aborts the sleep via `AbortController`, emits `auto_retry_end` with `finalError: "Retry cancelled"`, resets state
3. On `auto_retry_end`: Restores original escape handler

Key implementation detail: The sleep is wrapped with an `AbortController` that can be cancelled:

```typescript
// From agent-session.ts
this._retryAbortController = new AbortController();
try {
    await this._sleep(delayMs, this._retryAbortController.signal);
} catch {
    // Aborted during sleep
    this._emit({ type: "auto_retry_end", success: false, attempt, finalError: "Retry cancelled" });
    return false;
}
```

### 4. Retry at API layer vs Session layer?

**Answer: Session layer (not provider layer).**

Claude Code handles retries in `AgentSession._handleRetryableError()`, NOT in the provider. The provider just throws errors. Session decides whether to retry based on:
- Error message matching patterns: `overloaded|rate.?limit|429|500|502|503|504|connection.?error`
- User settings: `retry.enabled`, `retry.maxRetries`, `retry.baseDelayMs`

This is different from our current approach (Anthropic provider has inline retry loop). We should consider moving retry logic up to the agent/session layer for consistency.

---

## Summary of Changes from Claude Code Research

1. **Use Loader component with spinner** (like Claude Code), not static status line text
2. **Escape key cancels retry** via `trio.CancelScope` (Python equivalent of AbortController)
3. **Clear retry UI immediately** when retry succeeds or fails
4. **Retry logic inline in TUI** — don't abstract until eval/training also need it
5. **Providers throw, don't retry** — move retry decisions out of provider layer
