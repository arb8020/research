# Rate Limiter for LLM Providers

**DRI:** 
**Status:** Draft

## Context

When running concurrent evaluations (e.g., `max_concurrent=100`), we overwhelm LLM provider rate limits. Current behavior:

1. OpenAI/Anthropic SDKs have built-in retry with exponential backoff
2. We catch `RateLimitError` after exhausting retries and wrap as `ProviderError`
3. Current `max_retries=3` is too low for sustained rate limiting

**Observed behavior:**
```
[17:12:45] Retrying request in 1.709486 seconds
[17:12:47] Sample sample_0042 provider_error: ProviderError[openai]: 
           OpenAI Responses API error: Connection error. (attempts: 3)
```

**Goal:** Maximize throughput (tokens/second per API key) while ensuring all samples eventually complete. We accept 429s as long as retries succeed.

## Out of Scope (for now)

- Token-based rate limiting (requests/minute is simpler, add TPM later)
- Distributed rate limiting (single process only)
- LIFO queue ordering (use FIFO for now, revisit if needed)
- Per-model rate limit configuration (use header-based adaptive limits)

## Solution

A phased approach, implementing Phase 1 and 2 now:

### Phase 1: Increase Retry Resilience
- Bump `max_retries` default from 3 → 10
- SDK exponential backoff handles transient 429s
- Backoff naturally spreads thundering herd over time

### Phase 2: Header Parsing + Observability  
- Parse rate limit headers after each response
- Store remaining/total in module-level state per API key
- Log rate limit state for visibility (structured, tail-sampled)

### Phase 3 (Future): Proactive Concurrency Control
- Use `trio.CapacityLimiter` to cap in-flight requests
- Dynamically adjust limit based on header info
- Only implement if Phase 1-2 prove insufficient

### Sample-Level Retry
- If a sample fails due to `ProviderError` after exhausting request retries, mark as "failed_retryable"
- Eval loop automatically retries failed samples until all complete or max sample-retry count hit
- Separates request-scale retries (seconds) from sample-scale retries (minutes)

---

## Implementation Details

### Phase 1: Config Changes

```python
# In config/base.py - change default
max_retries: int = 10  # Was: 3

# In dtypes.py Endpoint class - change default  
max_retries: int = 10  # Was: 3
```

### Phase 2: Rate Limit Header Parsing

Rate limit headers by provider:

| Provider | Remaining Requests | Total Limit | Reset Time |
|----------|-------------------|-------------|------------|
| OpenAI | `x-ratelimit-remaining-requests` | `x-ratelimit-limit-requests` | `x-ratelimit-reset-requests` |
| Anthropic | `anthropic-ratelimit-requests-remaining` | `anthropic-ratelimit-requests-limit` | `anthropic-ratelimit-requests-reset` |
| Google | Not consistently exposed | - | - |

```python
# rollouts/_rate_limit.py

"""Rate limit tracking and observability.

Design notes:
- Module-level state keyed by API key (not model - limits are org-wide)
- Each request learns from its own response headers (no blocking for shared state)
- Accept initial thundering herd; headers guide subsequent requests
- Log only on interesting events (near limit, errors) per logging_sucks.md philosophy
- FIFO ordering for now; TODO: consider LIFO if we want recent requests prioritized
- Uses existing rollouts logging (init_rollout_logging quiets httpx/httpcore at WARNING)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any

# Use standard logging - rollouts.logging_utils.init_rollout_logging() configures
# appropriate levels for httpx/httpcore/openai/anthropic (WARNING) vs our code (INFO)
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
    """Return truncated hash for logging (don't log full keys)."""
    if len(api_key) < 8:
        return "***"
    return f"{api_key[:4]}...{api_key[-4:]}"


def update_rate_limit_from_headers(
    api_key: str,
    provider: str,
    headers: dict[str, str],
) -> None:
    """Update rate limit state from response headers.
    
    Called after each API response. Logs only on interesting events:
    - First time we learn the limit
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
    
    # Structured logging on interesting events
    utilization = state.utilization_pct
    
    if is_first_update:
        logger.info(
            "rate_limit_discovered",
            extra={
                "provider": provider,
                "api_key": key_hash,
                "remaining_requests": remaining,
                "total_requests": total,
                "utilization_pct": utilization,
            }
        )
    elif utilization is not None and utilization > 80:
        logger.warning(
            "rate_limit_high_utilization",
            extra={
                "provider": provider,
                "api_key": key_hash,
                "remaining_requests": remaining,
                "total_requests": total,
                "utilization_pct": utilization,
                "reset_time": reset_time,
            }
        )
    elif remaining == 0:
        logger.warning(
            "rate_limit_exhausted",
            extra={
                "provider": provider,
                "api_key": key_hash,
                "total_requests": total,
                "reset_time": reset_time,
            }
        )


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
    - ISO 8601 timestamps
    """
    if not value:
        return None
    
    try:
        # Relative time format: "30s", "1m", etc.
        if value.endswith("s"):
            seconds = float(value[:-1])
            return time.time() + seconds
        elif value.endswith("m"):
            minutes = float(value[:-1])
            return time.time() + minutes * 60
        elif value.endswith("ms"):
            ms = float(value[:-2])
            return time.time() + ms / 1000
        else:
            # Try ISO 8601
            from datetime import datetime
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt.timestamp()
    except (ValueError, TypeError):
        return None
```

### Provider Integration

Each provider parses headers after receiving response.

**OpenAI SDK Header Access:** The OpenAI SDK requires using `with_streaming_response` to access headers from streaming responses. This is a context manager pattern:

```python
# In openai_responses.py - restructured for header access:

from .._rate_limit import update_rate_limit_from_headers

async def stream_openai_responses(actor: Actor, on_chunk: OnChunk) -> tuple[Message, Completion]:
    client = AsyncOpenAI(**client_kwargs)
    
    # Use with_streaming_response to get access to headers
    async with client.responses.with_streaming_response.create(**params) as response:
        # Extract headers BEFORE consuming the stream
        headers = dict(response.headers)
        
        # Now iterate the stream
        stream = response.parse()
        final_message, usage_data = await aggregate_openai_responses_stream(stream, on_chunk)
        
        # Update rate limit state from headers
        update_rate_limit_from_headers(
            api_key=actor.endpoint.api_key,
            provider="openai",
            headers=headers,
        )
    
    return message, completion
```

**Anthropic SDK:** Similar pattern - use the response object to access headers after stream completes.

### Sample-Level Retry

Integrates into existing `evaluate()` function in `rollouts/evaluation.py`. The retry loop wraps the sample evaluation, not the entire eval:

```python
# In rollouts/evaluation.py, modify the evaluate() function:

async def evaluate(
    dataset: Iterator[dict[str, Any]],
    config: EvalConfig,
) -> EvalReport:
    # ... existing sample collection code ...
    
    # NEW: Add retry loop around sample evaluation
    max_sample_retries = getattr(config, 'max_sample_retries', 3)
    pending_samples = list(samples_to_eval)
    all_results: list[Sample] = []
    
    for attempt in range(max_sample_retries + 1):
        if not pending_samples:
            break
        
        if attempt > 0:
            logger.info(
                "retrying_failed_samples",
                extra={
                    "attempt": attempt,
                    "pending_count": len(pending_samples),
                    "completed_count": len(all_results),
                }
            )
            # Wait before retry to let rate limits reset
            await trio.sleep(min(30 * attempt, 120))
        
        # Run current batch (existing concurrent evaluation code)
        batch_results = await _evaluate_samples_concurrent(pending_samples, config)
        
        # Partition: completed vs needs-retry
        newly_completed = []
        still_pending = []
        
        for (sample_id, sample_data), result in zip(pending_samples, batch_results):
            if result.metadata.get("error_type") == "provider_error":
                still_pending.append((sample_id, sample_data))
            else:
                newly_completed.append(result)
        
        all_results.extend(newly_completed)
        pending_samples = still_pending
    
    # Mark remaining as permanent failures
    for sample_id, sample_data in pending_samples:
        all_results.append(Sample(
            id=sample_id,
            input=sample_data,
            metadata={"error": f"Failed after {max_sample_retries + 1} sample retries"},
        ))
    
    # ... existing report generation code ...
```

**Note:** Add `max_sample_retries: int = 3` field to `EvalConfig` dataclass.

---

## Files Modified

1. `rollouts/config/base.py` - Bump `max_retries` default: 3 → 10
2. `rollouts/dtypes.py` - Bump `max_retries` default in Endpoint: 3 → 10, add `max_sample_retries` to EvalConfig
3. `rollouts/_rate_limit.py` - New file: header parsing, state tracking, structured logging
4. `rollouts/providers/openai_responses.py` - Restructure to use `with_streaming_response`, call `update_rate_limit_from_headers`
5. `rollouts/providers/openai_completions.py` - Restructure to use `with_streaming_response`, call `update_rate_limit_from_headers`
6. `rollouts/providers/anthropic.py` - Add header parsing after stream completes
7. `rollouts/evaluation.py` - Add sample-level retry loop in `evaluate()`

---

## Testing

Per grugbrain philosophy: integration tests at cut points, manual testing first, skip unit tests for internals (they break on refactor and test implementation not API).

### Manual Testing (Do This First)

```bash
# Run eval with high concurrency, observe logs
VERBOSE=1 python -m rollouts.eval --max-concurrent=50 --samples=100

# Expected log output when near limits:
# INFO rate_limit_discovered provider=openai remaining=500 total=500
# WARN rate_limit_high_utilization provider=openai remaining=50 total=500 utilization_pct=90.0

# Verify sample-level retry by intentionally using a rate-limited key:
# - Should see some samples fail on first pass
# - Should see "retrying_failed_samples" log
# - Should see all samples eventually complete
```

### Integration Test (After Manual Verification Works)

One integration test that exercises the sample-level retry at the eval API boundary:

```python
# tests/test_eval_retry.py

async def test_sample_level_retry_on_provider_error():
    """Verify that samples failing with ProviderError are retried."""
    from rollouts.evaluation import evaluate
    from rollouts.providers.base import ProviderError
    
    # Create a mock provider that fails N times then succeeds
    fail_count = 0
    
    async def flaky_provider(*args, **kwargs):
        nonlocal fail_count
        fail_count += 1
        if fail_count <= 2:
            raise ProviderError("Rate limited", provider="mock", attempts=3)
        return mock_success_response()
    
    # Patch the provider and run eval
    # ... (inject flaky_provider)
    
    report = await evaluate(dataset, config)
    
    # All samples should eventually succeed
    assert all(s.score is not None for s in report.sample_results)
    # Should have retried
    assert fail_count > len(dataset)
```

### Regression Tests

If bugs are found during manual testing, write a regression test that reproduces the bug first, then fix it. This is the one case where "test first" makes sense.

---

## Future Enhancements (Phase 3+)

1. **Proactive concurrency control** - Use `trio.CapacityLimiter` to cap in-flight requests based on header info
2. **Token-based limiting** - Track input/output tokens per minute (TPM), not just RPM
3. **Adaptive backoff** - If we see many 429s, proactively slow down before SDK retries
4. **Per-provider tuning** - Different strategies for OpenAI (strict limits) vs Anthropic (softer)
5. **Metrics export** - Expose rate limit utilization as Prometheus metrics
6. **LIFO queue ordering** - Consider LIFO if we want to prioritize recent requests during overload
