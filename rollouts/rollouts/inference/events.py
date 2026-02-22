"""Wide event logging for inference engine.

Implements the "one event per request" pattern from loggingsucks.com.
Each request emits a single structured event when it completes,
containing all timing and context data accumulated during its lifecycle.

Usage:
    tracker = RequestTracker()

    # When request arrives
    tracker.start_request(uid, prompt_len=len(prompt_ids))

    # When prefill completes (first token generated)
    tracker.record_first_token(uid)

    # When request finishes
    event = tracker.finish_request(uid, output_len=output_len)
    logger.info(event)  # or emit to structured logging backend

The event dict can be:
- Logged as JSON for analysis
- Sent to observability backends (Honeycomb, etc.)
- Aggregated for metrics dashboards
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class RequestTiming:
    """Timing data accumulated during request lifecycle.

    All times are in seconds (float).
    """

    # Request lifecycle
    start_time: float = 0.0
    first_token_time: float | None = None
    end_time: float | None = None

    # Batch participation
    num_prefill_batches: int = 0
    num_decode_batches: int = 0

    # Context
    prompt_len: int = 0
    output_len: int = 0
    cached_len: int = 0  # Prefix cache hit

    # Batch sizes seen (for analyzing batching efficiency)
    prefill_batch_sizes: list[int] = field(default_factory=list)
    decode_batch_sizes: list[int] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float | None:
        """Time to first token in milliseconds."""
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.start_time) * 1000

    @property
    def e2e_ms(self) -> float | None:
        """End-to-end latency in milliseconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000

    @property
    def tpot_ms(self) -> float | None:
        """Time per output token (decode phase) in milliseconds.

        Excludes first token (TTFT). Returns None if < 2 tokens generated.
        """
        if self.first_token_time is None or self.end_time is None:
            return None
        if self.output_len <= 1:
            return None
        decode_time_ms = (self.end_time - self.first_token_time) * 1000
        return decode_time_ms / (self.output_len - 1)


class RequestTracker:
    """Tracks timing data for in-flight requests.

    Thread-safe for single-writer (engine) usage.
    """

    def __init__(self) -> None:
        self._timings: dict[int, RequestTiming] = {}

    def start_request(
        self,
        uid: int,
        prompt_len: int,
        cached_len: int = 0,
    ) -> None:
        """Record request start."""
        self._timings[uid] = RequestTiming(
            start_time=time.perf_counter(),
            prompt_len=prompt_len,
            cached_len=cached_len,
        )

    def record_first_token(self, uid: int) -> None:
        """Record when first token is generated (prefill complete)."""
        timing = self._timings.get(uid)
        if timing is not None and timing.first_token_time is None:
            timing.first_token_time = time.perf_counter()

    def record_batch(self, uid: int, batch_size: int, is_prefill: bool) -> None:
        """Record batch participation."""
        timing = self._timings.get(uid)
        if timing is None:
            return

        if is_prefill:
            timing.num_prefill_batches += 1
            timing.prefill_batch_sizes.append(batch_size)
        else:
            timing.num_decode_batches += 1
            timing.decode_batch_sizes.append(batch_size)

    def finish_request(
        self,
        uid: int,
        output_len: int,
        finish_reason: str = "stop",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Finish tracking and return the wide event.

        Returns a structured dict suitable for JSON logging.
        """
        timing = self._timings.pop(uid, None)
        if timing is None:
            return {"uid": uid, "error": "no_timing_data"}

        timing.end_time = time.perf_counter()
        timing.output_len = output_len

        event = {
            "event_type": "request_complete",
            "uid": uid,
            # Timing (ms)
            "ttft_ms": timing.ttft_ms,
            "tpot_ms": timing.tpot_ms,
            "e2e_ms": timing.e2e_ms,
            # Lengths
            "prompt_len": timing.prompt_len,
            "output_len": timing.output_len,
            "cached_len": timing.cached_len,
            "total_len": timing.prompt_len + timing.output_len,
            # Batching
            "num_prefill_batches": timing.num_prefill_batches,
            "num_decode_batches": timing.num_decode_batches,
            "total_batches": timing.num_prefill_batches + timing.num_decode_batches,
            # Outcome
            "finish_reason": finish_reason,
        }

        # Add average batch sizes if available
        if timing.prefill_batch_sizes:
            event["avg_prefill_batch_size"] = sum(timing.prefill_batch_sizes) / len(
                timing.prefill_batch_sizes
            )
        if timing.decode_batch_sizes:
            event["avg_decode_batch_size"] = sum(timing.decode_batch_sizes) / len(
                timing.decode_batch_sizes
            )

        # Merge any extra context
        if extra:
            event.update(extra)

        return event

    def cancel_request(self, uid: int) -> None:
        """Remove request without emitting event."""
        self._timings.pop(uid, None)


def emit_wide_event(event: dict[str, Any]) -> None:
    """Emit a wide event to the logging system.

    Uses structured logging (JSON) for easy parsing.
    """
    # Use the inference.events logger for filtering
    logger.info(event)
