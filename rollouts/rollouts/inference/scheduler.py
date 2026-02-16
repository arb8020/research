"""Scheduler for inference engine.

Decides which requests to prefill/decode each step.
All functions are pure — take state in, return new state out.

Design:
- PrefillQueue: requests waiting for prefill
- DecodeSet: requests currently decoding
- schedule_step(): returns next Batch to run
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .core import Batch, Req, make_batch

# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER STATE
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SchedulerState:
    """Immutable scheduler state.

    Fields:
        prefill_queue: Requests waiting for prefill (FIFO order)
        decode_set: Requests currently decoding
        finished: Requests that completed this step
    """

    prefill_queue: tuple[Req, ...]
    decode_set: frozenset[Req]
    finished: tuple[Req, ...]

    def __post_init__(self) -> None:
        # No request should be in both queues
        prefill_uids = {r.uid for r in self.prefill_queue}
        decode_uids = {r.uid for r in self.decode_set}
        assert prefill_uids.isdisjoint(decode_uids), "request in both prefill and decode"


def empty_scheduler_state() -> SchedulerState:
    """Create empty scheduler state."""
    return SchedulerState(
        prefill_queue=(),
        decode_set=frozenset(),
        finished=(),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULER CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration."""

    max_batch_size: int  # Max requests per batch
    max_tokens_per_batch: int  # Max total tokens per batch (for prefill)
    max_seq_len: int  # Max sequence length

    def __post_init__(self) -> None:
        assert self.max_batch_size > 0
        assert self.max_tokens_per_batch > 0
        assert self.max_seq_len > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ScheduleResult:
    """Result of scheduling step.

    Fields:
        batch: Batch to run (or None if nothing to do)
        new_state: Updated scheduler state
    """

    batch: Batch | None
    new_state: SchedulerState


# ═══════════════════════════════════════════════════════════════════════════════
# STATE TRANSITIONS
# ═══════════════════════════════════════════════════════════════════════════════


def add_request(state: SchedulerState, req: Req) -> SchedulerState:
    """Add a new request to the prefill queue."""
    assert req.cached_len == 0, "new request should have cached_len=0"
    assert req.uid not in {r.uid for r in state.prefill_queue}, "duplicate uid in prefill"
    assert req.uid not in {r.uid for r in state.decode_set}, "duplicate uid in decode"

    return SchedulerState(
        prefill_queue=state.prefill_queue + (req,),
        decode_set=state.decode_set,
        finished=(),  # clear finished from previous step
    )


def remove_finished(state: SchedulerState, finished_uids: frozenset[int]) -> SchedulerState:
    """Remove finished requests from decode set."""
    new_decode = frozenset(r for r in state.decode_set if r.uid not in finished_uids)
    finished = tuple(r for r in state.decode_set if r.uid in finished_uids)

    return SchedulerState(
        prefill_queue=state.prefill_queue,
        decode_set=new_decode,
        finished=finished,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCHEDULING LOGIC
# ═══════════════════════════════════════════════════════════════════════════════


def schedule_prefill(
    state: SchedulerState,
    config: SchedulerConfig,
    num_free_pages: int,
    device: torch.device,
    allocate_pages: callable,  # (n: int) -> Tensor
) -> ScheduleResult:
    """Try to schedule a prefill batch.

    Returns batch of requests to prefill, or None if nothing to prefill.
    """
    if not state.prefill_queue:
        return ScheduleResult(batch=None, new_state=state)

    # Select requests that fit
    selected: list[Req] = []
    total_tokens = 0
    pages_needed = 0

    for req in state.prefill_queue:
        # Check batch size limit
        if len(selected) >= config.max_batch_size:
            break

        # Check token budget
        req_tokens = req.extend_len
        if total_tokens + req_tokens > config.max_tokens_per_batch:
            break

        # Check page availability (1 page per token for now)
        req_pages = req_tokens
        if pages_needed + req_pages > num_free_pages:
            break

        # Check sequence length
        if req.device_len > config.max_seq_len:
            # Skip this request (too long)
            continue

        selected.append(req)
        total_tokens += req_tokens
        pages_needed += req_pages

    if not selected:
        return ScheduleResult(batch=None, new_state=state)

    # Allocate pages and build batch
    out_loc = allocate_pages(total_tokens)
    batch = make_batch(
        reqs=tuple(selected),
        phase="prefill",
        out_loc=out_loc,
        device=device,
    )

    # Update state: move selected from prefill to decode
    selected_uids = {r.uid for r in selected}
    new_prefill = tuple(r for r in state.prefill_queue if r.uid not in selected_uids)
    new_decode = state.decode_set | frozenset(selected)

    new_state = SchedulerState(
        prefill_queue=new_prefill,
        decode_set=new_decode,
        finished=(),
    )

    return ScheduleResult(batch=batch, new_state=new_state)


def schedule_decode(
    state: SchedulerState,
    config: SchedulerConfig,
    num_free_pages: int,
    device: torch.device,
    allocate_pages: callable,  # (n: int) -> Tensor
) -> ScheduleResult:
    """Schedule a decode batch.

    Returns batch of requests to decode, or None if nothing to decode.
    """
    if not state.decode_set:
        return ScheduleResult(batch=None, new_state=state)

    # Select up to max_batch_size requests
    # Each decode request needs 1 token processed, 1 page allocated
    decode_list = list(state.decode_set)
    selected = decode_list[: config.max_batch_size]

    # Check page availability
    pages_needed = len(selected)
    if pages_needed > num_free_pages:
        # Reduce batch to fit
        selected = selected[:num_free_pages]

    if not selected:
        return ScheduleResult(batch=None, new_state=state)

    # Allocate pages and build batch
    out_loc = allocate_pages(len(selected))
    batch = make_batch(
        reqs=tuple(selected),
        phase="decode",
        out_loc=out_loc,
        device=device,
    )

    # State unchanged for decode (requests stay in decode_set)
    return ScheduleResult(batch=batch, new_state=state)


def schedule_step(
    state: SchedulerState,
    config: SchedulerConfig,
    num_free_pages: int,
    device: torch.device,
    allocate_pages: callable,
) -> ScheduleResult:
    """Schedule next batch. Prefill has priority over decode.

    Args:
        state: Current scheduler state
        config: Scheduler configuration
        num_free_pages: Available KV cache pages
        device: GPU device
        allocate_pages: Function to allocate pages, (n: int) -> Tensor

    Returns:
        ScheduleResult with batch and new state
    """
    # Try prefill first
    result = schedule_prefill(state, config, num_free_pages, device, allocate_pages)
    if result.batch is not None:
        return result

    # Fall back to decode
    return schedule_decode(state, config, num_free_pages, device, allocate_pages)


# ═══════════════════════════════════════════════════════════════════════════════
# POST-FORWARD UPDATE
# ═══════════════════════════════════════════════════════════════════════════════


def update_after_forward(
    state: SchedulerState,
    batch: Batch,
    next_tokens: Tensor,  # [batch_size], sampled tokens
    eos_token_id: int,
) -> SchedulerState:
    """Update scheduler state after forward pass completes.

    Args:
        state: Current state
        batch: Batch that was just processed
        next_tokens: Sampled tokens for each request
        eos_token_id: Token ID that signals end of sequence

    Returns:
        New state with updated requests
    """
    from .core import req_after_decode_step

    assert len(next_tokens) == len(batch.reqs), "token count mismatch"

    finished_uids: set[int] = set()
    updated_reqs: list[Req] = []

    for req, next_token in zip(batch.reqs, next_tokens.tolist(), strict=False):
        # Update request state
        new_req = req_after_decode_step(req, next_token)
        updated_reqs.append(new_req)

        # Check if finished
        is_eos = next_token == eos_token_id and not req.sampling_params.ignore_eos
        is_max_len = not new_req.can_decode

        if is_eos or is_max_len:
            finished_uids.add(req.uid)

    # Build new decode set with updated requests
    new_decode_set: set[Req] = set()
    finished_reqs: list[Req] = []

    for old_req in state.decode_set:
        # Find updated version
        updated = next((r for r in updated_reqs if r.uid == old_req.uid), None)
        if updated is not None:
            if updated.uid in finished_uids:
                finished_reqs.append(updated)
            else:
                new_decode_set.add(updated)
        else:
            # Request wasn't in this batch, keep as-is
            new_decode_set.add(old_req)

    return SchedulerState(
        prefill_queue=state.prefill_queue,
        decode_set=frozenset(new_decode_set),
        finished=tuple(finished_reqs),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE
# ═══════════════════════════════════════════════════════════════════════════════


def has_pending_work(state: SchedulerState) -> bool:
    """Check if there's any work to do."""
    return bool(state.prefill_queue) or bool(state.decode_set)
