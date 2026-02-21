"""Core data types for inference engine.

Design principles:
- All dataclasses are frozen (immutable)
- State transitions are pure functions that return new state
- No global context — pass everything explicitly
- Assertions document invariants

Derived from mini-sglang (Apache 2.0), rewritten to follow our style.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

# ═══════════════════════════════════════════════════════════════════════════════
# SAMPLING PARAMS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration for a request. Immutable."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 1024
    ignore_eos: bool = False
    return_logprobs: bool = False  # Whether to compute and return logprobs

    def __post_init__(self) -> None:
        assert self.max_tokens > 0, "max_tokens must be positive"
        assert self.top_p > 0.0 and self.top_p <= 1.0, "top_p must be in (0, 1]"

    @property
    def is_greedy(self) -> bool:
        return (self.temperature <= 0.0 or self.top_k == 1) and self.top_p == 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Req:
    """A single inference request. Immutable.

    Fields:
        uid: Unique identifier for this request
        input_ids: All tokens so far (prompt + generated), on CPU
        cached_len: How many tokens have K,V in cache (already processed)
        max_len: Maximum sequence length (prompt + max_tokens)
        sampling_params: Sampling configuration
        table_idx: Index into page table (assigned by scheduler)
        logprobs: Per-token log probabilities for generated tokens (if requested)
    """

    uid: int
    input_ids: Tensor  # CPU tensor, int32
    cached_len: int
    max_len: int
    sampling_params: SamplingParams
    table_idx: int
    logprobs: Tensor | None = None  # CPU tensor, float32, shape [num_generated]

    def __post_init__(self) -> None:
        assert self.input_ids.device == torch.device("cpu"), "input_ids must be on CPU"
        assert self.input_ids.dtype == torch.int32, "input_ids must be int32"
        assert 0 <= self.cached_len <= len(self.input_ids), "cached_len out of bounds"
        assert len(self.input_ids) <= self.max_len, "input_ids exceeds max_len"
        assert self.table_idx >= 0, "table_idx must be non-negative"
        if self.logprobs is not None:
            assert self.logprobs.device == torch.device("cpu"), "logprobs must be on CPU"
            assert self.logprobs.dtype == torch.float32, "logprobs must be float32"

    @property
    def device_len(self) -> int:
        """How many tokens exist in the sequence."""
        return len(self.input_ids)

    @property
    def extend_len(self) -> int:
        """How many tokens need processing in next forward pass."""
        return self.device_len - self.cached_len

    @property
    def remain_len(self) -> int:
        """How many tokens left until max_len."""
        return self.max_len - self.device_len

    @property
    def can_decode(self) -> bool:
        """Can this request generate more tokens?"""
        return self.remain_len > 0


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST STATE TRANSITIONS (pure functions)
# ═══════════════════════════════════════════════════════════════════════════════


def req_after_forward(req: Req) -> Req:
    """Return new Req after a forward pass completes.

    The forward pass processed tokens [cached_len, device_len).
    Now those tokens are cached.
    """
    assert req.extend_len > 0, "nothing to process"

    return Req(
        uid=req.uid,
        input_ids=req.input_ids,
        cached_len=req.device_len,  # all current tokens are now cached
        max_len=req.max_len,
        sampling_params=req.sampling_params,
        table_idx=req.table_idx,
        logprobs=req.logprobs,
    )


def req_append_token(req: Req, next_token: int, logprob: float | None = None) -> Req:
    """Return new Req with token appended.

    Called after sampling. The new token is NOT yet cached
    (will be processed in next forward pass).

    Args:
        req: Current request state
        next_token: Sampled token ID
        logprob: Log probability of the sampled token (if requested)
    """
    assert req.can_decode, "request cannot generate more tokens"

    new_token_tensor = torch.tensor([next_token], dtype=torch.int32)
    new_input_ids = torch.cat([req.input_ids, new_token_tensor])

    # Append logprob if provided
    new_logprobs = req.logprobs
    if logprob is not None:
        logprob_tensor = torch.tensor([logprob], dtype=torch.float32)
        if req.logprobs is None:
            new_logprobs = logprob_tensor
        else:
            new_logprobs = torch.cat([req.logprobs, logprob_tensor])

    return Req(
        uid=req.uid,
        input_ids=new_input_ids,
        cached_len=req.cached_len,  # unchanged — new token not yet processed
        max_len=req.max_len,
        sampling_params=req.sampling_params,
        table_idx=req.table_idx,
        logprobs=new_logprobs,
    )


def req_after_decode_step(req: Req, next_token: int, logprob: float | None = None) -> Req:
    """Convenience: forward completed, then append token.

    This is what happens after a decode step:
    1. Forward pass caches the current last token
    2. Sampling produces next token
    3. Next token is appended (not yet cached)

    Args:
        req: Current request state
        next_token: Sampled token ID
        logprob: Log probability of the sampled token (if requested)
    """
    req = req_after_forward(req)
    req = req_append_token(req, next_token, logprob)
    return req


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Batch:
    """A batch of requests for one forward pass. Immutable, fully initialized.

    Fields:
        reqs: Tuple of requests (immutable sequence)
        phase: Whether this is prefill or decode
        input_ids: Token IDs to process, shape [total_tokens]
        positions: Position of each token, shape [total_tokens]
        out_loc: Where to write KV cache, shape [total_tokens]
    """

    reqs: tuple[Req, ...]
    phase: Literal["prefill", "decode"]
    input_ids: Tensor  # GPU tensor
    positions: Tensor  # GPU tensor
    out_loc: Tensor  # GPU tensor

    def __post_init__(self) -> None:
        assert len(self.reqs) > 0, "batch cannot be empty"
        assert self.phase in ("prefill", "decode"), f"invalid phase: {self.phase}"

        # All tensors should have same length
        total_tokens = sum(r.extend_len for r in self.reqs)
        assert len(self.input_ids) == total_tokens, "input_ids length mismatch"
        assert len(self.positions) == total_tokens, "positions length mismatch"
        assert len(self.out_loc) == total_tokens, "out_loc length mismatch"

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def size(self) -> int:
        """Number of requests in batch."""
        return len(self.reqs)

    @property
    def num_tokens(self) -> int:
        """Total tokens to process."""
        return len(self.input_ids)


# ═══════════════════════════════════════════════════════════════════════════════
# BATCH CONSTRUCTION (pure functions)
# ═══════════════════════════════════════════════════════════════════════════════


def make_batch(
    reqs: tuple[Req, ...],
    phase: Literal["prefill", "decode"],
    out_loc: Tensor,
    device: torch.device,
) -> Batch:
    """Construct a Batch from requests.

    Args:
        reqs: Requests to batch together
        phase: prefill or decode
        out_loc: Pre-allocated KV cache locations
        device: GPU device for tensors

    Returns:
        Fully initialized Batch
    """
    assert len(reqs) > 0, "cannot make empty batch"

    input_ids_list: list[Tensor] = []
    positions_list: list[Tensor] = []

    for req in reqs:
        extend_len = req.extend_len
        assert extend_len > 0, f"req {req.uid} has nothing to process"

        # Tokens to process: [cached_len, device_len)
        tokens = req.input_ids[req.cached_len : req.device_len]
        input_ids_list.append(tokens)

        # Positions for these tokens
        pos = torch.arange(req.cached_len, req.device_len, dtype=torch.int32)
        positions_list.append(pos)

    input_ids = torch.cat(input_ids_list).to(device)
    positions = torch.cat(positions_list).to(device)

    assert len(out_loc) == len(input_ids), "out_loc size mismatch"

    return Batch(
        reqs=reqs,
        phase=phase,
        input_ids=input_ids,
        positions=positions,
        out_loc=out_loc,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


def create_req(
    uid: int,
    prompt_ids: list[int],
    sampling_params: SamplingParams,
    table_idx: int,
) -> Req:
    """Create a new request from a prompt.

    Args:
        uid: Unique request ID
        prompt_ids: Token IDs for the prompt
        sampling_params: Sampling configuration
        table_idx: Assigned page table slot

    Returns:
        New Req ready for prefill (cached_len=0)
    """
    assert len(prompt_ids) > 0, "prompt cannot be empty"

    input_ids = torch.tensor(prompt_ids, dtype=torch.int32)
    max_len = len(prompt_ids) + sampling_params.max_tokens

    return Req(
        uid=uid,
        input_ids=input_ids,
        cached_len=0,
        max_len=max_len,
        sampling_params=sampling_params,
        table_idx=table_idx,
    )
