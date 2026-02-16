"""Chunked prefill for long sequences.

Long prompts can exhaust GPU memory or cause OOM during prefill.
Chunked prefill splits long prompts into chunks:

1. Process first chunk (e.g., tokens 0-1024)
2. Process second chunk (tokens 1024-2048), using cached K,V from step 1
3. Continue until all prompt tokens are processed
4. Only then start decoding

This is transparent to the request — it just sees a longer prefill phase.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core import Req


@dataclass(frozen=True)
class ChunkedReq:
    """A request being processed in chunks.

    Wraps a regular Req but tracks chunking progress separately.
    The underlying Req's cached_len advances as chunks complete.
    """

    req: Req
    chunk_size: int
    total_prompt_len: int

    @property
    def is_complete(self) -> bool:
        """True when all prompt tokens have been processed."""
        return self.req.cached_len >= self.total_prompt_len

    @property
    def next_chunk_len(self) -> int:
        """Length of the next chunk to process."""
        remaining = self.total_prompt_len - self.req.cached_len
        return min(self.chunk_size, remaining)

    @property
    def can_decode(self) -> bool:
        """Can only decode after all prompt chunks are processed."""
        return self.is_complete and self.req.can_decode


def should_chunk(prompt_len: int, max_chunk_size: int) -> bool:
    """Check if a prompt should be chunked."""
    return prompt_len > max_chunk_size


def create_chunked_req(
    req: Req,
    chunk_size: int,
) -> ChunkedReq:
    """Create a chunked request from a regular request.

    Args:
        req: Original request (cached_len should be 0)
        chunk_size: Maximum tokens per chunk

    Returns:
        ChunkedReq that will process prompt in chunks
    """
    assert req.cached_len == 0, "Chunked req should start with cached_len=0"
    return ChunkedReq(
        req=req,
        chunk_size=chunk_size,
        total_prompt_len=req.device_len,
    )


def advance_chunked_req(chunked: ChunkedReq, new_req: Req) -> ChunkedReq:
    """Advance chunked request after processing a chunk.

    Args:
        chunked: Current chunked request state
        new_req: Updated Req with advanced cached_len

    Returns:
        New ChunkedReq with updated state
    """
    return ChunkedReq(
        req=new_req,
        chunk_size=chunked.chunk_size,
        total_prompt_len=chunked.total_prompt_len,
    )


@dataclass(frozen=True)
class PrefillBudget:
    """Budget for prefill scheduling.

    Tracks remaining capacity for prefill tokens in a batch.
    """

    max_tokens: int  # Maximum tokens in this prefill batch
    used_tokens: int = 0  # Tokens already scheduled

    @property
    def remaining(self) -> int:
        return self.max_tokens - self.used_tokens

    def can_fit(self, num_tokens: int) -> bool:
        return num_tokens <= self.remaining

    def use(self, num_tokens: int) -> PrefillBudget:
        assert self.can_fit(num_tokens)
        return PrefillBudget(
            max_tokens=self.max_tokens,
            used_tokens=self.used_tokens + num_tokens,
        )


def schedule_prefill_chunk(
    chunked: ChunkedReq,
    budget: PrefillBudget,
) -> tuple[int, PrefillBudget]:
    """Schedule as much of a chunk as fits in the budget.

    Args:
        chunked: Chunked request to schedule
        budget: Remaining prefill budget

    Returns:
        (num_tokens_to_process, updated_budget)
        num_tokens_to_process may be 0 if budget exhausted
    """
    chunk_len = chunked.next_chunk_len
    tokens_to_process = min(chunk_len, budget.remaining)

    if tokens_to_process == 0:
        return 0, budget

    return tokens_to_process, budget.use(tokens_to_process)


class ChunkedPrefillManager:
    """Manages chunked prefill requests.

    Tracks which requests are being processed in chunks and
    coordinates with the scheduler.
    """

    def __init__(self, max_chunk_size: int) -> None:
        self.max_chunk_size = max_chunk_size

        # Requests currently being chunked
        # uid -> ChunkedReq
        self.active_chunks: dict[int, ChunkedReq] = {}

    def maybe_chunk(self, req: Req) -> Req | ChunkedReq:
        """Check if request needs chunking, create ChunkedReq if so."""
        prompt_len = req.device_len
        if should_chunk(prompt_len, self.max_chunk_size):
            chunked = create_chunked_req(req, self.max_chunk_size)
            self.active_chunks[req.uid] = chunked
            return chunked
        return req

    def is_chunking(self, uid: int) -> bool:
        """Check if a request is being processed in chunks."""
        return uid in self.active_chunks

    def get_chunk(self, uid: int) -> ChunkedReq | None:
        """Get chunked request state."""
        return self.active_chunks.get(uid)

    def advance(self, uid: int, new_req: Req) -> None:
        """Advance chunked request after processing a chunk."""
        if uid not in self.active_chunks:
            return

        chunked = self.active_chunks[uid]
        new_chunked = advance_chunked_req(chunked, new_req)

        if new_chunked.is_complete:
            # All chunks processed, remove from tracking
            del self.active_chunks[uid]
        else:
            # More chunks to process
            self.active_chunks[uid] = new_chunked

    def cleanup(self, uid: int) -> None:
        """Remove tracking for a completed/cancelled request."""
        self.active_chunks.pop(uid, None)
