"""Paged KV cache - vLLM-style block allocation.

Adapted from nano-vllm (GeeeekExplorer/nano-vllm).
Key features:
- Fixed-size blocks (default 16 tokens)
- Hash-based prefix caching with xxhash
- Reference counting for block sharing
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np
import xxhash


@dataclass
class Block:
    """A single KV cache block."""

    block_id: int
    ref_count: int = 0
    hash: int = -1  # -1 = incomplete block (can't be cached)
    token_ids: list[int] = field(default_factory=list)

    def update(self, hash: int, token_ids: list[int]) -> None:
        """Update block hash and tokens (when block is complete)."""
        self.hash = hash
        self.token_ids = token_ids

    def reset(self) -> None:
        """Reset block for reuse."""
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


def compute_block_hash(token_ids: list[int], prefix_hash: int = -1) -> int:
    """Compute hash for a complete block.

    Chained hashing: hash depends on prefix, enabling prefix cache lookup.

    Args:
        token_ids: Tokens in this block (must be exactly block_size)
        prefix_hash: Hash of previous block (-1 for first block)

    Returns:
        64-bit hash for this block
    """
    h = xxhash.xxh64()
    if prefix_hash != -1:
        h.update(prefix_hash.to_bytes(8, "little"))
    h.update(np.array(token_ids).tobytes())
    return h.intdigest()


class PagedKVCache:
    """vLLM-style paged KV cache with prefix caching.

    Why a class?
    - Owns GPU memory (the actual K/V tensors, allocated elsewhere)
    - Manages block lifecycle (alloc/dealloc)
    - Tracks sequence→block mappings

    The actual K/V tensors are stored in the model's attention layers.
    This class just manages the block allocation and lookup.
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        assert num_blocks > 0
        assert block_size > 0

        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()

        # seq_id -> list of block_ids
        self.seq_to_blocks: dict[int, list[int]] = {}
        # seq_id -> number of cached tokens (prefix hits)
        self.seq_cached_tokens: dict[int, int] = {}

    # ═══════════════════════════════════════════════════
    # PROTOCOL METHODS
    # ═══════════════════════════════════════════════════

    def num_free_blocks(self) -> int:
        """Number of free blocks available."""
        return len(self.free_block_ids)

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate for a sequence of this length."""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        return num_blocks_needed <= len(self.free_block_ids)

    def allocate(self, seq_id: int, token_ids: list[int]) -> tuple[int, ...]:
        """Allocate blocks for a new sequence.

        Uses prefix caching: if blocks match previous sequences, reuse them.

        Returns:
            Tuple of allocated block IDs
        """
        assert seq_id not in self.seq_to_blocks, f"Sequence {seq_id} already allocated"

        block_ids: list[int] = []
        cached_tokens = 0
        prefix_hash = -1
        cache_miss = False

        num_blocks = (len(token_ids) + self.block_size - 1) // self.block_size

        for i in range(num_blocks):
            # Get tokens for this block
            start = i * self.block_size
            end = min(start + self.block_size, len(token_ids))
            block_tokens = token_ids[start:end]

            # Only complete blocks can be cached
            is_complete = len(block_tokens) == self.block_size

            # Compute hash for complete blocks (needed for both lookup and storage)
            block_hash = -1
            if is_complete:
                block_hash = compute_block_hash(block_tokens, prefix_hash)

            # Try cache lookup (only if we haven't had a miss yet)
            if is_complete and not cache_miss:
                cached_block_id = self.hash_to_block_id.get(block_hash, -1)

                # Check for cache hit
                if cached_block_id != -1:
                    cached_block = self.blocks[cached_block_id]
                    if cached_block.token_ids == block_tokens:
                        # Cache hit! Reuse block
                        if cached_block_id in self.used_block_ids:
                            cached_block.ref_count += 1
                        else:
                            self._allocate_block(cached_block_id)
                        block_ids.append(cached_block_id)
                        cached_tokens += self.block_size
                        prefix_hash = block_hash
                        continue

                # Cache miss - stop trying to lookup
                cache_miss = True

            # Allocate new block
            new_block_id = self.free_block_ids[0]
            block = self._allocate_block(new_block_id)
            block_ids.append(new_block_id)

            # Update hash table for complete blocks
            if is_complete:
                block.update(block_hash, block_tokens)
                self.hash_to_block_id[block_hash] = new_block_id
                prefix_hash = block_hash

        self.seq_to_blocks[seq_id] = block_ids
        self.seq_cached_tokens[seq_id] = cached_tokens

        return tuple(block_ids)

    def deallocate(self, seq_id: int) -> None:
        """Free all blocks for a sequence."""
        if seq_id not in self.seq_to_blocks:
            return

        for block_id in reversed(self.seq_to_blocks[seq_id]):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

        del self.seq_to_blocks[seq_id]
        del self.seq_cached_tokens[seq_id]

    def append_token(self, seq_id: int, token_id: int) -> int | None:
        """Append a token to a sequence.

        Returns:
            New block ID if a new block was allocated, else None
        """
        assert seq_id in self.seq_to_blocks

        block_ids = self.seq_to_blocks[seq_id]
        last_block = self.blocks[block_ids[-1]]

        # Count current tokens in last block
        # (simplified: assumes we track this elsewhere or can compute it)
        # For now, check if we need a new block based on hash state

        if last_block.hash != -1:
            # Last block is complete, need new block
            if not self.free_block_ids:
                raise RuntimeError("No free blocks for token append")
            new_block_id = self.free_block_ids[0]
            self._allocate_block(new_block_id)
            block_ids.append(new_block_id)
            return new_block_id

        return None

    def get_block_ids(self, seq_id: int) -> tuple[int, ...]:
        """Get block IDs for a sequence."""
        return tuple(self.seq_to_blocks.get(seq_id, []))

    def get_cached_tokens(self, seq_id: int) -> int:
        """Get number of cached tokens for a sequence (prefix hits)."""
        return self.seq_cached_tokens.get(seq_id, 0)

    # ═══════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════

    def _allocate_block(self, block_id: int) -> Block:
        """Mark block as used."""
        block = self.blocks[block_id]
        assert block.ref_count == 0, f"Block {block_id} already in use"
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> None:
        """Return block to free list."""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
