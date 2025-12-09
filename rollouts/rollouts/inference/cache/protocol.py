"""KV cache protocol - interface for swappable backends.

PagedKVCache and RadixKVCache both implement this protocol.
"""

from typing import Protocol


class KVCacheManager(Protocol):
    """Protocol for KV cache backends.

    Why a protocol (not ABC)?
    - Structural subtyping: implementations don't need to inherit
    - Duck typing: any class with these methods works
    - Cleaner testing: easy to create mock implementations
    """

    def num_free_blocks(self) -> int:
        """Number of free blocks available for allocation."""
        ...

    def can_allocate(self, num_tokens: int) -> bool:
        """Check if we can allocate blocks for a sequence of this length."""
        ...

    def allocate(self, seq_id: int, token_ids: list[int]) -> tuple[int, ...]:
        """Allocate blocks for a sequence. Returns block IDs.

        May return cached blocks if prefix matches (prefix caching).
        """
        ...

    def deallocate(self, seq_id: int) -> None:
        """Free blocks for a sequence."""
        ...

    def append_token(self, seq_id: int, token_id: int) -> int | None:
        """Append token to sequence. Returns new block ID if allocated, else None."""
        ...

    def get_block_ids(self, seq_id: int) -> tuple[int, ...]:
        """Get current block IDs for a sequence."""
        ...
