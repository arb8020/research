"""Radix KV cache - SGLang-style prefix tree.

TODO: Implement this.

Key differences from PagedKVCache:
- Uses a radix tree (prefix tree) instead of hash table
- Better for RL: N rollouts from same prompt share prefix naturally
- O(prefix_length) lookup instead of O(1), but better sharing

For now, this is a stub that implements the protocol but raises NotImplementedError.
"""

from dataclasses import dataclass


@dataclass
class RadixNode:
    """Node in the radix tree."""

    token_ids: list[int]  # Tokens stored at this node
    block_id: int  # KV cache block ID
    children: dict[int, "RadixNode"]  # token -> child node
    ref_count: int = 0


class RadixKVCache:
    """SGLang-style radix tree KV cache.

    TODO: Implement this.

    Why radix tree for RL?
    - N rollouts from same prompt: all share the same prefix path
    - Tree structure makes prefix sharing automatic
    - No hash collisions to worry about

    Why not implemented yet?
    - PagedKVCache is simpler and works
    - Radix tree is optimization for specific RL workloads
    - Can add later when we need better prefix sharing
    """

    def __init__(self, num_blocks: int, block_size: int = 16):
        raise NotImplementedError(
            "RadixKVCache not yet implemented. Use PagedKVCache for now."
        )

    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def can_allocate(self, num_tokens: int) -> bool:
        raise NotImplementedError

    def allocate(self, seq_id: int, token_ids: list[int]) -> tuple[int, ...]:
        raise NotImplementedError

    def deallocate(self, seq_id: int) -> None:
        raise NotImplementedError

    def append_token(self, seq_id: int, token_id: int) -> int | None:
        raise NotImplementedError

    def get_block_ids(self, seq_id: int) -> tuple[int, ...]:
        raise NotImplementedError
