"""Context builder - wires PagedKVCache to InferenceContext.

Pure functions that convert block allocator output to attention context.
"""

from rollouts.inference.cache.paged import PagedKVCache
from rollouts.inference.types import InferenceContext


def build_prefill_context(
    cache: PagedKVCache,
    seq_ids: list[int],
    seq_lens: list[int],
    block_tables: list[tuple[int, ...]],
) -> InferenceContext:
    """Build context for prefill (processing prompt).

    Args:
        cache: Block allocator (for block_size)
        seq_ids: Sequence IDs being prefilled
        seq_lens: Length of each sequence
        block_tables: Block IDs for each sequence (from cache.allocate())

    Returns:
        InferenceContext for prefill forward pass
    """
    block_size = cache.block_size

    # Build slot mapping: [total_tokens] -> cache slot
    # For prefill, we map each token position to its cache slot
    slot_mapping: list[int] = []

    for _seq_idx, (seq_len, blocks) in enumerate(zip(seq_lens, block_tables, strict=False)):
        for pos in range(seq_len):
            block_idx = pos // block_size
            offset = pos % block_size
            if block_idx < len(blocks):
                slot = blocks[block_idx] * block_size + offset
                slot_mapping.append(slot)
            else:
                # Should not happen if allocation is correct
                slot_mapping.append(-1)

    return InferenceContext(
        is_prefill=True,
        slot_mapping=tuple(slot_mapping),
        block_tables=tuple(block_tables),
        seq_lens=tuple(seq_lens),
        max_seq_len=max(seq_lens) if seq_lens else 0,
    )


def build_decode_context(
    cache: PagedKVCache,
    seq_ids: list[int],
    seq_lens: list[int],
    block_tables: list[tuple[int, ...]],
) -> InferenceContext:
    """Build context for decode (generating one token per sequence).

    Args:
        cache: Block allocator (for block_size)
        seq_ids: Sequence IDs being decoded
        seq_lens: Current length of each sequence (including new token position)
        block_tables: Block IDs for each sequence

    Returns:
        InferenceContext for decode forward pass
    """
    block_size = cache.block_size

    # For decode, each sequence generates 1 token
    # slot_mapping has 1 entry per sequence (where to store the new K/V)
    slot_mapping: list[int] = []

    for seq_len, blocks in zip(seq_lens, block_tables, strict=False):
        # New token goes at position seq_len - 1 (0-indexed)
        pos = seq_len - 1
        block_idx = pos // block_size
        offset = pos % block_size
        if block_idx < len(blocks):
            slot = blocks[block_idx] * block_size + offset
            slot_mapping.append(slot)
        else:
            slot_mapping.append(-1)

    return InferenceContext(
        is_prefill=False,
        slot_mapping=tuple(slot_mapping),
        block_tables=tuple(block_tables),
        seq_lens=tuple(seq_lens),
        max_seq_len=max(seq_lens) if seq_lens else 0,
    )


def allocate_and_build_context(
    cache: PagedKVCache,
    seq_id: int,
    token_ids: list[int],
) -> tuple[InferenceContext, int]:
    """Allocate blocks and build prefill context for a single sequence.

    Convenience function that combines allocation + context building.

    Args:
        cache: Block allocator
        seq_id: Sequence ID
        token_ids: Prompt tokens

    Returns:
        (context, cached_tokens) - context for prefill, number of prefix-cached tokens
    """
    block_ids = cache.allocate(seq_id, token_ids)
    cached_tokens = cache.get_cached_tokens(seq_id)

    ctx = build_prefill_context(
        cache=cache,
        seq_ids=[seq_id],
        seq_lens=[len(token_ids)],
        block_tables=[block_ids],
    )

    return ctx, cached_tokens


def extend_and_build_context(
    cache: PagedKVCache,
    seq_id: int,
    new_token: int,
    current_len: int,
) -> InferenceContext:
    """Extend sequence by one token and build decode context.

    Args:
        cache: Block allocator
        seq_id: Sequence ID
        new_token: Token being generated
        current_len: Current sequence length (before this token)

    Returns:
        InferenceContext for decode forward pass
    """
    # Append may allocate a new block if current block is full
    cache.append_token(seq_id, new_token)

    block_ids = cache.get_block_ids(seq_id)
    new_len = current_len + 1

    return build_decode_context(
        cache=cache,
        seq_ids=[seq_id],
        seq_lens=[new_len],
        block_tables=[block_ids],
    )
