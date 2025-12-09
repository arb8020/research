"""Block mask utilities for FlexAttention."""

import torch


def create_causal_block_mask(
    batch_size: int,
    seq_len: int,
    block_size: int = 128,
    device: torch.device | str = "cuda",
):
    """Create causal block mask for FlexAttention.

    Returns a BlockMask object for use with flex_attention().

    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        block_size: FlexAttention block size (default 128)
        device: Device for mask tensors

    Returns:
        BlockMask for causal attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    return create_block_mask(
        causal_mask,
        B=batch_size,
        H=None,  # Same mask for all heads
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=block_size,
        device=device,
    )


def create_document_mask(
    batch_size: int,
    seq_lens: tuple[int, ...],
    total_len: int,
    block_size: int = 128,
    device: torch.device | str = "cuda",
):
    """Create document mask for packed sequences (FlexAttention).

    For packed sequences where multiple documents are concatenated,
    this mask ensures each position only attends to positions within
    the same document.

    Args:
        batch_size: Number of packed batches
        seq_lens: Length of each document in the packed sequence
        total_len: Total length of packed sequence
        block_size: FlexAttention block size
        device: Device for mask tensors

    Returns:
        BlockMask for document-aware causal attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    # Compute document boundaries
    boundaries = [0]
    for length in seq_lens:
        boundaries.append(boundaries[-1] + length)

    def document_mask(b, h, q_idx, kv_idx):
        # Find which document each position belongs to
        # q_idx and kv_idx must be in same document, and causal
        q_doc = 0
        kv_doc = 0
        for i, boundary in enumerate(boundaries[1:]):
            if q_idx >= boundary:
                q_doc = i + 1
            if kv_idx >= boundary:
                kv_doc = i + 1
        return (q_doc == kv_doc) & (q_idx >= kv_idx)

    return create_block_mask(
        document_mask,
        B=batch_size,
        H=None,
        Q_LEN=total_len,
        KV_LEN=total_len,
        BLOCK_SIZE=block_size,
        device=device,
    )
