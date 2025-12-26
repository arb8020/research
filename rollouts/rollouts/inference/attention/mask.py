"""Block mask utilities for FlexAttention.

Provides composable mask functions for use with PyTorch's flex_attention.
Masks can be combined using & (and) and | (or) operators.

References:
- PyTorch FlexAttention: https://pytorch.org/blog/flexattention/
- attention-gym: https://github.com/meta-pytorch/attention-gym
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask


def create_causal_block_mask(
    batch_size: int,
    seq_len: int,
    block_size: int = 128,
    device: torch.device | str = "cuda",
) -> BlockMask:
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

    def causal_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
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


def create_sliding_window_causal_mask(
    batch_size: int,
    seq_len: int,
    window_size: int,
    block_size: int = 128,
    device: torch.device | str = "cuda",
) -> BlockMask:
    """Create causal sliding window block mask for FlexAttention.

    Combines causal masking with sliding window attention. Each query token
    can only attend to:
    1. Tokens that come before it (causal)
    2. Tokens within window_size positions (sliding window)

    Used by models like Mistral, Qwen2.5, etc.

    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        window_size: Size of the sliding window (e.g., 4096 for Qwen2.5)
        block_size: FlexAttention block size (default 128)
        device: Device for mask tensors

    Returns:
        BlockMask for causal sliding window attention
    """
    from torch.nn.attention.flex_attention import create_block_mask

    def causal_sliding_window_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
        # Causal: can only attend to positions <= current position
        causal = q_idx >= kv_idx
        # Sliding window: can only attend to positions within window_size
        # q_idx - kv_idx gives the distance; must be < window_size
        in_window = (q_idx - kv_idx) < window_size
        return causal & in_window

    return create_block_mask(
        causal_sliding_window_mask,
        B=batch_size,
        H=None,  # Same mask for all heads
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        BLOCK_SIZE=block_size,
        device=device,
    )


def create_attention_mask(
    batch_size: int,
    seq_len: int,
    sliding_window: int | None = None,
    block_size: int = 128,
    device: torch.device | str = "cuda",
) -> BlockMask:
    """Create appropriate block mask based on attention configuration.

    Factory function that returns the right mask type based on parameters.

    Args:
        batch_size: Number of sequences in batch
        seq_len: Sequence length
        sliding_window: If set, use sliding window attention with this window size
        block_size: FlexAttention block size (default 128)
        device: Device for mask tensors

    Returns:
        BlockMask for the configured attention type
    """
    if sliding_window is not None:
        return create_sliding_window_causal_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            window_size=sliding_window,
            block_size=block_size,
            device=device,
        )
    else:
        return create_causal_block_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            block_size=block_size,
            device=device,
        )


def create_document_mask(
    batch_size: int,
    seq_lens: tuple[int, ...],
    total_len: int,
    block_size: int = 128,
    device: torch.device | str = "cuda",
) -> BlockMask:
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

    def document_mask(b: int, h: int, q_idx: int, kv_idx: int) -> bool:
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
