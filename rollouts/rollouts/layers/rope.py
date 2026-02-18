"""Rotary Position Embeddings (RoPE)."""

import torch
from torch import Tensor


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims for RoPE.

    Splits x into two halves along last dim, swaps them, and negates the first half.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor
) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key.

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]

    Returns:
        (q_rotated, k_rotated) with same shapes as inputs
    """
    assert q.ndim == 4, f"q must be 4D (batch, heads, seq, head_dim), got {q.shape}"
    assert k.ndim == 4, f"k must be 4D (batch, heads, seq, head_dim), got {k.shape}"
    assert cos.ndim == 2, f"cos must be 2D (seq, head_dim), got {cos.shape}"
    assert sin.ndim == 2, f"sin must be 2D (seq, head_dim), got {sin.shape}"
    assert q.shape[2] == cos.shape[0], f"seq_len mismatch: q={q.shape[2]}, cos={cos.shape[0]}"
    assert q.shape[3] == cos.shape[1], f"head_dim mismatch: q={q.shape[3]}, cos={cos.shape[1]}"

    # Broadcast cos/sin to [1, 1, seq_len, head_dim] for batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_rope_embeddings(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Compute RoPE cos/sin embeddings.

    Args:
        seq_len: Sequence length
        head_dim: Head dimension (must be even)
        device: Device for output tensors
        theta: RoPE base frequency (default 10000)
        dtype: Output dtype

    Returns:
        (cos, sin) each of shape [seq_len, head_dim]
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)
