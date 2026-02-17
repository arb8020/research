"""Functional Llama-style transformer.

Pure functions with weight dicts - no nn.Module inheritance.
Based on rollouts/tools/functional_extractor/llama_functional.py.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    assert x.ndim >= 2, f"x must be at least 2D, got {x.shape}"
    assert weight.ndim == 1, f"weight must be 1D, got {weight.shape}"
    assert x.shape[-1] == weight.shape[0], (
        f"dim mismatch: x={x.shape[-1]}, weight={weight.shape[0]}"
    )

    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normed.to(x.dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, Tensor]:
    """Apply RoPE to query and key.

    Args:
        q: [batch, n_heads, seq_len, head_dim]
        k: [batch, n_kv_heads, seq_len, head_dim]
        cos: [seq_len, head_dim]
        sin: [seq_len, head_dim]
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
    """Compute RoPE cos/sin embeddings."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def attention(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    config: ModelConfig,
) -> Tensor:
    """Self-attention with RoPE and GQA."""
    assert hidden_states.ndim == 3, (
        f"hidden_states must be 3D (batch, seq, dim), got {hidden_states.shape}"
    )

    batch_size, seq_len, dim = hidden_states.shape
    prefix = f"layers.{layer_idx}.self_attn"

    n_heads = config.n_heads
    n_kv_heads = config.n_kv_heads
    head_dim = config.head_dim
    n_kv_groups = n_heads // n_kv_heads

    assert dim == config.dim, f"hidden dim mismatch: got {dim}, expected {config.dim}"
    assert n_heads % n_kv_heads == 0, (
        f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})"
    )

    # Project Q, K, V (no bias in Llama)
    q = F.linear(hidden_states, weights[f"{prefix}.q_proj.weight"])
    k = F.linear(hidden_states, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden_states, weights[f"{prefix}.v_proj.weight"])

    assert q.shape == (batch_size, seq_len, n_heads * head_dim), f"q shape mismatch: {q.shape}"
    assert k.shape == (batch_size, seq_len, n_kv_heads * head_dim), f"k shape mismatch: {k.shape}"

    # Reshape to [batch, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat KV for GQA
    if n_kv_groups > 1:
        k = k.repeat_interleave(n_kv_groups, dim=1)
        v = v.repeat_interleave(n_kv_groups, dim=1)

    assert k.shape[1] == n_heads, (
        f"after GQA expansion, k heads should be {n_heads}, got {k.shape[1]}"
    )

    # Scaled dot-product attention
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, n_heads * head_dim)
    return F.linear(attn_output, weights[f"{prefix}.o_proj.weight"])


def mlp(
    hidden_states: Tensor, weights: dict[str, Tensor], layer_idx: int, config: ModelConfig
) -> Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""
    assert hidden_states.ndim == 3, f"hidden_states must be 3D, got {hidden_states.shape}"

    prefix = f"layers.{layer_idx}.mlp"
    gate = F.linear(hidden_states, weights[f"{prefix}.gate_proj.weight"])
    up = F.linear(hidden_states, weights[f"{prefix}.up_proj.weight"])

    assert gate.shape[-1] == config.mlp_dim, (
        f"gate dim: {gate.shape[-1]}, expected {config.mlp_dim}"
    )

    return F.linear(F.silu(gate) * up, weights[f"{prefix}.down_proj.weight"])


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    config: ModelConfig,
) -> Tensor:
    """Single transformer layer."""
    prefix = f"layers.{layer_idx}"

    # Pre-attention norm + attention + residual
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps
    )
    hidden_states = attention(hidden_states, weights, layer_idx, cos, sin, config)
    hidden_states = residual + hidden_states

    # Pre-MLP norm + MLP + residual
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
    )
    hidden_states = mlp(hidden_states, weights, layer_idx, config)
    hidden_states = residual + hidden_states

    return hidden_states


def forward(input_ids: Tensor, weights: dict[str, Tensor], config: ModelConfig) -> Tensor:
    """Full transformer forward pass.

    Args:
        input_ids: [batch, seq_len] input token ids
        weights: dict of all model weights
        config: model configuration

    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    assert input_ids.ndim == 2, f"input_ids must be 2D (batch, seq), got {input_ids.shape}"

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Embeddings
    hidden_states = F.embedding(input_ids, weights["embed_tokens.weight"])
    assert hidden_states.shape == (batch_size, seq_len, config.dim), (
        f"embed shape: {hidden_states.shape}"
    )

    # RoPE
    cos, sin = compute_rope_embeddings(
        seq_len, config.head_dim, device, config.rope_theta, dtype=hidden_states.dtype
    )
    assert cos.shape == (seq_len, config.head_dim), (
        f"cos shape: {cos.shape}, expected ({seq_len}, {config.head_dim})"
    )

    # Layers
    for i in range(config.n_layers):
        hidden_states = transformer_layer(hidden_states, weights, i, cos, sin, config)

    # Output
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)
    logits = F.linear(hidden_states, weights["lm_head.weight"])

    assert logits.shape == (batch_size, seq_len, config.vocab_size), f"logits shape: {logits.shape}"
    return logits


def init_weights(
    config: ModelConfig, device: torch.device, dtype: torch.dtype
) -> dict[str, Tensor]:
    """Initialize random model weights.

    Uses small init scale (0.02) following GPT-2/Llama convention.
    """
    weights = {}
    init_scale = 0.02

    # Embedding
    weights["embed_tokens.weight"] = (
        torch.randn(config.vocab_size, config.dim, device=device, dtype=dtype) * init_scale
    )

    # Layers
    for i in range(config.n_layers):
        prefix = f"layers.{i}"

        # Layer norms (init to 1)
        weights[f"{prefix}.input_layernorm.weight"] = torch.ones(
            config.dim, device=device, dtype=dtype
        )
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.ones(
            config.dim, device=device, dtype=dtype
        )

        # Attention projections
        weights[f"{prefix}.self_attn.q_proj.weight"] = (
            torch.randn(config.n_heads * config.head_dim, config.dim, device=device, dtype=dtype)
            * init_scale
        )
        weights[f"{prefix}.self_attn.k_proj.weight"] = (
            torch.randn(config.n_kv_heads * config.head_dim, config.dim, device=device, dtype=dtype)
            * init_scale
        )
        weights[f"{prefix}.self_attn.v_proj.weight"] = (
            torch.randn(config.n_kv_heads * config.head_dim, config.dim, device=device, dtype=dtype)
            * init_scale
        )
        weights[f"{prefix}.self_attn.o_proj.weight"] = (
            torch.randn(config.dim, config.n_heads * config.head_dim, device=device, dtype=dtype)
            * init_scale
        )

        # MLP projections
        weights[f"{prefix}.mlp.gate_proj.weight"] = (
            torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
        )
        weights[f"{prefix}.mlp.up_proj.weight"] = (
            torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
        )
        weights[f"{prefix}.mlp.down_proj.weight"] = (
            torch.randn(config.dim, config.mlp_dim, device=device, dtype=dtype) * init_scale
        )

    # Output
    weights["norm.weight"] = torch.ones(config.dim, device=device, dtype=dtype)
    weights["lm_head.weight"] = (
        torch.randn(config.vocab_size, config.dim, device=device, dtype=dtype) * init_scale
    )

    # Enable gradients
    for w in weights.values():
        w.requires_grad_(True)

    return weights


def count_parameters(weights: dict[str, Tensor]) -> int:
    """Count total parameters in weight dict."""
    return sum(w.numel() for w in weights.values())
