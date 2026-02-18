"""Functional Llama-style transformer.

Pure functions with weight dicts - no nn.Module inheritance.
Based on rollouts/tools/functional_extractor/llama_functional.py.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig
from ...layers import apply_rotary_pos_emb, compute_rope_embeddings, rms_norm

# Type alias for linear function (F.linear or fp8_linear)
LinearFn = Callable[[Tensor, Tensor], Tensor]


def attention(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    config: ModelConfig,
    linear_fn: LinearFn = F.linear,
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
    q = linear_fn(hidden_states, weights[f"{prefix}.q_proj.weight"])
    k = linear_fn(hidden_states, weights[f"{prefix}.k_proj.weight"])
    v = linear_fn(hidden_states, weights[f"{prefix}.v_proj.weight"])

    assert q.shape == (batch_size, seq_len, n_heads * head_dim), f"q shape mismatch: {q.shape}"
    assert k.shape == (batch_size, seq_len, n_kv_heads * head_dim), f"k shape mismatch: {k.shape}"

    # Reshape to [batch, num_heads, seq_len, head_dim]
    q = q.view(batch_size, seq_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, n_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # QK Norm (normalize after RoPE, per-head L2 norm)
    if config.use_qk_norm:
        q_norm_weight = weights.get(f"{prefix}.q_norm.weight")
        k_norm_weight = weights.get(f"{prefix}.k_norm.weight")
        if q_norm_weight is not None:
            # RMSNorm per head: normalize over head_dim
            q = rms_norm(q, q_norm_weight, config.rms_norm_eps)
            k = rms_norm(k, k_norm_weight, config.rms_norm_eps)

    # Repeat KV for GQA
    if n_kv_groups > 1:
        k = k.repeat_interleave(n_kv_groups, dim=1)
        v = v.repeat_interleave(n_kv_groups, dim=1)

    assert k.shape[1] == n_heads, (
        f"after GQA expansion, k heads should be {n_heads}, got {k.shape[1]}"
    )

    # Scaled dot-product attention
    # TODO: sliding window attention - if config.sliding_window_pattern:
    #   Use layer_idx to determine if this layer uses sliding window or global attention
    #   Pattern like "SSSL" means layers 0,1,2 use sliding window, layer 3 uses global
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, n_heads * head_dim)
    return linear_fn(attn_output, weights[f"{prefix}.o_proj.weight"])


def mlp(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    config: ModelConfig,
    linear_fn: LinearFn = F.linear,
) -> Tensor:
    """MLP: SwiGLU or ReLU².

    SwiGLU: down(silu(gate(x)) * up(x))
    ReLU²:  down(relu(up(x))²)
    """
    assert hidden_states.ndim == 3, f"hidden_states must be 3D, got {hidden_states.shape}"

    prefix = f"layers.{layer_idx}.mlp"

    if config.use_relu2:
        # ReLU² MLP (no gate, just up -> relu² -> down)
        up = linear_fn(hidden_states, weights[f"{prefix}.up_proj.weight"])
        hidden = F.relu(up).square()
        return linear_fn(hidden, weights[f"{prefix}.down_proj.weight"])
    else:
        # SwiGLU MLP
        gate = linear_fn(hidden_states, weights[f"{prefix}.gate_proj.weight"])
        up = linear_fn(hidden_states, weights[f"{prefix}.up_proj.weight"])

        assert gate.shape[-1] == config.mlp_dim, (
            f"gate dim: {gate.shape[-1]}, expected {config.mlp_dim}"
        )

        return linear_fn(F.silu(gate) * up, weights[f"{prefix}.down_proj.weight"])


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    config: ModelConfig,
    linear_fn: LinearFn = F.linear,
) -> Tensor:
    """Single transformer layer."""
    prefix = f"layers.{layer_idx}"

    # Pre-attention norm + attention + residual
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps
    )
    hidden_states = attention(hidden_states, weights, layer_idx, cos, sin, config, linear_fn)
    hidden_states = residual + hidden_states

    # Pre-MLP norm + MLP + residual
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
    )
    hidden_states = mlp(hidden_states, weights, layer_idx, config, linear_fn)
    hidden_states = residual + hidden_states

    return hidden_states


def forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    config: ModelConfig,
    linear_fn: LinearFn = F.linear,
) -> Tensor:
    """Full transformer forward pass.

    Args:
        input_ids: [batch, seq_len] input token ids
        weights: dict of all model weights
        config: model configuration
        linear_fn: Linear function to use (F.linear or fp8_linear)

    Returns:
        logits: [batch, seq_len, vocab_size]
    """
    assert input_ids.ndim == 2, f"input_ids must be 2D (batch, seq), got {input_ids.shape}"

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Embeddings
    # TODO: value embeddings (ResFormer) - add learned value vectors that get added to residual
    #   on alternating layers. See nanochat gpt.py for implementation.
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
        hidden_states = transformer_layer(hidden_states, weights, i, cos, sin, config, linear_fn)

    # Output
    hidden_states = rms_norm(hidden_states, weights["norm.weight"], config.rms_norm_eps)
    # lm_head uses standard F.linear (not FP8) - vocab_size often not divisible by 16
    logits = F.linear(hidden_states, weights["lm_head.weight"])

    # TODO: logit softcap - if config.logit_softcap is not None:
    #   logits = config.logit_softcap * torch.tanh(logits / config.logit_softcap)

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

        # QK Norm weights (per-head normalization)
        if config.use_qk_norm:
            weights[f"{prefix}.self_attn.q_norm.weight"] = torch.ones(
                config.head_dim, device=device, dtype=dtype
            )
            weights[f"{prefix}.self_attn.k_norm.weight"] = torch.ones(
                config.head_dim, device=device, dtype=dtype
            )

        # MLP projections
        if not config.use_relu2:
            # SwiGLU needs gate_proj
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
