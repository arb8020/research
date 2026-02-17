"""Functional layer implementations with multiple variants.

Each layer type has:
- A main function that dispatches based on config
- Variant-specific implementations
- Weight initialization helpers

Activations (from nmoe):
- swiglu: SiLU(gate) * up - standard Llama
- relu_squared: ReLU(x)² - 2-weight only, no gate
- squared_reglu: ReLU(gate)² * up - gated squared relu
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from ..config import ModelConfig

# Type alias for activation variants
Activation = Literal["swiglu", "relu_squared", "squared_reglu"]


# =============================================================================
# MLP Variants
# =============================================================================


def mlp_swiglu(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x)).

    Standard Llama/Llama2/Llama3 activation. Uses 3 weight matrices.
    """
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    return F.linear(F.silu(gate) * up, down_weight)


def mlp_relu_squared(
    hidden_states: Tensor,
    w1_weight: Tensor,
    w2_weight: Tensor,
) -> Tensor:
    """ReLU² MLP: down(relu(up(x))²).

    Uses only 2 weight matrices (no gate). From Primer paper.
    More parameter efficient but slightly worse quality.
    """
    h = F.linear(hidden_states, w1_weight)
    return F.linear(F.relu(h) ** 2, w2_weight)


def mlp_squared_reglu(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Squared ReGLU MLP: down(relu(gate(x))² * up(x)).

    Gated variant with squared relu. Used in some MoE configs.
    """
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    return F.linear(F.relu(gate) ** 2 * up, down_weight)


def mlp(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    config: ModelConfig,
    activation: Activation = "swiglu",
) -> Tensor:
    """MLP forward pass with configurable activation.

    Args:
        hidden_states: [batch, seq, dim]
        weights: weight dict
        layer_idx: layer index for weight lookup
        config: model config
        activation: one of "swiglu", "relu_squared", "squared_reglu"

    Returns:
        [batch, seq, dim]
    """
    assert hidden_states.ndim == 3, f"hidden_states must be 3D, got {hidden_states.shape}"

    prefix = f"layers.{layer_idx}.mlp"

    if activation == "swiglu":
        return mlp_swiglu(
            hidden_states,
            weights[f"{prefix}.gate_proj.weight"],
            weights[f"{prefix}.up_proj.weight"],
            weights[f"{prefix}.down_proj.weight"],
        )
    elif activation == "relu_squared":
        # relu_squared uses w1/w2 naming (2 weights only)
        return mlp_relu_squared(
            hidden_states,
            weights[f"{prefix}.w1.weight"],
            weights[f"{prefix}.w2.weight"],
        )
    elif activation == "squared_reglu":
        return mlp_squared_reglu(
            hidden_states,
            weights[f"{prefix}.gate_proj.weight"],
            weights[f"{prefix}.up_proj.weight"],
            weights[f"{prefix}.down_proj.weight"],
        )
    else:
        raise ValueError(f"Unknown activation: {activation}")


# =============================================================================
# Attention Variants
# =============================================================================


def attention_sdpa(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = True,
    window_size: int | None = None,
) -> Tensor:
    """Standard scaled dot-product attention.

    Args:
        q: [batch, n_heads, seq, head_dim]
        k: [batch, n_kv_heads, seq, head_dim]
        v: [batch, n_kv_heads, seq, head_dim]
        is_causal: apply causal mask
        window_size: if set, apply sliding window attention

    Returns:
        [batch, n_heads, seq, head_dim]
    """
    if window_size is not None:
        # Create sliding window + causal mask
        seq_len = q.shape[2]
        # Positions where attention is allowed: j <= i and j > i - window_size
        i = torch.arange(seq_len, device=q.device)[:, None]
        j = torch.arange(seq_len, device=q.device)[None, :]
        mask = (j <= i) & (j > i - window_size)
        # Convert to additive mask for SDPA
        attn_mask = torch.where(mask, 0.0, float("-inf"))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
        return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


# =============================================================================
# Weight Initialization
# =============================================================================


# =============================================================================
# Multi-head Latent Attention (MLA) - DeepSeek-V2 style
# =============================================================================


def mla_compress_kv(
    hidden_states: Tensor,
    kv_a_proj_weight: Tensor,
    kv_a_layernorm_weight: Tensor,
    kv_b_proj_weight: Tensor,
    eps: float = 1e-5,
) -> tuple[Tensor, Tensor]:
    """Compress KV using low-rank projection (MLA).

    DeepSeek-V2 style: K,V share a compressed latent representation.
    Reduces KV cache from 2 * n_heads * head_dim to kv_lora_rank.

    Args:
        hidden_states: [batch, seq, dim]
        kv_a_proj_weight: [kv_lora_rank, dim] - down projection
        kv_a_layernorm_weight: [kv_lora_rank] - norm weight
        kv_b_proj_weight: [n_kv_heads * (head_dim + head_dim), kv_lora_rank] - up projection

    Returns:
        k: [batch, seq, n_kv_heads, head_dim]
        v: [batch, seq, n_kv_heads, head_dim]
    """
    batch, seq, _ = hidden_states.shape

    # Down project to latent
    latent = F.linear(hidden_states, kv_a_proj_weight)  # [batch, seq, kv_lora_rank]

    # RMSNorm on latent
    latent_fp32 = latent.to(torch.float32)
    variance = latent_fp32.pow(2).mean(-1, keepdim=True)
    latent = kv_a_layernorm_weight * (latent_fp32 * torch.rsqrt(variance + eps)).to(latent.dtype)

    # Up project to K, V
    kv = F.linear(latent, kv_b_proj_weight)  # [batch, seq, n_kv_heads * 2 * head_dim]

    # Split into K and V
    n_kv_heads_x_head_dim = kv.shape[-1] // 2
    k, v = kv.split(n_kv_heads_x_head_dim, dim=-1)

    return k, v


def attention_mla(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    eps: float = 1e-5,
) -> Tensor:
    """Multi-head Latent Attention (DeepSeek-V2 style).

    Key differences from standard attention:
    - K,V share a compressed latent (reduces KV cache)
    - RoPE applied only to part of Q,K (rope_head_dim)
    - Rest of Q,K is "nope" (no position encoding)

    Args:
        hidden_states: [batch, seq, dim]
        weights: weight dict
        layer_idx: layer index
        cos, sin: RoPE embeddings [seq, rope_head_dim]
        n_heads: number of query heads
        n_kv_heads: number of KV heads
        head_dim: full head dimension
        kv_lora_rank: KV compression rank
        rope_head_dim: dimension for RoPE (rest is nope)
        eps: layernorm epsilon
    """
    assert hidden_states.ndim == 3
    batch, seq, dim = hidden_states.shape
    prefix = f"layers.{layer_idx}.self_attn"

    nope_head_dim = head_dim - rope_head_dim

    # Q projection
    q = F.linear(hidden_states, weights[f"{prefix}.q_proj.weight"])
    q = q.view(batch, seq, n_heads, head_dim)

    # Split Q into rope and nope parts
    q_nope = q[..., :nope_head_dim]
    q_rope = q[..., nope_head_dim:]

    # Compress KV
    k_compressed, v = mla_compress_kv(
        hidden_states,
        weights[f"{prefix}.kv_a_proj.weight"],
        weights[f"{prefix}.kv_a_layernorm.weight"],
        weights[f"{prefix}.kv_b_proj.weight"],
        eps,
    )

    # Reshape K, V
    k = k_compressed.view(batch, seq, n_kv_heads, head_dim)
    v = v.view(batch, seq, n_kv_heads, head_dim)

    # Split K into rope and nope parts
    k_nope = k[..., :nope_head_dim]
    k_rope = k[..., nope_head_dim:]

    # Apply RoPE to rope parts only
    # cos, sin: [seq, rope_head_dim]
    cos = cos[:seq, :rope_head_dim].unsqueeze(0).unsqueeze(2)  # [1, seq, 1, rope_head_dim]
    sin = sin[:seq, :rope_head_dim].unsqueeze(0).unsqueeze(2)

    # Rotate half for RoPE
    def rotate_half(x: Tensor) -> Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_rope = (q_rope * cos) + (rotate_half(q_rope) * sin)
    k_rope = (k_rope * cos) + (rotate_half(k_rope) * sin)

    # Recombine
    q = torch.cat([q_nope, q_rope], dim=-1)
    k = torch.cat([k_nope, k_rope], dim=-1)

    # Transpose for attention: [batch, heads, seq, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # GQA expansion
    n_kv_groups = n_heads // n_kv_heads
    if n_kv_groups > 1:
        k = k.repeat_interleave(n_kv_groups, dim=1)
        v = v.repeat_interleave(n_kv_groups, dim=1)

    # Attention
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch, seq, n_heads * head_dim)
    return F.linear(attn_output, weights[f"{prefix}.o_proj.weight"])


# =============================================================================
# Mixture of Experts (MoE)
# =============================================================================


def router(
    hidden_states: Tensor,
    gate_weight: Tensor,
    topk: int,
) -> tuple[Tensor, Tensor]:
    """Route tokens to top-k experts.

    Args:
        hidden_states: [batch, seq, dim]
        gate_weight: [n_experts, dim]
        topk: number of experts per token

    Returns:
        weights: [batch, seq, topk] - normalized expert weights
        indices: [batch, seq, topk] - expert indices
    """
    # Compute router logits
    logits = F.linear(hidden_states, gate_weight)  # [batch, seq, n_experts]

    # Top-k selection with sigmoid (following nmoe/DeepSeek-V3)
    scores = torch.sigmoid(logits)
    weights, indices = torch.topk(scores, k=topk, dim=-1)

    # Normalize weights
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-12)

    return weights, indices


def moe_naive(
    hidden_states: Tensor,
    router_weight: Tensor,
    expert_weights: list[tuple[Tensor, Tensor, Tensor]],
    topk: int,
    activation: Activation = "swiglu",
) -> Tensor:
    """Naive MoE: loop over experts, no fusion.

    This is correct but slow - O(n_experts) forward passes.
    Use for correctness testing, not production.

    Args:
        hidden_states: [batch, seq, dim]
        router_weight: [n_experts, dim]
        expert_weights: list of (gate, up, down) weight tuples per expert
        topk: number of experts per token
        activation: MLP activation type

    Returns:
        [batch, seq, dim]
    """
    batch, seq, dim = hidden_states.shape
    n_experts = len(expert_weights)

    assert router_weight.shape[0] == n_experts

    # Get routing decisions
    weights, indices = router(hidden_states, router_weight, topk)

    # Compute expert outputs (naive loop)
    # Shape: [batch, seq, n_experts, dim]
    expert_outputs = torch.zeros(
        batch, seq, n_experts, dim, device=hidden_states.device, dtype=hidden_states.dtype
    )

    for e, (gate_w, up_w, down_w) in enumerate(expert_weights):
        if activation == "swiglu":
            expert_outputs[:, :, e] = mlp_swiglu(hidden_states, gate_w, up_w, down_w)
        elif activation == "relu_squared":
            # For relu_squared, gate_w is actually w1, up_w is w2
            expert_outputs[:, :, e] = mlp_relu_squared(hidden_states, gate_w, up_w)
        elif activation == "squared_reglu":
            expert_outputs[:, :, e] = mlp_squared_reglu(hidden_states, gate_w, up_w, down_w)

    # Gather selected expert outputs and weight them
    # indices: [batch, seq, topk] -> gather from expert_outputs
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, -1, dim)  # [batch, seq, topk, dim]
    selected_outputs = torch.gather(
        expert_outputs, dim=2, index=indices_expanded
    )  # [batch, seq, topk, dim]

    # Weighted sum
    output = (selected_outputs * weights.unsqueeze(-1)).sum(dim=2)  # [batch, seq, dim]

    return output


def init_moe_weights(
    config: ModelConfig,
    layer_idx: int,
    n_experts: int,
    device: torch.device,
    dtype: torch.dtype,
    activation: Activation = "swiglu",
    init_scale: float = 0.02,
) -> dict[str, Tensor]:
    """Initialize MoE weights for a single layer.

    Returns:
        weight dict with router and per-expert MLP weights
    """
    prefix = f"layers.{layer_idx}.moe"
    weights = {}

    # Router
    weights[f"{prefix}.router.weight"] = (
        torch.randn(n_experts, config.dim, device=device, dtype=dtype) * init_scale
    )

    # Expert MLPs
    for e in range(n_experts):
        if activation == "relu_squared":
            weights[f"{prefix}.experts.{e}.w1.weight"] = (
                torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
            )
            weights[f"{prefix}.experts.{e}.w2.weight"] = (
                torch.randn(config.dim, config.mlp_dim, device=device, dtype=dtype) * init_scale
            )
        else:
            weights[f"{prefix}.experts.{e}.gate_proj.weight"] = (
                torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
            )
            weights[f"{prefix}.experts.{e}.up_proj.weight"] = (
                torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
            )
            weights[f"{prefix}.experts.{e}.down_proj.weight"] = (
                torch.randn(config.dim, config.mlp_dim, device=device, dtype=dtype) * init_scale
            )

    # Enable gradients
    for w in weights.values():
        w.requires_grad_(True)

    return weights


# =============================================================================
# Weight Initialization
# =============================================================================


def init_mlp_weights(
    config: ModelConfig,
    layer_idx: int,
    device: torch.device,
    dtype: torch.dtype,
    activation: Activation = "swiglu",
    init_scale: float = 0.02,
) -> dict[str, Tensor]:
    """Initialize MLP weights for a single layer.

    Returns weight dict with appropriate keys for the activation type.
    """
    prefix = f"layers.{layer_idx}.mlp"
    weights = {}

    if activation == "relu_squared":
        # 2-weight architecture
        weights[f"{prefix}.w1.weight"] = (
            torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
        )
        weights[f"{prefix}.w2.weight"] = (
            torch.randn(config.dim, config.mlp_dim, device=device, dtype=dtype) * init_scale
        )
    else:
        # 3-weight gated architecture (swiglu, squared_reglu)
        weights[f"{prefix}.gate_proj.weight"] = (
            torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
        )
        weights[f"{prefix}.up_proj.weight"] = (
            torch.randn(config.mlp_dim, config.dim, device=device, dtype=dtype) * init_scale
        )
        weights[f"{prefix}.down_proj.weight"] = (
            torch.randn(config.dim, config.mlp_dim, device=device, dtype=dtype) * init_scale
        )

    # Enable gradients
    for w in weights.values():
        w.requires_grad_(True)

    return weights
