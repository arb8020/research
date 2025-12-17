"""Functional implementation of GLM-4.5 MoE forward pass.

This is a single-file, pure PyTorch implementation with no classes.
Uses only torch and torch.nn.functional.

Usage:
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    import torch

    # Download only first few shards for testing
    snapshot_download(
        "zai-org/GLM-4.5",
        allow_patterns=["model-0000[1-5]-of-00093.safetensors", "config.json"],
        local_dir="./glm4_weights",
    )

    # Load weights from shards
    weights = {}
    for i in range(1, 6):
        with safe_open(f"./glm4_weights/model-{i:05d}-of-00093.safetensors", framework="pt") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = glm4_moe_forward(input_ids, weights, num_layers=5)

Architecture (GLM-4.5):
    hidden_size: 5120
    intermediate_size: 12288 (dense MLP)
    moe_intermediate_size: 1536 (per expert)
    num_layers: 92
    num_attention_heads: 96
    num_kv_heads: 8
    head_dim: 128
    vocab_size: 151552
    rope_theta: 10000.0
    rms_norm_eps: 1e-5

    MoE config:
        first_k_dense_replace: 3 (layers 0-2 use dense MLP)
        n_routed_experts: 160
        n_shared_experts: 1
        num_experts_per_tok: 8
        n_group: 8
        topk_group: 4
        routed_scaling_factor: 2.5
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Config constants for GLM-4.5
HIDDEN_SIZE = 5120
INTERMEDIATE_SIZE = 12288  # Dense MLP intermediate
MOE_INTERMEDIATE_SIZE = 1536  # Per-expert intermediate
NUM_LAYERS = 92
NUM_HEADS = 96
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151552
ROPE_THETA = 10000.0
RMS_NORM_EPS = 1e-5

# MoE config
FIRST_K_DENSE_REPLACE = 3  # Layers 0, 1, 2 use dense MLP
N_ROUTED_EXPERTS = 160
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 8
N_GROUP = 1  # GLM-4.5 uses n_group=1, NOT 8 like DeepSeek
TOPK_GROUP = 1  # GLM-4.5 uses topk_group=1, NOT 4 like DeepSeek
ROUTED_SCALING_FACTOR = 2.5
NORM_TOPK_PROB = True


def rms_norm(x: Tensor, weight: Tensor, eps: float = RMS_NORM_EPS) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight."""
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    return weight * x_normed.to(input_dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_rope_embeddings(
    positions: Tensor,
    head_dim: int = HEAD_DIM,
    theta: float = ROPE_THETA,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Compute rotary position embeddings (cos, sin)."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )

    positions_expanded = positions[:, None, :].float()
    inv_freq_expanded = inv_freq[None, :, None]

    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match number of query heads (for GQA)."""
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def attention(  # noqa: PLR0913
    hidden_states: Tensor,
    q_weight: Tensor,
    q_bias: Tensor,
    k_weight: Tensor,
    k_bias: Tensor,
    v_weight: Tensor,
    v_bias: Tensor,
    o_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    attention_mask: Tensor | None = None,
    num_heads: int = NUM_HEADS,
    num_kv_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
) -> Tensor:
    """Self-attention with QKNorm, RoPE and GQA.

    GLM-4 uses:
    - Attention bias on q/k/v projections
    - QKNorm (RMSNorm on q and k after projection)
    - No bias on output projection
    """
    batch_size, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    # Project Q, K, V (with bias)
    q = F.linear(hidden_states, q_weight, q_bias)
    k = F.linear(hidden_states, k_weight, k_bias)
    v = F.linear(hidden_states, v_weight, v_bias)

    # Reshape to (batch, seq_len, num_heads, head_dim) then apply QKNorm
    q = q.view(batch_size, seq_len, num_heads, head_dim)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim)

    # QKNorm: RMSNorm on head_dim (last dimension)
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)

    # Transpose to (batch, num_heads, seq_len, head_dim)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat KV for GQA
    k = repeat_kv(k, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)

    # Scaled dot-product attention
    attn_output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=(attention_mask is None),
    )

    # Reshape and project output
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, num_heads * head_dim)
    output = F.linear(attn_output, o_weight)  # No bias

    return output


def dense_mlp(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Dense SwiGLU MLP for layers 0-2: down(silu(gate(x)) * up(x))."""
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    hidden = F.silu(gate) * up
    output = F.linear(hidden, down_weight)
    return output


def moe_router(
    hidden_states: Tensor,
    router_weight: Tensor,
    e_score_correction_bias: Tensor,
    n_routed_experts: int = N_ROUTED_EXPERTS,
    num_experts_per_tok: int = NUM_EXPERTS_PER_TOK,
    n_group: int = N_GROUP,
    topk_group: int = TOPK_GROUP,
    routed_scaling_factor: float = ROUTED_SCALING_FACTOR,
    norm_topk_prob: bool = NORM_TOPK_PROB,
    debug: bool = False,
) -> tuple[Tensor, Tensor]:
    """Route tokens to experts using sigmoid + group-based top-k selection.

    Args:
        hidden_states: (batch * seq_len, hidden_size)
        router_weight: (n_routed_experts, hidden_size)
        e_score_correction_bias: (n_routed_experts,)

    Returns:
        topk_indices: (batch * seq_len, num_experts_per_tok)
        topk_weights: (batch * seq_len, num_experts_per_tok)
    """
    if debug:
        print(f"      [moe_router] n_group={n_group}, topk_group={topk_group}")

    # Compute router logits and apply sigmoid
    router_logits = F.linear(hidden_states.float(), router_weight.float())
    router_probs = router_logits.sigmoid()

    if debug:
        print(f"      [moe_router] router_probs sample: {router_probs[0, :5].tolist()}")

    # Add correction bias for expert selection
    router_logits_for_choice = router_probs + e_score_correction_bias

    if debug:
        print(f"      [moe_router] scores_for_choice sample: {router_logits_for_choice[0, :5].tolist()}")

    # Group-based selection: split experts into groups, pick top-k groups first
    # Shape: (batch*seq, n_group, experts_per_group)
    experts_per_group = n_routed_experts // n_group
    group_scores = (
        router_logits_for_choice.view(-1, n_group, experts_per_group)
        .topk(2, dim=-1)[0]  # Top 2 per group
        .sum(dim=-1)  # Sum to get group score
    )

    # Select top-k groups
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)

    # Create mask for experts in selected groups
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(-1, n_group, experts_per_group)
        .reshape(-1, n_routed_experts)
    )

    # Mask out experts not in selected groups
    scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)

    # Select top-k experts from remaining
    topk_indices = torch.topk(scores_for_choice, k=num_experts_per_tok, dim=-1, sorted=False)[1]

    # Get weights from original probabilities (not logits_for_choice)
    topk_weights = router_probs.gather(1, topk_indices)

    # Normalize probabilities if configured
    if norm_topk_prob:
        denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / denominator

    # Apply scaling factor
    topk_weights = topk_weights * routed_scaling_factor

    return topk_indices, topk_weights


def expert_forward(
    hidden_states: Tensor,
    topk_indices: Tensor,
    topk_weights: Tensor,
    expert_gate_weights: list[Tensor],
    expert_up_weights: list[Tensor],
    expert_down_weights: list[Tensor],
    n_routed_experts: int = N_ROUTED_EXPERTS,
) -> Tensor:
    """Execute selected experts on tokens.

    This is the "naive" implementation that iterates over active experts.
    More efficient implementations exist but this matches HF behavior.

    Args:
        hidden_states: (batch * seq_len, hidden_size)
        topk_indices: (batch * seq_len, num_experts_per_tok)
        topk_weights: (batch * seq_len, num_experts_per_tok)
        expert_*_weights: List of weight tensors per expert
    """
    final_hidden_states = torch.zeros_like(hidden_states)

    # Create one-hot mask for expert assignment
    # Shape: (n_experts, num_experts_per_tok, batch * seq_len)
    expert_mask = F.one_hot(topk_indices, num_classes=n_routed_experts)
    expert_mask = expert_mask.permute(2, 1, 0)

    # Find which experts are actually used
    expert_hit = (expert_mask.sum(dim=(1, 2)) > 0).nonzero(as_tuple=True)[0]

    for expert_idx in expert_hit:
        expert_idx = expert_idx.item()

        # Find which tokens go to this expert and at which top-k position
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

        # Get hidden states for these tokens
        current_state = hidden_states[token_idx]

        # Run through expert MLP
        gate = F.linear(current_state, expert_gate_weights[expert_idx])
        up = F.linear(current_state, expert_up_weights[expert_idx])
        current_hidden = F.silu(gate) * up
        current_hidden = F.linear(current_hidden, expert_down_weights[expert_idx])

        # Weight by routing probability
        current_hidden = current_hidden * topk_weights[token_idx, top_k_pos, None]

        # Accumulate into output
        final_hidden_states.index_add_(0, token_idx, current_hidden.to(final_hidden_states.dtype))

    return final_hidden_states


def moe_block(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
) -> Tensor:
    """Full MoE block: router + experts + shared expert.

    Args:
        hidden_states: (batch, seq_len, hidden_size)
        weights: Dict of all model weights
        layer_idx: Layer index for weight lookup
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Flatten for routing
    hidden_flat = hidden_states.view(-1, hidden_size)

    # Route tokens to experts
    topk_indices, topk_weights = moe_router(
        hidden_flat,
        router_weight=weights[f"{prefix}.gate.weight"],
        e_score_correction_bias=weights[f"{prefix}.gate.e_score_correction_bias"],
    )

    # Collect expert weights
    expert_gate_weights = []
    expert_up_weights = []
    expert_down_weights = []
    for i in range(N_ROUTED_EXPERTS):
        expert_gate_weights.append(weights[f"{prefix}.experts.{i}.gate_proj.weight"])
        expert_up_weights.append(weights[f"{prefix}.experts.{i}.up_proj.weight"])
        expert_down_weights.append(weights[f"{prefix}.experts.{i}.down_proj.weight"])

    # Execute routed experts
    routed_output = expert_forward(
        hidden_flat,
        topk_indices,
        topk_weights,
        expert_gate_weights,
        expert_up_weights,
        expert_down_weights,
    )

    # Reshape back
    routed_output = routed_output.view(batch_size, seq_len, hidden_size)

    # Shared expert (always runs on all tokens)
    shared_output = dense_mlp(
        hidden_states,
        gate_weight=weights[f"{prefix}.shared_experts.gate_proj.weight"],
        up_weight=weights[f"{prefix}.shared_experts.up_proj.weight"],
        down_weight=weights[f"{prefix}.shared_experts.down_proj.weight"],
    )

    # Combine routed + shared
    return routed_output + shared_output


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Single transformer decoder layer with dense or MoE MLP."""
    prefix = f"model.layers.{layer_idx}"

    # Pre-attention norm + attention + residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"])

    hidden_states = attention(
        hidden_states,
        q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
        q_bias=weights[f"{prefix}.self_attn.q_proj.bias"],
        k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
        k_bias=weights[f"{prefix}.self_attn.k_proj.bias"],
        v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
        v_bias=weights[f"{prefix}.self_attn.v_proj.bias"],
        o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
        q_norm_weight=weights[f"{prefix}.self_attn.q_norm.weight"],
        k_norm_weight=weights[f"{prefix}.self_attn.k_norm.weight"],
        cos=cos,
        sin=sin,
        attention_mask=attention_mask,
    )

    hidden_states = residual + hidden_states

    # Pre-MLP norm + MLP/MoE + residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"])

    # Use dense MLP for first k layers, MoE for the rest
    if layer_idx < FIRST_K_DENSE_REPLACE:
        hidden_states = dense_mlp(
            hidden_states,
            gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
            up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
            down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
        )
    else:
        hidden_states = moe_block(hidden_states, weights, layer_idx)

    hidden_states = residual + hidden_states

    return hidden_states


def glm4_moe_forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    attention_mask: Tensor | None = None,
    num_layers: int = NUM_LAYERS,
) -> Tensor:
    """GLM-4.5 MoE forward pass.

    Args:
        input_ids: Input token IDs of shape (batch, seq_len)
        weights: Dict of model weights (can be partial for testing)
        attention_mask: Optional padding mask
        num_layers: Number of layers to run (for partial testing)

    Returns:
        Logits tensor of shape (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    # Position IDs
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Compute RoPE embeddings
    cos, sin = compute_rope_embeddings(positions, dtype=hidden_states.dtype)

    # Transformer layers
    for layer_idx in range(num_layers):
        hidden_states = transformer_layer(
            hidden_states, weights, layer_idx, cos, sin, attention_mask
        )

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["model.norm.weight"])

    # LM head (tied to embedding weights)
    logits = F.linear(hidden_states, weights["model.embed_tokens.weight"])

    return logits


if __name__ == "__main__":
    print("GLM-4 MoE Functional Implementation")
    print("=" * 50)
    print(f"Hidden size: {HIDDEN_SIZE}")
    print(f"Layers: {NUM_LAYERS} ({FIRST_K_DENSE_REPLACE} dense, {NUM_LAYERS - FIRST_K_DENSE_REPLACE} MoE)")
    print(f"Experts: {N_ROUTED_EXPERTS} routed + {N_SHARED_EXPERTS} shared")
    print(f"Experts per token: {NUM_EXPERTS_PER_TOK}")
    print(f"Attention heads: {NUM_HEADS} ({NUM_KV_HEADS} KV heads)")
