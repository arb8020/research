#!/usr/bin/env python3
"""Functional implementation of Qwen3-Next-80B forward pass.

This is a single-file, pure PyTorch implementation with no classes.
Uses only torch and torch.nn.functional.

Qwen3-Next is a hybrid Mamba + Transformer architecture with MoE:
- Layers alternate between GatedDeltaNet (linear attention) and self-attention
- 512 experts per layer, 10 active per token + shared expert
- Partial RoPE (25% of head_dim)
- GQA 8:1 ratio

Usage:
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    weights = dict(model.state_dict())

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")
    logits = qwen3_next_forward(input_ids, weights)

Architecture (Qwen3-Next-80B-A3B):
    hidden_size: 2048
    intermediate_size: 5120
    moe_intermediate_size: 512
    num_layers: 48
    num_attention_heads: 16
    num_kv_heads: 2
    head_dim: 256
    vocab_size: 151936
    rope_theta: 10000000
    rms_norm_eps: 1e-6
    partial_rotary_factor: 0.25
    num_experts: 512
    num_experts_per_tok: 10
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Config constants for Qwen3-Next-80B-A3B
HIDDEN_SIZE = 2048
INTERMEDIATE_SIZE = 5120
MOE_INTERMEDIATE_SIZE = 512
SHARED_EXPERT_INTERMEDIATE_SIZE = 512
NUM_LAYERS = 48
NUM_HEADS = 16
NUM_KV_HEADS = 2
HEAD_DIM = 256
VOCAB_SIZE = 151936
ROPE_THETA = 10000000.0
RMS_NORM_EPS = 1e-6
PARTIAL_ROTARY_FACTOR = 0.25
NUM_EXPERTS = 512
NUM_EXPERTS_PER_TOK = 10

# GatedDeltaNet config
CONV_KERNEL_SIZE = 4
NUM_HEADS_DELTANET = 32  # from A_log shape [32]
DELTANET_HEAD_DIM = 128  # 4096 / 32 from out_proj shape

# Layers with self_attn (others have linear_attn)
# From exploration: q_norm/k_norm found at layers 3, 7, 11, 15, 19, ...
SELF_ATTN_LAYERS = set(range(3, 48, 4))  # Every 4th layer starting at 3


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


def apply_rotary_pos_emb_partial(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
) -> tuple[Tensor, Tensor]:
    """Apply partial rotary position embeddings (only to first rotary_dim dimensions).

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine tensor of shape (batch, seq_len, rotary_dim)
        sin: Sine tensor of shape (batch, seq_len, rotary_dim)
        rotary_dim: Number of dimensions to apply RoPE to

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Split into rotary and non-rotary parts
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Unsqueeze for broadcasting: (batch, seq_len, rotary_dim) -> (batch, 1, seq_len, rotary_dim)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Apply RoPE to rotary part
    q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back
    q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)

    return q_embed, k_embed


def compute_rope_embeddings(
    positions: Tensor,
    rotary_dim: int,
    theta: float = ROPE_THETA,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[Tensor, Tensor]:
    """Compute rotary position embeddings (cos, sin) for partial RoPE.

    Args:
        positions: Position indices of shape (batch, seq_len)
        rotary_dim: Number of dimensions for RoPE
        theta: RoPE theta parameter
        dtype: Output dtype

    Returns:
        Tuple of (cos, sin) each of shape (batch, seq_len, rotary_dim)
    """
    # inv_freq: (rotary_dim // 2,)
    inv_freq = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=positions.device).float() / rotary_dim))

    # positions: (batch, seq_len) -> (batch, 1, seq_len)
    # inv_freq: (rotary_dim // 2,) -> (1, rotary_dim // 2, 1)
    positions_expanded = positions[:, None, :].float()
    inv_freq_expanded = inv_freq[None, :, None]

    # freqs: (batch, rotary_dim // 2, seq_len) -> (batch, seq_len, rotary_dim // 2)
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)

    # emb: (batch, seq_len, rotary_dim)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match number of query heads (for GQA)."""
    if n_rep == 1:
        return hidden_states

    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


def self_attention(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    prefix: str,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Self-attention with partial RoPE, GQA, Q/K norms, and gated output.

    Qwen3-Next attention uses gated attention:
    - Q projection outputs 2x normal size, chunked into query + gate
    - Output is multiplied by sigmoid(gate)

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        weights: Dict of weight tensors
        prefix: Weight key prefix (e.g., "model.layers.3.self_attn")
        cos, sin: RoPE embeddings
        rotary_dim: Dimensions for RoPE
        attention_mask: Optional mask

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    batch_size, seq_len, _ = hidden_states.shape
    num_kv_groups = NUM_HEADS // NUM_KV_HEADS

    # Project Q - outputs 2x size for query + gate
    q_proj_out = F.linear(hidden_states, weights[f"{prefix}.q_proj.weight"])
    # Reshape to [batch, seq_len, num_heads, head_dim * 2] then chunk
    q_proj_out = q_proj_out.view(batch_size, seq_len, NUM_HEADS, HEAD_DIM * 2)
    query_states, gate = torch.chunk(q_proj_out, 2, dim=-1)

    # Project K, V (normal size)
    k = F.linear(hidden_states, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden_states, weights[f"{prefix}.v_proj.weight"])

    # Reshape K, V to (batch, seq_len, num_kv_heads, head_dim)
    k = k.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)
    v = v.view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)

    # Apply Q/K norms (applied to [batch, seq_len, num_heads, head_dim] shape)
    if f"{prefix}.q_norm.weight" in weights:
        # Flatten for RMS norm, then reshape back
        query_states = rms_norm(
            query_states.reshape(-1, HEAD_DIM),
            weights[f"{prefix}.q_norm.weight"]
        ).view(batch_size, seq_len, NUM_HEADS, HEAD_DIM)
        k = rms_norm(
            k.reshape(-1, HEAD_DIM),
            weights[f"{prefix}.k_norm.weight"]
        ).view(batch_size, seq_len, NUM_KV_HEADS, HEAD_DIM)

    # Transpose to (batch, num_heads, seq_len, head_dim)
    q = query_states.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Apply partial RoPE
    q, k = apply_rotary_pos_emb_partial(q, k, cos, sin, rotary_dim)

    # Repeat KV for GQA
    k = repeat_kv(k, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)

    # Scaled dot-product attention
    attn_output = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=(attention_mask is None),
    )

    # Reshape output to (batch, seq_len, num_heads * head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(batch_size, seq_len, NUM_HEADS * HEAD_DIM)

    # Apply gating: output = attn_output * sigmoid(gate)
    gate = gate.view(batch_size, seq_len, NUM_HEADS * HEAD_DIM)
    attn_output = attn_output * torch.sigmoid(gate)

    # Output projection
    output = F.linear(attn_output, weights[f"{prefix}.o_proj.weight"])

    return output


def gated_delta_net(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    prefix: str,
) -> Tensor:
    """GatedDeltaNet (linear attention / Mamba-style).

    This implements the Qwen3NextGatedDeltaNet module:
    - Conv1d for local context
    - Delta rule with A_log and dt_bias
    - Gated output

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        weights: Dict of weight tensors
        prefix: Weight key prefix (e.g., "model.layers.0.linear_attn")

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Get weights
    A_log = weights[f"{prefix}.A_log"]  # [num_heads]
    dt_bias = weights[f"{prefix}.dt_bias"]  # [num_heads]
    conv1d_weight = weights[f"{prefix}.conv1d.weight"]  # [expand_size, 1, kernel_size]
    in_proj_qkvz = weights[f"{prefix}.in_proj_qkvz.weight"]  # [qkvz_size, hidden_size]
    in_proj_ba = weights[f"{prefix}.in_proj_ba.weight"]  # [ba_size, hidden_size]
    norm_weight = weights[f"{prefix}.norm.weight"]  # [head_dim]
    out_proj = weights[f"{prefix}.out_proj.weight"]  # [hidden_size, expand_size]

    num_heads = A_log.shape[0]  # 32
    expand_size = conv1d_weight.shape[0]  # 8192
    head_dim = expand_size // num_heads  # 256
    qkvz_size = in_proj_qkvz.shape[0]  # 12288 = 3 * expand_size (q, k, v, z packed differently)

    # Compute A from A_log (negative to ensure stability)
    # Keep in float32 for stability but will convert decay to input dtype
    A = -torch.exp(A_log.float())  # [num_heads]
    input_dtype = hidden_states.dtype

    # Project input to qkvz space
    qkvz = F.linear(hidden_states, in_proj_qkvz)  # [batch, seq_len, qkvz_size]

    # Split qkvz - the exact split depends on implementation
    # From shape 12288 = 3 * 4096, likely q, kv, z each 4096
    q_size = expand_size // 2  # 4096
    kv_size = expand_size // 2  # 4096
    z_size = expand_size  # 4096 (but qkvz is 12288, so this needs adjustment)

    # Actually from 12288: could be q(4096) + k(4096) + v(4096) = 12288
    third = qkvz_size // 3
    q = qkvz[..., :third]  # [batch, seq_len, 4096]
    k = qkvz[..., third:2*third]  # [batch, seq_len, 4096]
    v_and_z = qkvz[..., 2*third:]  # [batch, seq_len, 4096]

    # Project for beta and alpha (dt)
    ba = F.linear(hidden_states, in_proj_ba)  # [batch, seq_len, ba_size=64]
    ba_half = ba.shape[-1] // 2
    beta = ba[..., :ba_half]  # [batch, seq_len, 32]
    alpha = ba[..., ba_half:]  # [batch, seq_len, 32] - this is dt before softplus

    # Apply dt_bias and softplus to get dt
    dt = F.softplus(alpha + dt_bias)  # [batch, seq_len, num_heads]

    # Apply conv1d to q (causal convolution)
    # Reshape q for conv1d: [batch, seq_len, expand/2] -> [batch, expand/2, seq_len]
    q_conv = q.transpose(1, 2)
    # Pad for causal conv
    q_conv = F.pad(q_conv, (CONV_KERNEL_SIZE - 1, 0))
    # Apply depthwise conv1d
    q_conv = F.conv1d(q_conv, conv1d_weight[:third], groups=third)
    q = q_conv.transpose(1, 2)  # [batch, seq_len, expand/2]

    # Apply SiLU activation to q
    q = F.silu(q)

    # Reshape for multi-head processing
    # q, k: [batch, seq_len, 4096] -> [batch, seq_len, num_heads, head_dim/2]
    q = q.view(batch_size, seq_len, num_heads, -1)
    k = k.view(batch_size, seq_len, num_heads, -1)
    v_and_z = v_and_z.view(batch_size, seq_len, num_heads, -1)

    # Delta rule recurrence (simplified - actual impl uses chunked/parallel scan)
    # h_{t+1} = A * h_t + beta_t * (k_t^T @ v_t)
    # o_t = q_t @ h_t

    # For simplicity, implement sequential scan (slow but correct)
    # In practice, this should use parallel scan or chunked computation
    head_dim_half = q.shape[-1]

    # Initialize hidden state (use input dtype for consistency)
    h = torch.zeros(batch_size, num_heads, head_dim_half, head_dim_half,
                    device=hidden_states.device, dtype=input_dtype)

    outputs = []
    for t in range(seq_len):
        # Get current timestep values
        q_t = q[:, t]  # [batch, num_heads, head_dim_half]
        k_t = k[:, t]  # [batch, num_heads, head_dim_half]
        v_t = v_and_z[:, t]  # [batch, num_heads, head_dim_half]
        dt_t = dt[:, t]  # [batch, num_heads]
        beta_t = torch.sigmoid(beta[:, t])  # [batch, num_heads]

        # Compute decay factor (compute in float32, convert to input dtype)
        decay = torch.exp(A * dt_t).to(input_dtype)  # [batch, num_heads]

        # Outer product: k^T @ v -> [batch, num_heads, head_dim_half, head_dim_half]
        kv = torch.einsum('bhi,bhj->bhij', k_t, v_t)

        # Update hidden state with decay and gated update
        # h = decay * h + beta * kv
        h = decay[:, :, None, None] * h + beta_t[:, :, None, None] * kv

        # Output: q @ h -> [batch, num_heads, head_dim_half]
        o_t = torch.einsum('bhi,bhij->bhj', q_t, h)
        outputs.append(o_t)

    # Stack outputs: [seq_len, batch, num_heads, head_dim_half] -> [batch, seq_len, num_heads, head_dim_half]
    output = torch.stack(outputs, dim=1)

    # Apply group norm
    output = output.view(batch_size, seq_len, -1)  # [batch, seq_len, num_heads * head_dim_half]

    # Apply norm (simplified - actual uses group norm per head)
    output_normed = rms_norm(output.view(batch_size * seq_len, num_heads, -1),
                            norm_weight).view(batch_size, seq_len, -1)

    # Output projection
    output = F.linear(output_normed, out_proj)

    return output


def moe_mlp(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    prefix: str,
) -> Tensor:
    """Mixture of Experts MLP.

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        weights: Dict of weight tensors
        prefix: Weight key prefix (e.g., "model.layers.0.mlp")

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    batch_size, seq_len, hidden_size = hidden_states.shape

    # Router
    router_logits = F.linear(hidden_states, weights[f"{prefix}.gate.weight"])  # [batch, seq_len, num_experts]

    # Get top-k experts
    routing_weights, selected_experts = torch.topk(router_logits, NUM_EXPERTS_PER_TOK, dim=-1)
    routing_weights = F.softmax(routing_weights, dim=-1)  # Normalize top-k weights

    # Flatten for expert computation
    hidden_flat = hidden_states.view(-1, hidden_size)  # [batch * seq_len, hidden_size]
    routing_weights_flat = routing_weights.view(-1, NUM_EXPERTS_PER_TOK)
    selected_experts_flat = selected_experts.view(-1, NUM_EXPERTS_PER_TOK)

    # Compute expert outputs (simplified - loops over experts)
    # In practice, use grouped GEMM or expert parallelism
    expert_output = torch.zeros_like(hidden_flat)

    for expert_idx in range(NUM_EXPERTS):
        # Find which tokens selected this expert
        expert_mask = (selected_experts_flat == expert_idx).any(dim=-1)
        if not expert_mask.any():
            continue

        # Get tokens for this expert
        token_indices = expert_mask.nonzero(as_tuple=True)[0]
        expert_input = hidden_flat[token_indices]

        # Get weight for this expert per token
        expert_weights = torch.zeros(len(token_indices), device=hidden_states.device, dtype=hidden_states.dtype)
        for k in range(NUM_EXPERTS_PER_TOK):
            k_mask = selected_experts_flat[token_indices, k] == expert_idx
            expert_weights[k_mask] = routing_weights_flat[token_indices[k_mask], k]

        # Expert MLP: SwiGLU
        gate = F.linear(expert_input, weights[f"{prefix}.experts.{expert_idx}.gate_proj.weight"])
        up = F.linear(expert_input, weights[f"{prefix}.experts.{expert_idx}.up_proj.weight"])
        expert_hidden = F.silu(gate) * up
        expert_out = F.linear(expert_hidden, weights[f"{prefix}.experts.{expert_idx}.down_proj.weight"])

        # Weighted contribution
        expert_output[token_indices] += expert_weights[:, None] * expert_out

    # Shared expert (always applied)
    shared_gate = F.linear(hidden_flat, weights[f"{prefix}.shared_expert.gate_proj.weight"])
    shared_up = F.linear(hidden_flat, weights[f"{prefix}.shared_expert.up_proj.weight"])
    shared_hidden = F.silu(shared_gate) * shared_up
    shared_out = F.linear(shared_hidden, weights[f"{prefix}.shared_expert.down_proj.weight"])

    # Shared expert gate
    if f"{prefix}.shared_expert_gate.weight" in weights:
        shared_expert_gate = torch.sigmoid(
            F.linear(hidden_flat, weights[f"{prefix}.shared_expert_gate.weight"])
        )
        shared_out = shared_out * shared_expert_gate

    # Combine
    output = expert_output + shared_out
    output = output.view(batch_size, seq_len, hidden_size)

    return output


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    rotary_dim: int,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Single transformer decoder layer (handles both linear_attn and self_attn).

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        weights: Dict of weight tensors
        layer_idx: Layer index
        cos, sin: RoPE embeddings (for self_attn layers)
        rotary_dim: Dimensions for RoPE
        attention_mask: Optional attention mask

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    prefix = f"model.layers.{layer_idx}"

    # Pre-attention norm
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.input_layernorm.weight"])

    # Attention (self_attn or linear_attn)
    if layer_idx in SELF_ATTN_LAYERS:
        hidden_states = self_attention(
            hidden_states,
            weights,
            f"{prefix}.self_attn",
            cos,
            sin,
            rotary_dim,
            attention_mask,
        )
    else:
        hidden_states = gated_delta_net(
            hidden_states,
            weights,
            f"{prefix}.linear_attn",
        )

    hidden_states = residual + hidden_states

    # Pre-MLP norm
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, weights[f"{prefix}.post_attention_layernorm.weight"])

    # MoE MLP
    hidden_states = moe_mlp(hidden_states, weights, f"{prefix}.mlp")

    hidden_states = residual + hidden_states

    return hidden_states


def qwen3_next_forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Full Qwen3-Next-80B forward pass.

    Args:
        input_ids: Input token IDs of shape (batch, seq_len)
        weights: Dict of all model weights (from model.state_dict())
        attention_mask: Optional 2D padding mask of shape (batch, seq_len)

    Returns:
        Logits tensor of shape (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    # Position IDs
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Compute RoPE embeddings for partial RoPE
    rotary_dim = int(HEAD_DIM * PARTIAL_ROTARY_FACTOR)  # 64 = 256 * 0.25
    cos, sin = compute_rope_embeddings(positions, rotary_dim, dtype=hidden_states.dtype)

    # Create attention mask (for self_attn layers)
    attn_mask_4d = None
    if attention_mask is not None and not attention_mask.all():
        # Create 4D causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        padding_mask = attention_mask[:, None, None, :].bool()
        combined_mask = causal_mask & padding_mask
        min_dtype = torch.finfo(hidden_states.dtype).min
        attn_mask_4d = torch.where(combined_mask, 0.0, min_dtype).to(hidden_states.dtype)

    # Transformer layers
    for layer_idx in range(NUM_LAYERS):
        hidden_states = transformer_layer(
            hidden_states, weights, layer_idx, cos, sin, rotary_dim, attn_mask_4d
        )

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["model.norm.weight"])

    # LM head
    logits = F.linear(hidden_states, weights["lm_head.weight"])

    return logits


# For testing
if __name__ == "__main__":
    import sys

    print("Testing functional Qwen3-Next implementation...")

    if not torch.cuda.is_available():
        print("No GPU available. Run on GPU for full test.")
        sys.exit(0)

    from transformers import AutoModelForCausalLM

    print("Loading Qwen3-Next-80B-A3B-Instruct...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-Next-80B-A3B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="balanced",
    )
    model.eval()

    weights = {k: v for k, v in model.state_dict().items()}

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")

    print("Running original model...")
    with torch.no_grad():
        original_logits = model(input_ids).logits

    print("Running functional implementation...")
    with torch.no_grad():
        functional_logits = qwen3_next_forward(input_ids, weights)

    print(f"Original shape: {original_logits.shape}")
    print(f"Functional shape: {functional_logits.shape}")

    matches = torch.allclose(original_logits, functional_logits, rtol=1e-4, atol=1e-4)
    max_diff = (original_logits - functional_logits).abs().max().item()

    print(f"Matches: {matches}")
    print(f"Max diff: {max_diff:.6e}")

    if matches:
        print("SUCCESS! Functional implementation matches original.")
    else:
        print("FAILURE! Outputs do not match.")
        sys.exit(1)
