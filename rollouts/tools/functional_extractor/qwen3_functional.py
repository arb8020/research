"""Functional implementation of Qwen3-0.6B forward pass.

This is a single-file, pure PyTorch implementation with no classes.
Uses only torch and torch.nn.functional.

Usage:
    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    weights = dict(model.state_dict())

    input_ids = torch.tensor([[1, 2, 3, 4]])
    logits = qwen3_forward(input_ids, weights)

Architecture (Qwen3-0.6B):
    hidden_size: 1024
    intermediate_size: 3072
    num_layers: 28
    num_attention_heads: 16
    num_kv_heads: 8
    head_dim: 128
    vocab_size: 151936
    rope_theta: 1000000.0
    rms_norm_eps: 1e-6

Key differences from Qwen2.5:
    - No attention bias (attention_bias=false)
    - QKNorm: RMSNorm applied to q and k after projection, before RoPE
    - Different head_dim (128 vs 64)
    - Different GQA ratio (2:1 vs 7:1)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# Config constants for Qwen3-0.6B
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
NUM_LAYERS = 28
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
VOCAB_SIZE = 151936
ROPE_THETA = 1000000.0
RMS_NORM_EPS = 1e-6


def rms_norm(x: Tensor, weight: Tensor, eps: float = RMS_NORM_EPS) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight.

    Args:
        x: Input tensor of shape (..., dim)
        weight: Norm weight of shape (dim,)
        eps: Epsilon for numerical stability

    Returns:
        Normalized tensor of same shape as x
    """
    input_dtype = x.dtype
    x_fp32 = x.to(torch.float32)
    variance = x_fp32.pow(2).mean(-1, keepdim=True)
    x_normed = x_fp32 * torch.rsqrt(variance + eps)
    # HF does: weight * hidden_states.to(input_dtype) - multiply after cast
    return weight * x_normed.to(input_dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate half the hidden dims for RoPE.

    Args:
        x: Tensor of shape (..., head_dim)

    Returns:
        Rotated tensor where first half and second half are swapped with negation
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor,
    k: Tensor,
    cos: Tensor,
    sin: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply rotary position embeddings to query and key.

    Args:
        q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
        k: Key tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        cos: Cosine tensor of shape (batch, seq_len, head_dim)
        sin: Sine tensor of shape (batch, seq_len, head_dim)

    Returns:
        Tuple of (rotated_q, rotated_k)
    """
    # Unsqueeze for broadcasting: (batch, seq_len, head_dim) -> (batch, 1, seq_len, head_dim)
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
    """Compute rotary position embeddings (cos, sin).

    Args:
        positions: Position indices of shape (batch, seq_len)
        head_dim: Dimension per head
        theta: RoPE theta parameter
        dtype: Output dtype

    Returns:
        Tuple of (cos, sin) each of shape (batch, seq_len, head_dim)
    """
    # inv_freq: (head_dim // 2,)
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim)
    )

    # positions: (batch, seq_len) -> (batch, 1, seq_len)
    # inv_freq: (head_dim // 2,) -> (1, head_dim // 2, 1)
    positions_expanded = positions[:, None, :].float()
    inv_freq_expanded = inv_freq[None, :, None]

    # freqs: (batch, head_dim // 2, seq_len) -> (batch, seq_len, head_dim // 2)
    freqs = (inv_freq_expanded @ positions_expanded).transpose(1, 2)

    # emb: (batch, seq_len, head_dim)
    emb = torch.cat((freqs, freqs), dim=-1)

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return cos, sin


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    """Repeat KV heads to match number of query heads (for GQA).

    Args:
        hidden_states: Tensor of shape (batch, num_kv_heads, seq_len, head_dim)
        n_rep: Number of times to repeat each KV head

    Returns:
        Tensor of shape (batch, num_kv_heads * n_rep, seq_len, head_dim)
    """
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
    k_weight: Tensor,
    v_weight: Tensor,
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

    Qwen3 differences from Qwen2.5:
    - No bias on q/k/v projections
    - QKNorm: RMSNorm applied to q and k after projection, before RoPE

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        q_weight: Query projection weight (no bias)
        k_weight: Key projection weight (no bias)
        v_weight: Value projection weight (no bias)
        o_weight: Output projection weight (no bias)
        q_norm_weight: Query norm weight of shape (head_dim,)
        k_norm_weight: Key norm weight of shape (head_dim,)
        cos, sin: RoPE embeddings
        attention_mask: Optional mask of shape (batch, 1, seq_len, seq_len) or None.
            Should be 0 for positions to attend to, -inf for masked positions.
            If None, uses causal masking.
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads
        head_dim: Dimension per head

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    batch_size, seq_len, _ = hidden_states.shape
    num_kv_groups = num_heads // num_kv_heads

    # Project Q, K, V (no bias in Qwen3)
    q = F.linear(hidden_states, q_weight)
    k = F.linear(hidden_states, k_weight)
    v = F.linear(hidden_states, v_weight)

    # Reshape to (batch, num_heads, seq_len, head_dim)
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2)

    # QKNorm: Apply RMSNorm to q and k (per head, on head_dim dimension)
    # Shape: (batch, num_heads, seq_len, head_dim) -> norm on last dim
    q = rms_norm(q, q_norm_weight)
    k = rms_norm(k, k_norm_weight)

    # Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # Repeat KV for GQA
    k = repeat_kv(k, num_kv_groups)
    v = repeat_kv(v, num_kv_groups)

    # Use scaled_dot_product_attention for numerical consistency with HF
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
    output = F.linear(attn_output, o_weight)

    return output


def mlp(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """SwiGLU MLP: down(silu(gate(x)) * up(x)).

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        gate_weight: Gate projection weight
        up_weight: Up projection weight
        down_weight: Down projection weight

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    hidden = F.silu(gate) * up
    output = F.linear(hidden, down_weight)
    return output


def transformer_layer(
    hidden_states: Tensor,
    layer_weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Single transformer decoder layer.

    Args:
        hidden_states: Input of shape (batch, seq_len, hidden_size)
        layer_weights: Dict of weight tensors for this layer
        layer_idx: Layer index (for weight key prefix)
        cos, sin: RoPE embeddings
        attention_mask: Optional attention mask

    Returns:
        Output tensor of shape (batch, seq_len, hidden_size)
    """
    prefix = f"model.layers.{layer_idx}"

    # Pre-attention norm + attention + residual
    residual = hidden_states
    hidden_states = rms_norm(hidden_states, layer_weights[f"{prefix}.input_layernorm.weight"])

    hidden_states = attention(
        hidden_states,
        q_weight=layer_weights[f"{prefix}.self_attn.q_proj.weight"],
        k_weight=layer_weights[f"{prefix}.self_attn.k_proj.weight"],
        v_weight=layer_weights[f"{prefix}.self_attn.v_proj.weight"],
        o_weight=layer_weights[f"{prefix}.self_attn.o_proj.weight"],
        q_norm_weight=layer_weights[f"{prefix}.self_attn.q_norm.weight"],
        k_norm_weight=layer_weights[f"{prefix}.self_attn.k_norm.weight"],
        cos=cos,
        sin=sin,
        attention_mask=attention_mask,
    )

    hidden_states = residual + hidden_states

    # Pre-MLP norm + MLP + residual
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states, layer_weights[f"{prefix}.post_attention_layernorm.weight"]
    )

    hidden_states = mlp(
        hidden_states,
        gate_weight=layer_weights[f"{prefix}.mlp.gate_proj.weight"],
        up_weight=layer_weights[f"{prefix}.mlp.up_proj.weight"],
        down_weight=layer_weights[f"{prefix}.mlp.down_proj.weight"],
    )

    hidden_states = residual + hidden_states

    return hidden_states


def create_causal_mask(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    attention_mask: Tensor | None = None,
) -> Tensor | None:
    """Create a 4D causal attention mask, optionally combined with padding mask.

    Args:
        seq_len: Sequence length
        device: Device for the mask
        dtype: Dtype for the mask (should match hidden states)
        attention_mask: Optional 2D padding mask of shape (batch, seq_len) where
            1 = attend, 0 = mask out

    Returns:
        4D mask of shape (batch, 1, seq_len, seq_len) with 0 for attend, min_dtype for mask,
        or None if no padding mask and we can use is_causal=True
    """
    if attention_mask is None:
        return None

    # If all values are 1 (no padding), use is_causal=True path for better numerical match
    if attention_mask.all():
        return None

    # Create causal mask: (1, 1, seq_len, seq_len)
    # Lower triangular = True (attend), upper triangular = False (mask)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # Expand padding mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
    # This masks out attending TO padded positions (columns)
    padding_mask = attention_mask[:, None, None, :].bool()  # (batch, 1, 1, seq_len)

    # Combine: attend only where BOTH causal and padding allow
    combined_mask = causal_mask & padding_mask  # (batch, 1, seq_len, seq_len)

    # Convert to float mask: 0 for attend, min_dtype for mask
    min_dtype = torch.finfo(dtype).min
    mask = torch.where(combined_mask, 0.0, min_dtype)
    mask = mask.to(dtype)

    # _unmask_unattended: For rows that are fully masked (padding positions),
    # make them attend to all tokens. Required by SDPA memory-efficient path.
    fully_masked_rows = torch.all(mask == min_dtype, dim=-1, keepdim=True)
    mask = mask * (~fully_masked_rows)

    return mask


def qwen3_forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    attention_mask: Tensor | None = None,
    num_layers: int = NUM_LAYERS,
) -> Tensor:
    """Full Qwen3-0.6B forward pass.

    Args:
        input_ids: Input token IDs of shape (batch, seq_len)
        weights: Dict of all model weights (from model.state_dict())
        attention_mask: Optional 2D padding mask of shape (batch, seq_len) where
            1 = real token, 0 = padding. If None, no padding is assumed.
        num_layers: Number of transformer layers

    Returns:
        Logits tensor of shape (batch, seq_len, vocab_size)
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Token embeddings
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    # Position IDs are always sequential [0, 1, 2, ..., seq_len-1]
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    # Compute RoPE embeddings once
    cos, sin = compute_rope_embeddings(positions, dtype=hidden_states.dtype)

    # Create attention mask if padding is provided
    attn_mask_4d = create_causal_mask(seq_len, device, hidden_states.dtype, attention_mask)

    # Transformer layers
    for layer_idx in range(num_layers):
        hidden_states = transformer_layer(hidden_states, weights, layer_idx, cos, sin, attn_mask_4d)

    # Final norm
    hidden_states = rms_norm(hidden_states, weights["model.norm.weight"])

    # LM head (tied to embedding weights in Qwen3)
    logits = F.linear(hidden_states, weights["model.embed_tokens.weight"])

    return logits


# For testing
if __name__ == "__main__":
    import sys

    print("Testing functional Qwen3 implementation...")

    # Check if we have GPU
    if not torch.cuda.is_available():
        print("No GPU available. Run on GPU for full test.")
        print("Testing shape logic with random weights...")

        # Create random weights for shape testing
        weights = {
            "model.embed_tokens.weight": torch.randn(VOCAB_SIZE, HIDDEN_SIZE),
            "model.norm.weight": torch.randn(HIDDEN_SIZE),
        }

        for i in range(NUM_LAYERS):
            prefix = f"model.layers.{i}"
            weights[f"{prefix}.input_layernorm.weight"] = torch.randn(HIDDEN_SIZE)
            weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(HIDDEN_SIZE)
            # Q, K, V projections (no bias in Qwen3)
            weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(
                NUM_HEADS * HEAD_DIM, HIDDEN_SIZE
            )
            weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(
                NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE
            )
            weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(
                NUM_KV_HEADS * HEAD_DIM, HIDDEN_SIZE
            )
            weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(
                HIDDEN_SIZE, NUM_HEADS * HEAD_DIM
            )
            # QKNorm weights (per head_dim)
            weights[f"{prefix}.self_attn.q_norm.weight"] = torch.randn(HEAD_DIM)
            weights[f"{prefix}.self_attn.k_norm.weight"] = torch.randn(HEAD_DIM)
            # MLP
            weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
            weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(INTERMEDIATE_SIZE, HIDDEN_SIZE)
            weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(HIDDEN_SIZE, INTERMEDIATE_SIZE)

        input_ids = torch.tensor([[1, 2, 3, 4]])
        logits = qwen3_forward(input_ids, weights)
        print(f"Output shape: {logits.shape}")
        print(f"Expected shape: (1, 4, {VOCAB_SIZE})")
        assert logits.shape == (1, 4, VOCAB_SIZE), "Shape mismatch!"
        print("Shape test passed!")
        sys.exit(0)

    # Full test with real model
    from transformers import AutoModelForCausalLM

    print("Loading Qwen3-0.6B...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B",
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()

    weights = {k: v for k, v in model.state_dict().items()}

    input_ids = torch.tensor([[1, 2, 3, 4]], device="cuda:0")

    print("Running original model...")
    with torch.no_grad():
        original_logits = model(input_ids).logits

    print("Running functional implementation...")
    with torch.no_grad():
        functional_logits = qwen3_forward(input_ids, weights)

    print(f"Original shape: {original_logits.shape}")
    print(f"Functional shape: {functional_logits.shape}")

    matches = torch.allclose(original_logits, functional_logits, rtol=1e-5, atol=1e-5)
    max_diff = (original_logits - functional_logits).abs().max().item()

    print(f"Matches: {matches}")
    print(f"Max diff: {max_diff:.6e}")

    if matches:
        print("SUCCESS! Functional implementation matches original.")
    else:
        print("FAILURE! Outputs do not match.")
        sys.exit(1)
