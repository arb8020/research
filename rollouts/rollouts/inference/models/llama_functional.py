"""Llama model as pure functions.

No classes, no nn.Module. Just functions that take (input_ids, weights, ...) -> logits.
This makes the computation explicit and debuggable.

Usage:
    from transformers import AutoModelForCausalLM
    weights = dict(AutoModelForCausalLM.from_pretrained(...).state_dict())
    logits = forward(input_ids, weights, config)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LlamaConfig:
    """Minimal config for functional forward."""

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position: int = 8192

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @property
    def num_q_per_kv(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads


def load_config(model_name: str) -> LlamaConfig:
    """Load config from HuggingFace model."""
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(model_name)
    rope_theta = getattr(hf_config, "rope_theta", None)
    if rope_theta is None:
        rope_params = getattr(hf_config, "rope_parameters", None)
        if isinstance(rope_params, dict):
            rope_theta = rope_params.get("rope_theta", rope_params.get("theta"))
    if rope_theta is None:
        rope_scaling = getattr(hf_config, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_scaling.get("theta"))
    if rope_theta is None:
        rope_theta = 10000.0

    return LlamaConfig(
        hidden_size=hf_config.hidden_size,
        num_hidden_layers=hf_config.num_hidden_layers,
        num_attention_heads=hf_config.num_attention_heads,
        num_key_value_heads=hf_config.num_key_value_heads,
        intermediate_size=hf_config.intermediate_size,
        vocab_size=hf_config.vocab_size,
        rms_norm_eps=hf_config.rms_norm_eps,
        rope_theta=float(rope_theta),
        max_position=getattr(hf_config, "max_position_embeddings", 8192),
    )


# =============================================================================
# PRIMITIVES
# =============================================================================


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """RMSNorm: x * rsqrt(mean(x^2) + eps) * weight"""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    # Match HuggingFace: convert to input_dtype BEFORE multiplying with weight
    return weight * x.to(input_dtype)


def rotary_embedding(
    positions: Tensor, head_dim: int, theta: float, device: torch.device
) -> tuple[Tensor, Tensor]:
    """Compute cos/sin for rotary embeddings.

    Returns:
        (cos, sin) each shape [seq_len, head_dim]
    """
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    # [seq_len, head_dim/2]
    freqs = torch.outer(positions.float(), inv_freq)
    # [seq_len, head_dim]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embedding to x.

    x: [seq_len, num_heads, head_dim]
    cos, sin: [seq_len, head_dim]
    """
    # Reshape for broadcasting: [seq_len, 1, head_dim]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)

    # Match HuggingFace exactly:
    # q_embed = (q * cos) + (rotate_half(q) * sin)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return (x * cos) + (torch.cat((-x2, x1), dim=-1) * sin)


def silu_mul(x: Tensor) -> Tensor:
    """SiLU(gate) * up for SwiGLU MLP."""
    gate, up = x.chunk(2, dim=-1)
    return F.silu(gate) * up


# =============================================================================
# ATTENTION (no KV cache - pure prefill)
# =============================================================================


def attention(
    hidden_states: Tensor,
    q_weight: Tensor,
    k_weight: Tensor,
    v_weight: Tensor,
    o_weight: Tensor,
    cos: Tensor,
    sin: Tensor,
    config: LlamaConfig,
) -> Tensor:
    """Self-attention as pure function.

    hidden_states: [seq_len, hidden_size]
    Returns: [seq_len, hidden_size]
    """
    seq_len = hidden_states.shape[0]

    # Project
    q = F.linear(hidden_states, q_weight)  # [seq_len, num_heads * head_dim]
    k = F.linear(hidden_states, k_weight)  # [seq_len, num_kv_heads * head_dim]
    v = F.linear(hidden_states, v_weight)  # [seq_len, num_kv_heads * head_dim]

    # Reshape to [seq_len, num_heads, head_dim]
    q = q.view(seq_len, config.num_attention_heads, config.head_dim)
    k = k.view(seq_len, config.num_key_value_heads, config.head_dim)
    v = v.view(seq_len, config.num_key_value_heads, config.head_dim)

    # Apply rotary embeddings
    q = apply_rotary(q, cos, sin)
    k = apply_rotary(k, cos, sin)

    # Expand KV for GQA: [seq_len, num_kv_heads, head_dim] -> [seq_len, num_heads, head_dim]
    if config.num_q_per_kv > 1:
        k = k.repeat_interleave(config.num_q_per_kv, dim=1)
        v = v.repeat_interleave(config.num_q_per_kv, dim=1)

    # Match HuggingFace SDPA input shape: [batch, num_heads, seq_len, head_dim]
    q = q.transpose(0, 1).unsqueeze(0)
    k = k.transpose(0, 1).unsqueeze(0)
    v = v.transpose(0, 1).unsqueeze(0)

    # Scaled dot-product attention with causal mask
    # Using PyTorch's optimized SDPA
    attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # Reshape back: [seq_len, num_heads * head_dim]
    attn_output = attn_output.squeeze(0).transpose(0, 1).contiguous().view(seq_len, -1)

    # Output projection
    return F.linear(attn_output, o_weight)


def mlp(
    hidden_states: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """SwiGLU MLP as pure function."""
    gate = F.linear(hidden_states, gate_weight)
    up = F.linear(hidden_states, up_weight)
    hidden = F.silu(gate) * up
    return F.linear(hidden, down_weight)


def transformer_layer(
    hidden_states: Tensor,
    weights: dict[str, Tensor],
    layer_idx: int,
    cos: Tensor,
    sin: Tensor,
    config: LlamaConfig,
) -> Tensor:
    """Single transformer layer."""
    prefix = f"model.layers.{layer_idx}"

    # Pre-attention norm
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states,
        weights[f"{prefix}.input_layernorm.weight"],
        config.rms_norm_eps,
    )

    # Self-attention
    hidden_states = attention(
        hidden_states,
        q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
        k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
        v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
        o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
        cos=cos,
        sin=sin,
        config=config,
    )
    hidden_states = residual + hidden_states

    # Pre-MLP norm
    residual = hidden_states
    hidden_states = rms_norm(
        hidden_states,
        weights[f"{prefix}.post_attention_layernorm.weight"],
        config.rms_norm_eps,
    )

    # MLP
    hidden_states = mlp(
        hidden_states,
        gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
        up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
        down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
    )
    hidden_states = residual + hidden_states

    return hidden_states


# =============================================================================
# FULL FORWARD
# =============================================================================


def forward(
    input_ids: Tensor,
    weights: dict[str, Tensor],
    config: LlamaConfig,
) -> Tensor:
    """Full forward pass as pure function.

    input_ids: [seq_len] or [batch, seq_len]
    weights: state_dict from HuggingFace model
    config: model configuration

    Returns: logits [seq_len, vocab_size] or [batch, seq_len, vocab_size]
    """
    # Handle batch dimension
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, seq_len = input_ids.shape
    device = input_ids.device

    # Embedding
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])

    # Compute rotary embeddings once
    positions = torch.arange(seq_len, device=device)
    cos, sin = rotary_embedding(positions, config.head_dim, config.rope_theta, device)
    # Cast to model dtype
    cos = cos.to(hidden_states.dtype)
    sin = sin.to(hidden_states.dtype)

    # Process each batch item separately (for simplicity)
    outputs = []
    for b in range(batch_size):
        h = hidden_states[b]  # [seq_len, hidden_size]

        # Transformer layers
        for layer_idx in range(config.num_hidden_layers):
            h = transformer_layer(h, weights, layer_idx, cos, sin, config)

        # Final norm
        h = rms_norm(h, weights["model.norm.weight"], config.rms_norm_eps)

        # LM head
        logits = F.linear(h, weights["lm_head.weight"])
        outputs.append(logits)

    logits = torch.stack(outputs, dim=0)

    if squeeze_output:
        logits = logits.squeeze(0)

    return logits


# =============================================================================
# TESTING
# =============================================================================


def find_divergence(model_name: str = "HuggingFaceTB/SmolLM2-135M") -> str | None:
    """Find exactly where our forward diverges from HuggingFace.

    Returns the name of the diverging component, or None if everything matches.
    """
    import logging

    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    logger.info(f"Finding divergence in {model_name}...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    weights = {k: v.to(device) for k, v in hf_model.state_dict().items()}
    config = load_config(model_name)

    prompt = "Hi"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Capture HF intermediate outputs
    hf_outputs = {}

    def make_hook(name: str) -> Callable[[Any, Any, Any], None]:
        def hook(module: Any, inp: Any, out: Any) -> None:
            if isinstance(out, tuple):
                hf_outputs[name] = out[0].detach().clone()
            else:
                hf_outputs[name] = out.detach().clone()

        return hook

    handles = []
    handles.append(hf_model.model.embed_tokens.register_forward_hook(make_hook("embed")))
    for i, layer in enumerate(hf_model.model.layers):
        handles.append(
            layer.input_layernorm.register_forward_hook(make_hook(f"layer_{i}_input_norm"))
        )
        handles.append(layer.self_attn.register_forward_hook(make_hook(f"layer_{i}_attn")))
        handles.append(
            layer.post_attention_layernorm.register_forward_hook(make_hook(f"layer_{i}_post_norm"))
        )
        handles.append(layer.mlp.register_forward_hook(make_hook(f"layer_{i}_mlp")))
        handles.append(layer.register_forward_hook(make_hook(f"layer_{i}")))
    handles.append(hf_model.model.norm.register_forward_hook(make_hook("final_norm")))

    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    for h in handles:
        h.remove()

    # Now run our forward step by step
    seq_len = input_ids.shape[1]
    positions = torch.arange(seq_len, device=device)

    # Test RoPE vs HuggingFace
    our_cos, our_sin = rotary_embedding(positions, config.head_dim, config.rope_theta, device)
    # HF's rotary values are captured in each layer's attention
    # For a simple test, we'll check the first layer's input vs our computation
    logger.info(f"Testing with seq_len={seq_len}")

    # Embedding
    hidden_states = F.embedding(input_ids, weights["model.embed_tokens.weight"])
    hf_embed = hf_outputs["embed"]
    diff = (hidden_states.float() - hf_embed.float()).abs().max().item()
    logger.info(f"embed: max_diff={diff:.2e}")
    if diff > 1e-5:
        return "embed"

    # RoPE
    cos, sin = rotary_embedding(positions, config.head_dim, config.rope_theta, device)
    cos = cos.to(dtype)
    sin = sin.to(dtype)

    h = hidden_states[0]  # [seq_len, hidden]

    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"

        # Input layernorm
        residual = h
        h_normed = rms_norm(h, weights[f"{prefix}.input_layernorm.weight"], config.rms_norm_eps)
        hf_normed = hf_outputs[f"layer_{layer_idx}_input_norm"]
        if hf_normed.dim() == 3:
            hf_normed = hf_normed[0]
        diff = (h_normed.float() - hf_normed.float()).abs().max().item()
        logger.info(f"layer_{layer_idx}_input_norm: max_diff={diff:.2e}")

        # Test if using HF's norm output fixes attention divergence
        if layer_idx == 0 and diff > 1e-5:
            # Test attention with HF's input_norm output
            attn_with_hf_input = attention(
                hf_normed.to(dtype),
                q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
                k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
                v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
                o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
                cos=cos,
                sin=sin,
                config=config,
            )
            hf_attn_check = hf_outputs[f"layer_{layer_idx}_attn"]
            if isinstance(hf_attn_check, tuple):
                hf_attn_check = hf_attn_check[0]
            if hf_attn_check.dim() == 3:
                hf_attn_check = hf_attn_check[0]
            attn_diff_with_hf_input = (
                (attn_with_hf_input.float() - hf_attn_check.float()).abs().max().item()
            )
            logger.info(f"  attn diff using HF norm output: {attn_diff_with_hf_input:.2e}")

        if diff > 1e-2:
            return f"layer_{layer_idx}_input_norm"

        # Attention
        attn_out = attention(
            h_normed,
            q_weight=weights[f"{prefix}.self_attn.q_proj.weight"],
            k_weight=weights[f"{prefix}.self_attn.k_proj.weight"],
            v_weight=weights[f"{prefix}.self_attn.v_proj.weight"],
            o_weight=weights[f"{prefix}.self_attn.o_proj.weight"],
            cos=cos,
            sin=sin,
            config=config,
        )
        hf_attn = hf_outputs[f"layer_{layer_idx}_attn"]
        if isinstance(hf_attn, tuple):
            hf_attn = hf_attn[0]
        if hf_attn.dim() == 3:
            hf_attn = hf_attn[0]
        diff = (attn_out.float() - hf_attn.float()).abs().max().item()
        logger.info(f"layer_{layer_idx}_attn: max_diff={diff:.2e}")
        if diff > 1e-2:
            return f"layer_{layer_idx}_attn"

        h = residual + attn_out

        # Post-attention layernorm
        residual = h
        h_normed = rms_norm(
            h, weights[f"{prefix}.post_attention_layernorm.weight"], config.rms_norm_eps
        )

        # Check post-attention norm
        hf_post_norm = hf_outputs[f"layer_{layer_idx}_post_norm"]
        if hf_post_norm.dim() == 3:
            hf_post_norm = hf_post_norm[0]
        diff = (h_normed.float() - hf_post_norm.float()).abs().max().item()
        logger.info(f"layer_{layer_idx}_post_norm: max_diff={diff:.2e}")
        if diff > 1e-2:
            return f"layer_{layer_idx}_post_norm"

        # MLP
        mlp_out = mlp(
            h_normed,
            gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
            up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
            down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
        )
        hf_mlp = hf_outputs[f"layer_{layer_idx}_mlp"]
        if hf_mlp.dim() == 3:
            hf_mlp = hf_mlp[0]
        diff = (mlp_out.float() - hf_mlp.float()).abs().max().item()
        logger.info(
            f"layer_{layer_idx}_mlp: max_diff={diff:.2e}, shapes: ours={mlp_out.shape}, hf={hf_mlp.shape}"
        )
        if diff > 1e-2:
            # Debug: print more info
            logger.info(
                f"  mlp_out range: [{mlp_out.min().item():.3f}, {mlp_out.max().item():.3f}]"
            )
            logger.info(f"  hf_mlp range: [{hf_mlp.min().item():.3f}, {hf_mlp.max().item():.3f}]")
            logger.info(
                f"  h_normed range: [{h_normed.min().item():.3f}, {h_normed.max().item():.3f}]"
            )
            # Check if it's an input issue
            hf_post_norm_for_mlp = hf_outputs[f"layer_{layer_idx}_post_norm"]
            if hf_post_norm_for_mlp.dim() == 3:
                hf_post_norm_for_mlp = hf_post_norm_for_mlp[0]
            mlp_input_diff = (h_normed.float() - hf_post_norm_for_mlp.float()).abs().max().item()
            logger.info(f"  MLP input diff: {mlp_input_diff:.2e}")

            # Test MLP with HF input to isolate the bug
            hf_mlp_input = hf_post_norm_for_mlp.to(dtype)
            our_mlp_with_hf_input = mlp(
                hf_mlp_input,
                gate_weight=weights[f"{prefix}.mlp.gate_proj.weight"],
                up_weight=weights[f"{prefix}.mlp.up_proj.weight"],
                down_weight=weights[f"{prefix}.mlp.down_proj.weight"],
            )
            diff_with_hf_input = (our_mlp_with_hf_input.float() - hf_mlp.float()).abs().max().item()
            logger.info(f"  MLP diff using HF input: {diff_with_hf_input:.2e}")

            return f"layer_{layer_idx}_mlp"

        h = residual + mlp_out

        # Check full layer output
        hf_layer = hf_outputs[f"layer_{layer_idx}"]
        if isinstance(hf_layer, tuple):
            hf_layer = hf_layer[0]
        if hf_layer.dim() == 3:
            hf_layer = hf_layer[0]
        diff = (h.float() - hf_layer.float()).abs().max().item()
        logger.info(f"layer_{layer_idx}: max_diff={diff:.2e}")

    # Final norm
    h = rms_norm(h, weights["model.norm.weight"], config.rms_norm_eps)
    hf_final = hf_outputs["final_norm"]
    if hf_final.dim() == 3:
        hf_final = hf_final[0]
    diff = (h.float() - hf_final.float()).abs().max().item()
    logger.info(f"final_norm: max_diff={diff:.2e}")
    if diff > 1e-2:
        return "final_norm"

    # LM head
    our_logits = F.linear(h, weights["lm_head.weight"])
    diff = (our_logits.float() - hf_logits[0].float()).abs().max().item()
    logger.info(f"logits: max_diff={diff:.2e}")
    if diff > 1e-2:
        return "lm_head"

    return None


def test_vs_huggingface(model_name: str = "HuggingFaceTB/SmolLM2-135M") -> bool:
    """Test functional forward matches HuggingFace exactly."""
    import logging

    logger = logging.getLogger(__name__)

    # First find where divergence occurs
    divergence = find_divergence(model_name)
    if divergence:
        logger.error(f"DIVERGES AT: {divergence}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    logger.info(f"Testing {model_name} on {device}...")

    # Load HuggingFace model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    )
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get weights as dict
    weights = {k: v.to(device) for k, v in hf_model.state_dict().items()}

    # Load config
    config = load_config(model_name)

    # Test input
    prompt = "Hello, world"
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # HuggingFace forward
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    # Our forward
    with torch.no_grad():
        our_logits = forward(input_ids, weights, config)

    # Compare
    if our_logits.dim() == 2:
        our_logits = our_logits.unsqueeze(0)

    max_diff = (our_logits.float() - hf_logits.float()).abs().max().item()
    mean_diff = (our_logits.float() - hf_logits.float()).abs().mean().item()

    logger.info(f"Max diff: {max_diff:.2e}")
    logger.info(f"Mean diff: {mean_diff:.2e}")

    # Check argmax
    our_argmax = our_logits.argmax(dim=-1)
    hf_argmax = hf_logits.argmax(dim=-1)
    argmax_match = (our_argmax == hf_argmax).all().item()
    logger.info(f"Argmax match: {argmax_match}")

    # Per-position analysis
    for pos in range(our_logits.shape[1]):
        pos_diff = (our_logits[0, pos].float() - hf_logits[0, pos].float()).abs().max().item()
        logger.info(f"  Position {pos}: max_diff={pos_diff:.2e}")

    # Pass if within bf16 tolerance
    if max_diff < 1e-2:
        logger.info("PASS: Functional forward matches HuggingFace")
        return True
    else:
        logger.error(f"FAIL: max_diff={max_diff:.2e} exceeds 1e-2")
        return False


if __name__ == "__main__":
    import sys

    from rollouts._logging import setup_logging

    setup_logging(
        use_color=True,
        logger_levels={"httpx": "WARNING"},
        use_queue_handler=(sys.version_info >= (3, 12)),
    )
    success = test_vs_huggingface()
    sys.exit(0 if success else 1)
