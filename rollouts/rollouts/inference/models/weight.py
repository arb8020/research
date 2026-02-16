"""Weight loading from HuggingFace models."""

from __future__ import annotations

import torch
from torch import Tensor


def load_weights(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Tensor]:
    """Load weights from HuggingFace model.

    Args:
        model_path: HuggingFace model name or local path
        device: Target device
        dtype: Target dtype

    Returns:
        Dict mapping parameter names to tensors
    """
    from safetensors import safe_open
    from transformers.utils import cached_file

    # Find safetensors files
    try:
        # Try single file first
        safetensors_path = cached_file(
            model_path,
            "model.safetensors",
        )
        files = [safetensors_path]
    except Exception:
        # Try sharded files
        import json

        index_path = cached_file(
            model_path,
            "model.safetensors.index.json",
        )
        with open(index_path) as f:
            index = json.load(f)

        # Get unique shard files
        shard_files = set(index["weight_map"].values())
        files = [cached_file(model_path, shard_file) for shard_file in sorted(shard_files)]

    # Load all weights
    state_dict: dict[str, Tensor] = {}
    for filepath in files:
        with safe_open(filepath, framework="pt", device=str(device)) as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                state_dict[key] = tensor.to(dtype)

    return state_dict


def remap_weights_llama(
    hf_state_dict: dict[str, Tensor],
    num_layers: int,
) -> dict[str, Tensor]:
    """Remap HuggingFace Llama weights to our model structure.

    HuggingFace naming:
        model.embed_tokens.weight
        model.layers.{i}.self_attn.q_proj.weight
        model.layers.{i}.self_attn.k_proj.weight
        model.layers.{i}.self_attn.v_proj.weight
        model.layers.{i}.self_attn.o_proj.weight
        model.layers.{i}.mlp.gate_proj.weight
        model.layers.{i}.mlp.up_proj.weight
        model.layers.{i}.mlp.down_proj.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight

    Our model structure (LlamaForCausalLM):
        model.embed_tokens.weight
        model.layers.{i}.self_attn.qkv_proj.linear.weight  (merged QKV)
        model.layers.{i}.self_attn.o_proj.linear.weight
        model.layers.{i}.mlp.gate_up_proj.linear.weight    (merged gate+up)
        model.layers.{i}.mlp.down_proj.linear.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.norm.weight
        lm_head.weight  (nn.Linear, not wrapped)
    """
    remapped: dict[str, Tensor] = {}

    # Embedding (under model.)
    remapped["model.embed_tokens.weight"] = hf_state_dict["model.embed_tokens.weight"]

    # Final norm (under model.)
    remapped["model.norm.weight"] = hf_state_dict["model.norm.weight"]

    # LM head (directly on LlamaForCausalLM, uses nn.Linear)
    if "lm_head.weight" in hf_state_dict:
        remapped["lm_head.weight"] = hf_state_dict["lm_head.weight"]
    else:
        # Tied embeddings
        remapped["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

    # Layers (under model.)
    for i in range(num_layers):
        prefix_hf = f"model.layers.{i}"
        prefix_ours = f"model.layers.{i}"

        # Merge Q, K, V projections
        q = hf_state_dict[f"{prefix_hf}.self_attn.q_proj.weight"]
        k = hf_state_dict[f"{prefix_hf}.self_attn.k_proj.weight"]
        v = hf_state_dict[f"{prefix_hf}.self_attn.v_proj.weight"]
        qkv = torch.cat([q, k, v], dim=0)
        remapped[f"{prefix_ours}.self_attn.qkv_proj.linear.weight"] = qkv

        # O projection
        remapped[f"{prefix_ours}.self_attn.o_proj.linear.weight"] = hf_state_dict[
            f"{prefix_hf}.self_attn.o_proj.weight"
        ]

        # Merge gate and up projections
        gate = hf_state_dict[f"{prefix_hf}.mlp.gate_proj.weight"]
        up = hf_state_dict[f"{prefix_hf}.mlp.up_proj.weight"]
        gate_up = torch.cat([gate, up], dim=0)
        remapped[f"{prefix_ours}.mlp.gate_up_proj.linear.weight"] = gate_up

        # Down projection
        remapped[f"{prefix_ours}.mlp.down_proj.linear.weight"] = hf_state_dict[
            f"{prefix_hf}.mlp.down_proj.weight"
        ]

        # Layer norms (no .linear suffix)
        remapped[f"{prefix_ours}.input_layernorm.weight"] = hf_state_dict[
            f"{prefix_hf}.input_layernorm.weight"
        ]
        remapped[f"{prefix_ours}.post_attention_layernorm.weight"] = hf_state_dict[
            f"{prefix_hf}.post_attention_layernorm.weight"
        ]

    return remapped
