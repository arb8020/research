"""Weight loading from HuggingFace models with tensor parallelism support."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..tp import TPConfig


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


def shard_weights_for_tp(
    state_dict: dict[str, Tensor],
    tp: TPConfig,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    intermediate_size: int,
) -> dict[str, Tensor]:
    """Shard weights for tensor parallelism.

    Each TP rank gets a subset of the weights based on layer type:
    - Column-parallel (QKV, gate_up): shard along output dim (dim 0)
    - Row-parallel (o_proj, down_proj): shard along input dim (dim 1)
    - Other weights (embeddings, norms, lm_head): keep full copy

    Args:
        state_dict: Full model weights (already remapped to our naming)
        tp: TP configuration with rank and world_size
        num_q_heads: Total number of Q heads
        num_kv_heads: Total number of KV heads
        head_dim: Head dimension
        intermediate_size: MLP intermediate size

    Returns:
        Sharded weights for this TP rank
    """
    if not tp.is_distributed:
        return state_dict

    from ..tp import get_local_size, shard_dim

    rank = tp.rank
    world_size = tp.world_size

    # Calculate local sizes
    local_q_heads = get_local_size(num_q_heads, world_size)
    local_kv_heads = get_local_size(num_kv_heads, world_size)
    local_intermediate = get_local_size(intermediate_size, world_size)

    # Calculate shard ranges
    q_start, _ = shard_dim(num_q_heads, rank, world_size)
    kv_start, _ = shard_dim(num_kv_heads, rank, world_size)
    intermediate_start, _ = shard_dim(intermediate_size, rank, world_size)

    sharded: dict[str, Tensor] = {}

    for name, tensor in state_dict.items():
        if "qkv_proj" in name:
            # QKV is column-parallel: [q_size + 2*kv_size, hidden] -> shard heads
            # Weight layout: [Q heads, K heads, V heads] along dim 0
            q_size = num_q_heads * head_dim
            kv_size = num_kv_heads * head_dim

            q_weight = tensor[:q_size]  # [q_size, hidden]
            k_weight = tensor[q_size : q_size + kv_size]  # [kv_size, hidden]
            v_weight = tensor[q_size + kv_size :]  # [kv_size, hidden]

            # Reshape to [num_heads, head_dim, hidden] to shard heads
            q_weight = q_weight.view(num_q_heads, head_dim, -1)
            k_weight = k_weight.view(num_kv_heads, head_dim, -1)
            v_weight = v_weight.view(num_kv_heads, head_dim, -1)

            # Take this rank's heads
            q_shard = q_weight[q_start : q_start + local_q_heads]
            k_shard = k_weight[kv_start : kv_start + local_kv_heads]
            v_shard = v_weight[kv_start : kv_start + local_kv_heads]

            # Reshape back and concatenate
            q_shard = q_shard.reshape(local_q_heads * head_dim, -1)
            k_shard = k_shard.reshape(local_kv_heads * head_dim, -1)
            v_shard = v_shard.reshape(local_kv_heads * head_dim, -1)

            sharded[name] = torch.cat([q_shard, k_shard, v_shard], dim=0)

        elif "o_proj" in name:
            # O proj is row-parallel: [hidden, q_size] -> shard input (heads)
            # Weight: [hidden, q_size] - shard along dim 1
            weight = tensor  # [hidden, q_size]
            q_size = num_q_heads * head_dim

            # Reshape to [hidden, num_heads, head_dim] to shard heads
            weight = weight.view(weight.shape[0], num_q_heads, head_dim)
            weight_shard = weight[:, q_start : q_start + local_q_heads, :]
            sharded[name] = weight_shard.reshape(weight.shape[0], local_q_heads * head_dim)

        elif "gate_up_proj" in name:
            # Gate+up is column-parallel: [2*intermediate, hidden] -> shard output
            # Layout: [gate, up] each of size intermediate_size
            gate_weight = tensor[:intermediate_size]  # [intermediate, hidden]
            up_weight = tensor[intermediate_size:]  # [intermediate, hidden]

            gate_shard = gate_weight[intermediate_start : intermediate_start + local_intermediate]
            up_shard = up_weight[intermediate_start : intermediate_start + local_intermediate]

            sharded[name] = torch.cat([gate_shard, up_shard], dim=0)

        elif "down_proj" in name:
            # Down proj is row-parallel: [hidden, intermediate] -> shard input
            weight = tensor  # [hidden, intermediate]
            weight_shard = weight[:, intermediate_start : intermediate_start + local_intermediate]
            sharded[name] = weight_shard

        else:
            # Embeddings, norms, lm_head: full copy on each rank
            sharded[name] = tensor

    return sharded
