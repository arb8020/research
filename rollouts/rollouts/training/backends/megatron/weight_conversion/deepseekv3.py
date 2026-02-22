"""DeepSeekV3 / GLM-4.7-Flash weight conversion.

Converts Megatron state dict keys to HuggingFace format.
GLM-4.7-Flash (glm4moelite) uses the DeepSeekV3 architecture.

Ported from SLIME with minimal changes.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def convert_deepseekv3_to_hf(
    args: Any,
    name: str,
    param: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    """Convert DeepSeekV3/GLM-4.7 Megatron param to HuggingFace format.

    Args:
        args: Namespace with model config (num_attention_heads, hidden_size, etc.)
        name: Megatron parameter name
        param: Parameter tensor

    Returns:
        List of (hf_name, tensor) tuples
    """
    import torch

    # Embedding and output layers
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    # Compute head dimensions
    try:
        head_dim = (
            args.kv_channels
            if args.kv_channels is not None
            else args.hidden_size // args.num_attention_heads
        )
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # Decoder layers
    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # MoE experts
        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                        gate_weight,
                    ),
                    (
                        f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                        up_weight,
                    ),
                ]
            elif rest == "linear_fc2":
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
            else:
                raise ValueError(f"Unknown expert parameter: {name}")

        # Shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter: {name}")

        # Attention projections
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_q_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_proj.weight", param)]
        elif rest == "self_attention.linear_q_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_q_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]

        # Indexer (for models with indexer attention)
        elif rest == "self_attention.wq_b.weight":
            wq_b = param
            wq_b = wq_b.view(-1, 128, wq_b.shape[-1])
            wq_b = torch.cat([wq_b[:, 64:], wq_b[:, :64]], dim=1).view(-1, wq_b.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", wq_b)]
        elif rest == "self_attention.wk.weight":
            wk = param
            wk = torch.cat([wk[64:], wk[:64]], dim=0).view(-1, wk.shape[-1])
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", wk)]
        elif rest == "self_attention.weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]
        elif rest == "self_attention.k_norm.weight":
            knorm_weight = torch.cat([param[64:], param[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", knorm_weight)]
        elif rest == "self_attention.k_norm.bias":
            knorm_bias = torch.cat([param[64:], param[:64]], dim=0)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", knorm_bias)]

        # QKV bias
        elif rest == "self_attention.linear_qkv.bias":
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias.contiguous().flatten()),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias.contiguous().flatten()),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias.contiguous().flatten()),
            ]

        # MLP (non-MoE)
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

        # Layer norms
        elif (
            rest == "self_attention.linear_qkv.layer_norm_weight"
            or rest == "input_layernorm.weight"
        ):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # KV projections (for latent attention)
        elif rest == "self_attention.linear_kv_down_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)]
        elif rest == "self_attention.linear_kv_up_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]

        # Router
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

    # MTP layers (multi-token prediction)
    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        layer_idx = int(layer_idx) + args.num_layers
        if rest == "eh_proj.weight":
            return [(f"model.layers.{layer_idx}.eh_proj.weight", param)]
        elif rest == "enorm.weight":
            return [(f"model.layers.{layer_idx}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            return [(f"model.layers.{layer_idx}.hnorm.weight", param)]
        elif rest == "final_layernorm.weight":
            return [(f"model.layers.{layer_idx}.shared_head.norm.weight", param)]
        else:
            # Recurse for transformer layer params within MTP
            new_name = f"module.module.decoder.layers.{layer_idx}.{rest}"
            new_name = new_name.replace("transformer_layer.", "")
            return convert_deepseekv3_to_hf(args, new_name, param)

    raise ValueError(f"Unknown parameter name: {name}")
