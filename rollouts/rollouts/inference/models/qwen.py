"""Qwen model implementations for inference.

Supports:
- Qwen2ForCausalLM: Similar to Llama with attention bias
- Qwen3ForCausalLM: Adds QK-Norm to attention

Architecture differences from Llama:
- Qwen2: Has attention bias (q, k, v, o projections have bias)
- Qwen3: Has QK-Norm (RMSNorm on Q and K before attention), no attention bias
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from ..attention.backend import AttentionBackend, AttentionMetadata
from ..layers.activation import silu_and_mul
from ..layers.linear import GateUpParallelLinear, QKVParallelLinear, RowParallelLinear
from ..layers.norm import RMSNorm
from ..layers.rotary import RotaryEmbedding, apply_rotary_pos_emb
from .config import ModelConfig

if TYPE_CHECKING:
    from ..tp import TPConfig


# ═══════════════════════════════════════════════════════════════════════════════
# QWEN ATTENTION (supports both Qwen2 and Qwen3 variants)
# ═══════════════════════════════════════════════════════════════════════════════


class QwenAttention(nn.Module):
    """Qwen attention layer with optional QK-Norm and attention bias.

    - Qwen2: has_attn_bias=True, has_qk_norm=False
    - Qwen3: has_attn_bias=False, has_qk_norm=True
    """

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
        has_attn_bias: bool = False,
        has_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.rotary = rotary
        self.has_qk_norm = has_qk_norm

        # QKV projection (merged) - column parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            bias=has_attn_bias,
            dtype=dtype,
            tp=tp,
        )

        # Store local head counts (after TP sharding)
        self.num_q_heads = self.qkv_proj.local_num_q_heads
        self.num_kv_heads = self.qkv_proj.local_num_kv_heads

        # QK-Norm (Qwen3 only) - applied per head
        if has_qk_norm:
            self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, dtype)
            self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, dtype)
        else:
            self.q_norm = None
            self.k_norm = None

        # Output projection - row parallel
        self.o_proj = RowParallelLinear(
            in_features=config.num_attention_heads * config.head_dim,
            out_features=config.hidden_size,
            bias=has_attn_bias,
            dtype=dtype,
            tp=tp,
        )

    def forward(
        self,
        hidden_states: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        # QKV projection
        q, k, v = self.qkv_proj(hidden_states)

        # Reshape for attention
        q = q.view(-1, self.num_q_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # Apply QK-Norm if enabled (before RoPE, per Qwen3 paper)
        if self.has_qk_norm:
            # Apply norm per head: [tokens, heads, dim] -> normalize over dim
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Apply rotary embeddings
        cos, sin = self.rotary(positions)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Attention
        attn_output = attn_backend.forward(
            q=q,
            k=k,
            v=v,
            layer_idx=self.layer_idx,
            metadata=attn_metadata,
            out_loc=out_loc,
        )

        # Reshape and project
        attn_output = attn_output.view(-1, self.num_q_heads * self.head_dim)
        return self.o_proj(attn_output)


# ═══════════════════════════════════════════════════════════════════════════════
# QWEN MLP (same as Llama - SwiGLU)
# ═══════════════════════════════════════════════════════════════════════════════


class QwenMLP(nn.Module):
    """Qwen MLP layer (SwiGLU) - identical to Llama."""

    def __init__(
        self,
        config: ModelConfig,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()

        self.gate_up_proj = GateUpParallelLinear(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=dtype,
            tp=tp,
        )

        self.down_proj = RowParallelLinear(
            in_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
            dtype=dtype,
            tp=tp,
        )

    def forward(self, x: Tensor) -> Tensor:
        gate_up = self.gate_up_proj(x)
        hidden = silu_and_mul(gate_up)
        return self.down_proj(hidden)


# ═══════════════════════════════════════════════════════════════════════════════
# QWEN2 MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class Qwen2DecoderLayer(nn.Module):
    """Qwen2 transformer layer."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        # Qwen2: has attention bias, no QK-Norm
        self.self_attn = QwenAttention(
            config, layer_idx, rotary, dtype, tp, has_attn_bias=True, has_qk_norm=False
        )
        self.mlp = QwenMLP(config, dtype, tp)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_states, residual = self.input_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.self_attn(
            hidden_states, positions, attn_backend, attn_metadata, out_loc
        )
        hidden_states, residual = self.post_attention_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):
    """Qwen2 model (without LM head)."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary = RotaryEmbedding(config.rotary_config, device)
        self.layers = nn.ModuleList([
            Qwen2DecoderLayer(config, i, self.rotary, dtype, tp)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual: Tensor | None = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, positions, attn_backend, attn_metadata, out_loc
            )

        hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):
    """Qwen2 for causal language modeling."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config, device, dtype, tp)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        hidden_states = self.model(input_ids, positions, attn_backend, attn_metadata, out_loc)
        return self.lm_head(hidden_states)

    def load_weights(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# QWEN3 MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class Qwen3DecoderLayer(nn.Module):
    """Qwen3 transformer layer with QK-Norm."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        # Qwen3: no attention bias, has QK-Norm
        self.self_attn = QwenAttention(
            config, layer_idx, rotary, dtype, tp, has_attn_bias=False, has_qk_norm=True
        )
        self.mlp = QwenMLP(config, dtype, tp)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> tuple[Tensor, Tensor]:
        hidden_states, residual = self.input_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.self_attn(
            hidden_states, positions, attn_backend, attn_metadata, out_loc
        )
        hidden_states, residual = self.post_attention_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3Model(nn.Module):
    """Qwen3 model (without LM head)."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary = RotaryEmbedding(config.rotary_config, device)
        self.layers = nn.ModuleList([
            Qwen3DecoderLayer(config, i, self.rotary, dtype, tp)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual: Tensor | None = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, residual, positions, attn_backend, attn_metadata, out_loc
            )

        hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual)
        return hidden_states


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 for causal language modeling."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device, dtype, tp)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        hidden_states = self.model(input_ids, positions, attn_backend, attn_metadata, out_loc)
        return self.lm_head(hidden_states)

    def load_weights(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# QWEN3-MOE MODEL
# ═══════════════════════════════════════════════════════════════════════════════


class Qwen3MoeDecoderLayer(nn.Module):
    """Qwen3-MoE transformer layer with MoE MLP."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        from ..layers.moe import MoeMLP

        # Qwen3-MoE: has QK-Norm, uses MoE MLP
        self.self_attn = QwenAttention(
            config, layer_idx, rotary, dtype, tp, has_attn_bias=False, has_qk_norm=True
        )

        # MoE MLP instead of standard MLP
        self.mlp = MoeMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            dtype=dtype,
            tp=tp,
        )

        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        hidden_states: Tensor,
        residual: Tensor | None,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
        return_routed_experts: bool = False,
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
        hidden_states, residual = self.input_layernorm.forward_with_residual(
            hidden_states, residual
        )
        hidden_states = self.self_attn(
            hidden_states, positions, attn_backend, attn_metadata, out_loc
        )
        hidden_states, residual = self.post_attention_layernorm.forward_with_residual(
            hidden_states, residual
        )

        if return_routed_experts:
            hidden_states, routed_experts = self.mlp(hidden_states, return_routed_experts=True)
            return hidden_states, residual, routed_experts
        else:
            hidden_states = self.mlp(hidden_states)
            return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    """Qwen3-MoE model (without LM head)."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.num_layers = config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)
        self.rotary = RotaryEmbedding(config.rotary_config, device)
        self.layers = nn.ModuleList([
            Qwen3MoeDecoderLayer(config, i, self.rotary, dtype, tp)
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
        return_routed_experts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        hidden_states = self.embed_tokens(input_ids)
        residual: Tensor | None = None

        if return_routed_experts:
            all_routed_experts = []
            for layer in self.layers:
                hidden_states, residual, routed = layer(
                    hidden_states,
                    residual,
                    positions,
                    attn_backend,
                    attn_metadata,
                    out_loc,
                    return_routed_experts=True,
                )
                all_routed_experts.append(routed)

            hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual)
            # Stack: [num_layers, num_tokens, top_k]
            routed_experts = torch.stack(all_routed_experts, dim=0)
            return hidden_states, routed_experts
        else:
            for layer in self.layers:
                hidden_states, residual = layer(
                    hidden_states, residual, positions, attn_backend, attn_metadata, out_loc
                )
            hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual)
            return hidden_states


class Qwen3MoeForCausalLM(nn.Module):
    """Qwen3-MoE for causal language modeling.

    Replaces standard MLP with MoE layer. For RL training, use
    return_routed_experts=True to track expert routing for replay.
    """

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config, device, dtype, tp)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
        return_routed_experts: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            return_routed_experts: If True, return (logits, routed_experts)
                where routed_experts is [num_layers, num_tokens, top_k] int32

        Returns:
            logits or (logits, routed_experts)
        """
        if return_routed_experts:
            hidden_states, routed_experts = self.model(
                input_ids,
                positions,
                attn_backend,
                attn_metadata,
                out_loc,
                return_routed_experts=True,
            )
            logits = self.lm_head(hidden_states)
            return logits, routed_experts
        else:
            hidden_states = self.model(input_ids, positions, attn_backend, attn_metadata, out_loc)
            return self.lm_head(hidden_states)

    def load_weights(self, state_dict: dict[str, Tensor]) -> None:
        self.load_state_dict(state_dict, strict=False)


# ═══════════════════════════════════════════════════════════════════════════════
# WEIGHT REMAPPING
# ═══════════════════════════════════════════════════════════════════════════════


def remap_weights_qwen(
    hf_weights: dict[str, Tensor],
    num_layers: int,
) -> dict[str, Tensor]:
    """Remap HuggingFace Qwen weights to our naming convention.

    Qwen uses same naming as Llama, so this is essentially the same mapping.
    The main difference is handling the QK-Norm weights for Qwen3.
    """
    remapped = {}

    # Direct mappings (same as Llama)
    direct_map = {
        "model.embed_tokens.weight": "model.embed_tokens.weight",
        "model.norm.weight": "model.norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    for hf_name, our_name in direct_map.items():
        if hf_name in hf_weights:
            remapped[our_name] = hf_weights[hf_name]

    # Per-layer mappings
    for i in range(num_layers):
        hf_prefix = f"model.layers.{i}"
        our_prefix = f"model.layers.{i}"

        layer_map = {
            # Attention
            f"{hf_prefix}.self_attn.q_proj.weight": f"{our_prefix}.self_attn.qkv_proj.q_weight",
            f"{hf_prefix}.self_attn.k_proj.weight": f"{our_prefix}.self_attn.qkv_proj.k_weight",
            f"{hf_prefix}.self_attn.v_proj.weight": f"{our_prefix}.self_attn.qkv_proj.v_weight",
            f"{hf_prefix}.self_attn.q_proj.bias": f"{our_prefix}.self_attn.qkv_proj.q_bias",
            f"{hf_prefix}.self_attn.k_proj.bias": f"{our_prefix}.self_attn.qkv_proj.k_bias",
            f"{hf_prefix}.self_attn.v_proj.bias": f"{our_prefix}.self_attn.qkv_proj.v_bias",
            f"{hf_prefix}.self_attn.o_proj.weight": f"{our_prefix}.self_attn.o_proj.weight",
            f"{hf_prefix}.self_attn.o_proj.bias": f"{our_prefix}.self_attn.o_proj.bias",
            # QK-Norm (Qwen3)
            f"{hf_prefix}.self_attn.q_norm.weight": f"{our_prefix}.self_attn.q_norm.weight",
            f"{hf_prefix}.self_attn.k_norm.weight": f"{our_prefix}.self_attn.k_norm.weight",
            # MLP
            f"{hf_prefix}.mlp.gate_proj.weight": f"{our_prefix}.mlp.gate_up_proj.gate_weight",
            f"{hf_prefix}.mlp.up_proj.weight": f"{our_prefix}.mlp.gate_up_proj.up_weight",
            f"{hf_prefix}.mlp.down_proj.weight": f"{our_prefix}.mlp.down_proj.weight",
            # Norms
            f"{hf_prefix}.input_layernorm.weight": f"{our_prefix}.input_layernorm.weight",
            f"{hf_prefix}.post_attention_layernorm.weight": f"{our_prefix}.post_attention_layernorm.weight",
        }

        for hf_name, our_name in layer_map.items():
            if hf_name in hf_weights:
                remapped[our_name] = hf_weights[hf_name]

    return remapped
