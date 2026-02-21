"""Llama model implementation for inference with tensor parallelism.

Compatible with Llama 2, Llama 3, and similar architectures.
For TP: heads are sharded across GPUs, row-parallel layers all_reduce.
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


class LlamaAttention(nn.Module):
    """Llama attention layer with tensor parallelism support."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.rotary = rotary

        # QKV projection (merged) - column parallel
        self.qkv_proj = QKVParallelLinear(
            hidden_size=config.hidden_size,
            num_q_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            bias=False,
            dtype=dtype,
            tp=tp,
        )

        # Store local head counts (after TP sharding)
        self.num_q_heads = self.qkv_proj.local_num_q_heads
        self.num_kv_heads = self.qkv_proj.local_num_kv_heads

        # Output projection - row parallel (input is sharded across heads)
        # Input is local_num_q_heads * head_dim from each rank
        self.o_proj = RowParallelLinear(
            in_features=config.num_attention_heads * config.head_dim,
            out_features=config.hidden_size,
            bias=False,
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
        """
        Args:
            hidden_states: [total_tokens, hidden_size]
            positions: [total_tokens]
            attn_backend: Attention backend
            attn_metadata: Attention metadata
            out_loc: KV cache slots [total_tokens]

        Returns:
            [total_tokens, hidden_size]
        """
        # QKV projection
        q, k, v = self.qkv_proj(hidden_states)

        # Reshape for attention (using local head counts)
        # q: [total_tokens, local_num_q_heads * head_dim] -> [total_tokens, local_num_q_heads, head_dim]
        # k: [total_tokens, local_num_kv_heads * head_dim] -> [total_tokens, local_num_kv_heads, head_dim]
        q = q.view(-1, self.num_q_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

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
        # attn_output: [total_tokens, local_num_q_heads, head_dim]
        attn_output = attn_output.view(-1, self.num_q_heads * self.head_dim)
        return self.o_proj(attn_output)


class LlamaMLP(nn.Module):
    """Llama MLP layer (SwiGLU) with tensor parallelism support."""

    def __init__(
        self,
        config: ModelConfig,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()

        # Gate and up projection (merged) - column parallel
        self.gate_up_proj = GateUpParallelLinear(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=False,
            dtype=dtype,
            tp=tp,
        )

        # Down projection - row parallel
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


class LlamaDecoderLayer(nn.Module):
    """Single Llama transformer layer with tensor parallelism support."""

    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        rotary: RotaryEmbedding,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config, layer_idx, rotary, dtype, tp)
        self.mlp = LlamaMLP(config, dtype, tp)
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
        """
        Returns:
            (hidden_states, residual)
        """
        # Pre-attention norm with residual
        hidden_states, residual = self.input_layernorm.forward_with_residual(
            hidden_states, residual
        )

        # Self attention
        hidden_states = self.self_attn(
            hidden_states, positions, attn_backend, attn_metadata, out_loc
        )

        # Pre-MLP norm with residual
        hidden_states, residual = self.post_attention_layernorm.forward_with_residual(
            hidden_states, residual
        )

        # MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual


class LlamaModel(nn.Module):
    """Llama model (without LM head) with tensor parallelism support."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp = tp

        # Token embeddings (not sharded - each rank has full embedding)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=dtype)

        # Shared rotary embeddings
        self.rotary = RotaryEmbedding(config.rotary_config, device)

        # Transformer layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i, self.rotary, dtype, tp)
            for i in range(config.num_hidden_layers)
        ])

        # Final norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, dtype)

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: AttentionBackend,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """
        Returns:
            [total_tokens, hidden_size]
        """
        hidden_states = self.embed_tokens(input_ids)
        residual: Tensor | None = None

        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states,
                residual,
                positions,
                attn_backend,
                attn_metadata,
                out_loc,
            )

        # Final norm with residual
        hidden_states, _ = self.norm.forward_with_residual(hidden_states, residual)
        return hidden_states


class LlamaForCausalLM(nn.Module):
    """Llama for causal language modeling with tensor parallelism support."""

    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype = torch.bfloat16,
        tp: TPConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.tp = tp
        self.model = LlamaModel(config, device, dtype, tp)

        # LM head (not sharded - each rank computes full logits)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=dtype)

        # Tie weights if configured
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
        """
        Returns:
            Logits, shape [total_tokens, vocab_size]
        """
        hidden_states = self.model(input_ids, positions, attn_backend, attn_metadata, out_loc)
        return self.lm_head(hidden_states)

    def load_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load weights from remapped state dict."""
        # The state dict should already be remapped to our naming
        # Use strict=False because rotary embeddings (inv_freq, cos_cache, sin_cache)
        # are computed buffers, not loaded weights
        self.load_state_dict(state_dict, strict=False)
