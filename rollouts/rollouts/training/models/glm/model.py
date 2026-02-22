"""GLM-4.7-Flash and GLM-5 model implementation.

Key architectural differences from Llama/Qwen:
- QKNorm: RMSNorm applied to Q and K after projection (before RoPE)
- Attention bias: Q/K/V have bias, output projection has no bias
- MoE routing: Sigmoid-gated with e_score_correction_bias
- Shared expert: Always runs on all tokens, combined additively with routed
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.utils import trunc_normal_
from torchtitan.protocols.model import AttentionMasksType
from torchtitan.protocols.train_spec import ModelProtocol

if TYPE_CHECKING:
    pass

from .args import GLMModelArgs


def precompute_rope_cache(dim: int, max_seq_len: int, base: float = 1_000_000.0) -> torch.Tensor:
    """Precompute RoPE cos/sin cache."""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=freqs.dtype, device=freqs.device)
    idx_theta = torch.outer(t, freqs).float()
    freqs = torch.cat([idx_theta, idx_theta], dim=-1)
    rope_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    return rope_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half the hidden dims for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def reshape_for_broadcast(
    rope_cache: torch.Tensor, x: torch.Tensor, positions: torch.Tensor | None = None
) -> torch.Tensor:
    """Reshape RoPE cache for broadcasting."""
    bz, seqlen, _, head_dim = x.shape
    if positions is None:
        rope_cache = rope_cache[0:seqlen]
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    elif positions.size(0) == 1:
        rope_cache = rope_cache[positions.squeeze(0)]
        shape = [-1, seqlen, 1, head_dim * 2]
        return rope_cache.view(*shape)
    else:
        rope_cache_expanded = rope_cache[None, :, None, :].expand(bz, -1, -1, -1)
        rope_cache = torch.gather(
            rope_cache_expanded,
            dim=1,
            index=positions.view(bz, seqlen, 1, 1).expand(bz, seqlen, 1, head_dim * 2),
        )
        return rope_cache


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    rope_cache: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to Q and K."""
    head_dim = xq.shape[-1]
    rope_cache = reshape_for_broadcast(rope_cache, xq, positions)
    cos = rope_cache[..., :head_dim].to(dtype=xq.dtype, device=xq.device)
    sin = rope_cache[..., head_dim:].to(dtype=xq.dtype, device=xq.device)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class GLMAttention(nn.Module):
    """GLM attention with QKNorm and bias on Q/K/V projections.

    Key differences from standard attention:
    - QKNorm: RMSNorm applied to Q and K after projection (before RoPE)
    - Bias on Q/K/V linear projections (no bias on output)
    """

    q_norm: nn.RMSNorm | None
    k_norm: nn.RMSNorm | None

    def __init__(self, model_args: GLMModelArgs) -> None:
        super().__init__()
        self.n_heads = model_args.n_heads
        self.n_kv_heads = (
            model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        )
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = model_args.head_dim
        self.scaling = self.head_dim**-0.5
        self.attn_type = getattr(model_args, "attn_type", "sdpa")
        self.enable_gqa = self.n_heads > self.n_kv_heads

        # GLM uses QKNorm
        if model_args.qk_norm:
            self.q_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
            self.k_norm = nn.RMSNorm(
                self.head_dim, eps=model_args.norm_eps, elementwise_affine=True
            )
        else:
            self.q_norm = None
            self.k_norm = None

        # GLM uses bias on Q/K/V but not on output
        use_bias = model_args.attn_bias
        self.wq = nn.Linear(model_args.dim, model_args.n_heads * self.head_dim, bias=use_bias)
        self.wk = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=use_bias)
        self.wv = nn.Linear(model_args.dim, self.n_kv_heads * self.head_dim, bias=use_bias)
        self.wo = nn.Linear(model_args.n_heads * self.head_dim, model_args.dim, bias=False)

    def init_weights(self, init_std: float) -> None:
        for linear in (self.wq, self.wk, self.wv):
            trunc_normal_(linear.weight, mean=0.0, std=0.02)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        trunc_normal_(self.wo.weight, mean=0.0, std=init_std)
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        if self.k_norm is not None:
            self.k_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bs, seqlen, -1, self.head_dim)
        xk = xk.view(bs, seqlen, -1, self.head_dim)
        xv = xv.view(bs, seqlen, -1, self.head_dim)

        # GLM QKNorm: apply BEFORE RoPE
        if self.q_norm:
            xq = self.q_norm(xq)
        if self.k_norm:
            xk = self.k_norm(xk)

        # Apply rotary embeddings
        xq, xk = apply_rotary_emb(xq, xk, rope_cache, positions)

        xq = xq.transpose(1, 2)  # (bs, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Use SDPA - handles GQA automatically with enable_gqa
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            is_causal=True,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        )
        output = output.transpose(1, 2).contiguous()
        output = output.view(bs, seqlen, -1)
        return self.wo(output)


class GLMFeedForward(nn.Module):
    """SwiGLU MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)  # gate_proj
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)  # down_proj
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # up_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def init_weights(self, init_std: float) -> None:
        trunc_normal_(self.w1.weight, mean=0.0, std=0.02)
        for linear in (self.w2, self.w3):
            trunc_normal_(linear.weight, mean=0.0, std=init_std)


class GLMSharedExpert(nn.Module):
    """Shared expert that always runs on all tokens."""

    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    def init_weights(self, init_std: float) -> None:
        trunc_normal_(self.gate_proj.weight, mean=0.0, std=0.02)
        trunc_normal_(self.up_proj.weight, mean=0.0, std=0.02)
        trunc_normal_(self.down_proj.weight, mean=0.0, std=init_std)


class GLMMoE(nn.Module):
    """GLM Mixture of Experts with sigmoid routing and shared expert.

    Key differences from standard MoE:
    - Sigmoid routing (not softmax)
    - e_score_correction_bias for expert selection
    - Shared expert that always runs on all tokens
    - routed_scaling_factor applied to routed outputs
    """

    def __init__(self, model_args: GLMModelArgs, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.n_routed_experts = model_args.n_routed_experts
        self.n_shared_experts = model_args.n_shared_experts
        self.num_experts_per_tok = model_args.num_experts_per_tok
        self.routed_scaling_factor = model_args.routed_scaling_factor

        # Router with e_score_correction_bias
        self.router = nn.Linear(dim, self.n_routed_experts, bias=False)
        self.e_score_correction_bias = nn.Parameter(torch.zeros(self.n_routed_experts))

        # Individual experts (each is a SwiGLU MLP)
        self.experts = nn.ModuleList([
            GLMFeedForward(dim, hidden_dim) for _ in range(self.n_routed_experts)
        ])

        # Shared expert
        if self.n_shared_experts > 0:
            self.shared_expert = GLMSharedExpert(dim, hidden_dim)
        else:
            self.shared_expert = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)

        # Route: sigmoid + correction bias
        router_logits = self.router(x_flat.float())
        router_probs = router_logits.sigmoid()
        scores_for_choice = router_probs + self.e_score_correction_bias

        # Select top-k experts
        _, topk_indices = torch.topk(scores_for_choice, k=self.num_experts_per_tok, dim=-1)

        # Get weights from original probs (not scores_for_choice)
        topk_weights = router_probs.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor

        # Execute experts
        final_hidden = torch.zeros_like(x_flat)
        expert_mask = F.one_hot(topk_indices, num_classes=self.n_routed_experts)
        expert_mask = expert_mask.permute(2, 1, 0)  # (n_experts, top_k, batch*seq)

        for expert_idx in range(self.n_routed_experts):
            expert = self.experts[expert_idx]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            if token_idx.numel() == 0:
                continue

            current_state = x_flat[token_idx]
            current_hidden = expert(current_state)
            current_hidden = current_hidden * topk_weights[token_idx, top_k_pos, None]
            final_hidden.index_add_(0, token_idx, current_hidden.to(final_hidden.dtype))

        routed_output = final_hidden.view(batch_size, seq_len, hidden_size)

        # Add shared expert output
        if self.shared_expert is not None:
            shared_output = self.shared_expert(x)
            return routed_output + shared_output

        return routed_output

    def init_weights(self, init_std: float, buffer_device: torch.device) -> None:
        trunc_normal_(self.router.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.e_score_correction_bias)
        for expert in self.experts:
            expert.init_weights(init_std)
        if self.shared_expert is not None:
            self.shared_expert.init_weights(init_std)


class GLMTransformerBlock(nn.Module):
    """GLM transformer block with MoE support."""

    def __init__(self, layer_id: int, model_args: GLMModelArgs) -> None:
        super().__init__()
        self.n_heads = model_args.n_heads
        self.dim = model_args.dim

        self.attention = GLMAttention(model_args)

        self.moe_enabled = model_args.moe_enabled
        if self.moe_enabled:
            self.moe = GLMMoE(
                model_args,
                dim=model_args.dim,
                hidden_dim=model_args.moe_inter_dim,
            )
        else:
            self.feed_forward = GLMFeedForward(dim=model_args.dim, hidden_dim=model_args.hidden_dim)

        self.attention_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.ffn_norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)

        if model_args.depth_init:
            self.weight_init_std = 0.02 / (2 * (layer_id + 1)) ** 0.5
        else:
            self.weight_init_std = 0.02 / (2 * model_args.n_layers) ** 0.5

    def forward(
        self,
        x: torch.Tensor,
        rope_cache: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x), rope_cache, attention_masks, positions)

        if self.moe_enabled:
            x = x + self.moe(self.ffn_norm(x))
        else:
            x = x + self.feed_forward(self.ffn_norm(x))
        return x

    def init_weights(self, buffer_device: torch.device) -> None:
        for norm in (self.attention_norm, self.ffn_norm):
            norm.reset_parameters()
        self.attention.init_weights(self.weight_init_std)
        if self.moe_enabled:
            self.moe.init_weights(self.weight_init_std, buffer_device)
        else:
            self.feed_forward.init_weights(self.weight_init_std)


class GLMModel(ModelProtocol):
    """GLM-4.7-Flash / GLM-5 model."""

    def __init__(self, model_args: GLMModelArgs) -> None:
        super().__init__(model_args)
        self.model_args = model_args
        self.vocab_size = model_args.vocab_size
        self.n_layers = model_args.n_layers
        self.head_dim = model_args.head_dim
        self.enable_weight_tying = model_args.enable_weight_tying

        self.tok_embeddings = nn.Embedding(model_args.vocab_size, model_args.dim)

        self.register_buffer("rope_cache", self._precompute_rope_cache(), persistent=False)

        self.layers = torch.nn.ModuleDict()
        for layer_id in range(model_args.n_layers):
            self.layers[str(layer_id)] = GLMTransformerBlock(layer_id, model_args)

        self.norm = nn.RMSNorm(model_args.dim, eps=model_args.norm_eps)
        self.output = nn.Linear(model_args.dim, model_args.vocab_size, bias=False)

        if self.enable_weight_tying:
            self.output.weight = self.tok_embeddings.weight

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.rope_cache.device
        with torch.device(buffer_device):
            self.rope_cache = self._precompute_rope_cache()
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            if layer is not None:
                cast(GLMTransformerBlock, layer).init_weights(buffer_device)
        if self.norm is not None:
            self.norm.reset_parameters()

        final_out_std = self.model_args.dim**-0.5
        cutoff_factor = 3

        if self.enable_weight_tying:
            assert self.tok_embeddings is not None and self.output is not None
            self.output.weight = self.tok_embeddings.weight

        if self.output is not None:
            trunc_normal_(
                self.output.weight,
                mean=0.0,
                std=final_out_std,
                a=-cutoff_factor * final_out_std,
                b=cutoff_factor * final_out_std,
            )

    def _precompute_rope_cache(self) -> torch.Tensor:
        return precompute_rope_cache(
            self.model_args.head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_masks: AttentionMasksType | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Passthrough for pipeline parallelism
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        for layer in self.layers.values():
            h = layer(h, self.rope_cache, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h
        return output
