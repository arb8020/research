"""Example: How to wire up attention backend with a model.

This shows the pattern for integrating FlexAttentionBackend
with a transformer model, demonstrating explicit context threading.
"""

import torch
import torch.nn as nn
from torch import Tensor

from rollouts.inference.attention.config import CacheConfig
from rollouts.inference.attention.flex_backend import FlexAttentionBackend
from rollouts.inference.attention.layer import Attention
from rollouts.inference.types import InferenceContext


class ExampleTransformerBlock(nn.Module):
    """Example transformer block showing context threading pattern.

    Key points:
    - Attention layer receives shared backend at init
    - Forward takes ctx: InferenceContext explicitly
    - No global state, everything passed through
    """

    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        backend: FlexAttentionBackend,
    ):
        super().__init__()
        self.layer_idx = layer_idx

        # Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        # Attention with shared backend
        self.attn = Attention(layer_idx=layer_idx, backend=backend)

        # MLP (simplified)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size, bias=False),
        )

        # Norms
        self.input_layernorm = nn.RMSNorm(hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(hidden_size)

        # For reshaping
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def forward(self, x: Tensor, ctx: InferenceContext) -> Tensor:
        """Forward pass with explicit context.

        Args:
            x: [batch, seq_len, hidden_size]
            ctx: Inference context (passed through, not stored)

        Returns:
            Output: [batch, seq_len, hidden_size]
        """
        batch, seq_len, _ = x.shape

        # Pre-norm
        residual = x
        x = self.input_layernorm(x)

        # Q/K/V projections
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch, seq_len, self.num_kv_heads, self.head_dim)

        # Attention (ctx threaded through)
        attn_out = self.attn(q, k, v, ctx)

        # Output projection
        attn_out = attn_out.view(batch, seq_len, -1)
        x = residual + self.o_proj(attn_out)

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        x = residual + self.mlp(x)

        return x


class ExampleModel(nn.Module):
    """Example model showing full backend setup.

    Pattern:
    1. Create CacheConfig from model config
    2. Create FlexAttentionBackend (owns all KV cache)
    3. Pass backend to each transformer block
    4. Thread ctx through forward pass
    """

    def __init__(
        self,
        num_layers: int = 4,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_kv_heads: int = 2,
        head_dim: int = 64,
        vocab_size: int = 32000,
        num_blocks: int = 256,
        block_size: int = 16,
        device: str = "cuda",
    ):
        super().__init__()

        # 1. Create cache config
        cache_config = CacheConfig(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            device=device,
        )

        # 2. Create shared backend
        self.backend = FlexAttentionBackend(cache_config)
        self.backend.set_num_heads(num_heads)

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # 3. Create layers, each gets reference to shared backend
        self.layers = nn.ModuleList([
            ExampleTransformerBlock(
                layer_idx=i,
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                backend=self.backend,
            )
            for i in range(num_layers)
        ])

        self.norm = nn.RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: Tensor, ctx: InferenceContext) -> Tensor:
        """Forward pass with explicit context threading.

        Args:
            input_ids: [batch, seq_len]
            ctx: Inference context (threaded through all layers)

        Returns:
            Logits: [batch, seq_len, vocab_size]
        """
        # Embed
        x = self.embed_tokens(input_ids)

        # 4. Thread ctx through each layer
        for layer in self.layers:
            x = layer(x, ctx)

        # Final norm and output
        x = self.norm(x)
        return self.lm_head(x)


def example_usage():
    """Show how to use the model with context."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    model = ExampleModel(device=device).to(device)

    # Create input
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, 32000, (batch_size, seq_len), device=device)

    # Create inference context for prefill
    # In real usage, slot_mapping comes from the block allocator
    slot_mapping = tuple(range(batch_size * seq_len))
    ctx = InferenceContext(
        is_prefill=True,
        slot_mapping=slot_mapping,
        seq_lens=tuple([seq_len] * batch_size),
        max_seq_len=seq_len,
    )

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids, ctx)

    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Cache k shape: {model.backend.k_cache.shape}")

    return logits


if __name__ == "__main__":
    example_usage()
