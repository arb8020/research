"""FlexAttention backend - PyTorch stdlib implementation."""

import torch
from torch import Tensor

from ...inference.attention.config import CacheConfig
from ...inference.attention.mask import create_attention_mask
from ...inference.types import InferenceContext


class FlexAttentionBackend:
    """FlexAttention backend using PyTorch stdlib.

    Supports:
    - Standard causal attention
    - Sliding window attention (for Mistral, Qwen2.5, etc.)
    - KV caching with paged memory

    Why a class (owns state)?
    - Owns KV cache tensors (shared across all layers)
    - Caches block masks for efficiency
    - Needs initialization with config

    Why not nn.Module?
    - Not part of model parameters (cache is transient)
    - Doesn't need register_buffer/load_state_dict machinery
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.num_heads_per_kv = 1  # Set by model, updated via set_num_heads()
        self.scale = config.head_dim**-0.5

        # Cache block masks by (batch_size, seq_len) to avoid recomputation
        # create_block_mask is expensive, so we cache results
        self._mask_cache: dict[tuple[int, int], object] = {}

        # KV cache: [num_layers, total_slots, num_kv_heads, head_dim]
        # All layers share one allocation (indexed by layer_idx)
        self.k_cache = torch.zeros(config.cache_shape, dtype=config.dtype, device=config.device)
        self.v_cache = torch.zeros(config.cache_shape, dtype=config.dtype, device=config.device)

    def set_num_heads(self, num_heads: int) -> None:
        """Set number of query heads (for GQA expansion)."""
        self.num_heads_per_kv = num_heads // self.config.num_kv_heads

    def get_block_mask(self, batch_size: int, seq_len: int) -> object:
        """Get or create block mask for attention.

        Creates appropriate mask based on config (causal or sliding window).
        Caches masks to avoid expensive recomputation.

        Args:
            batch_size: Number of sequences
            seq_len: Sequence length

        Returns:
            BlockMask for flex_attention, or None if no special masking needed
        """
        # For simple causal without sliding window, we can use is_causal=True
        # which is faster than creating a block mask
        if self.config.sliding_window is None:
            return None

        # Check cache
        cache_key = (batch_size, seq_len)
        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # Create mask
        mask = create_attention_mask(
            batch_size=batch_size,
            seq_len=seq_len,
            sliding_window=self.config.sliding_window,
            block_size=self.config.block_size,
            device=self.config.device,
        )

        # Cache for reuse (limit cache size to prevent memory bloat)
        if len(self._mask_cache) < 100:
            self._mask_cache[cache_key] = mask

        return mask

    def clear_mask_cache(self) -> None:
        """Clear cached block masks (call if sequence lengths change significantly)."""
        self._mask_cache.clear()

    def forward(
        self,
        q: Tensor,  # [batch, seq_len, num_heads, head_dim]
        k: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        v: Tensor,  # [batch, seq_len, num_kv_heads, head_dim]
        layer_idx: int,
        ctx: InferenceContext,
    ) -> Tensor:
        """Compute attention with KV cache.

        Args:
            q, k, v: Query, key, value tensors
            layer_idx: Which layer (for indexing into shared cache)
            ctx: Inference context with cache slot mapping

        Returns:
            Attention output: [batch, seq_len, num_heads, head_dim]
        """
        # Store K/V into cache
        if ctx.slot_mapping is not None:
            self._store_kv(k, v, layer_idx, ctx.slot_mapping)

        if ctx.is_prefill:
            return self._prefill_attention(q, k, v, ctx)
        else:
            return self._decode_attention(q, layer_idx, ctx)

    def _store_kv(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        slot_mapping: tuple[int, ...],
    ) -> None:
        """Store K/V into cache at specified slots.

        Args:
            k: [batch, seq_len, num_kv_heads, head_dim]
            v: [batch, seq_len, num_kv_heads, head_dim]
            layer_idx: Which layer's cache to write to
            slot_mapping: [total_tokens] mapping to cache slots
        """
        # Flatten batch and seq dimensions
        k_flat = k.reshape(-1, self.config.num_kv_heads, self.config.head_dim)
        v_flat = v.reshape(-1, self.config.num_kv_heads, self.config.head_dim)

        # Convert to tensor for indexing
        slots = torch.tensor(slot_mapping, device=k.device, dtype=torch.long)

        # Store into cache (only non-negative slots)
        valid_mask = slots >= 0
        valid_slots = slots[valid_mask]
        self.k_cache[layer_idx, valid_slots] = k_flat[valid_mask]
        self.v_cache[layer_idx, valid_slots] = v_flat[valid_mask]

    def _prefill_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        ctx: InferenceContext,
    ) -> Tensor:
        """Prefill attention: compute over full input sequence.

        For prefill, we use the input K/V directly (they're also stored in cache).
        Uses FlexAttention with appropriate masking (causal or sliding window).
        """
        batch, seq_len, num_heads, head_dim = q.shape

        # Transpose to [batch, heads, seq, dim] for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads if using GQA
        if self.num_heads_per_kv > 1:
            k = k.repeat_interleave(self.num_heads_per_kv, dim=1)
            v = v.repeat_interleave(self.num_heads_per_kv, dim=1)

        # Get block mask: use provided mask, or create from config
        block_mask = ctx.block_mask
        if block_mask is None:
            block_mask = self.get_block_mask(batch, seq_len)

        # Use FlexAttention if we have a block_mask (e.g., sliding window),
        # else standard SDPA with is_causal=True
        if block_mask is not None:
            from torch.nn.attention.flex_attention import flex_attention

            out = flex_attention(q, k, v, block_mask=block_mask, scale=self.scale)
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=self.scale
            )

        # Transpose back to [batch, seq, heads, dim]
        return out.transpose(1, 2)

    def _decode_attention(
        self,
        q: Tensor,
        layer_idx: int,
        ctx: InferenceContext,
    ) -> Tensor:
        """Decode attention: query attends to cached K/V.

        For decode, we read K/V from cache using block tables.
        Each sequence has 1 query token attending to all cached tokens.
        """
        batch, seq_len, num_heads, head_dim = q.shape
        assert seq_len == 1, "Decode should have seq_len=1"

        if ctx.block_tables is None or ctx.seq_lens is None:
            raise ValueError("Decode requires block_tables and seq_lens in context")

        # Gather K/V from cache using block tables
        k_gathered, v_gathered = self._gather_kv_from_cache(
            layer_idx, ctx.block_tables, ctx.seq_lens
        )

        # Transpose to [batch, heads, seq, dim]
        q = q.transpose(1, 2)  # [batch, heads, 1, dim]
        k_gathered = k_gathered.transpose(1, 2)  # [batch, heads, cache_len, dim]
        v_gathered = v_gathered.transpose(1, 2)

        # Expand KV heads if using GQA
        if self.num_heads_per_kv > 1:
            k_gathered = k_gathered.repeat_interleave(self.num_heads_per_kv, dim=1)
            v_gathered = v_gathered.repeat_interleave(self.num_heads_per_kv, dim=1)

        # Compute attention
        if ctx.block_mask is not None:
            from torch.nn.attention.flex_attention import flex_attention

            out = flex_attention(
                q, k_gathered, v_gathered, block_mask=ctx.block_mask, scale=self.scale
            )
        else:
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k_gathered, v_gathered, is_causal=False, scale=self.scale
            )

        return out.transpose(1, 2)

    def _gather_kv_from_cache(
        self,
        layer_idx: int,
        block_tables: tuple[tuple[int, ...], ...],
        seq_lens: tuple[int, ...],
    ) -> tuple[Tensor, Tensor]:
        """Gather K/V from cache using block tables.

        This is a simplified implementation. Real paged attention uses
        specialized kernels that do this more efficiently.

        Args:
            layer_idx: Which layer's cache to read from
            block_tables: [num_seqs, max_blocks] block IDs per sequence
            seq_lens: [num_seqs] number of tokens per sequence

        Returns:
            k, v: [num_seqs, max_seq_len, num_kv_heads, head_dim]
        """
        num_seqs = len(block_tables)
        max_seq_len = max(seq_lens)

        # Allocate output tensors
        k_out = torch.zeros(
            num_seqs,
            max_seq_len,
            self.config.num_kv_heads,
            self.config.head_dim,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        v_out = torch.zeros_like(k_out)

        # Gather from cache (this is slow, just for correctness)
        # TODO: Replace with FlexAttention block_mask approach or FlashInfer
        block_size = self.config.block_size

        for seq_idx, (blocks, seq_len) in enumerate(zip(block_tables, seq_lens, strict=False)):
            for pos in range(seq_len):
                block_idx = pos // block_size
                offset = pos % block_size
                if block_idx < len(blocks):
                    slot = blocks[block_idx] * block_size + offset
                    k_out[seq_idx, pos] = self.k_cache[layer_idx, slot]
                    v_out[seq_idx, pos] = self.v_cache[layer_idx, slot]

        return k_out, v_out

    def clear_cache(self) -> None:
        """Zero out all cache tensors and clear mask cache."""
        self.k_cache.zero_()
        self.v_cache.zero_()
        self._mask_cache.clear()
