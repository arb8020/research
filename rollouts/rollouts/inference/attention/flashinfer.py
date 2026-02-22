"""FlashInfer attention backend.

FlashInfer provides optimized attention kernels, especially for decode.
Better throughput than FlashAttention for long sequences due to:
- Tensor core utilization for GQA
- Efficient paged KV cache access
- Optimized CUDA graph support

Requirements:
- pip install flashinfer (Linux only, CUDA 11.8+)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from .backend import AttentionMetadata

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
    )

logger = logging.getLogger(__name__)


def is_flashinfer_available() -> bool:
    """Check if FlashInfer is available."""
    try:
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


def _next_power_of_2(n: int) -> int:
    """Round up to next power of 2."""
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


@dataclass
class FlashInferMetadata:
    """FlashInfer-specific metadata for attention.

    Extends base AttentionMetadata with FlashInfer wrapper state.
    """

    # Base metadata
    cu_seqlens_q_cpu: Tensor  # CPU, pinned
    cu_seqlens_k_cpu: Tensor  # CPU, pinned
    cu_seqlens_q_gpu: Tensor  # GPU
    seq_lens_cpu: Tensor  # CPU, pinned - key sequence lengths
    last_page_len_cpu: Tensor  # CPU, pinned - always 1s for page_size=1
    indices: Tensor  # GPU - flattened page table indices

    # Config
    num_q_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int  # Currently only 1 is supported
    dtype: torch.dtype

    # Wrapper (set after plan())
    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper | None = None
    is_decode: bool = False
    initialized: bool = False


class FlashInferBackend:
    """FlashInfer attention backend for high-performance inference.

    Uses FlashInfer's paged KV cache kernels which are optimized for:
    - Decode (single token per sequence): uses tensor cores for GQA
    - Prefill (many tokens): uses standard flash attention

    Key features:
    - page_size=1 (token-level granularity, simpler than block-paged)
    - Supports CUDA graphs via CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    - Workspace buffers shared between prefill/decode wrappers
    """

    def __init__(
        self,
        k_cache: Tensor,
        v_cache: Tensor,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        """Initialize FlashInfer backend.

        Args:
            k_cache: Key cache tensor [num_layers, num_slots, num_kv_heads, head_dim]
            v_cache: Value cache tensor [num_layers, num_slots, num_kv_heads, head_dim]
            num_q_heads: Number of query heads
            num_kv_heads: Number of KV heads (for GQA)
            head_dim: Dimension per head
        """
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.k_cache = k_cache
        self.v_cache = v_cache
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.device = k_cache.device
        self.dtype = k_cache.dtype

        # Workspace buffer (128MB, shared between wrappers)
        self.workspace_buffer = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.device
        )

        # Use tensor cores for decode when GQA ratio >= 4
        gqa_ratio = num_q_heads // num_kv_heads
        use_tensor_cores = gqa_ratio >= 4

        # Create wrappers
        # Using fa2 backend (FlashAttention-2) as fa3 can be slower
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.workspace_buffer,
            kv_layout="NHD",
            backend="fa2",
        )
        self.decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer,
            use_tensor_cores=use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",
        )

        # Share int workspace buffer between wrappers
        self.decode_wrapper._int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer

        # Cache for ones tensor (last_page_len is always 1 for page_size=1)
        self._cached_ones: Tensor | None = None

        # CUDA graph state
        self._graph_wrappers: dict[int, Any] = {}
        self._graph_capture_data: dict[str, Tensor] | None = None

        logger.debug(
            f"FlashInfer backend initialized: "
            f"{num_q_heads}q/{num_kv_heads}kv heads, "
            f"head_dim={head_dim}, tensor_cores={use_tensor_cores}"
        )

    def _get_ones_cpu(self, size: int) -> Tensor:
        """Get a pinned CPU tensor of ones (for last_page_len)."""
        if self._cached_ones is None or len(self._cached_ones) < size:
            new_size = _next_power_of_2(size)
            self._cached_ones = torch.ones(new_size, dtype=torch.int32, pin_memory=True)
        return self._cached_ones[:size]

    def store_kv(
        self,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        out_loc: Tensor,
    ) -> None:
        """Store K,V to cache slots."""
        self.k_cache[layer_idx, out_loc] = k
        self.v_cache[layer_idx, out_loc] = v

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        layer_idx: int,
        metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Compute attention using FlashInfer.

        Args:
            q: Query tensor [total_q_tokens, num_q_heads, head_dim]
            k: Key tensor [total_q_tokens, num_kv_heads, head_dim]
            v: Value tensor [total_q_tokens, num_kv_heads, head_dim]
            layer_idx: Current layer index
            metadata: Attention metadata
            out_loc: KV cache slots for storing new K,V

        Returns:
            Attention output [total_q_tokens, num_q_heads, head_dim]
        """
        # Store new K,V to cache
        self.store_kv(k, v, layer_idx, out_loc)

        # Get cache for this layer, reshape for FlashInfer page_size=1
        # Original: [num_slots, num_kv_heads, head_dim]
        # FlashInfer expects: [num_pages, page_size, num_kv_heads, head_dim]
        # With page_size=1: [num_slots, 1, num_kv_heads, head_dim]
        k_cache = self.k_cache[layer_idx].unsqueeze(1)
        v_cache = self.v_cache[layer_idx].unsqueeze(1)

        # Get or create FlashInfer metadata
        fi_meta = self._get_or_create_fi_metadata(metadata)

        # Initialize wrapper if needed (calls plan())
        self._ensure_wrapper_initialized(fi_meta)

        # Run attention
        return fi_meta.wrapper.run(q=q, paged_kv_cache=(k_cache, v_cache))

    def _get_or_create_fi_metadata(self, metadata: AttentionMetadata) -> FlashInferMetadata:
        """Convert base metadata to FlashInfer metadata."""
        # Check if already converted (stored as attribute)
        if hasattr(metadata, "_flashinfer_meta"):
            return metadata._flashinfer_meta

        batch_size = metadata.batch_size
        is_decode = metadata.max_seqlen_q == 1

        # Build flattened indices from page table
        # page_table shape: [batch_size, max_seqlen_k]
        # We need to flatten it based on actual sequence lengths
        indices_list = []
        for i in range(batch_size):
            seq_len = int(metadata.cache_seqlens[i].item())
            indices_list.append(metadata.page_table[i, :seq_len])
        indices = (
            torch.cat(indices_list)
            if indices_list
            else torch.tensor([], device=self.device, dtype=torch.int32)
        )

        # CPU tensors (pinned for async transfer)
        cu_seqlens_q_cpu = metadata.cu_seqlens_q.cpu().pin_memory()
        cu_seqlens_k_cpu = metadata.cu_seqlens_k.cpu().pin_memory()
        seq_lens_cpu = metadata.cache_seqlens.cpu().pin_memory()
        last_page_len_cpu = self._get_ones_cpu(batch_size)

        fi_meta = FlashInferMetadata(
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            cu_seqlens_q_gpu=metadata.cu_seqlens_q,
            seq_lens_cpu=seq_lens_cpu,
            last_page_len_cpu=last_page_len_cpu,
            indices=indices,
            num_q_heads=self.num_q_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            page_size=1,
            dtype=self.dtype,
            wrapper=self.decode_wrapper if is_decode else self.prefill_wrapper,
            is_decode=is_decode,
            initialized=False,
        )

        # Cache on metadata object
        metadata._flashinfer_meta = fi_meta
        return fi_meta

    def _ensure_wrapper_initialized(self, fi_meta: FlashInferMetadata) -> None:
        """Initialize FlashInfer wrapper with plan() if needed."""
        if fi_meta.initialized:
            return

        fi_meta.initialized = True

        if fi_meta.is_decode:
            # Decode path
            fi_meta.wrapper.plan(
                indptr=fi_meta.cu_seqlens_k_cpu,
                indices=fi_meta.indices,
                last_page_len=fi_meta.last_page_len_cpu,
                num_qo_heads=fi_meta.num_q_heads,
                num_kv_heads=fi_meta.num_kv_heads,
                head_dim=fi_meta.head_dim,
                page_size=fi_meta.page_size,
                pos_encoding_mode="NONE",  # RoPE applied before attention
                seq_lens=fi_meta.seq_lens_cpu,
                data_type=fi_meta.dtype,
                q_data_type=fi_meta.dtype,
                kv_data_type=fi_meta.dtype,
                non_blocking=True,
            )
        else:
            # Prefill path
            fi_meta.wrapper.plan(
                qo_indptr=fi_meta.cu_seqlens_q_cpu,
                paged_kv_indptr=fi_meta.cu_seqlens_k_cpu,
                paged_kv_indices=fi_meta.indices,
                paged_kv_last_page_len=fi_meta.last_page_len_cpu,
                num_qo_heads=fi_meta.num_q_heads,
                num_kv_heads=fi_meta.num_kv_heads,
                head_dim_qk=fi_meta.head_dim,
                page_size=fi_meta.page_size,
                pos_encoding_mode="NONE",
                seq_lens=fi_meta.seq_lens_cpu,
                q_data_type=fi_meta.dtype,
                kv_data_type=fi_meta.dtype,
                non_blocking=True,
                causal=True,
            )

    # ═══════════════════════════════════════════════════════════════════════════
    # CUDA GRAPH SUPPORT
    # ═══════════════════════════════════════════════════════════════════════════

    def init_cuda_graph_capture(
        self,
        max_batch_size: int,
        max_seq_len: int,
        batch_sizes: list[int],
    ) -> None:
        """Initialize buffers for CUDA graph capture.

        Call this before capturing graphs. Creates persistent buffers
        that will be reused across graph replays.

        Args:
            max_batch_size: Maximum batch size to support
            max_seq_len: Maximum sequence length
            batch_sizes: List of batch sizes to capture graphs for
        """
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        # Create capture buffers
        self._graph_capture_data = {
            "cu_seqlens_k": torch.zeros(max_batch_size + 1, dtype=torch.int32, device=self.device),
            "seq_lens": torch.zeros(max_batch_size, dtype=torch.int32, device=self.device),
            "page_table": torch.zeros(
                max_batch_size * max_seq_len, dtype=torch.int32, device=self.device
            ),
            "last_page_len": torch.ones(max_batch_size, dtype=torch.int32, device=self.device),
        }

        # Create graph wrappers for each batch size
        for bs in batch_sizes:
            capture = self._graph_capture_data
            self._graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer,
                kv_layout="NHD",
                use_tensor_cores=(self.num_q_heads // self.num_kv_heads) >= 4,
                indptr_buffer=capture["cu_seqlens_k"][: bs + 1],
                indices_buffer=capture["page_table"],
                last_page_len_buffer=capture["last_page_len"][:bs],
            )
            self._graph_wrappers[bs]._backend = "fa2"
            self._graph_wrappers[
                bs
            ]._int_workspace_buffer = self.prefill_wrapper._int_workspace_buffer

        logger.debug(f"FlashInfer CUDA graph capture initialized for batch sizes: {batch_sizes}")

    def get_graph_wrapper(self, batch_size: int) -> Any:
        """Get CUDA graph wrapper for a batch size."""
        assert batch_size in self._graph_wrappers, f"No graph wrapper for batch_size={batch_size}"
        return self._graph_wrappers[batch_size]

    def has_graph_wrapper(self, batch_size: int) -> bool:
        """Check if graph wrapper exists for batch size."""
        return batch_size in self._graph_wrappers
