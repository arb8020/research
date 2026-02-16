"""CUDA Graph capture and replay for decode optimization.

CUDA graphs capture a sequence of GPU operations and replay them
with minimal CPU overhead. This significantly speeds up decode
(where we run many small operations).

Requirements:
- Static tensor shapes (we pad to fixed batch sizes)
- Decode only (prefill has variable shapes)
- No dynamic control flow in the captured region

State lives in caller-provided dict to keep this module stateless.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

from .attention.backend import AttentionBackend, AttentionMetadata
from .core import Batch


@dataclass
class GraphBuffers:
    """Pre-allocated buffers for CUDA graph capture and replay.

    These tensors are copied into before replaying the graph.
    """

    input_ids: Tensor  # [max_bs]
    positions: Tensor  # [max_bs]
    out_loc: Tensor  # [max_bs]
    logits: Tensor  # [max_bs, vocab_size]

    # Attention metadata buffers
    cu_seqlens_q: Tensor  # [max_bs + 1]
    cu_seqlens_k: Tensor  # [max_bs + 1]
    cache_seqlens: Tensor  # [max_bs]
    page_table: Tensor  # [max_bs, max_seq_len]


def create_graph_buffers(
    max_bs: int,
    max_seq_len: int,
    vocab_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> GraphBuffers:
    """Create pre-allocated buffers for graph capture/replay."""
    return GraphBuffers(
        input_ids=torch.zeros(max_bs, dtype=torch.int32, device=device),
        positions=torch.zeros(max_bs, dtype=torch.int32, device=device),
        out_loc=torch.zeros(max_bs, dtype=torch.int32, device=device),
        logits=torch.empty(max_bs, vocab_size, dtype=dtype, device=device),
        cu_seqlens_q=torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
        cu_seqlens_k=torch.zeros(max_bs + 1, dtype=torch.int32, device=device),
        cache_seqlens=torch.zeros(max_bs, dtype=torch.int32, device=device),
        page_table=torch.zeros(max_bs, max_seq_len, dtype=torch.int32, device=device),
    )


def get_default_batch_sizes() -> list[int]:
    """Get default batch sizes for graph capture."""
    return [1, 2, 4] + list(range(8, 129, 8))


def capture_graphs(
    state: dict,
    model_forward: Callable,
    attn_backend: AttentionBackend,
    device: torch.device,
    max_seq_len: int,
    vocab_size: int,
    batch_sizes: list[int] | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Capture CUDA graphs for decode at various batch sizes.

    Call this after model warmup. Populates state with:
    - "buffers": GraphBuffers
    - "graphs": dict[int, CUDAGraph]
    - "batch_sizes": sorted list
    - "stream": capture stream

    Args:
        state: Caller-provided state dict
        model_forward: Function(input_ids, positions, attn_backend, attn_metadata, out_loc) -> logits
        attn_backend: Attention backend
        device: CUDA device
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        batch_sizes: Batch sizes to capture (default: 1,2,4,8,16,...,128)
        dtype: Model dtype
    """
    if batch_sizes is None:
        batch_sizes = get_default_batch_sizes()
    batch_sizes = sorted(batch_sizes)
    max_bs = max(batch_sizes)

    # Create buffers
    buffers = create_graph_buffers(max_bs, max_seq_len, vocab_size, device, dtype)

    # Create capture stream
    stream = torch.cuda.Stream(device=device)

    # Capture graphs (largest first to share memory pool)
    graphs: dict[int, torch.cuda.CUDAGraph] = {}
    pool = None

    torch.cuda.synchronize(device)

    for bs in sorted(batch_sizes, reverse=True):
        # Setup dummy inputs
        _setup_dummy_inputs(buffers, bs, device)

        # Warmup
        with torch.cuda.stream(stream):
            _run_decode_step(buffers, bs, max_seq_len, model_forward, attn_backend)

        # Capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            _run_decode_step(buffers, bs, max_seq_len, model_forward, attn_backend)

        if pool is None:
            pool = graph.pool()

        graphs[bs] = graph

    # Store in state
    state["buffers"] = buffers
    state["graphs"] = graphs
    state["batch_sizes"] = batch_sizes
    state["stream"] = stream
    state["max_bs"] = max_bs
    state["max_seq_len"] = max_seq_len


def _setup_dummy_inputs(buffers: GraphBuffers, bs: int, device: torch.device) -> None:
    """Setup dummy inputs for graph capture."""
    buffers.cu_seqlens_q[: bs + 1] = torch.arange(bs + 1, device=device)
    buffers.cu_seqlens_k[: bs + 1] = torch.arange(bs + 1, device=device)
    buffers.cache_seqlens[:bs] = 1


def _run_decode_step(
    buffers: GraphBuffers,
    bs: int,
    max_seq_len: int,
    model_forward: Callable,
    attn_backend: AttentionBackend,
) -> None:
    """Run a decode step (used for capture)."""
    metadata = AttentionMetadata(
        cu_seqlens_q=buffers.cu_seqlens_q[: bs + 1],
        cu_seqlens_k=buffers.cu_seqlens_k[: bs + 1],
        cache_seqlens=buffers.cache_seqlens[:bs],
        max_seqlen_q=1,
        max_seqlen_k=max_seq_len,
        page_table=buffers.page_table[:bs],
    )

    logits = model_forward(
        input_ids=buffers.input_ids[:bs],
        positions=buffers.positions[:bs],
        attn_backend=attn_backend,
        attn_metadata=metadata,
        out_loc=buffers.out_loc[:bs],
    )

    buffers.logits[:bs] = logits


def can_use_graph(state: dict, batch: Batch) -> bool:
    """Check if we can use CUDA graph for this batch.

    Args:
        state: Graph state dict (must have "graphs" and "max_bs")
        batch: Decode batch to check

    Returns:
        True if graph replay is available for this batch
    """
    if "graphs" not in state:
        return False

    if not batch.is_decode:
        return False

    for req in batch.reqs:
        if req.extend_len != 1:
            return False

    return batch.size <= state["max_bs"]


def get_padded_size(state: dict, batch_size: int) -> int:
    """Get the next available graph batch size.

    Args:
        state: Graph state dict (must have "batch_sizes")
        batch_size: Actual batch size

    Returns:
        Padded batch size for graph replay
    """
    for bs in state["batch_sizes"]:
        if bs >= batch_size:
            return bs
    return batch_size


def replay_graph(
    state: dict,
    batch: Batch,
    attn_metadata: AttentionMetadata,
) -> Tensor:
    """Replay captured graph for a decode batch.

    Args:
        state: Graph state dict
        batch: Decode batch (each request has 1 new token)
        attn_metadata: Attention metadata

    Returns:
        Logits for each request, shape [batch_size, vocab_size]
    """
    buffers: GraphBuffers = state["buffers"]
    graphs: dict[int, torch.cuda.CUDAGraph] = state["graphs"]
    device = buffers.input_ids.device

    bs = batch.size
    padded_bs = get_padded_size(state, bs)

    assert padded_bs in graphs, f"No graph for batch size {padded_bs}"

    # Copy inputs to buffers
    buffers.input_ids[:bs].copy_(batch.input_ids)
    buffers.positions[:bs].copy_(batch.positions)
    buffers.out_loc[:bs].copy_(batch.out_loc)

    # Copy attention metadata
    buffers.cu_seqlens_q[: bs + 1].copy_(attn_metadata.cu_seqlens_q)
    buffers.cu_seqlens_k[: bs + 1].copy_(attn_metadata.cu_seqlens_k)
    buffers.cache_seqlens[:bs].copy_(attn_metadata.cache_seqlens)
    buffers.page_table[:bs, : attn_metadata.max_seqlen_k].copy_(attn_metadata.page_table)

    # Pad with dummy values if needed
    if padded_bs > bs:
        buffers.cu_seqlens_q[bs + 1 : padded_bs + 1] = torch.arange(
            bs + 1, padded_bs + 1, device=device
        )
        buffers.cu_seqlens_k[bs + 1 : padded_bs + 1] = torch.arange(
            bs + 1, padded_bs + 1, device=device
        )
        buffers.cache_seqlens[bs:padded_bs] = 1

    # Replay graph
    graphs[padded_bs].replay()

    # Return only the real batch's logits
    return buffers.logits[:bs]
