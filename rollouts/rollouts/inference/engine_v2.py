"""Inference engine v2 - with custom model and FlashAttention.

This version uses:
- Custom Llama model (not HuggingFace)
- FlashAttention with paged KV cache (or reference fallback)
- Batched forward with proper attention metadata
- Page table for KV cache management
- CUDA graphs for decode optimization
- Overlap scheduling for CPU/GPU parallelism
- Radix cache for prefix sharing

State for CUDA graphs, overlap, and radix cache lives in dicts to keep
the modules stateless (following nmoe patterns).

The v1 engine (engine.py) is kept for HuggingFace compatibility.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor

from .attention.backend import AttentionMetadata, build_attention_metadata
from .attention.flash import FlashAttentionBackend, is_flash_attn_available
from .attention.reference import ReferenceAttentionBackend
from .chunked_prefill import ChunkedPrefillManager
from .core import (
    Batch,
    Req,
    SamplingParams,
    create_req,
    req_after_decode_step,
    req_after_forward,
    req_append_token,
)
from .graph import can_use_graph, capture_graphs, replay_graph
from .kv_cache import CacheConfig, KVCachePool
from .models.config import load_model_config
from .models.llama import LlamaForCausalLM
from .models.llama_functional import (
    LlamaConfig as FunctionalLlamaConfig,
)
from .models.llama_functional import (
    forward as functional_forward,
)
from .models.llama_functional import (
    load_config as load_functional_config,
)
from .models.weight import load_weights, remap_weights_llama, shard_weights_for_tp
from .overlap import (
    ForwardInput,
    ForwardOutput,
    create_forward_output,
    overlap_step,
)
from .radix import CacheHandle, evict, init_radix_state, insert_prefix, lock, match_prefix, unlock
from .scheduler import (
    SchedulerConfig,
    SchedulerState,
    add_request,
    empty_scheduler_state,
    has_pending_work,
    schedule_step,
)
from .tp import TPConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EngineConfig:
    """Engine configuration."""

    model_path: str
    max_batch_size: int = 32
    max_tokens_per_batch: int = 4096
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.bfloat16
    attention_backend: Literal["auto", "flash", "flashinfer", "reference"] = "auto"

    # CUDA graph settings
    enable_cuda_graphs: bool = True
    graph_batch_sizes: list[int] | None = None

    # Overlap scheduling settings
    enable_overlap: bool = True

    # Radix cache settings
    enable_radix_cache: bool = True

    # Chunked prefill settings
    enable_chunked_prefill: bool = True
    prefill_chunk_size: int | None = None

    # Model implementation
    model_impl: Literal["module", "functional"] = "module"

    # Tensor parallelism settings
    tp_rank: int = 0
    tp_size: int = 1

    def __post_init__(self) -> None:
        assert self.max_batch_size > 0
        assert self.max_tokens_per_batch > 0
        assert self.max_seq_len > 0
        if self.prefill_chunk_size is not None:
            assert self.prefill_chunk_size > 0
        assert self.tp_rank >= 0 and self.tp_rank < self.tp_size
        assert self.tp_size >= 1


class PageTableManager:
    """Manages page table mapping (request, position) -> cache slot.

    The page table is a 2D tensor: [max_batch_size, max_seq_len]
    Entry [i, j] = slot index for request i's token at position j.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.device = device

        self.page_table = torch.zeros(
            (max_batch_size, max_seq_len),
            dtype=torch.int32,
            device=device,
        )

        self.free_indices: list[int] = list(range(max_batch_size))

    def allocate_index(self) -> int:
        assert self.free_indices, "no free table indices"
        return self.free_indices.pop()

    def free_index(self, idx: int) -> None:
        assert 0 <= idx < self.max_batch_size
        self.free_indices.append(idx)

    def update_slots(self, table_idx: int, start_pos: int, slots: Tensor) -> None:
        end_pos = start_pos + len(slots)
        assert end_pos <= self.max_seq_len
        self.page_table[table_idx, start_pos:end_pos] = slots


class InferenceEngineV2:
    """Inference engine v2 with custom model and attention.

    Key differences from v1:
    - Uses custom LlamaForCausalLM (not HuggingFace)
    - Uses FlashAttention or reference attention backend
    - Proper batched forward with attention metadata
    - Page table for KV cache management
    - CUDA graphs for decode optimization
    - Overlap scheduling for CPU/GPU parallelism
    - Radix cache for prefix sharing
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self._use_functional_model = config.model_impl == "functional"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._is_cuda = self.device.type == "cuda"

        # Tensor parallelism config
        self.tp: TPConfig | None = None
        if config.tp_size > 1:
            self.tp = TPConfig(rank=config.tp_rank, world_size=config.tp_size)
            # Note: caller must call engine.init_tp_group() before using the engine

        # Load model config
        self.model_config = load_model_config(config.model_path)

        # Load tokenizer
        self.tokenizer = self._load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        # KV cache (use local head count for TP)
        num_slots = config.max_batch_size * config.max_seq_len
        num_kv_heads = self.model_config.num_key_value_heads
        if self.tp is not None and self.tp.is_distributed:
            from .tp import get_local_size

            num_kv_heads = get_local_size(num_kv_heads, self.tp.world_size)

        cache_config = CacheConfig(
            num_layers=self.model_config.num_hidden_layers,
            num_heads=num_kv_heads,
            head_dim=self.model_config.head_dim,
            num_slots=num_slots,
            dtype=config.dtype,
        )
        self.kv_pool = KVCachePool(cache_config, self.device)

        # Attention backend
        self.attn_backend = self._create_attention_backend(config.attention_backend)

        # Load model
        self.model: LlamaForCausalLM | None = None
        self._functional_config: FunctionalLlamaConfig | None = None
        self._functional_weights: dict[str, Tensor] | None = None
        if self._use_functional_model:
            self._functional_config = load_functional_config(config.model_path)
            self._functional_weights = load_weights(config.model_path, self.device, config.dtype)
            if "lm_head.weight" not in self._functional_weights:
                self._functional_weights["lm_head.weight"] = self._functional_weights[
                    "model.embed_tokens.weight"
                ]
        else:
            self.model = self._load_model(config.model_path, config.dtype)

        # Page table manager
        self.page_table_mgr = PageTableManager(
            config.max_batch_size,
            config.max_seq_len,
            self.device,
        )

        # Scheduler
        self.scheduler_config = SchedulerConfig(
            max_batch_size=config.max_batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
            max_seq_len=config.max_seq_len,
        )
        self.state = empty_scheduler_state()

        # Request counter
        self.next_uid = 0

        # State dicts for stateless modules (nmoe pattern)
        self._graph_state: dict = {}
        self._overlap_state: dict = {}
        self._radix_state: dict = {}

        # Request -> cache handle mapping
        self._cache_handles: dict[int, CacheHandle] = {}
        self._chunk_pending_tokens: dict[int, Tensor] = {}

        # Feature flags
        self._use_cuda_graphs = config.enable_cuda_graphs and self._is_cuda
        self._use_overlap = config.enable_overlap and self._is_cuda
        self._use_radix_cache = config.enable_radix_cache and self._is_cuda
        self._use_chunked_prefill = config.enable_chunked_prefill
        self._prefill_chunk_size = config.prefill_chunk_size or config.max_tokens_per_batch
        self._chunked_prefill_mgr = ChunkedPrefillManager(self._prefill_chunk_size)

        if self._use_functional_model:
            if self._use_cuda_graphs:
                logger.info("Disabling CUDA graphs for functional model path")
            if self._use_overlap:
                logger.info("Disabling overlap execution for functional model path")
            self._use_cuda_graphs = False
            self._use_overlap = False

        # Initialize radix cache state
        if self._use_radix_cache:
            init_radix_state(self._radix_state, self.device)
            logger.info("Radix cache enabled")

        if self._use_cuda_graphs:
            logger.info("CUDA graphs enabled (will capture on first decode)")

        if self._use_overlap:
            logger.info("Overlap scheduling enabled")

        if self._use_chunked_prefill:
            logger.info(f"Chunked prefill enabled (chunk_size={self._prefill_chunk_size})")

    def _load_tokenizer(self, model_path: str) -> Any:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path)

    def _create_attention_backend(
        self, backend_type: Literal["auto", "flash", "flashinfer", "reference"]
    ) -> Any:
        from .attention.flashinfer import FlashInferBackend, is_flashinfer_available

        # Use local head counts for TP
        num_q_heads = self.model_config.num_attention_heads
        num_kv_heads = self.model_config.num_key_value_heads
        if self.tp is not None and self.tp.is_distributed:
            from .tp import get_local_size

            num_q_heads = get_local_size(num_q_heads, self.tp.world_size)
            num_kv_heads = get_local_size(num_kv_heads, self.tp.world_size)

        # Determine which backend to use
        # Priority: flashinfer > flash > reference
        if backend_type == "flashinfer" or (backend_type == "auto" and is_flashinfer_available()):
            logger.info("Using FlashInfer attention backend")
            return FlashInferBackend(
                k_cache=self.kv_pool.k_cache,
                v_cache=self.kv_pool.v_cache,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.model_config.head_dim,
            )
        elif backend_type == "flash" or (backend_type == "auto" and is_flash_attn_available()):
            logger.info("Using FlashAttention backend")
            return FlashAttentionBackend(
                k_cache=self.kv_pool.k_cache,
                v_cache=self.kv_pool.v_cache,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.model_config.head_dim,
            )
        else:
            logger.info("Using reference attention backend")
            return ReferenceAttentionBackend(
                k_cache=self.kv_pool.k_cache,
                v_cache=self.kv_pool.v_cache,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim=self.model_config.head_dim,
            )

    def _load_model(self, model_path: str, dtype: torch.dtype) -> LlamaForCausalLM:
        model = LlamaForCausalLM(self.model_config, self.device, dtype, tp=self.tp)
        model.to(self.device)

        hf_weights = load_weights(model_path, self.device, dtype)
        remapped = remap_weights_llama(hf_weights, self.model_config.num_hidden_layers)

        # Shard weights for TP
        if self.tp is not None and self.tp.is_distributed:
            remapped = shard_weights_for_tp(
                remapped,
                self.tp,
                num_q_heads=self.model_config.num_attention_heads,
                num_kv_heads=self.model_config.num_key_value_heads,
                head_dim=self.model_config.head_dim,
                intermediate_size=self.model_config.intermediate_size,
            )

        model.load_weights(remapped)

        model.eval()
        return model

    def init_tp_group(
        self,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ) -> None:
        """Initialize tensor parallelism process group.

        Call this before using the engine for multi-GPU inference.
        Must be called once per process.
        """
        if self.tp is not None:
            self.tp.init_process_group(master_addr, master_port)
            logger.info(f"TP initialized: rank={self.tp.rank}/{self.tp.world_size}")

    def _model_forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        attn_backend: Any,
        attn_metadata: AttentionMetadata,
        out_loc: Tensor,
    ) -> Tensor:
        """Model forward pass (used by graph capture)."""
        assert self.model is not None
        return self.model(
            input_ids=input_ids,
            positions=positions,
            attn_backend=attn_backend,
            attn_metadata=attn_metadata,
            out_loc=out_loc,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═══════════════════════════════════════════════════════════════════════════

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams | None = None,
    ) -> int:
        """Add a request. Returns request uid."""
        if sampling_params is None:
            sampling_params = SamplingParams()

        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = list(prompt)

        assert len(prompt_ids) > 0, "prompt cannot be empty"
        if len(prompt_ids) > self.scheduler_config.max_seq_len:
            raise ValueError(
                f"Prompt length {len(prompt_ids)} exceeds max_seq_len="
                f"{self.scheduler_config.max_seq_len}"
            )
        table_idx = self.page_table_mgr.allocate_index()

        uid = self.next_uid
        self.next_uid += 1

        # Check radix cache for prefix hit
        cached_len = 0
        if self._use_radix_cache:
            prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int32, device=self.device)
            handle, matched_slots = match_prefix(self._radix_state, prompt_tensor)

            if handle.cached_len > 0:
                # For full prefix match, we still need to process last token to get logits.
                # Cap cached_len to prompt_len - 1 so there's always at least 1 token.
                # Note: This means full prefix matches will recompute the last token's KV,
                # which may cause minor non-determinism vs the cached version.
                if handle.cached_len == len(prompt_ids):
                    cached_len = len(prompt_ids) - 1
                    matched_slots = matched_slots[:-1]
                else:
                    cached_len = handle.cached_len

                lock(self._radix_state, handle)
                self._cache_handles[uid] = handle
                self.page_table_mgr.update_slots(table_idx, 0, matched_slots)
                logger.debug(f"Radix cache hit: {cached_len} tokens for request {uid}")

        full_req = create_req(
            uid=uid,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            table_idx=table_idx,
        )

        req = full_req

        if self._use_chunked_prefill and len(prompt_ids) > self._prefill_chunk_size:
            first_chunk = prompt_ids[: self._prefill_chunk_size]
            pending_chunk = prompt_ids[self._prefill_chunk_size :]

            req = Req(
                uid=full_req.uid,
                input_ids=torch.tensor(first_chunk, dtype=torch.int32),
                cached_len=0,
                max_len=full_req.max_len,
                sampling_params=full_req.sampling_params,
                table_idx=full_req.table_idx,
            )
            self._chunked_prefill_mgr.maybe_chunk(full_req)
            self._chunk_pending_tokens[uid] = torch.tensor(pending_chunk, dtype=torch.int32)

        if cached_len > 0:
            req = Req(
                uid=req.uid,
                input_ids=req.input_ids,
                cached_len=cached_len,
                max_len=req.max_len,
                sampling_params=req.sampling_params,
                table_idx=req.table_idx,
            )

        self.state = add_request(self.state, req)
        return uid

    def step(self) -> list[Req]:
        """Run one forward pass. Returns finished requests."""
        if not has_pending_work(self.state):
            return []

        result = schedule_step(
            state=self.state,
            config=self.scheduler_config,
            num_free_pages=self._get_num_free_pages(),
            device=self.device,
            allocate_pages=self._allocate_pages,
        )

        if result.batch is None:
            return []

        batch = result.batch
        self.state = result.new_state

        if self._use_functional_model:
            next_tokens, logprobs = self._forward_functional(batch)
        else:
            self._update_page_table(batch)
            attn_metadata = self._build_attention_metadata(batch)
            next_tokens, logprobs = self._forward_with_graphs(batch, attn_metadata)

        self.state = self._update_state_after_batch(
            state=self.state,
            batch=batch,
            next_tokens=next_tokens,
            logprobs=logprobs,
        )

        for req in self.state.finished:
            self._handle_finished_request(req)

        return list(self.state.finished)

    def run_to_completion(self) -> list[Req]:
        """Run until all requests complete."""
        all_finished: list[Req] = []
        while has_pending_work(self.state):
            finished = self.step()
            all_finished.extend(finished)
        return all_finished

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | None = None,
    ) -> list[Req]:
        """Add requests and run to completion."""
        for prompt in prompts:
            self.add_request(prompt, sampling_params)
        return self.run_to_completion()

    def has_pending(self) -> bool:
        return has_pending_work(self.state)

    def shutdown(self) -> None:
        """Cleanup resources."""
        if self._use_radix_cache:
            for handle in self._cache_handles.values():
                unlock(self._radix_state, handle)
            self._cache_handles.clear()
            # Reinitialize radix state to clear all cached prefixes
            init_radix_state(self._radix_state, self.device)
        self._chunk_pending_tokens.clear()

        self.state = empty_scheduler_state()
        self.kv_pool.reset()

    def reload_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Hot-reload model weights without restarting the engine.

        This is the PipelineRL-style weight sync primitive. Training broadcasts
        updated weights, and inference engines call this to load them.

        IMPORTANT: This must be called when no requests are in flight.
        The caller should pause generation before calling this.

        Args:
            state_dict: New weights in HuggingFace format (will be remapped internally)

        What gets invalidated:
        - CUDA graphs (they have weight pointers baked in)
        - Radix cache (cached KV was computed with old weights)
        - KV cache pool is NOT cleared (in-flight requests keep their cache)
        """
        assert not has_pending_work(self.state), (
            "Cannot reload weights while requests are pending. "
            "Call run_to_completion() or shutdown() first."
        )

        logger.info("Reloading weights...")

        # 1. Remap HF weights to our naming convention
        remapped = remap_weights_llama(state_dict, self.model_config.num_hidden_layers)

        # 2. Load into model (in-place copy)
        if self._use_functional_model:
            # Functional path: just replace the dict
            assert self._functional_weights is not None
            for name, param in remapped.items():
                if name in self._functional_weights:
                    self._functional_weights[name].copy_(param)
                else:
                    self._functional_weights[name] = param.to(self.device)
            # Handle tied embeddings
            if "lm_head.weight" not in self._functional_weights:
                self._functional_weights["lm_head.weight"] = self._functional_weights[
                    "model.embed_tokens.weight"
                ]
        else:
            # Module path: use load_state_dict
            assert self.model is not None
            self.model.load_weights(remapped)

        # 3. Invalidate CUDA graphs (they have weight pointers baked in)
        if self._use_cuda_graphs:
            self._graph_state.clear()
            logger.info("CUDA graphs invalidated (will re-capture on next decode)")

        # 4. Invalidate radix cache (cached KV was computed with old weights)
        if self._use_radix_cache:
            # Release all handles and reinitialize
            for handle in self._cache_handles.values():
                unlock(self._radix_state, handle)
            self._cache_handles.clear()
            init_radix_state(self._radix_state, self.device)
            logger.info("Radix cache invalidated")

        # 5. Reset KV cache pool (old KV was computed with old weights)
        self.kv_pool.reset()

        logger.info("Weights reloaded successfully")

    def reload_weights_from_path(self, model_path: str) -> None:
        """Reload weights from a checkpoint path (HuggingFace format).

        Convenience wrapper around reload_weights() that loads from disk.

        Args:
            model_path: Path to HuggingFace checkpoint or model ID
        """
        state_dict = load_weights(model_path, self.device, self.config.dtype)
        self.reload_weights(state_dict)

    # ═══════════════════════════════════════════════════════════════════════════
    # NCCL WEIGHT SYNC (PipelineRL-style)
    # ═══════════════════════════════════════════════════════════════════════════

    def init_weight_sync(
        self,
        master_addr: str,
        master_port: int,
        rank: int,
        world_size: int,
        group_name: str = "weight_sync",
    ) -> None:
        """Initialize NCCL weight sync to receive weights from training.

        Call this once at startup to join the weight sync group.
        The trainer (rank 0) broadcasts weights, inference engines receive.

        Args:
            master_addr: IP address of trainer
            master_port: Port for NCCL rendezvous
            rank: This engine's rank (1-indexed, 0 is trainer)
            world_size: Total processes (trainer + inference GPUs)
            group_name: Name for the NCCL group
        """
        from .weight_sync import WeightSyncReceiver

        assert rank > 0, "Rank 0 is reserved for trainer"

        self._weight_sync_receiver = WeightSyncReceiver(
            master_addr=master_addr,
            master_port=master_port,
            rank=rank,
            world_size=world_size,
            group_name=group_name,
            device=self.device,
        )
        self._weight_sync_receiver.init_group()
        logger.info(f"Weight sync initialized (rank={rank})")

    def receive_weights_from_nccl(self) -> None:
        """Receive weights from trainer via NCCL and reload.

        Call this after the trainer broadcasts weights.
        Blocks until weights are received, then reloads model.

        For TP: receives full weights, shards them, and loads shards.
        """
        receiver = getattr(self, "_weight_sync_receiver", None)
        assert receiver is not None, "Call init_weight_sync() first"

        if self.tp is not None and self.tp.is_distributed:
            # TP mode: receive full weights, then shard
            self._receive_weights_tp(receiver)
        else:
            # Single GPU: receive directly into model
            self._receive_weights_single_gpu(receiver)

        # Invalidate caches
        if self._use_cuda_graphs:
            self._graph_state.clear()
            logger.info("CUDA graphs invalidated (will re-capture)")

        if self._use_radix_cache:
            from .radix import init_radix_state, unlock

            for handle in self._cache_handles.values():
                unlock(self._radix_state, handle)
            self._cache_handles.clear()
            init_radix_state(self._radix_state, self.device)
            logger.info("Radix cache invalidated")

        self.kv_pool.reset()
        logger.info(f"Weights received and loaded (v{receiver.weight_version})")

    def _receive_weights_single_gpu(self, receiver: Any) -> None:
        """Receive weights directly into model (non-TP path)."""
        if self._use_functional_model:
            assert self._functional_weights is not None
            receiver.receive_weights_into(self._functional_weights)
        else:
            assert self.model is not None
            receiver.receive_weights_into(self.model.state_dict())

    def _receive_weights_tp(self, receiver: Any) -> None:
        """Receive full weights, shard them for this TP rank, and load."""

        assert self.tp is not None

        # Build template with FULL (unsharded) shapes for receiving
        param_infos = self._build_full_param_infos()

        # Receive full weights
        full_weights = receiver.receive_weights(param_infos)

        # Remap from HF naming to our naming
        remapped = remap_weights_llama(full_weights, self.model_config.num_hidden_layers)

        # Shard for this TP rank
        sharded = shard_weights_for_tp(
            remapped,
            self.tp,
            num_q_heads=self.model_config.num_attention_heads,
            num_kv_heads=self.model_config.num_key_value_heads,
            head_dim=self.model_config.head_dim,
            intermediate_size=self.model_config.intermediate_size,
        )

        # Load into model
        if self._use_functional_model:
            assert self._functional_weights is not None
            for name, param in sharded.items():
                if name in self._functional_weights:
                    self._functional_weights[name].copy_(param)
        else:
            assert self.model is not None
            self.model.load_weights(sharded)

    def _build_full_param_infos(self) -> list[Any]:
        """Build ParamInfo list with full (unsharded) weight shapes.

        Used for receiving full weights from trainer before sharding.
        """
        from .weight_sync import ParamInfo

        # Get shapes from model config (full sizes, not sharded)
        hidden_size = self.model_config.hidden_size
        num_q_heads = self.model_config.num_attention_heads
        num_kv_heads = self.model_config.num_key_value_heads
        head_dim = self.model_config.head_dim
        intermediate_size = self.model_config.intermediate_size
        vocab_size = self.model_config.vocab_size
        num_layers = self.model_config.num_hidden_layers

        q_size = num_q_heads * head_dim
        kv_size = num_kv_heads * head_dim
        dtype = self.config.dtype

        infos = []

        # Embeddings
        infos.append(ParamInfo("model.embed_tokens.weight", (vocab_size, hidden_size), dtype))

        # Layers
        for i in range(num_layers):
            prefix = f"model.layers.{i}"

            # QKV (separate in HF format)
            infos.append(
                ParamInfo(f"{prefix}.self_attn.q_proj.weight", (q_size, hidden_size), dtype)
            )
            infos.append(
                ParamInfo(f"{prefix}.self_attn.k_proj.weight", (kv_size, hidden_size), dtype)
            )
            infos.append(
                ParamInfo(f"{prefix}.self_attn.v_proj.weight", (kv_size, hidden_size), dtype)
            )
            infos.append(
                ParamInfo(f"{prefix}.self_attn.o_proj.weight", (hidden_size, q_size), dtype)
            )

            # MLP (separate in HF format)
            infos.append(
                ParamInfo(f"{prefix}.mlp.gate_proj.weight", (intermediate_size, hidden_size), dtype)
            )
            infos.append(
                ParamInfo(f"{prefix}.mlp.up_proj.weight", (intermediate_size, hidden_size), dtype)
            )
            infos.append(
                ParamInfo(f"{prefix}.mlp.down_proj.weight", (hidden_size, intermediate_size), dtype)
            )

            # Norms
            infos.append(ParamInfo(f"{prefix}.input_layernorm.weight", (hidden_size,), dtype))
            infos.append(
                ParamInfo(f"{prefix}.post_attention_layernorm.weight", (hidden_size,), dtype)
            )

        # Final norm
        infos.append(ParamInfo("model.norm.weight", (hidden_size,), dtype))

        # LM head
        infos.append(ParamInfo("lm_head.weight", (vocab_size, hidden_size), dtype))

        return infos

    def cleanup_weight_sync(self) -> None:
        """Cleanup weight sync resources."""
        receiver = getattr(self, "_weight_sync_receiver", None)
        if receiver is not None:
            receiver.cleanup()
            self._weight_sync_receiver = None

    def capture_cuda_graphs_now(self) -> None:
        """Capture CUDA graphs for decode optimization.

        Call this after warmup. If not called explicitly,
        graphs are captured automatically on first decode batch.
        """
        if not self._use_cuda_graphs:
            return
        if "graphs" in self._graph_state:
            return

        logger.info("Capturing CUDA graphs...")
        capture_graphs(
            state=self._graph_state,
            model_forward=self._model_forward,
            attn_backend=self.attn_backend,
            device=self.device,
            max_seq_len=self.config.max_seq_len,
            vocab_size=self.model_config.vocab_size,
            batch_sizes=self.config.graph_batch_sizes,
            dtype=self.config.dtype,
        )
        logger.info("CUDA graphs captured")

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_num_free_pages(self) -> int:
        base_free = self.kv_pool.num_free_slots
        if self._use_radix_cache:
            return base_free + self._radix_state["evictable_tokens"]
        return base_free

    def _allocate_pages(self, n: int) -> Tensor:
        if self.kv_pool.num_free_slots >= n:
            return self.kv_pool.allocate_slots(n)

        if self._use_radix_cache:
            needed = n - self.kv_pool.num_free_slots
            if needed <= self._radix_state["evictable_tokens"]:
                evicted_slots = evict(self._radix_state, needed)
                remaining = n - len(evicted_slots)
                if remaining > 0:
                    new_slots = self.kv_pool.allocate_slots(remaining)
                    return torch.cat([evicted_slots, new_slots])
                return evicted_slots[:n]

        return self.kv_pool.allocate_slots(n)

    def _update_page_table(self, batch: Batch) -> None:
        offset = 0
        for req in batch.reqs:
            extend_len = req.extend_len
            slots = batch.out_loc[offset : offset + extend_len]
            self.page_table_mgr.update_slots(req.table_idx, req.cached_len, slots)
            offset += extend_len

    def _handle_finished_request(self, req: Req) -> None:
        self.page_table_mgr.free_index(req.table_idx)
        self._chunked_prefill_mgr.cleanup(req.uid)
        self._chunk_pending_tokens.pop(req.uid, None)

        if req.uid in self._cache_handles:
            handle = self._cache_handles.pop(req.uid)
            if self._use_radix_cache:
                unlock(self._radix_state, handle)

        if self._use_radix_cache:
            seq_len = req.device_len
            slots = self.page_table_mgr.page_table[req.table_idx, :seq_len].clone()
            tokens = req.input_ids.to(self.device)
            insert_prefix(self._radix_state, tokens, slots)

    def _append_next_prompt_chunk(self, req: Req) -> Req:
        pending = self._chunk_pending_tokens.get(req.uid)
        if pending is None or len(pending) == 0:
            return req

        chunk_len = min(self._prefill_chunk_size, len(pending))
        next_chunk = pending[:chunk_len]
        remaining = pending[chunk_len:]

        if len(remaining) == 0:
            self._chunk_pending_tokens.pop(req.uid, None)
        else:
            self._chunk_pending_tokens[req.uid] = remaining

        new_input_ids = torch.cat([req.input_ids, next_chunk])
        return Req(
            uid=req.uid,
            input_ids=new_input_ids,
            cached_len=req.cached_len,
            max_len=req.max_len,
            sampling_params=req.sampling_params,
            table_idx=req.table_idx,
        )

    def _update_state_after_batch(
        self,
        state: SchedulerState,
        batch: Batch,
        next_tokens: Tensor,
        logprobs: list[float | None],
    ) -> SchedulerState:
        decode_by_uid = {req.uid: req for req in state.decode_set}
        prefill_queue = list(state.prefill_queue)
        finished: list[Req] = []

        for i, (req, next_token) in enumerate(zip(batch.reqs, next_tokens.tolist(), strict=False)):
            logprob = logprobs[i]
            decode_by_uid.pop(req.uid, None)

            if batch.is_prefill and self._chunked_prefill_mgr.is_chunking(req.uid):
                chunk_prefill_req = req_after_forward(req)
                self._chunked_prefill_mgr.advance(req.uid, chunk_prefill_req)

                if self._chunked_prefill_mgr.is_chunking(req.uid):
                    next_chunk_req = self._append_next_prompt_chunk(chunk_prefill_req)
                    prefill_queue.append(next_chunk_req)
                    continue

                new_req = req_append_token(chunk_prefill_req, next_token, logprob)
            else:
                new_req = req_after_decode_step(req, next_token, logprob)

            is_eos = next_token == self.eos_token_id and not req.sampling_params.ignore_eos
            is_max_len = not new_req.can_decode
            if is_eos or is_max_len:
                finished.append(new_req)
            else:
                decode_by_uid[new_req.uid] = new_req

        return SchedulerState(
            prefill_queue=tuple(prefill_queue),
            decode_set=frozenset(decode_by_uid.values()),
            finished=tuple(finished),
        )

    def _build_attention_metadata(self, batch: Batch) -> AttentionMetadata:
        cached_lens = [req.cached_len for req in batch.reqs]
        extend_lens = [req.extend_len for req in batch.reqs]
        table_indices = [req.table_idx for req in batch.reqs]

        return build_attention_metadata(
            cached_lens=cached_lens,
            extend_lens=extend_lens,
            page_table=self.page_table_mgr.page_table,
            device=self.device,
            table_indices=table_indices,
        )

    def _forward_with_graphs(
        self, batch: Batch, attn_metadata: AttentionMetadata
    ) -> tuple[Tensor, list[float | None]]:
        use_graph = (
            self._use_cuda_graphs and batch.is_decode and can_use_graph(self._graph_state, batch)
        )

        if use_graph:
            if "graphs" not in self._graph_state:
                self.capture_cuda_graphs_now()
            return self._forward_with_graph(batch, attn_metadata)
        else:
            return self._forward(batch, attn_metadata)

    def _forward_with_graph(
        self, batch: Batch, attn_metadata: AttentionMetadata
    ) -> tuple[Tensor, list[float | None]]:
        with torch.no_grad():
            logits = replay_graph(self._graph_state, batch, attn_metadata)

        return self._sample_batch(logits, batch)

    def _forward(
        self, batch: Batch, attn_metadata: AttentionMetadata
    ) -> tuple[Tensor, list[float | None]]:
        assert self.model is not None
        with torch.no_grad():
            logits = self.model(
                input_ids=batch.input_ids,
                positions=batch.positions,
                attn_backend=self.attn_backend,
                attn_metadata=attn_metadata,
                out_loc=batch.out_loc,
            )

            last_indices = attn_metadata.cu_seqlens_q[1:] - 1
            last_logits = logits[last_indices]

        return self._sample_batch(last_logits, batch)

    def _forward_functional(self, batch: Batch) -> tuple[Tensor, list[float | None]]:
        # TODO: This functional path is incomplete - it recomputes full KV every step (O(n²)).
        # To make it production-ready:
        # 1. Pass k_cache/v_cache as explicit arguments to functional_forward
        # 2. Pass attn_metadata and out_loc for paged attention
        # 3. Re-enable CUDA graphs and overlap scheduling
        # For now, focus on the module path. Revisit functional once that's solid.
        assert self._functional_config is not None
        assert self._functional_weights is not None

        with torch.no_grad():
            rows: list[Tensor] = []
            for req in batch.reqs:
                req_input_ids = req.input_ids.to(self.device, dtype=torch.long)
                logits = functional_forward(
                    req_input_ids.unsqueeze(0),
                    self._functional_weights,
                    self._functional_config,
                )
                rows.append(logits[0, -1])
            last_logits = torch.stack(rows, dim=0)

        return self._sample_batch(last_logits, batch)

    def _sample_batch(self, logits: Tensor, batch: Batch) -> tuple[Tensor, list[float | None]]:
        """Sample next tokens from logits.

        Returns:
            (next_tokens, logprobs) where logprobs[i] is the log probability
            of next_tokens[i], or None if return_logprobs=False for that request.
        """
        next_tokens: list[int] = []
        logprobs: list[float | None] = []

        for i, req in enumerate(batch.reqs):
            token, logprob = self._sample_one(logits[i : i + 1], req.sampling_params)
            next_tokens.append(token)
            logprobs.append(logprob)

        tokens_tensor = torch.tensor(next_tokens, dtype=torch.int32, device="cpu")
        return tokens_tensor, logprobs

    def _sample_one(self, logits: Tensor, params: SamplingParams) -> tuple[int, float | None]:
        """Sample a single token from logits.

        Returns:
            (token_id, logprob) where logprob is None if return_logprobs=False.
        """
        # Compute log probabilities before any masking (for accurate logprobs)
        log_probs = torch.log_softmax(logits, dim=-1)

        if params.is_greedy:
            token = logits.argmax(dim=-1).item()
            logprob = log_probs[0, token].item() if params.return_logprobs else None
            return token, logprob

        # Apply temperature
        if params.temperature > 0:
            logits = logits / params.temperature

        # Apply top-k filtering
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Apply top-p (nucleus) filtering
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        # Sample from the filtered distribution
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()

        # Return logprob from the ORIGINAL distribution (not the filtered one)
        logprob = log_probs[0, token].item() if params.return_logprobs else None
        return token, logprob

    # ═══════════════════════════════════════════════════════════════════════════
    # OVERLAP EXECUTION
    # ═══════════════════════════════════════════════════════════════════════════

    def run_to_completion_overlap(self) -> list[Req]:
        """Run until all requests complete using overlap scheduling.

        Uses dual-stream execution to hide CPU latency.
        """
        if not self._use_overlap:
            return self.run_to_completion()

        if self._use_cuda_graphs and "graphs" not in self._graph_state:
            self.capture_cuda_graphs_now()

        all_finished: list[Req] = []
        last_output: ForwardOutput | None = None

        while has_pending_work(self.state) or last_output is not None:
            current_output = overlap_step(
                state=self._overlap_state,
                device=self.device,
                last_output=last_output,
                forward_fn=self._overlap_forward,
                schedule_fn=self._overlap_schedule,
                process_fn=self._overlap_process,
            )

            if last_output is not None:
                last_output.copy_done.synchronize()
                all_finished.extend(list(self.state.finished))

            last_output = current_output

        if last_output is not None:
            last_output.copy_done.synchronize()
            all_finished.extend(list(self.state.finished))

        return all_finished

    def _overlap_schedule(self) -> ForwardInput | None:
        if not has_pending_work(self.state):
            return None

        result = schedule_step(
            state=self.state,
            config=self.scheduler_config,
            num_free_pages=self._get_num_free_pages(),
            device=self.device,
            allocate_pages=self._allocate_pages,
        )

        if result.batch is None:
            return None

        batch = result.batch
        self.state = result.new_state

        self._update_page_table(batch)
        attn_metadata = self._build_attention_metadata(batch)

        return ForwardInput(batch=batch, attn_metadata=attn_metadata)

    def _overlap_forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch = forward_input.batch
        attn_metadata = forward_input.attn_metadata

        use_graph = (
            self._use_cuda_graphs
            and batch.is_decode
            and "graphs" in self._graph_state
            and can_use_graph(self._graph_state, batch)
        )

        with torch.no_grad():
            if use_graph:
                logits = replay_graph(self._graph_state, batch, attn_metadata)
            else:
                logits = self.model(
                    input_ids=batch.input_ids,
                    positions=batch.positions,
                    attn_backend=self.attn_backend,
                    attn_metadata=attn_metadata,
                    out_loc=batch.out_loc,
                )
                last_indices = attn_metadata.cu_seqlens_q[1:] - 1
                logits = logits[last_indices]

            # Note: overlap path doesn't support logprobs yet
            next_tokens_gpu, _logprobs = self._sample_batch_gpu(logits, batch)

        # Get stream from overlap state
        from .overlap import get_or_create_streams

        _, engine_stream = get_or_create_streams(self._overlap_state, self.device)
        return create_forward_output(next_tokens_gpu, engine_stream)

    def _overlap_process(self, output: ForwardOutput) -> None:
        # Processing happens in run_to_completion_overlap after synchronize
        pass

    def _sample_batch_gpu(self, logits: Tensor, batch: Batch) -> tuple[Tensor, list[float | None]]:
        """Sample batch on GPU, returning tokens and logprobs."""
        all_greedy = all(req.sampling_params.is_greedy for req in batch.reqs)
        any_logprobs = any(req.sampling_params.return_logprobs for req in batch.reqs)

        if all_greedy and not any_logprobs:
            tokens = logits.argmax(dim=-1).to(torch.int32)
            logprobs_list: list[float | None] = [None] * len(batch.reqs)
            return tokens, logprobs_list

        # Need per-token sampling and/or logprobs
        next_tokens: list[int] = []
        logprobs_list = []
        for i, req in enumerate(batch.reqs):
            token, logprob = self._sample_one(logits[i : i + 1], req.sampling_params)
            next_tokens.append(token)
            logprobs_list.append(logprob)

        tokens = torch.tensor(next_tokens, dtype=torch.int32, device=self.device)
        return tokens, logprobs_list
