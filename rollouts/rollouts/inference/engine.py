"""Inference engine - orchestrates model, cache, and scheduler.

This is a class because it owns GPU resources.
Pure functions in scheduler.py and core.py do the actual logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .core import Batch, Req, SamplingParams, create_req
from .kv_cache import (
    CacheConfig,
    KVCachePool,
    RequestCache,
    empty_request_cache,
    extend_request_cache,
    gather_past_key_values,
    store_new_key_values,
)
from .scheduler import (
    SchedulerConfig,
    add_request,
    empty_scheduler_state,
    has_pending_work,
    schedule_step,
    update_after_forward,
)

# ═══════════════════════════════════════════════════════════════════════════════
# ENGINE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EngineConfig:
    """Engine configuration."""

    model_path: str
    max_batch_size: int = 32
    max_tokens_per_batch: int = 4096
    max_seq_len: int = 2048
    dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        assert self.max_batch_size > 0
        assert self.max_tokens_per_batch > 0
        assert self.max_seq_len > 0


# ═══════════════════════════════════════════════════════════════════════════════
# TABLE MANAGER (simple version)
# ═══════════════════════════════════════════════════════════════════════════════


class SimpleTableManager:
    """Assigns table indices to requests.

    Each request gets a slot in the page table.
    """

    def __init__(self, max_requests: int) -> None:
        assert max_requests > 0
        self.max_requests = max_requests
        self.free_indices: list[int] = list(range(max_requests))

    def allocate(self) -> int:
        """Get a free table index."""
        assert self.free_indices, "no free table indices"
        return self.free_indices.pop()

    def free(self, idx: int) -> None:
        """Return table index to pool."""
        assert 0 <= idx < self.max_requests
        assert idx not in self.free_indices, "double free"
        self.free_indices.append(idx)


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════


class InferenceEngine:
    """Main inference engine.

    Owns:
    - Model (GPU)
    - KV cache pool (GPU memory)
    - Table manager

    State:
    - Scheduler state (immutable, replaced each step)
    - Request caches (immutable per-request, replaced on update)

    Why a class? Owns GPU resources, needs cleanup.
    """

    def __init__(self, config: EngineConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(config.model_path, config.dtype)
        self.tokenizer = self._load_tokenizer(config.model_path)
        self.eos_token_id = self.tokenizer.eos_token_id

        # Extract model config for KV cache
        model_config = self.model.config
        num_layers = model_config.num_hidden_layers
        num_heads = model_config.num_key_value_heads  # For GQA models
        head_dim = model_config.hidden_size // model_config.num_attention_heads

        # Scheduler config
        self.scheduler_config = SchedulerConfig(
            max_batch_size=config.max_batch_size,
            max_tokens_per_batch=config.max_tokens_per_batch,
            max_seq_len=config.max_seq_len,
        )

        # KV cache pool
        num_slots = config.max_batch_size * config.max_seq_len
        cache_config = CacheConfig(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_slots=num_slots,
            dtype=config.dtype,
        )
        self.kv_pool = KVCachePool(cache_config, self.device)

        # Table manager (for request slot assignment)
        self.table_manager = SimpleTableManager(config.max_batch_size)

        # Scheduler state (immutable, replaced each step)
        self.state = empty_scheduler_state()

        # Per-request cache state: uid -> RequestCache (immutable, replaced on update)
        self.request_caches: dict[int, RequestCache] = {}

        # Request counter
        self.next_uid = 0

    def _load_model(self, model_path: str, dtype: torch.dtype) -> nn.Module:
        """Load HuggingFace model."""
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=self.device,
        )
        model.eval()
        return model

    def _load_tokenizer(self, model_path: str):
        """Load tokenizer."""
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_path)

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

        # Tokenize if string
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        else:
            prompt_ids = list(prompt)

        assert len(prompt_ids) > 0, "prompt cannot be empty"

        # Allocate table slot
        table_idx = self.table_manager.allocate()

        # Create request
        uid = self.next_uid
        self.next_uid += 1

        req = create_req(
            uid=uid,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            table_idx=table_idx,
        )

        # Add to scheduler
        self.state = add_request(self.state, req)

        return uid

    def step(self) -> list[Req]:
        """Run one forward pass. Returns finished requests."""
        if not has_pending_work(self.state):
            return []

        # Schedule
        result = schedule_step(
            state=self.state,
            config=self.scheduler_config,
            num_free_pages=self.kv_pool.num_free_slots,
            device=self.device,
            allocate_pages=self.kv_pool.allocate_slots,
        )

        if result.batch is None:
            return []

        batch = result.batch
        self.state = result.new_state

        # Forward pass with KV cache
        next_tokens = self._forward_with_cache(batch)

        # Update state
        self.state = update_after_forward(
            state=self.state,
            batch=batch,
            next_tokens=next_tokens,
            eos_token_id=self.eos_token_id,
        )

        # Free table slots and cleanup cache for finished requests
        for req in self.state.finished:
            self.table_manager.free(req.table_idx)
            if req.uid in self.request_caches:
                del self.request_caches[req.uid]

        return list(self.state.finished)

    def run_to_completion(self) -> list[Req]:
        """Run until all requests complete. Returns all finished requests."""
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
        """Convenience: add requests and run to completion."""
        for prompt in prompts:
            self.add_request(prompt, sampling_params)

        return self.run_to_completion()

    def has_pending(self) -> bool:
        """Check if there's pending work."""
        return has_pending_work(self.state)

    def shutdown(self) -> None:
        """Cleanup resources."""
        self.state = empty_scheduler_state()
        self.kv_pool.reset()
        self.request_caches.clear()
        # Model cleanup handled by garbage collection

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════════════════════════════

    def _forward_with_cache(self, batch: Batch) -> Tensor:
        """Run forward pass with KV cache, return sampled tokens.

        For each request:
        1. Gather past K,V from cache (if any)
        2. Run forward on new tokens only
        3. Store new K,V in allocated slots
        4. Sample next token

        Processing requests individually for now (batching would require padding).
        """
        next_tokens: list[int] = []

        # out_loc tells us where to store new K,V; split by request
        out_loc_offset = 0

        for req in batch.reqs:
            extend_len = req.extend_len
            assert extend_len > 0

            # Get slots for this request's new tokens
            req_out_loc = batch.out_loc[out_loc_offset : out_loc_offset + extend_len]
            out_loc_offset += extend_len

            # Get or create request cache
            if req.uid not in self.request_caches:
                self.request_caches[req.uid] = empty_request_cache(req.uid, self.device)
            req_cache = self.request_caches[req.uid]

            # Sanity check: cached slots should match cached_len
            assert req_cache.cached_len == req.cached_len, (
                f"cache mismatch for req {req.uid}: "
                f"cache has {req_cache.cached_len}, req has {req.cached_len}"
            )

            # Gather past K,V for cached tokens
            past_key_values = gather_past_key_values(self.kv_pool, req_cache.slots)

            # New tokens to process: [cached_len, device_len)
            new_tokens = req.input_ids[req.cached_len : req.device_len]
            input_ids = new_tokens.to(self.device).unsqueeze(0)  # [1, extend_len]

            with torch.no_grad():
                outputs = self.model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                # Logits for last position
                last_logits = outputs.logits[:, -1, :]  # [1, vocab]

                # Store new K,V in cache
                # outputs.past_key_values contains K,V for full sequence
                # We extract only the new tokens' K,V
                store_new_key_values(
                    self.kv_pool,
                    req_out_loc,
                    outputs.past_key_values,
                    cached_len=req.cached_len,
                )

            # Update request cache with new slots
            self.request_caches[req.uid] = extend_request_cache(req_cache, req_out_loc)

            # Sample
            token = self._sample(last_logits, req.sampling_params)
            next_tokens.append(token)

        return torch.tensor(next_tokens, dtype=torch.int32, device="cpu")

    def _sample(self, logits: Tensor, params: SamplingParams) -> int:
        """Sample from logits."""
        if params.is_greedy:
            return logits.argmax(dim=-1).item()

        # Apply temperature
        if params.temperature > 0:
            logits = logits / params.temperature

        # Apply top-k
        if params.top_k > 0:
            top_k = min(params.top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Apply top-p
        if params.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > params.top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # Sample
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return token.item()
