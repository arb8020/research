# nano-inference: Minimal Inference Engine for RL

> **Design Principles Applied:**
> - Write usage code first (Casey)
> - Frozen dataclasses for config/data (CLASSES_VS_FUNCTIONAL)
> - Classes only for legitimate state (resources, lifecycle)
> - Pure functions for computation
> - Functions orchestrate stateful objects
> - Minimize stateful components (Sean)
> - Boring, well-tested components (Sean)

---

## Usage Code First

What we *want* the API to look like:

```python
# ═══════════════════════════════════════════════════
# USAGE: RL Training Loop
# ═══════════════════════════════════════════════════

from nano_inference import InferenceEngine, EngineConfig, SamplingParams

# Config is frozen dataclass - immutable, serializable
config = EngineConfig(
    model_path="Qwen/Qwen2.5-0.5B",
    cache_type="radix",  # or "paged"
    block_size=16,
)

# Engine is a class - owns GPU resources, needs cleanup
engine = InferenceEngine(config)

# Generate rollouts (the hot path)
samples = engine.generate(
    prompts=[[1, 2, 3, 4], [5, 6, 7, 8]],  # token IDs
    sampling_params=SamplingParams(temperature=0.7, max_tokens=256),
    num_samples_per_prompt=4,  # GRPO: 4 completions per prompt
)

# Each sample has what training needs
for sample in samples:
    print(sample.completion_tokens)  # [9, 10, 11, ...]
    print(sample.logprobs)           # [-1.2, -0.8, ...] per token
    print(sample.weight_version)     # 0 (which model generated this)

# Weight sync (PipelineRL-style, non-blocking)
engine.update_weights(new_state_dict, blocking=False)

# Or Miles-style blocking
engine.update_weights(new_state_dict, blocking=True)
engine.flush_cache()  # Clear KV cache after weights change

# Cleanup
engine.shutdown()
```

```python
# ═══════════════════════════════════════════════════
# USAGE: Low-Level Control (Continuous Granularity)
# ═══════════════════════════════════════════════════

# High-level: engine.generate() does everything
samples = engine.generate(prompts, params, num_samples=4)

# Mid-level: step-by-step control
for prompt in prompts:
    engine.add_request(prompt, params)

while engine.has_pending():
    finished = engine.step()  # One scheduling + forward pass
    for sample in finished:
        process(sample)

# Low-level: use pure functions directly
from nano_inference import schedule, sample_with_logprobs, model_forward

# Schedule is a pure function
sched_output = schedule(waiting, running, cache, config)

# Sampling is a pure function
tokens, logprobs = sample_with_logprobs(logits, temperatures)
```

---

## Requirements for RL

| Feature | Why |
|---------|-----|
| Logprobs | Policy gradient: `loss = -logprob * advantage` |
| Prefix caching | N rollouts share same prompt computation |
| Weight sync | Update policy weights without restart |
| Continuous batching | Maximize GPU utilization |

---

## Architecture: Functions Orchestrate Objects

```
┌─────────────────────────────────────────────────────────────────────┐
│                        InferenceEngine (class)                       │
│   Legitimate state: GPU resources, model weights, KV cache          │
│                                                                      │
│   ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│   │ KVCache      │  │ Model        │  │ WeightSyncManager      │   │
│   │ (class)      │  │ (nn.Module)  │  │ (class)                │   │
│   │ owns blocks  │  │ owns params  │  │ owns NCCL group        │   │
│   └──────────────┘  └──────────────┘  └────────────────────────┘   │
│                                                                      │
│   Orchestrated by pure functions:                                    │
│   - schedule(waiting, running, cache, config) -> SchedulerOutput    │
│   - model_forward(model, inputs, cache) -> logits                   │
│   - sample_with_logprobs(logits, temps) -> tokens, logprobs         │
└─────────────────────────────────────────────────────────────────────┘
```

**Why classes here?**
- `InferenceEngine`: Owns GPU memory, model weights, needs `shutdown()`
- `KVCache`: Owns block allocation state, ref counts
- `WeightSyncManager`: Owns NCCL process group, pending sync state

**Why pure functions?**
- `schedule()`: Stateless decision - takes lists, returns what to run
- `sample_with_logprobs()`: Pure math - logits in, tokens out
- `model_forward()`: Wraps stateful model but is itself deterministic

---

## Data Types (Frozen Dataclasses)

```python
from dataclasses import dataclass
from typing import Literal

# ═══════════════════════════════════════════════════
# CONFIG: Immutable, serializable
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class EngineConfig:
    """Engine configuration. Immutable after creation."""
    model_path: str
    cache_type: Literal["paged", "radix"]
    block_size: int = 16
    max_batch_size: int = 256
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9


@dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration. Immutable."""
    temperature: float = 1.0
    max_tokens: int = 256
    stop_token_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration. Immutable."""
    max_batch_size: int
    max_tokens_per_batch: int
    block_size: int


# ═══════════════════════════════════════════════════
# OUTPUT: Immutable results
# ═══════════════════════════════════════════════════

@dataclass(frozen=True)
class TrainingSample:
    """Output from inference engine for RL training. Immutable."""
    prompt_tokens: tuple[int, ...]      # tuple for immutability
    completion_tokens: tuple[int, ...]
    logprobs: tuple[float, ...]         # per-token logprob
    ref_logprobs: tuple[float, ...] | None  # for KL penalty
    weight_version: int
    finish_reason: Literal["stop", "length"]


@dataclass(frozen=True)
class CacheAllocation:
    """Result of cache allocation. Immutable."""
    block_ids: tuple[int, ...]
    cached_length: int


@dataclass(frozen=True)
class SchedulerOutput:
    """Result of scheduling decision. Immutable."""
    prefill_seqs: tuple[int, ...]   # seq_ids to prefill
    decode_seqs: tuple[int, ...]    # seq_ids to decode
    preempted_seqs: tuple[int, ...] # seq_ids evicted
```

---

## Pure Functions (Computation)

```python
import torch

# ═══════════════════════════════════════════════════
# SCHEDULING: Pure function
# ═══════════════════════════════════════════════════

def schedule(
    waiting: list[Sequence],
    running: list[Sequence],
    num_free_blocks: int,
    config: SchedulerConfig,
) -> SchedulerOutput:
    """Pure function: decide what to run this step.

    No side effects. Takes state, returns decision.
    """
    assert config.max_batch_size > 0
    assert config.block_size > 0

    prefill_seqs = []
    decode_seqs = []
    preempted_seqs = []

    blocks_available = num_free_blocks
    batch_tokens = 0

    # Decode running sequences first (they're already allocated)
    for seq in running:
        assert seq.status == SequenceStatus.RUNNING
        if len(decode_seqs) < config.max_batch_size:
            decode_seqs.append(seq.seq_id)
            batch_tokens += 1  # decode = 1 token

    # Then prefill waiting sequences
    for seq in waiting:
        assert seq.status == SequenceStatus.WAITING
        blocks_needed = (len(seq.token_ids) + config.block_size - 1) // config.block_size

        if blocks_needed > blocks_available:
            break  # Can't fit, stop
        if len(prefill_seqs) + len(decode_seqs) >= config.max_batch_size:
            break
        if batch_tokens + len(seq.token_ids) > config.max_tokens_per_batch:
            break

        prefill_seqs.append(seq.seq_id)
        blocks_available -= blocks_needed
        batch_tokens += len(seq.token_ids)

    return SchedulerOutput(
        prefill_seqs=tuple(prefill_seqs),
        decode_seqs=tuple(decode_seqs),
        preempted_seqs=tuple(preempted_seqs),
    )


# ═══════════════════════════════════════════════════
# SAMPLING: Pure function
# ═══════════════════════════════════════════════════

def sample_with_logprobs(
    logits: torch.Tensor,  # [batch, vocab]
    temperatures: torch.Tensor,  # [batch]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure function: sample tokens and compute their logprobs.

    Args:
        logits: Raw model output [batch, vocab_size]
        temperatures: Per-sequence temperature [batch]

    Returns:
        tokens: Sampled token IDs [batch]
        logprobs: Log probability of each sampled token [batch]
    """
    assert logits.dim() == 2
    assert temperatures.dim() == 1
    assert logits.size(0) == temperatures.size(0)

    # Scale by temperature
    scaled_logits = logits / temperatures.unsqueeze(-1).clamp(min=1e-8)

    # Compute probabilities
    probs = torch.softmax(scaled_logits, dim=-1)

    # Sample
    tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

    # Get logprob of sampled token
    logprobs = torch.log(torch.gather(probs, -1, tokens.unsqueeze(-1))).squeeze(-1)

    assert tokens.shape == (logits.size(0),)
    assert logprobs.shape == (logits.size(0),)

    return tokens, logprobs


# ═══════════════════════════════════════════════════
# BLOCK HASHING: Pure function
# ═══════════════════════════════════════════════════

def compute_block_hash(token_ids: tuple[int, ...], prefix_hash: int = -1) -> int:
    """Pure function: compute hash for KV cache block.

    Used by PagedKVCache to detect cache hits.
    """
    assert len(token_ids) > 0

    import xxhash
    h = xxhash.xxh64()
    if prefix_hash != -1:
        h.update(prefix_hash.to_bytes(8, "little"))
    h.update(bytes(token_ids))
    return h.intdigest()
```

---

## Stateful Classes (Resources + Lifecycle)

```python
from enum import Enum, auto

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# ═══════════════════════════════════════════════════
# SEQUENCE: Mutable state during generation
# ═══════════════════════════════════════════════════

@dataclass
class Sequence:
    """Per-sequence state. Mutable during generation.

    Why a class (not frozen dataclass)?
    - token_ids grows during generation
    - status changes (WAITING -> RUNNING -> FINISHED)
    - output_logprobs accumulates
    """
    seq_id: int
    token_ids: list[int]
    block_ids: list[int]

    num_prompt_tokens: int
    status: SequenceStatus

    # Config (frozen after creation)
    temperature: float
    max_tokens: int
    stop_token_ids: frozenset[int]

    # Accumulated output
    output_logprobs: list[float]

    def append_token(self, token_id: int, logprob: float) -> None:
        """Mutate: add generated token."""
        assert self.status == SequenceStatus.RUNNING
        self.token_ids.append(token_id)
        self.output_logprobs.append(logprob)


# ═══════════════════════════════════════════════════
# KV CACHE: Owns block allocation
# ═══════════════════════════════════════════════════

class KVCacheManager(Protocol):
    """Protocol for KV cache. Paged or Radix implementation."""

    def allocate(self, token_ids: list[int]) -> CacheAllocation:
        """Allocate blocks, return cached prefix length."""
        ...

    def deallocate(self, block_ids: list[int]) -> None:
        """Free blocks back to pool."""
        ...

    def num_free_blocks(self) -> int:
        """Available blocks for scheduling."""
        ...

    def flush(self) -> None:
        """Clear all cache (call after weight update)."""
        ...


class PagedKVCache:
    """vLLM-style block cache with hash lookup.

    Why a class?
    - Owns block pool (GPU memory)
    - Maintains hash_to_block mapping
    - Tracks ref_counts for sharing
    """

    def __init__(self, num_blocks: int, block_size: int):
        assert num_blocks > 0
        assert block_size > 0

        self.block_size = block_size
        self.blocks = [Block(i) for i in range(num_blocks)]
        self.hash_to_block: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def allocate(self, token_ids: list[int]) -> CacheAllocation:
        """Allocate blocks with prefix caching."""
        assert len(token_ids) > 0

        block_ids = []
        cached_length = 0
        prefix_hash = -1

        for i in range(0, len(token_ids), self.block_size):
            block_tokens = tuple(token_ids[i:i + self.block_size])

            # Only hash complete blocks
            if len(block_tokens) == self.block_size:
                h = compute_block_hash(block_tokens, prefix_hash)
                prefix_hash = h

                # Check cache hit
                if h in self.hash_to_block:
                    block_id = self.hash_to_block[h]
                    self.blocks[block_id].ref_count += 1
                    block_ids.append(block_id)
                    cached_length += self.block_size
                    continue

            # Cache miss: allocate new block
            assert len(self.free_block_ids) > 0, "OOM: no free blocks"
            block_id = self.free_block_ids.popleft()
            block = self.blocks[block_id]
            block.ref_count = 1

            if len(block_tokens) == self.block_size:
                block.hash = prefix_hash
                self.hash_to_block[prefix_hash] = block_id

            block_ids.append(block_id)

        return CacheAllocation(
            block_ids=tuple(block_ids),
            cached_length=cached_length,
        )

    def deallocate(self, block_ids: list[int]) -> None:
        """Free blocks, respecting ref counts."""
        for block_id in block_ids:
            block = self.blocks[block_id]
            assert block.ref_count > 0
            block.ref_count -= 1

            if block.ref_count == 0:
                if block.hash in self.hash_to_block:
                    del self.hash_to_block[block.hash]
                block.hash = -1
                self.free_block_ids.append(block_id)

    def num_free_blocks(self) -> int:
        return len(self.free_block_ids)

    def flush(self) -> None:
        """Clear all cache."""
        self.hash_to_block.clear()
        self.free_block_ids = deque(range(len(self.blocks)))
        for block in self.blocks:
            block.ref_count = 0
            block.hash = -1


@dataclass
class Block:
    """Single KV cache block."""
    block_id: int
    ref_count: int = 0
    hash: int = -1


# ═══════════════════════════════════════════════════
# WEIGHT SYNC: Owns NCCL group
# ═══════════════════════════════════════════════════

class WeightSyncManager:
    """Manages async weight updates.

    Why a class?
    - Owns NCCL process group (resource)
    - Tracks pending sync state
    - Needs cleanup
    """

    def __init__(self, model: nn.Module, process_group):
        assert model is not None

        self.model = model
        self.pg = process_group
        self.pending_sync: Future | None = None
        self.version = 0

    def start_sync(self, new_weights: dict[str, torch.Tensor]) -> None:
        """Begin async weight broadcast."""
        assert self.pending_sync is None, "sync already in progress"
        assert new_weights, "empty weights"

        # Launch NCCL broadcast in background
        self.pending_sync = self._broadcast_weights_async(new_weights)

    def maybe_finish_sync(self) -> bool:
        """Check if sync complete, apply if so."""
        if self.pending_sync is None:
            return False
        if not self.pending_sync.done():
            return False

        # Apply weights
        self.model.load_state_dict(self.pending_sync.result())
        self.pending_sync = None
        self.version += 1
        return True

    def sync_blocking(self, new_weights: dict[str, torch.Tensor]) -> None:
        """Blocking weight sync (Miles-style)."""
        assert new_weights, "empty weights"

        self.model.load_state_dict(new_weights)
        self.version += 1


# ═══════════════════════════════════════════════════
# ENGINE: Orchestrates everything
# ═══════════════════════════════════════════════════

class InferenceEngine:
    """Main inference engine.

    Why a class?
    - Owns GPU resources (model, KV cache)
    - Manages sequence lifecycle
    - Needs shutdown() for cleanup

    Pure functions do the work:
    - schedule() decides what to run
    - sample_with_logprobs() does sampling
    - model_forward() runs the model
    """

    def __init__(self, config: EngineConfig):
        assert config.block_size > 0
        assert config.max_batch_size > 0

        self.config = config
        self.model = self._load_model(config.model_path)
        self.cache = self._create_cache(config)
        self.weight_sync = WeightSyncManager(self.model, ...)

        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.seq_counter = 0

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams,
        num_samples_per_prompt: int = 1,
    ) -> list[TrainingSample]:
        """High-level: generate completions with logprobs."""
        assert len(prompts) > 0
        assert num_samples_per_prompt > 0

        # Add requests
        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                self.add_request(prompt, sampling_params)

        # Run until done
        results = []
        while self.has_pending():
            finished = self.step()
            results.extend(finished)

        assert len(results) == len(prompts) * num_samples_per_prompt
        return results

    def add_request(self, prompt_tokens: list[int], params: SamplingParams) -> int:
        """Mid-level: add single request."""
        assert len(prompt_tokens) > 0

        seq = Sequence(
            seq_id=self.seq_counter,
            token_ids=list(prompt_tokens),
            block_ids=[],
            num_prompt_tokens=len(prompt_tokens),
            status=SequenceStatus.WAITING,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stop_token_ids=params.stop_token_ids,
            output_logprobs=[],
        )
        self.seq_counter += 1
        self.waiting.append(seq)
        return seq.seq_id

    def step(self) -> list[TrainingSample]:
        """Mid-level: one scheduling + forward pass."""
        # Check weight sync
        self.weight_sync.maybe_finish_sync()

        # Schedule (pure function)
        sched_config = SchedulerConfig(
            max_batch_size=self.config.max_batch_size,
            max_tokens_per_batch=self.config.max_batch_size * 512,
            block_size=self.config.block_size,
        )
        sched_out = schedule(
            self.waiting,
            self.running,
            self.cache.num_free_blocks(),
            sched_config,
        )

        # ... forward pass, sampling, update state ...

        return []  # finished samples

    def has_pending(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def update_weights(self, state_dict: dict, blocking: bool = False) -> None:
        """Update model weights."""
        assert state_dict, "empty state_dict"

        if blocking:
            self.weight_sync.sync_blocking(state_dict)
        else:
            self.weight_sync.start_sync(state_dict)

    def get_weight_version(self) -> int:
        return self.weight_sync.version

    def flush_cache(self) -> None:
        """Clear KV cache (call after weight update)."""
        self.cache.flush()

    def shutdown(self) -> None:
        """Cleanup resources."""
        # Free GPU memory, close NCCL groups, etc.
        pass
```

---

## Protocol for RL Integration

Based on analysis of Miles (SLIME fork) and PipelineRL:

| Aspect | Miles | PipelineRL |
|--------|-------|------------|
| Weight Sync | Blocking (pause→NCCL→resume) | In-flight (NCCL during decode) |
| Communication | Ray actors + NCCL | HTTP handshake + NCCL |
| Version Tracking | `weight_versions` per sample | `model_version` per rollout |

```python
class InferenceEngine(Protocol):
    """Contract between rollouts and inference engine."""

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams,
        num_samples_per_prompt: int = 1,
    ) -> list[TrainingSample]: ...

    def update_weights(self, state_dict: dict, blocking: bool = False) -> None: ...
    def get_weight_version(self) -> int: ...
    def flush_cache(self) -> None: ...
    def shutdown(self) -> None: ...
```

---

## Implementation Plan

### Phase 1: Minimal Working Engine (~500 LOC)
- [x] Frozen dataclasses for config/output
- [x] Sequence mutable state
- [x] Simple scheduler (pure function)
- [x] Sampling with logprobs (pure function)
- [x] No KV cache (recompute everything)
- [ ] **FIX: Correctness bug** - greedy decode diverges from Transformers
  - Test: `rollouts/tests/test_inference_correctness.py`
  - 2/3 prompts match text, 1/3 diverges after "Paris."
  - All prompts have logprob diffs 0.18-0.91 (should be <0.01)
  - Possible causes: attention mask, forward pass vs generate(), numerical precision

### Phase 2: PagedAttention (~300 LOC)
- [ ] Block allocator class
- [ ] Hash-based cache lookup
- [ ] Integrate with scheduler

### Phase 3: RadixAttention (~400 LOC)
- [ ] Radix tree (prefix sharing)
- [ ] Swap in place of Paged

### Phase 4: Weight Sync (~200 LOC)
- [ ] WeightSyncManager class
- [ ] NCCL broadcast
- [ ] Version tracking

### Phase 5: Multi-GPU (~200 LOC)
- [ ] Tensor parallelism
- [ ] Distributed cache

### Phase 6: HTTP API (~300 LOC)
- [ ] OpenAI-compatible `/v1/completions`, `/v1/chat/completions`
- [ ] Async request handling
- [ ] Goal: vLLM/SGLang parity with cleaner code

---

## Non-Goals

- Speculative decoding
- Quantization/LoRA (can add later)
- Multi-modal

---

## Open Questions

1. **FlexAttention vs Flash-Attn?**
   - FlexAttention: PyTorch stdlib, no compile deps
   - Flash-Attn: Faster, but Triton/CUDA deps
   - Decision: Start with FlexAttention (boring, well-tested)

2. **Block size?**
   - Start with 16, make configurable

3. **Preemption strategy?**
   - Start with recompute (simpler than swap)

---

## Verified Components

### Sliding Window Attention (SWA)
- FlexAttention implementation with BlockMask
- Tested against reference explicit-mask implementation
- Max diff: 8.34e-07 (within fp tolerance)
- Full causal mode also verified (diff: 1.19e-06)

Key files:
- `rollouts/inference/attention/mask.py` - `create_sliding_window_causal_mask()`, `create_attention_mask()`
- `rollouts/inference/attention/flex_backend.py` - FlexAttentionBackend
- `rollouts/tools/functional_extractor/debug_swa.py` - Test script

---

## Running Tests

```bash
# Correctness test vs Transformers (provisions GPU)
uv run python rollouts/tests/test_inference_correctness.py --provision --keep-alive

# Reuse existing instance
uv run python rollouts/tests/test_inference_correctness.py --node-id runpod:abc123

# SWA test
uv run python -m rollouts.tools.functional_extractor.debug_swa
```
