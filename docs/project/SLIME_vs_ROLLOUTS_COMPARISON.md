# SLIME vs Rollouts: Gap Analysis & Implementation Roadmap

**Date:** Nov 9, 2025 (Updated: 2025)
**Purpose:** Identify what rollouts is missing compared to THUDM SLIME and what needs to be implemented

---

## Executive Summary

Your **rollouts** implementation has the core architecture right and already includes SLIME's async rollout manager with over-sampling. Many gaps have been filled since initial analysis.

### ✅ What You Already Have
- **AsyncRolloutManager** with dynamic over-sampling (SLIME's D4)
- Partial rollout caching
- Data buffer with epoch management
- Protocol-based backend abstraction
- Weight synchronization to inference servers
- Basic training loop orchestration
- **✅ Config system** - Full config module with protocols (`rollouts/rollouts/config/`)
- **✅ Metadata support** - `Sample.metadata: dict[str, Any]` field
- **✅ Loss mask** - `Sample.loss_mask: list[float]` per-token weights
- **✅ Dynamic filters** - `RolloutConfig.filter_fn` for quality control
- **✅ Custom reward functions** - `RolloutConfig.reward_fn` pluggable
- **✅ Custom generate functions** - `RolloutConfig.generate_fn` support
- **✅ Status enum** - `Sample.Status` (PENDING/COMPLETED/TRUNCATED/ABORTED)

### ❌ What You're Missing (High Priority)
1. **Buffer filters** - Custom strategies for partial rollout selection (separate from dynamic filters)
2. **Multimodal support** - Image handling in rollouts
3. **Multi-turn continuation API** - Append to response, track response_length separately
4. **Ray-based distributed orchestration** - Multi-node coordination

### ⚠️ What You're Missing (Medium Priority)
5. Reward model hub (rule-based, remote, batched)
6. Router middleware - RadixTree caching for prefix sharing
7. Checkpoint management for data state (DataBuffer/AsyncRolloutManager state)
8. Deterministic seeds - Per-sample seed control
9. Memory management (offload/onload)
10. Speculative decoding tracking

---

## Detailed Component Comparison

### 1. Rollout Generation System

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Async generation** | ✅ asyncio + Ray | ✅ trio | Minor: Different concurrency lib |
| **Over-sampling** | ✅ configurable factor | ✅ configurable factor | ✅ SAME |
| **Partial caching** | ✅ buffer-based | ✅ partial_samples list | ✅ SAME |
| **Dynamic filters** | ✅ pluggable filter_fn | ✅ `RolloutConfig.filter_fn` | ✅ **COMPLETE** |
| **Buffer filters** | ✅ pop_first, custom | ❌ No buffer filter strategy | **HIGH PRIORITY** |
| **Multimodal prompts** | ✅ Image encoding | ❌ Text only | **MEDIUM** |
| **Custom generate_fn** | ✅ Full async support | ✅ `RolloutConfig.generate_fn` | ✅ **COMPLETE** |
| **Multi-turn continuation** | ✅ Append to response | ❌ No continuation API | **MEDIUM** |
| **Deterministic seeds** | ✅ Per-sample seeds | ❌ No seed control | **LOW** |

**Files to reference:**
- SLIME: `references/slime/slime/rollout/sglang_rollout.py:73-86` (submit_generate_tasks)
- SLIME: `references/slime/slime/rollout/filter_hub/dynamic_sampling_filters.py` (filters)
- Yours: `rollouts/rollouts/training/rollout_gen/async_rollout_manager.py:75-150` (generate_batch)

---

### 2. Data Buffer Architecture

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Epoch tracking** | ✅ epoch_id | ✅ epoch_id | ✅ SAME |
| **Shuffling** | ✅ per-epoch shuffle | ✅ per-epoch shuffle | ✅ SAME |
| **Partial rollout buffer** | ✅ RolloutDataSourceWithBuffer | ⚠️ In AsyncRolloutManager | **REFACTOR** |
| **Buffer filter strategy** | ✅ Pluggable via buffer_filter_path | ❌ FIFO only | **HIGH PRIORITY** |
| **Checkpoint save/load** | ✅ Full state dict | ⚠️ Basic support | **MEDIUM** |
| **Metadata tracking** | ✅ Custom metadata_key | ✅ `Sample.metadata: dict` | ✅ **COMPLETE** |

**Key difference:** SLIME separates buffer logic into `RolloutDataSourceWithBuffer`, you have it inline in `AsyncRolloutManager`.

**Files to reference:**
- SLIME: `references/slime/slime/ray/rollout_data_source.py:122-177` (RolloutDataSourceWithBuffer)
- SLIME: `references/slime/slime/ray/rollout_data_source.py:179-183` (pop_first filter)
- Yours: `rollouts/rollouts/training/datasets/data_buffer.py`

---

### 3. Custom Functions & Extensibility

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Custom generate_fn** | ✅ Full async + metadata | ✅ `RolloutConfig.generate_fn` | ✅ **COMPLETE** |
| **Custom reward_fn** | ✅ Pluggable via --custom-rm-path | ✅ `RolloutConfig.reward_fn` | ✅ **COMPLETE** |
| **Dynamic filter_fn** | ✅ Per-group filtering | ✅ `RolloutConfig.filter_fn` | ✅ **COMPLETE** |
| **Buffer filter_fn** | ✅ Custom buffer selection | ❌ No buffer filter | **HIGH PRIORITY** |
| **Metadata passing** | ✅ sample.metadata dict | ✅ `Sample.metadata: dict` | ✅ **COMPLETE** |

**SLIME's killer feature:** Users can specify custom functions for:
1. `--custom-generate-function-path your_module.generate` - Multi-turn agents
2. `--custom-rm-path your_module.reward` - Custom reward logic
3. `--dynamic-sampling-filter-path slime.rollout.filter_hub...` - Quality filters
4. `--buffer-filter-path your_module.buffer_filter` - Buffer selection strategy

**Files to reference:**
- SLIME: `references/slime/docs/en/get_started/quick_start.md:413-543` (Custom functions guide)
- SLIME: `references/slime/slime/rollout/filter_hub/` (Filter hub)
- Yours: `rollouts/rollouts/training/types.py:115-125` (RolloutConfig)

---

### 4. Reward Computation

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Rule-based rewards** | ✅ Hub of implementations | ❌ No built-in rewards | **MEDIUM** |
| **Remote HTTP rewards** | ✅ async_rm() | ❌ No remote support | **MEDIUM** |
| **Batched rewards** | ✅ batched_async_rm() | ❌ No batching | **MEDIUM** |
| **Custom user rewards** | ✅ Pluggable | ✅ `RolloutConfig.reward_fn` | ✅ **COMPLETE** |

**SLIME reward types:**
- `deepscaler` - DeepSeek-style scaling rewards
- `remote` - HTTP POST to external server
- `rule_based` - Math/GPQA/Code correctness
- `custom` - User-defined function

**Files to reference:**
- SLIME: `references/slime/slime/rollout/rm_hub/` (Reward model hub)
- Yours: Currently reward_fn is passed to AsyncRolloutManager.generate_batch()

---

### 5. Multi-Turn & Agentic Support

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Multi-turn conversation** | ✅ Append to response | ❌ No continuation API | **HIGH PRIORITY** |
| **Tool calling** | ✅ via custom generate_fn | ❌ No tool support | **HIGH PRIORITY** |
| **Loss masking** | ✅ sample.loss_mask per-token | ✅ `Sample.loss_mask: list[float]` | ✅ **COMPLETE** |
| **Metadata passing** | ✅ session_id, tool_code, etc. | ✅ `Sample.metadata: dict` | ✅ **COMPLETE** |

**SLIME's approach:**
1. Dataset includes `metadata_key` column (JSON string)
2. Custom `generate_fn` accesses `sample.metadata['session_id']`
3. Generate action → Execute tool → Append observation
4. Set `loss_mask=1` for model tokens, `loss_mask=0` for tool outputs

**Files to reference:**
- SLIME: `references/slime/docs/en/get_started/quick_start.md:413-543` (Multiturn adaptation)
- SLIME: `references/slime/examples/search-r1/` (Search-R1 implementation)
- Yours: No equivalent yet

---

### 6. Distributed Orchestration

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Ray-based coordination** | ✅ Ray actors | ❌ No Ray integration | **MEDIUM** |
| **Multi-node training** | ✅ Ray cluster | ❌ No multi-node | **MEDIUM** |
| **Placement groups** | ✅ GPU affinity | ❌ No placement | **LOW** |
| **Fault tolerance** | ✅ Engine restart | ❌ No fault handling | **LOW** |

**SLIME's architecture:**
- Training backend is Ray actor
- Rollout manager is Ray actor
- SGLang engines are Ray actors
- All communicate via Ray object store

**Files to reference:**
- SLIME: `references/slime/slime/ray/` (Ray actors)
- SLIME: `references/slime/slime/ray/placement_group.py` (GPU placement)
- Yours: No Ray integration

---

### 7. Router & Caching

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **SGLang router** | ✅ Built-in | ⚠️ Direct HTTP | ✅ SAME (different interface) |
| **SLIME router** | ✅ Custom FastAPI | ❌ No custom router | **LOW** |
| **RadixTree middleware** | ✅ Prefix caching | ❌ No caching | **MEDIUM** |
| **Round-robin LB** | ✅ Request-based | ❌ No LB | **LOW** |

**SLIME's RadixTree:**
- Caches prompt prefixes (e.g., few-shot examples)
- Retrieves cached tokens + logprobs
- Reduces redundant computation for multi-turn

**Files to reference:**
- SLIME: `references/slime/slime/router/router.py` (Custom router)
- SLIME: `references/slime/slime/router/middleware_hub/radix_tree_middleware.py`
- Yours: `rollouts/rollouts/training/weight_sync.py:145-177` (Direct SGLang HTTP)

---

### 8. Sample Data Structure

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Basic fields** | ✅ prompt, response, tokens | ✅ Same | ✅ SAME |
| **Metadata field** | ✅ dict for custom data | ✅ `Sample.metadata: dict` | ✅ **COMPLETE** |
| **Loss mask** | ✅ per-token list[int] | ✅ `Sample.loss_mask: list[float]` | ✅ **COMPLETE** |
| **Rollout log probs** | ✅ for off-policy | ✅ `Sample.rollout_log_probs: list[float]` | ✅ **COMPLETE** |
| **Status enum** | ✅ PENDING/COMPLETED/ABORTED | ✅ `Sample.Status` enum | ✅ **COMPLETE** |
| **Spec info** | ✅ Speculative decoding | ❌ No spec tracking | **LOW** |

**SLIME Sample type:**
```python
@dataclass
class Sample:
    group_index: int
    prompt: Union[str, list]  # Multimodal
    tokens: list[int]
    response: str
    response_length: int
    reward: Union[float, dict]
    loss_mask: list[int]  # Per-token
    rollout_log_probs: list[float]
    metadata: dict  # Custom data!
    status: Status  # State tracking
    spec_info: SpecInfo  # Spec decoding
```

**Your Sample type (Updated):**
```python
@dataclass
class Sample:
    prompt: str | list[dict[str, str]]
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[float] = field(default_factory=list)  # ✅ Added
    reward: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)  # ✅ Added
    rollout_log_probs: Optional[list[float]] = None  # ✅ Added
    status: Status = Status.PENDING  # ✅ Added
    group_index: Optional[int] = None
    index: Optional[int] = None
```

**Files to reference:**
- SLIME: `references/slime/slime/utils/types.py` (Sample class)
- Yours: `rollouts/rollouts/training/types.py:20-35` (Sample class)

---

## Implementation Roadmap

### Phase 1: Core Extensibility ✅ **COMPLETE**

**Goal:** Enable custom functions and metadata passing

1. ✅ **Add metadata field to Sample** - `Sample.metadata: dict[str, Any]`
2. ✅ **Implement dynamic sampling filters** - `RolloutConfig.filter_fn`
3. ⚠️ **Implement buffer filters** - Still missing (separate from dynamic filters)
4. ✅ **Support custom reward functions** - `RolloutConfig.reward_fn`
5. ✅ **Enhance custom generate_fn support** - `RolloutConfig.generate_fn` with metadata

**Status:** Most core extensibility features are complete. Buffer filters remain.

---

### Phase 2: Multi-Turn & Agentic (PARTIALLY COMPLETE)

**Goal:** Support tool-calling agents and multi-turn interactions

1. ✅ **Add loss_mask to Sample** - `Sample.loss_mask: list[float]` per-token weights
2. ❌ **Add continuation API** - Still missing (append to response, track response_length)
3. ⚠️ **Tool calling support** - Basic support exists, but multi-turn continuation API missing

**Status:** Loss masking complete. Multi-turn continuation API still needed.

---

### Phase 3: Infrastructure (MEDIUM PRIORITY)

**Goal:** Production-grade features

1. **Multimodal support**
   - Support list[dict] for prompts (text + images)
   - Image encoding utilities
   - Update generate_fn to handle multimodal

2. **Checkpoint management**
   - Save/load DataBuffer state dict
   - Save/load AsyncRolloutManager state
   - Versioned checkpoints

3. **Reward model hub**
   - Remote HTTP rewards (async_rm)
   - Batched async rewards
   - Rule-based rewards (math, code)

4. **Router middleware**
   - Optional RadixTree caching
   - Prefix cache integration
   - Load balancing strategies

**Files to create:**
- `rollouts/rollouts/reward_hub/` (reward models)
- `rollouts/rollouts/router/` (optional router)

---

### Phase 4: Distributed (OPTIONAL)

**Goal:** Multi-node training support

1. **Ray integration**
   - Optional Ray backend for orchestration
   - Ray actors for training/rollout
   - Placement groups for GPU affinity

2. **Fault tolerance**
   - Engine restart on failure
   - Checkpoint recovery
   - Partial rollout collection on abort

**Files to create:**
- `rollouts/rollouts/distributed/` (Ray actors)

---

## Immediate Next Steps

**What to implement next (in order):**

### 1. ✅ Add metadata to Sample - **COMPLETE**
- `Sample.metadata: dict[str, Any]` exists

### 2. ✅ Add dynamic filter to AsyncRolloutManager - **COMPLETE**
- `RolloutConfig.filter_fn` exists and is used in `AsyncRolloutManager`

### 3. ❌ Implement buffer filter in DataBuffer - **STILL NEEDED**
```python
class DataBuffer:
    def __init__(self, ..., buffer_filter_fn=None):
        self.buffer: list[Sample] = []
        self.buffer_filter_fn = buffer_filter_fn or pop_first

    def get_prompts(self, n):
        # Try buffer first with filter
        from_buffer = self.buffer_filter_fn(self.buffer, n)
        # Then fresh data
        ...
```

### 4. ✅ Filter functions exist - **COMPLETE**
- `rollouts/rollouts/training/filters.py` contains filter implementations

### 5. ✅ Make reward_fn pluggable - **COMPLETE**
- `RolloutConfig.reward_fn` exists and is used in training loops

---

## Code Examples from SLIME

### Example 1: Dynamic Sampling Filter
**File:** `references/slime/slime/rollout/filter_hub/dynamic_sampling_filters.py:9-15`

```python
def check_reward_nonzero_std(args, samples: list[Sample], **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
```

### Example 2: Buffer Filter
**File:** `references/slime/slime/ray/rollout_data_source.py:179-183`

```python
def pop_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    num_to_pop = min(len(buffer), num_samples)
    samples = buffer[:num_to_pop]
    del buffer[:num_to_pop]
    return samples
```

### Example 3: Custom Generate Function (Multi-turn)
**File:** `references/slime/docs/en/get_started/quick_start.md:492-521`

```python
async def generate(args, sample: Sample, sampling_params) -> Sample:
    prompt, full_response, loss_masks = sample.prompt, "", []

    for _ in range(max_turns):
        # 1. Model generates action
        model_output = await call_sglang(prompt + full_response, ...)
        loss_masks += [1] * len(model_tokens)  # loss_mask = 1
        full_response += model_output

        # 2. Parse and execute action
        action, content = parse_action(model_output)
        if action == "search":
            # 3. Get observation from tool
            tool_output = await google_search(content)
            loss_masks += [0] * len(tool_tokens)  # loss_mask = 0
            full_response += tool_output
        elif action == "answer":
            break

    sample.response = full_response
    sample.loss_mask = loss_masks
    return sample
```

---

## Summary

**You have the foundation right:** AsyncRolloutManager with over-sampling is SLIME's core innovation.

**Progress Update (2025):**

✅ **Completed:**
1. **Metadata & extensibility** - `Sample.metadata`, `RolloutConfig.filter_fn`, `RolloutConfig.reward_fn`, `RolloutConfig.generate_fn`
2. **Loss masking** - `Sample.loss_mask: list[float]` per-token weights
3. **Status tracking** - `Sample.Status` enum
4. **Config system** - Full config module with protocols
5. **Logprobs** - `Sample.rollout_log_probs` for off-policy

❌ **Still Missing (prioritized):**
1. **Buffer filters** (HIGH) - Separate buffer filter strategy in DataBuffer
2. **Multi-turn continuation API** (HIGH) - Append to response, track response_length
3. **Reward model hub** (MEDIUM) - Rule-based, remote, batched rewards
4. **Multimodal support** (MEDIUM) - Image handling
5. **Infrastructure** (MEDIUM) - DataBuffer/AsyncRolloutManager checkpointing
6. **Distributed** (OPTIONAL) - Ray integration for multi-node

**Next Steps:**
1. Implement buffer filters in DataBuffer (2 hours)
2. Add multi-turn continuation API (2-3 hours)
3. Create reward model hub (3-4 hours)

**Total remaining:** ~7-9 hours of focused work

Great progress! Most core extensibility features are complete. Focus areas: buffer filters and multi-turn continuation.
