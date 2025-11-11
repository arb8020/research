# SLIME vs Rollouts: Gap Analysis & Implementation Roadmap

**Date:** Nov 9, 2025
**Purpose:** Identify what rollouts is missing compared to THUDM SLIME and what needs to be implemented

---

## Executive Summary

Your **rollouts** implementation has the core architecture right and already includes SLIME's async rollout manager with over-sampling. The main gaps are:

### ✅ What You Already Have
- **AsyncRolloutManager** with dynamic over-sampling (SLIME's D4)
- Partial rollout caching
- Data buffer with epoch management
- Protocol-based backend abstraction
- Weight synchronization to inference servers
- Basic training loop orchestration

### ❌ What You're Missing (High Priority)
1. **Dynamic sampling filters** - Quality control for over-sampled rollouts
2. **Buffer filters** - Custom strategies for partial rollout selection
3. **Multimodal support** - Image handling in rollouts
4. **Custom generation functions** - User-defined rollout logic for agents
5. **Custom reward functions** - Pluggable reward computation
6. **Ray-based distributed orchestration** - Multi-node coordination
7. **Advanced sampling features** - Deterministic seeds, multi-turn continuation
8. **Router middleware** - RadixTree caching for prefix sharing

### ⚠️ What You're Missing (Medium Priority)
9. Reward model hub (rule-based, remote, batched)
10. Evaluation pipeline integration
11. Checkpoint management for data state
12. Memory management (offload/onload)
13. Speculative decoding tracking

---

## Detailed Component Comparison

### 1. Rollout Generation System

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Async generation** | ✅ asyncio + Ray | ✅ trio | Minor: Different concurrency lib |
| **Over-sampling** | ✅ configurable factor | ✅ configurable factor | ✅ SAME |
| **Partial caching** | ✅ buffer-based | ✅ partial_samples list | ✅ SAME |
| **Dynamic filters** | ✅ pluggable filter_fn | ❌ Missing | **HIGH PRIORITY** |
| **Buffer filters** | ✅ pop_first, custom | ❌ No buffer filter strategy | **HIGH PRIORITY** |
| **Multimodal prompts** | ✅ Image encoding | ❌ Text only | **MEDIUM** |
| **Custom generate_fn** | ✅ Full async support | ⚠️ Basic support | **HIGH PRIORITY** |
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
| **Metadata tracking** | ✅ Custom metadata_key | ❌ No metadata support | **HIGH PRIORITY** |

**Key difference:** SLIME separates buffer logic into `RolloutDataSourceWithBuffer`, you have it inline in `AsyncRolloutManager`.

**Files to reference:**
- SLIME: `references/slime/slime/ray/rollout_data_source.py:122-177` (RolloutDataSourceWithBuffer)
- SLIME: `references/slime/slime/ray/rollout_data_source.py:179-183` (pop_first filter)
- Yours: `rollouts/rollouts/training/datasets/data_buffer.py`

---

### 3. Custom Functions & Extensibility

| Feature | SLIME | Rollouts | Gap |
|---------|-------|----------|-----|
| **Custom generate_fn** | ✅ Full async + metadata | ⚠️ Basic callable | **HIGH PRIORITY** |
| **Custom reward_fn** | ✅ Pluggable via --custom-rm-path | ❌ Hardcoded in loop | **HIGH PRIORITY** |
| **Dynamic filter_fn** | ✅ Per-group filtering | ❌ No filter support | **HIGH PRIORITY** |
| **Buffer filter_fn** | ✅ Custom buffer selection | ❌ No buffer filter | **HIGH PRIORITY** |
| **Metadata passing** | ✅ sample.metadata dict | ❌ No metadata field | **HIGH PRIORITY** |

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
| **Custom user rewards** | ✅ Pluggable | ❌ Inline in loop | **HIGH PRIORITY** |

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
| **Loss masking** | ✅ sample.loss_mask per-token | ⚠️ Basic support | **MEDIUM** |
| **Metadata passing** | ✅ session_id, tool_code, etc. | ❌ No metadata | **HIGH PRIORITY** |

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
| **Metadata field** | ✅ dict for custom data | ❌ No metadata | **HIGH PRIORITY** |
| **Loss mask** | ✅ per-token list[int] | ⚠️ Basic support | **MEDIUM** |
| **Rollout log probs** | ✅ for off-policy | ⚠️ Not stored | **MEDIUM** |
| **Status enum** | ✅ PENDING/COMPLETED/ABORTED | ❌ No status | **MEDIUM** |
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

**Your Sample type:**
```python
@dataclass
class Sample:
    prompt_str: str
    response_str: str
    reward: float
    # Missing: metadata, loss_mask, status, etc.
```

**Files to reference:**
- SLIME: `references/slime/slime/utils/types.py` (Sample class)
- Yours: `rollouts/rollouts/training/types.py:20-35` (Sample class)

---

## Implementation Roadmap

### Phase 1: Core Extensibility (HIGH PRIORITY)

**Goal:** Enable custom functions and metadata passing

1. **Add metadata field to Sample**
   - Add `metadata: dict[str, Any]` to Sample dataclass
   - Update DataBuffer to accept `metadata_key` parameter
   - Pass metadata through rollout pipeline

2. **Implement dynamic sampling filters**
   - Add `filter_fn` to RolloutConfig
   - Apply filter in AsyncRolloutManager after generation
   - Return `DynamicFilterOutput(keep: bool, reason: str)`
   - Example: `check_reward_nonzero_std` filter

3. **Implement buffer filters**
   - Move buffer from AsyncRolloutManager to DataBuffer
   - Add `buffer_filter_fn` to DataBuffer
   - Implement `pop_first` default filter
   - Support custom filter via callable

4. **Support custom reward functions**
   - Make reward_fn pluggable in training loop
   - Support both sync and async reward functions
   - Add reward model hub (remote, rule-based)

5. **Enhance custom generate_fn support**
   - Allow async generate_fn
   - Pass metadata to generate_fn
   - Support multi-turn continuation API

**Files to modify:**
- `rollouts/rollouts/training/types.py` (Sample, RolloutConfig)
- `rollouts/rollouts/training/datasets/data_buffer.py` (buffer filter)
- `rollouts/rollouts/training/rollout_gen/async_rollout_manager.py` (dynamic filter)
- `rollouts/rollouts/training/loops/rl_loop.py` (reward_fn)

---

### Phase 2: Multi-Turn & Agentic (HIGH PRIORITY)

**Goal:** Support tool-calling agents and multi-turn interactions

1. **Add loss_mask to Sample**
   - Add `loss_mask: list[int]` field
   - Update batch conversion to use loss_mask
   - Default to all 1s for backward compat

2. **Add continuation API**
   - Support appending to sample.response
   - Track response_length separately from total tokens
   - Handle tool outputs in conversation

3. **Add tool calling examples**
   - Implement Search-R1 style agent
   - Example custom generate_fn with tools
   - Documentation for agent training

**Files to create:**
- `rollouts/rollouts/environments/tool_calling.py`
- `examples/search_agent/custom_generate.py`
- `docs/MULTI_TURN_GUIDE.md`

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

**What to implement first (in order):**

### 1. Add metadata to Sample ⭐ (30 min)
```python
@dataclass
class Sample:
    prompt_str: str
    response_str: str
    reward: float
    metadata: dict[str, Any] = field(default_factory=dict)  # NEW
    loss_mask: list[int] | None = None  # NEW
```

### 2. Add dynamic filter to AsyncRolloutManager ⭐ (1 hour)
```python
@dataclass
class RolloutConfig:
    batch_size: int
    generate_fn: Callable
    filter_fn: Callable[[list[Sample]], bool] | None = None  # NEW
    over_sampling_factor: float = 1.5
```

### 3. Implement buffer filter in DataBuffer ⭐ (2 hours)
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

### 4. Create filter hub ⭐ (2 hours)
```python
# rollouts/rollouts/filters/dynamic_filters.py
def check_reward_nonzero_std(samples: list[Sample]) -> bool:
    rewards = [s.reward for s in samples]
    return torch.tensor(rewards).std() > 0.0

# rollouts/rollouts/filters/buffer_filters.py
def pop_first(buffer: list[Sample], n: int) -> list[Sample]:
    result = buffer[:n]
    del buffer[:n]
    return result
```

### 5. Make reward_fn pluggable ⭐ (1 hour)
Move reward computation out of training loop into configurable function.

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

**Missing pieces (prioritized):**

1. **Metadata & extensibility** (HIGH) - Add metadata field, filter functions
2. **Multi-turn support** (HIGH) - loss_mask, continuation API
3. **Reward hub** (MEDIUM) - Pluggable reward functions
4. **Infrastructure** (MEDIUM) - Checkpointing, multimodal
5. **Distributed** (OPTIONAL) - Ray integration for multi-node

**Start here:**
1. Add metadata to Sample (30 min)
2. Add filter_fn to RolloutConfig (1 hour)
3. Move buffer to DataBuffer with filter strategy (2 hours)
4. Create filter hub (2 hours)
5. Make reward_fn pluggable (1 hour)

**Total for Phase 1:** ~6-8 hours of focused work

Good luck! Your architecture is already solid - just need to add SLIME's extensibility features.
