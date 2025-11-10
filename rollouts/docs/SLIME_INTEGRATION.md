# SLIME Integration Guide

This document explains what we've added from SLIME and how to use it.

## What We Added

### 1. Filter Functions (`rollouts/training/filters.py`)

SLIME-style quality control for rollout generation.

**SLIME's default filter:**
```python
from rollouts.training import check_reward_nonzero_std

# Keep only if reward variance > 0
# Why: GRPO/PPO need variance to compute advantages
keep = check_reward_nonzero_std(samples)
```

**Other filters:**
```python
from rollouts.training import (
    check_min_reward,           # At least one good sample
    check_response_diversity,   # Avoid repetitive responses
    check_reasonable_length,    # Reasonable token count
    check_any_success,          # At least one completed
    check_quality_and_diversity, # Composite filter
)
```

### 2. Agent Integration (`rollouts/training/agent_integration.py`)

Bridge between agent framework and RL training.

**Based on:**
- `~/wafer_stuff/clicker/run_rollouts.py` (agent execution)
- `~/wafer_stuff/clicker/rollouts/training/sample_prep.py` (trajectory → sample)

**High-level API:**
```python
from rollouts.training import agent_rollout_to_sample
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.dtypes import Endpoint

endpoint = Endpoint(provider="sglang", model="Qwen/Qwen2.5-7B-Instruct")

sample = await agent_rollout_to_sample(
    prompt="What is 5 + 3?",
    environment_cls=CalculatorEnvironment,
    endpoint=endpoint,
    tokenizer=my_tokenizer,
    max_turns=10,
)

# sample has:
# - loss_mask: [0.0, 0.0, 1.0, 1.0, ...] (train on assistant, not tools)
# - tokens: tokenized conversation
# - response: full agent trajectory
```

**Batch API (for SLIME-style training):**
```python
from rollouts.training import generate_rollout_batch
from functools import partial

# Create generate_fn
generate_fn = partial(
    generate_rollout_batch,
    environment_cls=CalculatorEnvironment,
    endpoint=endpoint,
    tokenizer=tokenizer,
    max_turns=10,
)

# Use in RolloutConfig
config = RolloutConfig(
    batch_size=32,
    n_samples_per_prompt=8,
    over_sampling_factor=1.5,
    generate_fn=generate_fn,
    filter_fn=check_reward_nonzero_std,  # SLIME's filter!
)
```

## Key Features

### 1. Loss Mask (Multi-turn Agents)

The integration automatically builds loss masks for tool-using agents:

```python
# Example trajectory:
# User: "What is 5+3?"
# Assistant: "Let me calculate"  ← Train on this
# Tool: "8"                       ← Don't train on this
# Assistant: "The answer is 8"    ← Train on this

# loss_mask will be:
# [0, 0, ..., 1, 1, ..., 0, 0, ..., 1, 1, ...]
#  user      assistant   tool      assistant
```

### 2. Dynamic Sampling Filters

Filters decide whether to keep or discard sample groups:

```python
def my_filter(samples: list[Sample]) -> bool:
    """
    Args:
        samples: Group (same prompt, different responses)

    Returns:
        True = keep, False = discard
    """
    # Example: Keep if variance > 0
    rewards = [s.reward for s in samples]
    return torch.tensor(rewards).std() > 0.0
```

**Why filtering?**
- GRPO/PPO need reward variance to learn
- If all samples have same reward (std=0), advantages=0, no learning signal
- Better to discard and generate new prompt

### 3. SLIME-Compatible Sample Type

We already have a compatible `Sample` type in `rollouts/training/types.py`:

```python
@dataclass
class Sample:
    prompt: str | list[dict[str, str]]
    response: str
    tokens: list[int]
    loss_mask: list[float]  # Per-token weights!
    reward: float
    metadata: dict[str, Any]
    group_index: Optional[int]
    status: Status  # PENDING/COMPLETED/ABORTED
    rollout_log_probs: Optional[list[float]]
```

## Usage Examples

### Example 1: Calculator Agent with RL

```python
from functools import partial
from rollouts.training import (
    generate_rollout_batch,
    check_reward_nonzero_std,
    RolloutConfig,
)
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.dtypes import Endpoint

# Setup
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
endpoint = Endpoint(provider="sglang", model="Qwen/Qwen2.5-7B-Instruct")

# Create generate_fn (bound to config)
generate_fn = partial(
    generate_rollout_batch,
    environment_cls=CalculatorEnvironment,
    endpoint=endpoint,
    tokenizer=tokenizer,
    max_turns=10,
)

# Define reward_fn
def calc_reward(sample: Sample) -> float:
    """Reward: 1.0 if correct, 0.0 otherwise."""
    ground_truth = sample.metadata.get("ground_truth")
    return 1.0 if ground_truth in sample.response else 0.0

# Create config (SLIME-style!)
config = RolloutConfig(
    batch_size=32,
    n_samples_per_prompt=8,
    over_sampling_factor=1.5,
    generate_fn=generate_fn,
    reward_fn=calc_reward,
    filter_fn=check_reward_nonzero_std,
)

# Use with AsyncRolloutManager
manager = AsyncRolloutManager(config, data_buffer)
samples = await manager.generate_batch()
```

### Example 2: Custom Filter

```python
from rollouts.training import Sample

def check_math_quality(samples: list[Sample]) -> bool:
    """Custom filter for math problems."""
    # Require: variance AND at least one correct answer
    rewards = [s.reward for s in samples]
    std = torch.tensor(rewards).std()
    max_reward = max(rewards)

    return std > 0.0 and max_reward >= 0.8

config = RolloutConfig(
    batch_size=32,
    generate_fn=my_generate_fn,
    filter_fn=check_math_quality,  # Custom filter!
)
```

## Comparison to SLIME

| Feature | SLIME | Rollouts | Status |
|---------|-------|----------|--------|
| Sample type | ✅ | ✅ | Same structure |
| Async rollout manager | ✅ | ✅ | Already had it |
| Over-sampling | ✅ | ✅ | Already had it |
| Dynamic filters | ✅ | ✅ | **Added** |
| Agent integration | ✅ (custom_generate_fn) | ✅ | **Added** |
| Loss mask | ✅ | ✅ | **Added** |
| Metadata passing | ✅ | ✅ | Already had it |
| Buffer filters | ✅ | ❌ | TODO (Phase 2) |
| Multimodal | ✅ | ❌ | TODO (Phase 3) |

## What's Still Missing

From our comparison doc (`SLIME_vs_ROLLOUTS_COMPARISON.md`):

**Phase 2 (Multi-turn):**
- Tool calling examples (we have the code, just need examples)
- Multi-turn continuation API

**Phase 3 (Infrastructure):**
- Buffer filters (custom strategies for partial rollout selection)
- Multimodal support (images in prompts)
- Reward model hub (remote, batched, rule-based)
- Router middleware (RadixTree caching)

**Phase 4 (Optional):**
- Ray-based distributed orchestration

## Files Created

```
rollouts/rollouts/training/
├── filters.py              ← SLIME-style filters (NEW)
├── agent_integration.py    ← Agent → Sample bridge (NEW)
└── __init__.py            ← Updated exports
```

## References

- SLIME repo: `references/slime/`
- Clicker pattern: `~/wafer_stuff/clicker/run_rollouts.py`
- Comparison doc: `SLIME_vs_ROLLOUTS_COMPARISON.md`
