# Class Audit: What Should Be Refactored?

**Date**: 2025-11-09
**Goal**: Identify unnecessary stateful classes before building more on top of them

## Audit Methodology

For each class, ask:
1. **What mutable state does it manage?**
2. **Is this state that MUST persist across calls?**
3. **Or is it just a bag of methods / config wrapper?**

If #3, refactor to functions + dataclasses.

---

## 1. Training Components

### ✅ KEEP: Legitimate Stateful Classes

| Class | Mutable State | Justification | Verdict |
|-------|---------------|---------------|---------|
| `DataBuffer` | `epoch_id`, `sample_offset`, `_rng` | Tracks iteration through dataset | ✅ **KEEP** - Legitimate iterator state |
| `AsyncRolloutManager` | `partial_samples`, `_step_count`, `_abort_requested` | Caches incomplete rollouts, handles async coordination | ✅ **KEEP** - Legitimate async state management |
| `PyTorchTrainingBackend` | `model`, `optimizer`, `weight_version`, `current_step`, `_poisoned` | Training state, PyTorch is inherently stateful | ✅ **KEEP** - Legitimate training state |

### ⚠️ REFACTOR: Unnecessary Stateful Class

| Class | Current State | Issue | Refactor To |
|-------|---------------|-------|-------------|
| `RolloutManager` | `data_buffer`, `config`, `_step_count` | Only wraps DataBuffer + pure functions | ⚠️ **REFACTOR** to pure functions |

**Analysis of `RolloutManager`:**

```python
# Current (unnecessary class):
class RolloutManager:
    def __init__(self, data_buffer, config, **kwargs):
        self.data_buffer = data_buffer  # Dependency
        self.config = config            # Immutable config
        self.rollout_kwargs = kwargs    # Immutable config
        self._step_count = 0            # Just a counter!

    def __next__(self):
        prompts = self.data_buffer.get_prompts(self.config.batch_size)
        samples = self.config.generate_fn(prompts, **self.rollout_kwargs)
        samples = _apply_sample_transforms(samples, self.config)  # Pure!
        batch = convert_to_batch(samples, ...)  # Pure!
        self._step_count += 1  # Increment counter
        return batch
```

**Problems:**
- `_step_count` is just a loop counter, not real state
- `data_buffer` and `config` are just dependencies
- All the work is done by pure functions (`_apply_sample_transforms`, `convert_to_batch`)
- The class is just a wrapper around `for` loop iteration

**Refactor to:**

```python
# Better: Pure function
async def generate_rollout_batches(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    num_batches: int,
    **rollout_kwargs,
) -> list[RolloutBatch]:
    """Generate N rollout batches (pure orchestration).

    Args:
        data_buffer: DataBuffer (has state, we just use it)
        config: RolloutConfig (immutable)
        num_batches: How many batches to generate
        **rollout_kwargs: Kwargs for generate_fn

    Returns:
        List of RolloutBatch objects
    """
    batches = []

    for step in range(num_batches):
        # Get prompts (data_buffer manages its own state)
        prompts = data_buffer.get_prompts(config.batch_size)

        # Generate samples
        samples = config.generate_fn(prompts, **rollout_kwargs)

        # Apply transforms (pure function)
        samples = apply_sample_transforms(samples, config)

        # Convert to batch (pure function)
        batch = convert_to_batch(
            samples,
            epoch_id=data_buffer.epoch_id,
            step_id=step,
        )

        batches.append(batch)

    return batches


# Or even simpler: Generator function
def rollout_batch_generator(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    **rollout_kwargs,
) -> Iterator[RolloutBatch]:
    """Generate rollout batches indefinitely (generator).

    Yields:
        RolloutBatch objects
    """
    step = 0
    while True:
        prompts = data_buffer.get_prompts(config.batch_size)
        samples = config.generate_fn(prompts, **rollout_kwargs)
        samples = apply_sample_transforms(samples, config)
        batch = convert_to_batch(samples, data_buffer.epoch_id, step)
        yield batch
        step += 1


# Usage:
batches = rollout_batch_generator(data_buffer, config)
for batch in batches:
    # Train on batch
    ...
```

**Benefits of refactor:**
- ✅ No unnecessary class
- ✅ Clear data flow (generator pattern)
- ✅ Easy to test (pure functions + simple generator)
- ✅ No hidden state (step counter is local variable)

**Recommendation**: ⚠️ **Deprecate `RolloutManager`, use generator function**

Note: `AsyncRolloutManager` is DIFFERENT - it has legitimate state (`partial_samples` cache for async abort/resume). Keep that one.

---

## 2. Data Types (`dtypes.py`)

### ✅ KEEP: All Are Pure Data

All classes in `dtypes.py` are either:
- `@dataclass(frozen=True)` - immutable value objects
- `Enum` - immutable constants
- `Protocol` - just type hints

**Examples:**
- `Message`, `ToolCall`, `Usage`, `ChatCompletion` - All frozen dataclasses ✅
- `StopReason` - Enum ✅
- `JsonSerializable` - Mixin for serialization (no state) ✅

**Verdict**: ✅ **KEEP ALL** - These are pure data, correctly designed

---

## 3. Checkpoint Storage

### ⚠️ REFACTOR: `FileCheckpointStore` (Questionable)

```python
class FileCheckpointStore:
    def __init__(self, environment_registry, directory):
        self.directory = Path(directory)  # Just config!
        self.environment_registry = environment_registry  # Just config!

    async def save(self, checkpoint_id, state):
        # Pure I/O, no state mutation
        data = await serialize_agent_state(state)  # Pure function
        await trio.Path(path).write_text(json.dumps(data))  # Pure I/O

    async def load(self, checkpoint_id):
        # Pure I/O, no state mutation
        data = json.loads(await trio.Path(path).read_text())
        return await deserialize_agent_state(data, self.environment_registry)
```

**Analysis:**
- `directory` and `environment_registry` are just **configuration**, not mutable state
- All methods are **pure I/O** (read/write files, no internal state changes)
- No state is mutated between calls

**This is just a namespace for related functions!**

**Refactor to:**

```python
# Pure functions (no class needed)

async def save_agent_checkpoint(
    checkpoint_id: str,
    state: AgentState,
    directory: Path,
) -> None:
    """Save agent state to checkpoint (pure I/O)."""
    data = await serialize_agent_state(state)
    data["_metadata"] = {
        "checkpoint_id": checkpoint_id,
        "timestamp": time.time(),
    }
    path = directory / f"{checkpoint_id}.json"
    await trio.Path(path).write_text(json.dumps(data, indent=2))


async def load_agent_checkpoint(
    checkpoint_id: str,
    directory: Path,
    environment_registry: Dict[str, type[Environment]],
) -> AgentState:
    """Load agent state from checkpoint (pure I/O)."""
    path = directory / f"{checkpoint_id}.json"
    data = json.loads(await trio.Path(path).read_text())
    data.pop("_metadata", None)
    return await deserialize_agent_state(data, environment_registry)


async def list_agent_checkpoints(directory: Path) -> list[str]:
    """List all checkpoint IDs (pure I/O)."""
    return sorted(path.stem for path in directory.glob("*.json"))


# Usage (more explicit):
await save_agent_checkpoint(
    checkpoint_id="step_100",
    state=agent_state,
    directory=Path("/tmp/checkpoints"),
)

state = await load_agent_checkpoint(
    checkpoint_id="step_100",
    directory=Path("/tmp/checkpoints"),
    environment_registry=env_registry,
)
```

**Benefits:**
- ✅ No class wrapper for config
- ✅ Explicit parameters (no hidden instance variables)
- ✅ Easier to test (just pass different directories)
- ✅ More functional style

**Verdict**: ⚠️ **REFACTOR** to pure functions (low priority - not critical path)

---

## 4. Weight Sync (`weight_sync.py`)

### ✅ KEEP: Adapters Are Minimal

```python
@dataclass
class SGLangEngine:
    base_url: str
    timeout: float = 300.0

    async def update_weights_from_checkpoint(self, checkpoint_path):
        return await update_sglang_weights_from_disk(
            self.base_url, checkpoint_path, timeout=self.timeout
        )
```

**Analysis:**
- `base_url` and `timeout` are **immutable config**, not mutable state
- These are just **adapters** implementing `InferenceEngine` protocol
- Very thin wrappers around pure functions

**Verdict**: ✅ **KEEP** - Minimal adapters for polymorphism (acceptable use of classes)

These could be refactored to pure functions:
```python
# Alternative (more functional):
def sglang_engine(base_url: str, timeout: float = 300.0):
    """Factory for SGLang engine functions."""
    async def update_weights_from_checkpoint(checkpoint_path: str):
        return await update_sglang_weights_from_disk(
            base_url, checkpoint_path, timeout=timeout
        )
    return update_weights_from_checkpoint
```

But the current dataclass approach is fine for protocol implementation.

---

## 5. Config Classes

### ✅ KEEP: All Are Immutable Config

```python
@dataclass(frozen=True)
class RolloutConfig: ...

@dataclass(frozen=True)
class TrainingConfig: ...

@dataclass
class RunConfig: ...  # Could be frozen
```

**Verdict**: ✅ **KEEP** - Pure configuration data

**Suggestion**: Make `RunConfig` frozen too for consistency.

---

## Summary: What to Refactor

### 🔴 High Priority (Do Before Building More)

1. **`RolloutManager`** → Pure functions / Generator
   - **Why**: Building SFT/RL loops will use this pattern
   - **Effort**: ~1 hour
   - **Impact**: Prevents propagating bad pattern

### 🟡 Medium Priority (Nice to Have)

2. **`FileCheckpointStore`** → Pure functions
   - **Why**: Cleaner, more testable
   - **Effort**: ~30 minutes
   - **Impact**: Moderate (not on critical path)

### 🟢 Low Priority (Optional)

3. **Make `RunConfig` frozen** - Consistency
4. **Consider refactoring SGLang/VLLM adapters** - More functional (but current is fine)

---

## Refactoring Plan

### Step 1: Deprecate `RolloutManager` (Do Now)

**Create**: `training/rollout_generation.py`

```python
"""Pure functions for rollout generation.

Replaces the stateful RolloutManager class with pure functions.
"""

from typing import Iterator
from training.data_buffer import DataBuffer
from training.types import RolloutBatch, RolloutConfig


def generate_rollout_batches(
    data_buffer: DataBuffer,
    config: RolloutConfig,
    **rollout_kwargs,
) -> Iterator[RolloutBatch]:
    """Generate rollout batches indefinitely (generator).

    Args:
        data_buffer: DataBuffer for prompt iteration
        config: RolloutConfig with batch_size and generate_fn
        **rollout_kwargs: Additional kwargs for generate_fn

    Yields:
        RolloutBatch objects ready for training

    Example:
        >>> data_buffer = DataBuffer(prompts=[...])
        >>> config = RolloutConfig(batch_size=4, generate_fn=my_fn)
        >>> batches = generate_rollout_batches(data_buffer, config)
        >>> for batch in batches:
        ...     train_on_batch(batch)
    """
    step = 0
    while True:
        # Get prompts (data_buffer manages its own state)
        prompts = data_buffer.get_prompts(config.batch_size)

        # Generate samples
        samples = config.generate_fn(prompts, **rollout_kwargs)

        # Apply transforms
        samples = apply_sample_transforms(samples, config)

        # Convert to batch
        batch = convert_to_batch(
            samples,
            epoch_id=data_buffer.epoch_id,
            step_id=step,
        )

        yield batch
        step += 1


def apply_sample_transforms(samples, config):
    """Pure function: Apply filters/transforms to samples."""
    # Move from rollout_manager.py (already pure!)
    ...


def convert_to_batch(samples, epoch_id, step_id):
    """Pure function: Convert samples to RolloutBatch."""
    # Move from rollout_manager.py (already pure!)
    ...
```

**Update**: `AsyncRolloutManager` to use these pure functions internally

**Deprecate**: `RolloutManager` class (add deprecation warning)

### Step 2: Update Examples/Tests

Update examples to use generator pattern:

```python
# Old (class-based):
manager = RolloutManager(data_buffer, config)
for batch in manager:
    train_on_batch(batch)

# New (generator-based):
batches = generate_rollout_batches(data_buffer, config)
for batch in batches:
    train_on_batch(batch)
```

### Step 3: Refactor `FileCheckpointStore` (Optional)

Create `checkpoint_io.py` with pure functions.

---

## Design Philosophy Reinforced

**Classes are for STATE MANAGEMENT, not code organization.**

If your class just holds config and calls functions → **use a module with functions instead**.

Good uses of classes:
- ✅ `DataBuffer` - Manages iteration state
- ✅ `PyTorchTrainingBackend` - Manages training state
- ✅ `AsyncRolloutManager` - Manages async coordination state

Bad uses of classes:
- ❌ `RolloutManager` - Just wraps pure functions
- ❌ `FileCheckpointStore` - Just namespaces I/O functions
- ❌ "Helper" classes that are just bags of static methods

**Rule**: If you can implement it as a generator or pure function, DO THAT FIRST.

Only reach for classes when you have real mutable state to manage.
