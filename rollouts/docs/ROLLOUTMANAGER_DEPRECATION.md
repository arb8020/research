# RolloutManager: Should We Delete It?

**Date**: 2025-11-09
**Question**: Is there any reason RolloutManager was made stateful? Are we ever going to use it now that we have AsyncRolloutManager?

## Answer: NO - Delete It

### Current Usage

**Where it's imported:**
1. `training/__init__.py` - Exported in public API
2. `examples/test_rollout_manager.py` - Test file
3. `examples/sft_pipeline_example.py` - Example file
4. `training/async_rollout_manager.py` - Uses `convert_to_batch()` helper

**Actual usage**: Only in **examples/tests**, not in any production code.

### Why RolloutManager Was Created (Git History)

Commit `2db65d3`: "Add SFT data pipeline and RolloutManager (D2-D4)"

It was created as a **synchronous iterator** for simple use cases:

```python
# Original intent:
manager = RolloutManager(data_buffer, config)
for batch in manager:
    train_step(batch)
```

**Purpose**: Simple synchronous rollout generation (no async needed).

### Why It Has State

Looking at the code:

```python
class RolloutManager:
    def __init__(self, data_buffer, config):
        self.data_buffer = data_buffer  # Dependency
        self.config = config            # Config
        self._step_count = 0            # Counter for tracking

    def state_dict(self):
        """Save manager state for checkpointing."""
        return {
            "buffer_state": self.data_buffer.save_state(),
            "step_count": self._step_count,  # ← This is why it's "stateful"
        }
```

**The "state" exists for checkpointing** - so you can resume training from where you left off.

**But wait...** is `_step_count` real state or just a derived value?

```python
# _step_count is just incremented on each __next__() call
def __next__(self):
    # ... generate batch ...
    self._step_count += 1
    return batch
```

**This is NOT real state!** It's just a loop counter. The REAL state is in `data_buffer`.

You could derive `_step_count` from `data_buffer.sample_offset / config.batch_size`.

### AsyncRolloutManager vs RolloutManager

| Feature | RolloutManager | AsyncRolloutManager |
|---------|----------------|---------------------|
| **Execution** | Synchronous | Async (parallel) |
| **Use case** | Simple SFT | RL with dynamic sampling |
| **Over-sampling** | No | Yes (SLIME feature) |
| **Filtering** | Basic | Advanced (abort/resume) |
| **Real state** | None (just counter) | Yes (`partial_samples` cache) |
| **Dependencies** | `convert_to_batch()` | Same helper functions |

**AsyncRolloutManager is strictly more powerful:**
- Can do everything RolloutManager does
- Plus async parallel generation
- Plus dynamic over-sampling
- Plus partial sample caching (real state!)

### The Real Question: Do We Need Both?

**No. Here's why:**

1. **AsyncRolloutManager can replace RolloutManager**
   ```python
   # RolloutManager (sync):
   manager = RolloutManager(data_buffer, config)
   for batch in manager:
       train_step(batch)

   # AsyncRolloutManager (async, but same API):
   async with AsyncRolloutManager(data_buffer, config) as manager:
       for step in range(num_steps):
           batch = await manager.generate_batch()
           train_step(batch)
   ```

2. **The only difference is async vs sync**
   - But we're using Trio everywhere!
   - All our training loops are async
   - There's no use case for sync-only rollout generation

3. **Shared code: `convert_to_batch()` and helpers**
   - These are pure functions
   - Both managers use them
   - They should be standalone functions, not tied to either class

### Recommendation: Delete RolloutManager

**Reasons:**
1. ❌ Not used in production code (only examples/tests)
2. ❌ No legitimate mutable state (just a loop counter)
3. ❌ AsyncRolloutManager can do everything it does + more
4. ❌ We're async-first (Trio), no need for sync variant
5. ❌ Confusing to have two similar classes

**Migration path:**

```python
# OLD: RolloutManager (sync)
manager = RolloutManager(data_buffer, config)
for batch in manager:
    train_step(batch)

# NEW: AsyncRolloutManager (async)
async with AsyncRolloutManager(data_buffer, config) as manager:
    for step in range(num_steps):
        batch = await manager.generate_batch()
        train_step(batch)

# OR: Pure function generator (even simpler)
batches = generate_rollout_batches(data_buffer, config)
for batch in batches:
    train_step(batch)
```

### Action Plan

1. **Extract shared helpers** to `rollout_generation.py`:
   ```python
   # Pure functions (used by both managers):
   def apply_sample_transforms(samples, config): ...
   def convert_to_batch(samples, epoch_id, step_id): ...
   def extract_sample_fields(samples): ...
   def compute_response_lengths(loss_masks): ...
   ```

2. **Delete `RolloutManager` class**
   - Remove from `training/rollout_manager.py`
   - Keep the file for helper functions

3. **Update `AsyncRolloutManager`** to import helpers:
   ```python
   from training.rollout_generation import convert_to_batch, apply_sample_transforms
   ```

4. **Update examples** to use `AsyncRolloutManager`:
   - `examples/test_rollout_manager.py` → `examples/test_rollout_generation.py`
   - `examples/sft_pipeline_example.py` → Use AsyncRolloutManager

5. **Remove from public API**:
   - Delete export from `training/__init__.py`

### Why This Happened (Lessons Learned)

**Classic over-engineering pattern:**

1. Started with simple problem: "iterate over rollouts"
2. Made it a class: "easier to organize methods"
3. Added state tracking: "might need checkpointing later"
4. Built async version: "need parallelism for RL"
5. Now have two classes that do same thing

**What should have been:**
1. Pure functions: `convert_to_batch()`, etc.
2. Simple generator: `def generate_rollouts(): while True: yield ...`
3. Async manager only when needed: `AsyncRolloutManager` (has real state!)

**Lesson**: Start with functions, only add classes when you have REAL mutable state.

## Conclusion

**Delete RolloutManager.**

It was created before AsyncRolloutManager existed, has no legitimate state, and serves no purpose now that we have the async version.

Keep the pure helper functions (`convert_to_batch`, etc.) in a standalone module.
