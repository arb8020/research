# D5: Weight Sync Protocol - Architecture Clarification

**Based on SLIME code study (slime/backends/megatron_utils/)**

## Key Insight: Weight Version State Lives in Training Backend

After studying SLIME's implementation, the weight version tracking belongs in the **Training Backend**, not in a separate WeightSyncManager.

### SLIME's Architecture

```python
# slime/backends/megatron_utils/actor.py
class MegatronTrainRayActor:
    def __init__(self):
        self.model = ...  # PyTorch model
        self.weights = {"actor": {...}}  # CPU copy

        # Weight updater is OWNED by training backend
        self.weight_updater = UpdateWeightFromTensor(
            args, model, self.weights, ...
        )

    def update_weights(self):
        """Called after training step."""
        # Weight updater increments its own version
        self.weight_updater.update_weights()


# slime/backends/megatron_utils/update_weight_utils.py
class UpdateWeightFromTensor:
    def __init__(self, ...):
        self.weight_version = 0  # ← STATE LIVES HERE

    def update_weights(self):
        self.weight_version += 1  # Increment on each sync
        # ... sync to inference engines with version
```

**Key insight:** The training backend owns the weight updater, which owns the version state.

## Correct Stateful Component Map

For each major function, the ONE stateful piece:

1. **Data iteration:** `DataBuffer` (tracks epoch_id, sample_offset)
2. **Rollout generation:** `AsyncRolloutManager` (caches partial_samples)
3. **Weight sync:** NO separate state! Version tracking in Training Backend (D6)
4. **Training:** `TrainingBackend` (model weights, optimizer, weight_version)

## D5: Stateless Weight Sync Functions

D5 should provide **pure functions** for weight sync, with NO state:

```python
# ══════════════════════════════════════════════════════════════
# Fine-grained immediate mode (Casey Muratori style)
# ══════════════════════════════════════════════════════════════

async def update_sglang_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Update SGLang server weights from checkpoint (pure function).

    Tiger Style: Assert preconditions, no hidden state.

    Args:
        base_url: SGLang server URL (e.g. "http://localhost:30000")
        checkpoint_path: Path to checkpoint (local or HF model ID)
        timeout: Request timeout in seconds

    Returns:
        Response dict with success status

    Raises:
        httpx.HTTPError: If request fails
    """
    assert base_url, "base_url cannot be empty"
    assert checkpoint_path, "checkpoint_path cannot be empty"
    assert timeout > 0, f"timeout must be > 0, got {timeout}"

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/update_weights_from_disk",
            json={"model_path": checkpoint_path},
        )
        response.raise_for_status()
        return response.json()


async def update_vllm_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
    *,
    timeout: float = 300.0,
) -> dict[str, Any]:
    """Update vLLM server weights via reload_weights() RPC."""
    assert base_url and checkpoint_path and timeout > 0

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(
            f"{base_url}/collective_rpc",
            json={
                "method": "reload_weights",
                "params": {"model_path": checkpoint_path},
            },
        )
        response.raise_for_status()
        return response.json()


# ══════════════════════════════════════════════════════════════
# Minimal Protocol (only what's needed for polymorphism)
# ══════════════════════════════════════════════════════════════

class InferenceEngine(Protocol):
    """Minimal protocol for weight updates (Tiger Style: just type hints).

    Removed from design doc:
    - async def offload() -> None  # SGLang/vLLM don't have this
    - async def reload() -> None   # SGLang/vLLM don't have this

    Casey Muratori: Don't add methods that don't exist in implementations!
    """

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update model weights from checkpoint on disk."""
        ...


# ══════════════════════════════════════════════════════════════
# Adapters (Casey: redundancy - multiple ways to achieve goal)
# ══════════════════════════════════════════════════════════════

@dataclass
class SGLangEngine:
    """SGLang adapter implementing InferenceEngine protocol."""

    base_url: str
    timeout: float = 300.0

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        return await update_sglang_weights_from_disk(
            self.base_url,
            checkpoint_path,
            timeout=self.timeout,
        )


@dataclass
class VLLMEngine:
    """vLLM adapter implementing InferenceEngine protocol."""

    base_url: str
    timeout: float = 300.0

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        return await update_vllm_weights_from_disk(
            self.base_url,
            checkpoint_path,
            timeout=self.timeout,
        )


# ══════════════════════════════════════════════════════════════
# Stateless orchestration (Sean Goedecke: boring coordination)
# ══════════════════════════════════════════════════════════════

async def sync_weights_to_engines(
    engines: list[InferenceEngine],
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    """Sync checkpoint to multiple engines in parallel (pure function).

    No state! No retention! Just coordinates the sync.

    Args:
        engines: List of inference engines (SGLang or vLLM)
        checkpoint_path: Path to checkpoint directory

    Returns:
        List of responses from each engine

    Example:
        >>> engines = [
        ...     SGLangEngine("http://localhost:30000"),
        ...     VLLMEngine("http://localhost:30001"),
        ... ]
        >>> responses = await sync_weights_to_engines(engines, "/ckpt/step_100")
    """
    # Tiger Style: assert preconditions
    assert len(engines) > 0, "Must provide at least one engine"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Parallel sync (trio for structured concurrency)
    results = []
    async with trio.open_nursery() as nursery:
        async def sync_one(engine):
            response = await engine.update_weights_from_checkpoint(checkpoint_path)
            results.append(response)

        for engine in engines:
            nursery.start_soon(sync_one, engine)

    return results
```

## D6: Training Backend Owns Weight Version

Weight version tracking happens in D6 (Training Backend):

```python
# D6: Training Backend (has weight version state)
@dataclass
class PyTorchTrainingBackend:
    """PyTorch training backend with weight version tracking.

    This is where weight_version state lives (SLIME-inspired).
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    checkpoint_dir: Path

    # State: weight version counter (SLIME has this)
    weight_version: int = 0

    async def train_step(self, batch: RolloutBatch) -> dict[str, Any]:
        """Training step."""
        # ... training logic ...
        return {"loss": loss, "grad_norm": grad_norm}

    async def save_checkpoint(self, step: int) -> Path:
        """Save checkpoint with version metadata."""
        # Increment version (SLIME does this in UpdateWeightFromTensor)
        self.weight_version += 1

        checkpoint_path = self.checkpoint_dir / f"step_{step}"

        # Save model + metadata
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "weight_version": self.weight_version,  # Include version
        }, checkpoint_path / "pytorch_model.bin")

        return checkpoint_path
```

## D5 Acceptance Criteria (Updated)

- [x] `InferenceEngine` protocol defined (minimal, no offload/reload)
- [x] Fine-grained functions: `update_sglang_weights_from_disk()`, `update_vllm_weights_from_disk()`
- [x] Adapters: `SGLangEngine`, `VLLMEngine` implement protocol
- [x] Stateless orchestration: `sync_weights_to_engines()` coordinates parallel sync
- [x] No weight version state (that's D6!)
- [x] Casey Muratori: Granularity (fine + coarse), no retention
- [x] Tiger Style: Assert preconditions, explicit control flow
- [x] Sean Goedecke: Stateless coordination, boring patterns

## Usage Example

```python
# D5: Weight sync (stateless)
engines = [
    SGLangEngine("http://localhost:30000"),
    VLLMEngine("http://localhost:30001"),
]

# D6: Training backend (stateful - owns version)
training_backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    checkpoint_dir=Path("/checkpoints"),
)

# Training loop
for step in range(1000):
    batch = await rollout_manager.generate_batch()
    metrics = await training_backend.train_step(batch)

    if step % 100 == 0:
        # Save checkpoint (increments weight_version internally)
        checkpoint_path = await training_backend.save_checkpoint(step)

        # Sync to inference engines (stateless!)
        await sync_weights_to_engines(engines, str(checkpoint_path))
```

## Summary

**What changed from original design doc:**

1. ❌ Removed `InferenceEngine.offload()` and `reload()` - SGLang/vLLM don't have these
2. ❌ Removed `WeightSyncManager` as a stateful class
3. ✅ Added fine-grained pure functions (Casey Muratori style)
4. ✅ Made orchestration stateless (Sean Goedecke style)
5. ✅ Moved `weight_version` state to Training Backend (SLIME-inspired)

**Stateful components remain:**
1. DataBuffer (epoch_id, sample_offset)
2. AsyncRolloutManager (partial_samples cache)
3. TrainingBackend (model weights, optimizer, weight_version) ← D6

D5 is now **purely stateless coordination**, which is correct per SLIME's architecture!
