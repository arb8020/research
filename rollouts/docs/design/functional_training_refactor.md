# Functional Training Refactor

> Explicit state passing, pure functions, nmoe-style training loop.

## Current State

`PyTorchTrainingBackend` is a stateful class with methods that mutate `self`:

```python
@dataclass
class PyTorchTrainingBackend:
    model: nn.Module
    optimizer: Optimizer
    weight_version: int = 0
    current_step: int = 0
    _poisoned: bool = False

    def forward_backward(self, batch) -> TrainFuture[dict]:
        # mutates model grads
        ...

    def optim_step(self) -> TrainFuture[dict]:
        self.current_step += 1  # mutation
        ...

    async def save_checkpoint(self, step):
        self.weight_version += 1  # mutation
        ...
```

Problems:
- Hidden state changes (`_poisoned`, `current_step`, `weight_version`)
- Methods return futures that are immediately resolved (sync pretending to be async)
- Hard to test: need to mock entire class
- State changes not visible at call sites

## Design Principles

From nmoe and `code_style/FAVORITES.md`:
- **Local variables over instance state** - state visible in function scope
- **Explicit state threading** - pass state in, return state out
- **Frozen dataclasses for bundles** - group related data without behavior
- **Pure functions for transforms** - inputs determine outputs

## Proposed Design

### Core Types

```python
from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass(frozen=True)
class TrainingState:
    """Immutable training state snapshot.

    All mutable training state captured here.
    Functions return new TrainingState instead of mutating.
    """
    step: int = 0
    weight_version: int = 0
    tokens_seen: int = 0

@dataclass
class TrainingComponents:
    """Mutable components bundle (model, optimizer).

    These are inherently stateful (PyTorch requirement).
    Grouped for convenience, not behavior.
    """
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    device: torch.device
    checkpoint_dir: Path

@dataclass(frozen=True)
class TrainerConfig:
    """Training hyperparameters - immutable."""
    micro_batch_size: int | None = None
    num_minibatches: int | None = None
    max_grad_norm: float | None = 1.0
```

### Pure Functions

#### Forward/Backward

```python
def forward_backward(
    components: TrainingComponents,
    batch: dict[str, torch.Tensor],
    config: TrainerConfig,
) -> dict[str, float]:
    """Compute loss and gradients.

    Pure-ish: mutates model grads (PyTorch requirement).
    Returns metrics dict, no hidden state changes.
    """
    components.optimizer.zero_grad()

    # Move batch to device
    batch = {k: v.to(components.device) for k, v in batch.items()}

    # Gradient accumulation
    num_minibatches = config.get_num_minibatches(batch["input_ids"].shape[0])
    total_loss = 0.0

    for i in range(num_minibatches):
        micro_batch = slice_batch(batch, i, num_minibatches)
        logits = components.model(micro_batch["input_ids"])
        loss = compute_loss(logits, micro_batch)
        (loss / num_minibatches).backward()
        total_loss += loss.item()

    grad_norm = compute_grad_norm(components.model)

    return {
        "loss": total_loss / num_minibatches,
        "grad_norm": grad_norm,
    }
```

#### Optimizer Step

```python
def optim_step(
    components: TrainingComponents,
    state: TrainingState,
    config: TrainerConfig,
) -> TrainingState:
    """Apply gradients and return new state.

    Explicit state in, new state out.
    """
    # Clip gradients
    if config.max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            components.model.parameters(),
            config.max_grad_norm,
        )

    components.optimizer.step()

    # Return new state (no mutation)
    return TrainingState(
        step=state.step + 1,
        weight_version=state.weight_version,
        tokens_seen=state.tokens_seen,
    )
```

#### Checkpointing

```python
def save_checkpoint(
    components: TrainingComponents,
    state: TrainingState,
    metrics: dict[str, float],
) -> TrainingState:
    """Save checkpoint and return new state with incremented version."""
    ckpt_dir = components.checkpoint_dir / f"step_{state.step:04d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    torch.save(components.model.state_dict(), ckpt_dir / "model.bin")
    torch.save(components.optimizer.state_dict(), ckpt_dir / "optimizer.bin")

    metadata = {
        "step": state.step,
        "weight_version": state.weight_version + 1,
        "metrics": metrics,
    }
    (ckpt_dir / "metadata.json").write_text(json.dumps(metadata))

    return TrainingState(
        step=state.step,
        weight_version=state.weight_version + 1,
        tokens_seen=state.tokens_seen,
    )

def load_checkpoint(
    components: TrainingComponents,
    checkpoint_path: Path,
) -> TrainingState:
    """Load checkpoint and return restored state."""
    components.model.load_state_dict(
        torch.load(checkpoint_path / "model.bin", map_location="cpu")
    )
    components.optimizer.load_state_dict(
        torch.load(checkpoint_path / "optimizer.bin", map_location="cpu")
    )

    metadata = json.loads((checkpoint_path / "metadata.json").read_text())

    return TrainingState(
        step=metadata["step"],
        weight_version=metadata["weight_version"],
        tokens_seen=metadata.get("tokens_seen", 0),
    )
```

#### Weight Sync

```python
def get_weights(components: TrainingComponents) -> dict[str, torch.Tensor]:
    """Get model weights for inference sync. Pure accessor."""
    return components.model.state_dict()

def load_weights(
    components: TrainingComponents,
    weights: dict[str, torch.Tensor],
) -> None:
    """Load weights into model. Mutates model (PyTorch requirement)."""
    components.model.load_state_dict(weights)
```

### Training Loop (nmoe-style)

```python
async def grpo_train(
    config: GRPOConfig,
    prompts: list[dict],
    score_fn: Callable,
) -> dict[str, Any]:
    """GRPO training with explicit state passing."""

    # Initialize components (mutable PyTorch objects)
    model = load_model(config.model_name)
    optimizer = create_optimizer(model, config.lr)
    components = TrainingComponents(
        model=model,
        optimizer=optimizer,
        device=torch.device("cuda"),
        checkpoint_dir=Path(config.checkpoint_dir),
    )

    # Initialize state (immutable, threaded through)
    state = TrainingState()
    buffer_state = BufferState(seed=config.seed)
    trainer_config = TrainerConfig(max_grad_norm=config.max_grad_norm)

    # Maybe restore from checkpoint
    if config.resume:
        latest = find_latest_checkpoint(components.checkpoint_dir)
        if latest:
            state = load_checkpoint(components, latest)

    # Training loop - state flows explicitly
    for step_num in range(state.step, config.num_steps):
        # Get batch (functional)
        samples, buffer_state = get_samples(prompts, buffer_state, config.batch_size)

        # Generate rollouts
        batch = await generate_rollouts(components, samples, score_fn)

        # Forward/backward (mutates grads)
        metrics = forward_backward(components, batch, trainer_config)

        # Optimizer step (returns new state)
        state = optim_step(components, state, trainer_config)
        state = TrainingState(
            step=state.step,
            weight_version=state.weight_version,
            tokens_seen=state.tokens_seen + batch["input_ids"].numel(),
        )

        # Log
        log_metrics(step_num, metrics)

        # Checkpoint (returns new state)
        if step_num % config.checkpoint_every == 0:
            state = save_checkpoint(components, state, metrics)

        # Sync to inference
        if step_num % config.sync_every == 0:
            weights = get_weights(components)
            await sync_to_inference(weights, state.weight_version)

    return {"final_step": state.step, "tokens_seen": state.tokens_seen}
```

## Migration Path

### Phase 1: Add functions alongside class
- Add `forward_backward()`, `optim_step()`, `save_checkpoint()` as standalone functions
- Keep `PyTorchTrainingBackend` working for existing callers
- New code uses functions directly

### Phase 2: Refactor class to use functions
- `PyTorchTrainingBackend.forward_backward()` calls `forward_backward(self.components, ...)`
- Class becomes thin wrapper over functions
- Tests can use either

### Phase 3: Remove class (optional)
- If class provides no value, remove it
- Or keep as convenience bundle for users who prefer OOP

## Comparison

| Aspect | Current (class) | Proposed (functions) |
|--------|-----------------|---------------------|
| State visibility | Hidden in `self` | Explicit at call sites |
| Testing | Mock entire class | Pass test data directly |
| Composition | Subclass or wrap | Compose functions |
| Checkpointing | Method mutates version | Function returns new state |
| Type safety | Mutable state hard to track | Frozen dataclass enforced |

## Files to Change

- `rollouts/training/backends/pytorch.py` - add functions, refactor class
- `rollouts/training/backends/pytorch_factory.py` - update factories
- `rollouts/training/grpo.py` - use functional API
- `rollouts/training/loops/sft_loop.py` - use functional API
- `rollouts/training/loops/rl_loop.py` - use functional API

## Related

- nmoe's `train.py` - reference implementation
- `DataBuffer` refactor (commit `eccd507`) - same pattern applied to data loading
- `code_style/FAVORITES.md` - design principles
