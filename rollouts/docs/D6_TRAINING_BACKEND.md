# D6: Training Backend

**Status**: In Progress (PyTorch v1)
**Based on research**: SLIME, nanochat, torchforge, Tinker

## Overview

D6 provides training backend implementations that follow the `TrainingBackend` protocol. The protocol is backend-agnostic, supporting multiple backend options:

**D6v1** (now): Standard PyTorch (OOP, stateful) - **IMPLEMENTING NOW**
**D6v2** (later): torch.func + torchopt (functional PyTorch)
**D6v3** (later): Raw JAX (pure functional, TPU support)
**D6v4** (experimental): torchax (PyTorch syntax → JAX backend)

All backends implement the same protocol, allowing users to swap backends easily.

## Design Principles

Based on research into SLIME, nanochat, torchforge, and internal "Tinker" philosophy:

1. **SLIME**: Weight version tracking lives in training backend (`self.weight_version`)
2. **nanochat**: Simple checkpoint format (`step_N/pytorch_model.bin`, `metadata.json`)
3. **torchforge**: Clean separation between algorithm and infrastructure
4. **Tinker**: Minimal surface area, future-based pipelining, token-level control

## Protocol (Backend-Agnostic)

Already defined in `rollouts/training/backends/protocol.py`:

```python
class TrainingBackend(Protocol):
    """Protocol for training backends.

    Implementations: PyTorchBackend (D6v1), JAXBackend (D6v2)
    """

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients (returns future immediately)."""
        ...

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients and update weights (returns future)."""
        ...

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference."""
        ...

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint."""
        ...
```

## Key Design Decisions

### 1. Execution Model
**Decision**: Trio-based async with futures (Option B)

Training operations return `TrainFuture[T]` immediately, enabling pipelining:
```python
fwd_bwd_future = backend.forward_backward(batch)
# Do other work here...
metrics = await fwd_bwd_future.result()
```

Uses `trio.to_thread.run_sync()` to run PyTorch ops in background threads.

### 2. Training Step Pipelining
**Decision**: Sequential (one operation in flight at a time)

Simpler model for D6v1. Cannot call `forward_backward()` until previous operation completes.

Later versions may support overlapping operations for gradient accumulation.

### 3. Checkpoint Format
**Decision**: Simple PyTorch format (nanochat-inspired)

```
checkpoint_dir/
  step_0100/
    pytorch_model.bin      # torch.save(model.state_dict())
    optimizer.bin          # torch.save(optimizer.state_dict())
    metadata.json          # {step, weight_version, timestamp, metrics}
```

Metadata includes:
- `step`: Training step number
- `weight_version`: Weight version counter (SLIME-inspired)
- `timestamp`: Unix timestamp
- `metrics`: Dict of training metrics (loss, grad_norm, etc.)

### 4. Weight Version Semantics
**Decision**: Increment on `save_checkpoint()` (SLIME pattern)

```python
self.weight_version = 0  # Init

async def save_checkpoint(self, step: int):
    self.weight_version += 1  # Increment on save
    # Save checkpoint with version in metadata
```

On `load_checkpoint()`, `weight_version` is restored from metadata.

### 5. Integration with D5 Weight Sync
**Decision**: Decoupled (no direct integration)

D6 saves checkpoints, D5 syncs them to inference engines. User orchestrates:

```python
# D6: Save checkpoint
ckpt_path = await backend.save_checkpoint(step)

# D5: Sync to inference engines (stateless!)
from training import sync_weights_to_engines
await sync_weights_to_engines(engines, str(ckpt_path))
```

Clean separation of concerns.

### 6. Model Initialization
**Decision**: User provides pre-created model/optimizer (Option A)

```python
model = GPT(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=my_loss_fn,
    checkpoint_dir=Path("/checkpoints"),
)
```

Explicit is better than implicit. User handles device placement.

### 7. Device Management
**Decision**: User handles `.to(device)` before passing model (Option A)

Backend doesn't touch device placement. User is responsible for ensuring model, optimizer, and data are on same device.

### 8. Gradient Accumulation
**Decision**: Not in D6v1 (single batch per step)

```python
# Simple training loop (D6v1)
await backend.forward_backward(batch).result()
await backend.optim_step().result()
```

Later versions may support micro-batching for gradient accumulation.

### 9. Error Handling in Futures
**Decision**: Poison backend on error

If a training operation fails, backend enters poisoned state and rejects new operations. User must create new backend instance to recover.

```python
future = backend.forward_backward(batch)
try:
    metrics = await future.result()
except TrainingError as e:
    # Backend is now poisoned
    # Must create new backend to continue
```

### 10. Loss Computation
**Decision**: User provides loss function (Option B)

```python
def my_loss_fn(
    logits: torch.Tensor,      # [batch, seq_len, vocab_size]
    labels: torch.Tensor,      # [batch, seq_len]
    loss_mask: torch.Tensor,   # [batch, seq_len]
) -> torch.Tensor:
    """Compute masked loss (Tinker: token-level control)."""
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
    )
    loss = loss.reshape(labels.shape) * loss_mask
    return loss.sum() / loss_mask.sum()

backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=my_loss_fn,  # User-provided
    checkpoint_dir=Path("/ckpts"),
)
```

Provides flexibility for custom loss functions (e.g., token-level weighting, RL losses).

## D6v1: PyTorchTrainingBackend

### Implementation

```python
@dataclass
class PyTorchTrainingBackend:
    """Future-based PyTorch training backend (D6v1).

    Features:
    - Async training via Trio futures
    - Weight version tracking (SLIME-inspired)
    - Simple checkpoint format (nanochat-inspired)
    - Minimal surface area (Tinker-inspired)

    Tiger Style: Explicit state, assert preconditions.
    Casey Muratori: Minimal coupling, futures for pipelining.
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    loss_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    checkpoint_dir: Path

    # State (SLIME-inspired)
    weight_version: int = 0
    current_step: int = 0

    # Execution state
    _nursery: Optional[trio.Nursery] = None
    _poisoned: bool = False

    def __post_init__(self):
        """Validate initialization (Tiger Style)."""
        assert self.model is not None, "model cannot be None"
        assert self.optimizer is not None, "optimizer cannot be None"
        assert self.loss_fn is not None, "loss_fn cannot be None"
        assert self.checkpoint_dir is not None, "checkpoint_dir cannot be None"

        # Create checkpoint directory if needed
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients (returns future immediately).

        Args:
            batch: {
                "input_ids": torch.Tensor,  # [batch, seq_len]
                "labels": torch.Tensor,     # [batch, seq_len]
                "loss_mask": torch.Tensor,  # [batch, seq_len]
            }

        Returns:
            Future resolving to {"loss": float, "grad_norm": float}

        Raises:
            AssertionError: If backend is poisoned or batch is invalid
        """
        ...

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients and update weights (returns future).

        Returns:
            Future resolving to {"lr": float, "step": int}
        """
        ...

    async def save_checkpoint(
        self,
        step: int,
        metrics: Dict[str, float] = {},
    ) -> Path:
        """Save checkpoint with version (increments weight_version).

        Args:
            step: Training step number
            metrics: Optional training metrics to save

        Returns:
            Path to checkpoint directory (e.g., checkpoint_dir/step_0100)

        Side effects:
            - Increments self.weight_version
            - Creates checkpoint_dir/step_{step:04d}/
            - Saves pytorch_model.bin, optimizer.bin, metadata.json
        """
        ...

    async def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint and restore weight_version.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Metadata dict from checkpoint

        Side effects:
            - Loads model and optimizer state
            - Restores self.weight_version from metadata
            - Updates self.current_step
        """
        ...

    def get_weights(self) -> TrainFuture[Dict[str, Any]]:
        """Get model weights for syncing to inference.

        Returns:
            Future resolving to model.state_dict()
        """
        ...

    def load_weights(self, weights: Dict[str, Any]) -> TrainFuture[None]:
        """Load model weights from inference or checkpoint.

        Args:
            weights: state_dict to load
        """
        ...
```

### Usage Example

```python
import torch
import trio
from pathlib import Path
from rollouts.training.backends import PyTorchTrainingBackend

# User creates model and optimizer
model = GPT(config).to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# User provides loss function
def loss_fn(logits, labels, loss_mask):
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        reduction='none',
    )
    loss = loss.reshape(labels.shape) * loss_mask
    return loss.sum() / loss_mask.sum()

# Create backend
backend = PyTorchTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    checkpoint_dir=Path("/checkpoints"),
)

# Training loop
async def train():
    for step in range(1000):
        # Get batch from data buffer
        batch = await data_buffer.get_batch()

        # Forward + backward (returns future)
        fwd_bwd_future = backend.forward_backward(batch)
        metrics = await fwd_bwd_future.result()

        # Optimizer step (returns future)
        step_future = backend.optim_step()
        step_metrics = await step_future.result()

        print(f"Step {step}: loss={metrics['loss']:.4f}")

        # Save checkpoint every 100 steps
        if step % 100 == 0:
            ckpt_path = await backend.save_checkpoint(step, metrics)
            print(f"Saved checkpoint to {ckpt_path}")

            # D5: Sync to inference engines
            await sync_weights_to_engines(engines, str(ckpt_path))

trio.run(train)
```

## Backend Comparison Matrix

| Backend | Paradigm | Complexity | Maturity | Use Case |
|---------|----------|------------|----------|----------|
| **D6v1: PyTorch** | OOP, stateful | Low | Stable | Default, production |
| **D6v2: torch.func** | Functional | Medium | Stable (PyTorch 2.0+) | Bridge to JAX patterns |
| **D6v3: JAX** | Functional | Medium-High | Stable | TPU, XLA, max performance |
| **D6v4: torchax** | PyTorch syntax, JAX runtime | Low | **Experimental (v0.0.4)** | TPU with PyTorch code |

## D6v2: TorchFuncTrainingBackend (Future)

### Overview

Uses `torch.func` (formerly functorch) with `torchopt` for JAX-style functional training in PyTorch.

**Benefits**:
- Functional programming (same patterns as JAX)
- Stays in PyTorch ecosystem (models, checkpoints work)
- Enables `vmap` for batching, `grad` for autodiff
- Easy migration path to JAX (same mental model)

**Dependencies**:
- `torch >= 2.0` (torch.func is built-in)
- `torchopt` (functional optimizers)

### Key Differences from Standard PyTorch

```python
# Standard PyTorch (D6v1)
class PyTorchTrainingBackend:
    model: torch.nn.Module  # Stateful
    optimizer: torch.optim.Optimizer  # Stateful

# torch.func (D6v2)
class TorchFuncTrainingBackend:
    model: torch.nn.Module  # Used as template
    params: tuple  # Extracted parameters (immutable)
    opt_state: Any  # Optimizer state (immutable)
    optimizer: torchopt.Optimizer  # Functional optimizer
```

### Implementation Plan

```python
import torch
import torchopt
from torch.func import functional_call, grad_and_value

@dataclass
class TorchFuncTrainingBackend:
    """Functional PyTorch training backend (D6v2).

    JAX-style functional training using torch.func + torchopt.
    """

    model: torch.nn.Module
    params: tuple  # Immutable parameters
    buffers: dict  # Buffers (BatchNorm, etc.)
    optimizer: torchopt.Optimizer  # Functional optimizer
    opt_state: Any  # Optimizer state (immutable)
    loss_fn: Callable
    checkpoint_dir: Path

    weight_version: int = 0
    current_step: int = 0

    @classmethod
    def from_model(
        cls,
        model: torch.nn.Module,
        optimizer_fn: Callable,  # e.g., torchopt.adam(lr=1e-4)
        loss_fn: Callable,
        checkpoint_dir: Path,
    ):
        """Create from PyTorch model (extracts functional params)."""
        # Extract parameters as immutable tuple
        func_model, params, buffers = functorch.make_functional_with_buffers(model)

        # Initialize optimizer state
        optimizer = optimizer_fn
        opt_state = optimizer.init(params)

        return cls(
            model=model,
            params=params,
            buffers=buffers,
            optimizer=optimizer,
            opt_state=opt_state,
            loss_fn=loss_fn,
            checkpoint_dir=checkpoint_dir,
        )

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients (functional style).

        Uses torch.func.grad_and_value for JAX-style autodiff.
        """
        def loss_fn_wrapper(params):
            # Functional model call (no state mutation)
            logits = functional_call(
                self.model,
                (params, self.buffers),
                (batch["input_ids"],)
            )
            return self.loss_fn(logits, batch["labels"], batch["loss_mask"])

        # JAX-style: grad_and_value returns (gradients, loss)
        grads, loss = grad_and_value(loss_fn_wrapper)(self.params)

        # Functional optimizer update (immutable)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.params = torchopt.apply_updates(self.params, updates)

        # Compute grad norm
        grad_norm = sum(g.norm().item()**2 for g in grads) ** 0.5

        return TrainFuture(result={"loss": loss.item(), "grad_norm": grad_norm})
```

### Estimated Effort

**~1-2 days**:
- Functional param extraction (~2 hours)
- torchopt integration (~3 hours)
- Checkpointing (reuse PyTorch format) (~1 hour)
- Testing (~2 hours)

## D6v3: JAXTrainingBackend (Future)

### Key Differences from PyTorch

| Aspect | PyTorch | JAX |
|--------|---------|-----|
| **Paradigm** | OOP, stateful | Functional, pure functions |
| **Model State** | `model.parameters()` | Pytree of arrays |
| **Optimizer** | `optimizer.step()` (mutates) | Returns new params (immutable) |
| **Training Step** | `loss.backward()` (implicit) | `jax.grad(loss_fn)` (explicit) |
| **Device** | `.to(device)` | Automatic, or `jax.device_put()` |
| **Checkpointing** | `torch.save(state_dict)` | `orbax-checkpoint` |
| **Compilation** | `torch.compile()` (optional) | `jax.jit()` (expected) |

### Implementation Plan

```python
@dataclass
class JAXTrainingBackend:
    """Raw JAX training backend (D6v2).

    Features:
    - Pure functional training (JAX-native)
    - JIT-compiled train step
    - Immutable state management
    - Orbax checkpointing
    """

    apply_fn: Callable  # Pure function: (params, x) -> logits
    params: PyTree      # Immutable parameter tree
    opt_state: PyTree   # Optimizer state (immutable)
    optimizer: optax.GradientTransformation  # Stateless transform
    loss_fn: Callable
    checkpoint_dir: Path

    # State (immutable updates)
    weight_version: int = 0
    current_step: int = 0

    # JIT-compiled train step
    _train_step_fn: Optional[Callable] = None

    def __post_init__(self):
        """Initialize JIT-compiled train step."""
        @jax.jit
        def train_step(params, opt_state, batch):
            def loss_fn_wrapper(params):
                logits = self.apply_fn(params, batch["input_ids"])
                return self.loss_fn(logits, batch["labels"], batch["loss_mask"])

            loss, grads = jax.value_and_grad(loss_fn_wrapper)(params)
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            grad_norm = jax.tree_util.tree_reduce(
                lambda acc, x: acc + jnp.sum(x**2),
                grads,
                0.0
            ) ** 0.5

            return params, opt_state, {"loss": loss, "grad_norm": grad_norm}

        self._train_step_fn = train_step

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Compute loss and gradients using JAX.

        Note: In JAX, forward and backward are combined (jax.value_and_grad).
        This matches the PyTorch API but runs the full train step.

        Returns new params and opt_state (immutable update).
        """
        ...

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """Apply gradients (no-op in JAX, already done in forward_backward).

        This is a compatibility shim for the protocol. In JAX, the optimizer
        update happens in the same JIT-compiled function as the gradient
        computation.
        """
        ...

    async def save_checkpoint(self, step: int, metrics: Dict[str, float] = {}) -> Path:
        """Save checkpoint using Orbax.

        Increments weight_version (same as PyTorch).
        """
        ...

    async def load_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Load checkpoint using Orbax."""
        ...
```

### Challenges with JAX

1. **Functional State Management** (Medium Pain):
   - JAX requires immutable updates: `params, opt_state = train_step(params, opt_state, batch)`
   - Need to update `self.params` and `self.opt_state` on each step
   - Must be careful not to mutate state

2. **JIT Compilation** (Low Pain):
   - `jax.jit` expects pure functions
   - Pre-compile train step in `__post_init__`
   - Cache compiled function

3. **Async + JAX** (Medium Pain):
   - JAX training is typically synchronous
   - Futures still work (run JIT function in thread)
   - But JAX JIT cache is global (potential race conditions)

4. **Checkpointing** (Medium Pain):
   - Use `orbax-checkpoint` library
   - Different format from PyTorch (but similar API)
   - Metadata still stored as JSON

5. **Protocol Mismatch** (Low Pain):
   - `forward_backward()` and `optim_step()` don't map 1:1 to JAX
   - In JAX, both happen in single JIT-compiled function
   - Can make `optim_step()` a no-op (already done in forward_backward)

### Estimated Effort

**D6v2 (JAX backend): ~1-2 days work**

- Functional state management (~4 hours)
- JIT-compiled train step (~2 hours)
- Orbax checkpointing (~3 hours)
- Testing with async/futures (~2 hours)
- Documentation and examples (~1 hour)

### Decision

**Proceed with PyTorch-only for D6v1**. Keep JAX support as future option (D6v2).

Protocol is already backend-agnostic, so JAX can be added later without breaking changes.

## Acceptance Criteria (D6v1)

- [ ] `PyTorchTrainingBackend` implements `TrainingBackend` protocol
- [ ] Future-based API: `forward_backward()`, `optim_step()` return `TrainFuture[T]`
- [ ] Weight version tracking: `self.weight_version` increments on `save_checkpoint()`
- [ ] Simple checkpoint format: `step_N/{pytorch_model.bin, optimizer.bin, metadata.json}`
- [ ] Checkpoint save/load: `save_checkpoint()`, `load_checkpoint()`
- [ ] User-provided loss function: `loss_fn(logits, labels, loss_mask) -> loss`
- [ ] Tiger Style: Assert preconditions, explicit state
- [ ] Tinker: Minimal surface area, future-based pipelining
- [ ] Example/test demonstrating full training loop
- [ ] Integration with D5 weight sync (decoupled, user-orchestrated)

## D6v4: TorchaxTrainingBackend (Experimental)

### Overview

Uses `torchax` to write PyTorch code that runs on JAX runtime. Relocated to https://github.com/google/torchax as of Oct 2025.

**Current Status**: **v0.0.4 (Experimental)** - NOT recommended for production

**Benefits**:
- Write PyTorch code, get JAX performance/TPU support
- "Best of both worlds" - familiar API, XLA backend
- Automatic TPU support

**Risks**:
- Very early stage (v0.0.4)
- Recently relocated repo (Oct 2025)
- Unknown production readiness
- Might hit bugs/unsupported operations

### Implementation Plan

```python
import torch
import torchax

@dataclass
class TorchaxTrainingBackend:
    """torchax training backend (D6v4 - EXPERIMENTAL).

    PyTorch syntax running on JAX/XLA backend.
    """

    model: torch.nn.Module  # PyTorch model (runs on JAX!)
    optimizer: torch.optim.Optimizer  # PyTorch optimizer (runs on JAX!)
    loss_fn: Callable
    checkpoint_dir: Path

    weight_version: int = 0
    current_step: int = 0

    def __post_init__(self):
        """Move model to JAX device via torchax."""
        # torchax translates PyTorch ops to JAX under the hood
        # User writes: model.to("jax")
        assert str(next(self.model.parameters()).device) == "jax", \
            "Model must be on 'jax' device for torchax backend"

    def forward_backward(self, batch: Dict[str, Any]) -> TrainFuture[Dict[str, float]]:
        """Standard PyTorch training step (runs on JAX runtime).

        torchax automatically translates PyTorch ops to JAX.
        """
        self.optimizer.zero_grad()

        # PyTorch code, JAX execution!
        logits = self.model(batch["input_ids"])
        loss = self.loss_fn(logits, batch["labels"], batch["loss_mask"])

        # PyTorch backward, JAX autodiff!
        loss.backward()

        # Compute grad norm
        grad_norm = sum(
            p.grad.norm().item()**2
            for p in self.model.parameters()
            if p.grad is not None
        ) ** 0.5

        return TrainFuture(result={"loss": loss.item(), "grad_norm": grad_norm})

    def optim_step(self) -> TrainFuture[Dict[str, float]]:
        """PyTorch optimizer step (runs on JAX)."""
        self.optimizer.step()
        self.current_step += 1

        return TrainFuture(result={
            "lr": self.optimizer.param_groups[0]["lr"],
            "step": self.current_step,
        })
```

### Usage

```python
import torch
import torchax

# Standard PyTorch model
model = GPT(config)

# Move to JAX device (torchax magic!)
model = model.to("jax")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

backend = TorchaxTrainingBackend(
    model=model,
    optimizer=optimizer,
    loss_fn=loss_fn,
    checkpoint_dir=Path("/ckpts"),
)

# Standard PyTorch training loop, runs on JAX/TPU!
await backend.forward_backward(batch).result()
await backend.optim_step().result()
```

### Estimated Effort

**~1 day** (if it works, ~1 week if debugging edge cases):
- Model device handling (~1 hour)
- Testing what ops are supported (~4 hours)
- Checkpointing (should work via PyTorch) (~1 hour)
- Debugging unsupported ops (~varies, potentially days)

### Recommendation

**Skip for now, revisit in 6-12 months**:
- Too early (v0.0.4)
- Unknown stability
- Better to use torch.func (D6v2) as bridge, then JAX (D6v3) if needed

## Future Work (D6v2+)

- [ ] D6v2: torch.func + torchopt backend (functional PyTorch)
- [ ] D6v3: JAX backend implementation (raw JAX, JIT-compiled)
- [ ] D6v4: torchax backend (experimental, revisit when mature)
- [ ] Gradient accumulation support (micro-batching)
- [ ] Mixed precision training (AMP for PyTorch, JAX native)
- [ ] Distributed training (FSDP for PyTorch, pjit for JAX)
- [ ] HuggingFace checkpoint format conversion
- [ ] TensorBoard/WandB logging integration
- [ ] Profiling and performance optimization
