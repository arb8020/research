# Classes vs Functional: When to Use What

> **The Rule:** If you can make it a pure function, make it a pure function. Only use a class when you have legitimate persistent state.

---

## The Decision Tree *(Rollouts + Miniray + Casey + Tiger Style)*

**Classes are for legitimate, persistent state that needs encapsulation.**
**Functions are for computation and transformation.**

From ROLLOUTS_ANALYSIS.md:
> "Classes only for legitimate state: DataBuffer (prompt iteration), AsyncRolloutManager (async coordination), PyTorchTrainingBackend (model/optimizer state). Pure functions for: training loops, batch preparation, loss computation, reward functions."

**Use a class when you have:**
1. **State that must persist across multiple operations**
2. **State that needs cleanup/lifecycle management**
3. **State that benefits from encapsulation**

---

## Use Frozen Dataclasses For: Data & Configuration

**When:** You need to group related data that doesn't change.

```python
# GOOD - frozen dataclass for config
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration data - immutable, serializable."""
    learning_rate: float
    batch_size: int
    num_epochs: int
    model_name: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**d)

# Used like immutable data
config = TrainingConfig(learning_rate=0.001, batch_size=32, num_epochs=10, model_name="gpt-4")
# config.learning_rate = 0.01  # Error! Can't mutate frozen dataclass
```

**From rollouts/config/base.py:**
- `BaseModelConfig` - API endpoint config
- `BaseEnvironmentConfig` - environment settings
- `BaseEvaluationConfig` - evaluation parameters
- All frozen, all serializable, all immutable

**Benefits:**
- Type-safe (mypy validates)
- Serializable (JSON in/out for free)
- Hashable (can use in sets/dicts)
- Thread-safe (no synchronization needed)
- Clear boundaries (can't leak mutations)

---

## Use Regular Classes For: Stateful Resources

**When:** You have state that persists and changes across operations.

```python
# GOOD - class for stateful coordination
class AsyncRolloutManager:
    """Manages async rollout generation.

    Legitimate use of class:
    - Maintains async event loop
    - Manages queue of pending requests
    - Tracks in-flight generations
    - Needs cleanup (shutdown)
    """

    def __init__(self, endpoint: Endpoint, max_concurrent: int):
        self.endpoint = endpoint
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._tasks: list[asyncio.Task] = []

    async def generate_batch(self, prompts: list[str]) -> list[str]:
        """State changes: tracks tasks, manages semaphore."""
        async with self._semaphore:
            # ... async generation logic
            pass

    async def shutdown(self):
        """Cleanup: cancel pending tasks."""
        for task in self._tasks:
            task.cancel()

# From miniray/worker.py
@dataclass
class Worker:
    """Manages process lifecycle and IPC.

    Legitimate use of class:
    - Maintains socket connection
    - Owns child process PID
    - Needs cleanup (wait/kill)
    """
    pid: int
    _sock: socket.socket
    r: Any
    w: Any

    def recv(self, max_size: int) -> Any:
        # Reads from socket (stateful I/O)
        pass

    def send(self, msg: Any) -> None:
        # Writes to socket (stateful I/O)
        pass

    def wait(self) -> None:
        # Cleanup: wait for child to exit
        os.waitpid(self.pid, 0)
```

**Examples from rollouts/miniray:**
- `Worker` - manages process + socketpair
- `DataBuffer` - iterates through prompts with state
- `PyTorchTrainingBackend` - owns model/optimizer/scaler state
- `AsyncRolloutManager` - coordinates async operations

**When to use a class:**
1. **Resource ownership** (sockets, files, processes)
2. **Lifecycle management** (needs `__init__`/cleanup)
3. **Coordinated mutation** (multiple methods update same state)
4. **Async coordination** (event loops, queues)

---

## Use Pure Functions For: Everything Else

**When:** Computation, transformation, no persistent state needed.

```python
# GOOD - pure functions for training loops
def sft_training_step(
    backend: PyTorchTrainingBackend,
    batch: SFTBatch,
    config: TrainingConfig,
) -> tuple[Metrics, str | None]:
    """Pure function: explicit inputs/outputs, no hidden state.

    Args:
        backend: Training backend (has model/optimizer state)
        batch: Current batch
        config: Hyperparameters

    Returns:
        (metrics, error): Explicit success/failure
    """
    # Compute loss
    loss, err = backend.compute_loss(batch)
    if err:
        return None, f"Loss computation failed: {err}"

    # Update weights
    backend.optimizer.zero_grad()
    loss.backward()
    backend.optimizer.step()

    # Return metrics
    metrics = Metrics(loss=loss.item(), step=backend.step)
    return metrics, None


# GOOD - pure function for batch preparation
def prepare_sft_batch(
    samples: list[ConversationSample],
    tokenizer: Tokenizer,
    max_length: int,
) -> SFTBatch:
    """Pure function: transforms data, no side effects."""
    input_ids = []
    labels = []

    for sample in samples:
        tokens = tokenizer.encode(sample.text)
        input_ids.append(tokens[:max_length])
        labels.append(tokens[:max_length])

    return SFTBatch(
        input_ids=torch.tensor(input_ids),
        labels=torch.tensor(labels),
    )


# GOOD - pure function for loss computation
def compute_grpo_loss(
    logprobs: Tensor,
    ref_logprobs: Tensor,
    advantages: Tensor,
    beta: float,
) -> Tensor:
    """Pure function: math, no state."""
    ratio = torch.exp(logprobs - ref_logprobs)
    policy_loss = -(ratio * advantages).mean()
    kl_penalty = beta * (logprobs - ref_logprobs).mean()
    return policy_loss + kl_penalty
```

**From rollouts/training/:**
- `sft_loop.py` - pure function training loop
- `rl_loop.py` - pure function RL training
- `sft.py` - pure functions for batch prep
- `rl_losses.py` - pure math functions
- `rollout_generation.py` - pure generation functions

**When to use pure functions:**
1. **Stateless computation** (math, transforms)
2. **Batch preparation** (data → tensors)
3. **Training loops** (orchestrate stateful objects)
4. **Loss functions** (pure math)
5. **Validation/metrics** (aggregate results)

---

## The Pattern: Functions Orchestrate Objects

**Key insight:** Pure functions call methods on stateful objects.

```python
# PATTERN: Function uses stateful objects, but is itself pure
async def run_sft_training(
    config: TrainingConfig,
    backend: PyTorchTrainingBackend,  # Stateful
    data_buffer: DataBuffer,          # Stateful
) -> TrainingResult:
    """Pure function that orchestrates stateful objects.

    - config: immutable data
    - backend: owns model/optimizer (stateful)
    - data_buffer: owns dataset iteration (stateful)

    But the FUNCTION is pure:
    - Explicit inputs/outputs
    - No hidden state
    - Deterministic given same inputs
    """
    metrics_history = []

    for step in range(config.num_steps):
        # Get batch from stateful buffer
        batch = data_buffer.next_batch()

        # Run step using stateful backend
        metrics, err = sft_training_step(backend, batch, config)
        if err:
            return None, err

        metrics_history.append(metrics)

    return TrainingResult(metrics=metrics_history), None
```

**The principle:**
- **Objects encapsulate state** (backend owns model, buffer owns iteration)
- **Functions orchestrate** (training loop coordinates objects)
- **Control flow is explicit** (you see every state change)

**From rollouts/:**
```python
# Objects (stateful)
backend = PyTorchTrainingBackend(model, optimizer, device)
data_buffer = DataBuffer(dataset, batch_size)
manager = AsyncRolloutManager(endpoint, max_concurrent=4)

# Functions orchestrate (pure)
result, err = run_sft_training(config, backend, data_buffer)
trajectories, err = generate_rollouts(manager, prompts, config)
loss = compute_grpo_loss(logprobs, ref_logprobs, advantages, beta)
```

---

## Anti-Patterns to Avoid

### ❌ BAD: Class when you should use frozen dataclass
```python
# DON'T DO THIS
class Config:
    def __init__(self, lr, batch_size):
        self.lr = lr  # Mutable! Can change unexpectedly
        self.batch_size = batch_size

    def set_lr(self, lr):
        self.lr = lr  # Hidden state change

# DO THIS INSTEAD
@dataclass(frozen=True)
class Config:
    lr: float
    batch_size: int

    # To "change" config, create new one:
    # new_config = replace(old_config, lr=0.001)
```

### ❌ BAD: Class when you should use function
```python
# DON'T DO THIS
class LossComputer:
    def __init__(self, beta):
        self.beta = beta  # Why is this instance state?

    def compute(self, logprobs, ref_logprobs, advantages):
        # Pure computation disguised as a class
        ratio = torch.exp(logprobs - ref_logprobs)
        return -(ratio * advantages).mean() + self.beta * (logprobs - ref_logprobs).mean()

# DO THIS INSTEAD
def compute_grpo_loss(logprobs, ref_logprobs, advantages, beta):
    ratio = torch.exp(logprobs - ref_logprobs)
    return -(ratio * advantages).mean() + beta * (logprobs - ref_logprobs).mean()
```

### ❌ BAD: Mutable state in function
```python
# DON'T DO THIS
_global_metrics = []  # Hidden global state

def training_step(batch):
    loss = compute_loss(batch)
    _global_metrics.append(loss.item())  # Hidden side effect
    return loss

# DO THIS INSTEAD
def training_step(batch) -> tuple[Tensor, float]:
    loss = compute_loss(batch)
    metric = loss.item()
    return loss, metric  # Explicit output
```

### ❌ BAD: God object with unrelated state
```python
# DON'T DO THIS
class Trainer:
    def __init__(self):
        self.model = None
        self.optimizer = None
        self.dataset = None
        self.metrics = []
        self.config = {}
        self.checkpoints = []
        # Too much unrelated state!

    def do_everything(self):
        pass  # 500 lines of intermingled logic

# DO THIS INSTEAD
@dataclass(frozen=True)
class TrainingConfig: ...

class PyTorchBackend:
    # ONLY model/optimizer state
    pass

class DataBuffer:
    # ONLY dataset iteration
    pass

# Pure function orchestrates
def run_training(config, backend, data_buffer):
    # Explicit, clear separation
    pass
```

---

## Summary: The Decision Matrix

| Use Case | Pattern | Example |
|----------|---------|---------|
| **Configuration/Data** | `@dataclass(frozen=True)` | `TrainingConfig`, `Endpoint`, `Metrics` |
| **Resource Ownership** | Regular class | `Worker`, `AsyncRolloutManager` |
| **State + Lifecycle** | Regular class | `PyTorchTrainingBackend`, `DataBuffer` |
| **Computation** | Pure function | `compute_loss()`, `prepare_batch()` |
| **Orchestration** | Pure function | `run_training()`, `sft_training_step()` |
| **Transformations** | Pure function | `tokenize()`, `normalize()`, `filter_valid()` |

**The test:**
1. **Does it own a resource?** → Class (e.g., socket, process, file)
2. **Does it need cleanup?** → Class (e.g., `shutdown()`, `close()`)
3. **Is it just data?** → Frozen dataclass (e.g., config, metrics)
4. **Is it computation?** → Pure function (e.g., loss, transform)
5. **Does it orchestrate?** → Pure function calling objects

**When in doubt, start with a function. Upgrade to a class only when you have legitimate persistent state.**
