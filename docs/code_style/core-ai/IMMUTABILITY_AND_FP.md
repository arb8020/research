# Immutability & Functional Style

> **Core Insight:** Immutability makes state boundaries explicit. You can see where state lives, when it changes, what changed, and who can change it.

---

## The Principle

**Frozen Dataclasses + Pure Functions** *(Joe Duffy Midori + Tiger Style + Carmack SSA)*

> "All statics were immutable... This meant that static values were evaluated at compile-time, written to the readonly segment of the resulting binary image, and shared across all processes. More importantly for code quality, all runtime initialization checks were removed... a 10% reduction in code size." *(Joe Duffy)*

**The connection to other patterns:**
- **Tuple returns** = return new state, don't mutate
- **SSA** = don't reassign, create new bindings
- **Push ifs up, fors down** = centralize mutation, keep helpers pure
- **Minimize stateful components** = state is visible and contained

---

## The Pattern

```python
# GOOD - frozen dataclass, functional style
@dataclass(frozen=True)
class Config:
    learning_rate: float
    batch_size: int
    model_size: int

def train(config: Config, data: Dataset) -> tuple[Model, Metrics]:
    """Pure function: no mutation, explicit inputs/outputs."""
    # Can't mutate config - it's frozen
    model = build_model(config.model_size)
    metrics = run_training(model, data, config)
    return model, metrics

def adjust_learning_rate(config: Config, new_lr: float) -> Config:
    """Returns NEW config, original unchanged."""
    return replace(config, learning_rate=new_lr)

# State changes are explicit
config1 = Config(learning_rate=0.01, batch_size=32, model_size=1024)
config2 = adjust_learning_rate(config1, 0.001)  # config1 unchanged
model, metrics = train(config2, dataset)
```

---

## Anti-Pattern

```python
# BAD - mutable state, hidden changes
class Trainer:
    def __init__(self):
        self.config = {}  # What's in here?
        self.state = {}   # What's in here?
        self.model = None

    def set_learning_rate(self, lr):
        self.config['lr'] = lr  # When did this change?

    def train(self, data):
        self._update_state()  # What changed? Can't tell from call site
        self._adjust_params() # More hidden mutations
        return self.model

# Hidden state changes everywhere:
trainer = Trainer()
trainer.set_learning_rate(0.01)  # Mutation 1
trainer.train(data)               # Mutations 2, 3, 4... who knows?
```

---

## Why Frozen Dataclasses

1. **Explicit state** - all fields visible in definition
2. **No hidden mutations** - can't change after creation
3. **Serializable** - JSON in/out for free
4. **Type safe** - mypy checks fields
5. **Hash stable** - can use in sets/dicts safely
6. **Thread safe** - no synchronization needed
7. **Debuggable** - state never changes under you

---

## Pure Functions + Frozen Data

```python
# Helpers are PURE - no mutation, no side effects
def validate_config(config: Config) -> list[str]:
    """Returns errors, doesn't mutate anything."""
    errors = []
    if config.learning_rate <= 0:
        errors.append("learning_rate must be positive")
    if config.batch_size <= 0:
        errors.append("batch_size must be positive")
    return errors

def scale_batch_size(config: Config, factor: int) -> Config:
    """Returns NEW config with scaled batch size."""
    return replace(config, batch_size=config.batch_size * factor)

# Composition is clean
config = Config(learning_rate=0.01, batch_size=32, model_size=1024)
errors = validate_config(config)
if not errors:
    large_batch_config = scale_batch_size(config, 4)
    # config is unchanged, large_batch_config is new
```

---

## Ray Design Connection

```python
# From ray_design.txt:
# 1. Use Protocols, Not Concrete Classes
# 2. Async by Default
# 3. Message Passing, Not Shared Memory
# 4. Dependency Injection
# 5. Serializable Configuration ← frozen dataclasses!
# 6. Abstract Storage

@dataclass(frozen=True)
class TrainingConfig:
    model: ModelConfig
    optimizer: OptimizerConfig
    data: DataConfig

    def to_dict(self) -> dict:
        """Serialize for Ray workers."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Deserialize from Ray workers."""
        return cls(**d)

# Ray can serialize this across workers
# No mutable shared state
# Message passing instead of mutation
```

---

## Midori's Lessons

- **No mutable statics in user code** - eliminated 10% code size
- **Structs immutable by default** - `immutable struct S { ... }`
- **Static values in readonly segment** - shared across processes
- **No defensive copies needed** - immutability removes need

---

## Tiger Style Connection

```python
# Tiger Style: "Keep leaf functions pure"
# Helpers compute, don't mutate. Parent manages state.

def process_batch(config: Config, batch: Batch) -> ProcessedBatch:
    """Parent function - manages state flow."""

    # Call pure helpers
    normalized = normalize_inputs(batch.inputs)
    augmented = apply_augmentation(normalized, config.augment_params)
    encoded = encode_features(augmented)

    # All helpers are pure - return new data, don't mutate
    return ProcessedBatch(
        inputs=encoded,
        labels=batch.labels,
        metadata=batch.metadata,
    )

# Pure helpers - no mutation, no side effects
def normalize_inputs(inputs: Array) -> Array:
    mean = inputs.mean()
    std = inputs.std()
    return (inputs - mean) / std

def apply_augmentation(data: Array, params: AugmentParams) -> Array:
    if params.flip:
        data = flip(data)
    if params.rotate:
        data = rotate(data, params.angle)
    return data  # Returns new array
```

---

## The Throughline

**Immutability = Explicit State Boundaries**

When data is immutable:
- You see state creation (`config = Config(...)`)
- You see state transformation (`new_config = replace(old_config, ...)`)
- You see what changed (only the fields you specified)
- You see what stayed the same (everything else)
- Helpers can't surprise you with mutations
- Debugging is deterministic (state doesn't change under you)
- Tests are reliable (no hidden state between tests)

**This is the same pattern as:**
- **SSA**: `filtered_data = filter(raw_data)` not `data = filter(data)`
- **Tuple returns**: `(result, err)` not `throw Exception()`
- **Pure helpers**: Functions compute, don't mutate
- **Minimize state**: Stateless services > stateful services

**All say: Make state changes visible and explicit.**
