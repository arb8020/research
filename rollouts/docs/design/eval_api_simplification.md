# Eval API Simplification

**DRI:**
**Claude:** [this conversation]

## Context
Simplify `evaluate()` signature from 6 args to 2 args.

## Out of Scope
- Changing `DataBuffer` / training data loading (already functional, works well)
- Prompt optimization (separate feature, can be added later)
- Changing `run_agent` internals

## Solution
**Input:** `dataset: Iterable[dict]`, `config: EvalConfig`
**Output:** `EvalReport`

## Usage
```python
# Before (6 args)
report = await evaluate(
    dataset=iter(dataset),
    prepare_messages=prepare_messages,
    endpoint=endpoint,
    config=eval_config,
    dataset_path="gsm8k",
    environment_factory=environment_factory,
)

# After (2 args)
def prepare_messages(sample: dict) -> list[Message]:
    return [
        Message(role="system", content="You are a math tutor."),
        Message(role="user", content=sample["question"]),
    ]

config = EvalConfig(
    endpoint=endpoint,
    score_fn=my_score_fn,
    prepare_messages=prepare_messages,
    environment_factory=lambda s: CalculatorEnvironment(),
)

report = await evaluate(dataset, config)
```

---

## Details

### Type Changes

```python
@dataclass(frozen=True)
class EvalConfig:
    # Required
    endpoint: Endpoint
    score_fn: ScoreFn
    prepare_messages: PrepareMessagesFn  # Returns initial trajectory messages

    # Environment
    environment_factory: EnvironmentFactory | None = None

    # Execution
    run_config: RunConfig | None = None  # If None, use silent default
    max_concurrent: int = 1
    max_samples: int | None = None

    # Output
    eval_name: str = "evaluation"
    output_dir: Path | None = None
    verbose: bool = True
    show_progress: bool = False

# Type aliases
PrepareMessagesFn = Callable[[dict[str, Any]], list[Message]]
EnvironmentFactory = Callable[[dict[str, Any]], Awaitable[Environment]]
```

### Flow

1. `evaluate(dataset, config)` called
2. For each sample in dataset:
   - `messages = config.prepare_messages(sample)`
3. If `config.environment_factory`: `env = await config.environment_factory(sample)`
4. Run agent: `states = await run_agent(initial_state, run_config)`
5. Score: `score = config.score_fn(trajectory, sample)`
6. Collect results → `EvalReport`

### Files Modified
- `rollouts/rollouts/dtypes.py` - EvalConfig with prepare_messages required
- `rollouts/rollouts/evaluation.py` - Simplified evaluate() signature
- `rollouts/examples/eval/gsm8k/base_config.py` - Updated usage
- `rollouts/examples/eval/browsecomp/base_config.py` - Updated usage
