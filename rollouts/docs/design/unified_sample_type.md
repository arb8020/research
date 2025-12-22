# Unified Sample Type

**DRI:**
**Claude:** [this conversation]

## Context
Unify `Sample` and `EvalSample` into one overcomplete type so eval, training, and SFT trace generation share the same data structure.

## Out of Scope
- Changing `run_agent()` internals
- Config consolidation (EvalConfig, GRPOConfig, RunConfig) - separate refactor
- Provider/endpoint unification

## Solution
**Input:** Add `trajectory: Trajectory | None` and `score: Score | None` to `Sample`
**Output:** Single `Sample` type flows through entire pipeline; delete `EvalSample`

## Usage

```python
# ═══════════════════════════════════════════════════════════════════════
# SCORE FUNCTION - unified signature for eval and training
# ═══════════════════════════════════════════════════════════════════════

from rollouts import Sample, Score, Metric

def score_fn(sample: Sample) -> Score:
    """Works for both evaluation and training."""
    # Multi-turn: extract from trajectory
    if sample.trajectory:
        answer = extract_final_answer(sample.trajectory)
    # Single-turn: use response directly
    else:
        answer = sample.response

    correct = answer.strip() == sample.ground_truth
    return Score(metrics=(Metric("correct", float(correct), weight=1.0),))


# ═══════════════════════════════════════════════════════════════════════
# EVALUATION - sample.trajectory populated by run_agent()
# ═══════════════════════════════════════════════════════════════════════

from rollouts import evaluate, EvalConfig

config = EvalConfig(
    endpoint=my_endpoint,
    score_fn=score_fn,  # Same signature as training
    prepare_messages=lambda s: [Message(role="user", content=s["question"])],
    environment_factory=lambda s: CalculatorEnvironment(),
)

report = await evaluate(dataset, config)
# report.samples[0].trajectory  <- full execution trace
# report.samples[0].score       <- computed score
# report.samples[0].reward      <- score.reward (convenience)


# ═══════════════════════════════════════════════════════════════════════
# TRAINING - sample.trajectory available, tokens/loss_mask populated
# ═══════════════════════════════════════════════════════════════════════

from rollouts.training import grpo_train, GRPOConfig

config = GRPOConfig(model_name="Qwen/Qwen3-0.6B", num_steps=100)
prompts = [{"messages": [...], "answer": "42"}, ...]

results = grpo_train(
    config=config,
    prompts=prompts,
    score_fn=score_fn,  # Same function as evaluation!
    environment_cls=BasicEnvironment,
)


# ═══════════════════════════════════════════════════════════════════════
# SFT TRACE GENERATION - get sample with trajectory + tokens
# ═══════════════════════════════════════════════════════════════════════

from rollouts.training import agent_rollout_to_sample

sample = await agent_rollout_to_sample(
    prompt="What is 5 + 3?",
    environment_cls=CalculatorEnvironment,
    endpoint=endpoint,
    tokenizer=tokenizer,
)
# sample.trajectory  <- full multi-turn trace
# sample.tokens      <- tokenized for training
# sample.loss_mask   <- 1.0 for assistant, 0.0 for user/tool
```

---

## Details

### Type Changes

```python
# training/types.py - MODIFY Sample

@dataclass
class Sample:
    """Unified sample for evaluation, rollouts, and training.

    Overcomplete: optional fields for different use cases.
    - Evaluation: trajectory, score, ground_truth populated
    - Training: tokens, loss_mask, reward populated
    - Both: all fields available
    """

    # Identity
    id: str = ""
    index: int | None = None
    group_index: int | None = None  # GRPO grouping

    # Input
    input: dict[str, Any] = field(default_factory=dict)
    prompt: str | list[dict[str, str]] = ""
    ground_truth: Any | None = None

    # Generated
    trajectory: Trajectory | None = None  # Full execution trace (multi-turn)

    # Training-specific (optional, populated for RL/SFT)
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[float] = field(default_factory=list)
    reward: float = 0.0
    rollout_log_probs: list[float] | None = None  # Off-policy correction

    # Evaluation-specific (optional)
    score: Score | None = None

    # Status
    status: Status = Status.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response(self) -> str:
        """Extract final assistant response from trajectory."""
        if not self.trajectory or not self.trajectory.messages:
            return ""
        for msg in reversed(self.trajectory.messages):
            if msg.role == "assistant":
                if isinstance(msg.content, str):
                    return msg.content
                # Handle content blocks
                return "".join(
                    block.text for block in msg.content
                    if hasattr(block, "text")
                )
        return ""
```

```python
# evaluation.py - DELETE EvalSample entirely
# evaluate() returns EvalReport with samples: list[Sample]
```

### Flow

1. `run_agent()` produces `AgentState` with `actor.trajectory`
2. Convert to `Sample(trajectory=trajectory, input=input, ground_truth=gt)`
3. Call `score_fn(sample) -> Score`
4. Set `sample.score = score` and `sample.reward = score.reward`
5. For training: `sample.tokens`, `sample.loss_mask` also populated by tokenizer

### Implementation

1. Add `trajectory: Trajectory | None = None` and `score: Score | None = None` to `Sample`
2. Add `response` property that extracts from trajectory
3. Delete `EvalSample` class
4. Update `EvalReport.samples` type from `list[EvalSample]` to `list[Sample]`
5. Update `evaluate_sample()` to build `Sample` with trajectory, set `sample.score`
6. Change score_fn signature: `(Trajectory, Sample) -> Score` becomes `(Sample) -> Score`
7. Update `agent_rollout_to_sample()` to populate `sample.trajectory`
8. Update all eval configs (gsm8k, browsecomp) to new signature

**Breaking change:** All existing score_fn implementations must update.

### Files
**Modify:**
- `rollouts/training/types.py` - Add trajectory, score fields; add response property
- `rollouts/evaluation.py` - Delete EvalSample, update EvalReport, update score_fn calls
- `rollouts/dtypes.py` - Update EvalConfig.score_fn type hint
- `rollouts/training/agent_integration.py` - Populate sample.trajectory
- `rollouts/examples/eval/gsm8k/base_config.py` - Update score_fn signature
- `rollouts/examples/eval/browsecomp/base_config.py` - Update score_fn signature

---

## Future Cleanup (Out of Scope)

Other type duplications that could be compressed in future refactors:

### Endpoint Types (3 types for same concept)
- `Endpoint` (dtypes.py) - runtime, frozen, has api_key
- `EndpointConfig` (dtypes.py) - serializable for sessions, different thinking field format
- `BaseModelConfig` (config/base.py) - has api_key_env_var, to_endpoint()

Could merge into one `Endpoint` with `to_dict(exclude_secrets=True)`.

### Message Types (2 types)
- `Message` (dtypes.py) - runtime with content blocks
- `SessionMessage` (dtypes.py) - serializable for sessions

Could delete `SessionMessage`, use `Message.to_dict()`/`from_dict()`.

### EnvironmentConfig
- Just `type: str` + `config: dict`
- Could delete, use dict directly
