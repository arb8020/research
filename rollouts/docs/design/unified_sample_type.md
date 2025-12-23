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

## Phase 2: Unified Message and Endpoint Types

### Context

Two more type duplications following the same pattern as Sample/EvalSample:

**Message vs SessionMessage:**
- `Message`: runtime type with `content: str | list[ContentBlock]`, provider metadata
- `SessionMessage`: serializable for sessions, `content: str | list[dict]`, adds `timestamp`
- Conversion via `_message_to_session_message()` in agents.py

**Endpoint vs EndpointConfig:**
- `Endpoint`: runtime type with api_key, oauth_token, `thinking: dict | None`
- `EndpointConfig`: serializable for sessions, no secrets, `thinking: bool` + `thinking_budget: int`
- Lossy conversion - `thinking: dict` is more expressive than `bool + int`

### Solution

Same pattern as Sample: overcomplete types with optional fields.

**Message:**
- Add `timestamp: str | None = None` field
- Delete `SessionMessage` entirely
- `AgentSession.messages: list[Message]` - keeps type safety

**Endpoint:**
- Already has all fields (overcomplete)
- Delete `EndpointConfig` entirely
- Add `to_dict(exclude_secrets=True)` method
- Add `from_dict(data, api_key="", oauth_token="")` class method
- `AgentSession.endpoint: Endpoint` with secrets injected at runtime

### Why overcomplete > separate types

1. **Consistent with Sample refactor** - one type with optional fields
2. **Type safety preserved** - `AgentSession.messages: list[Message]` not `list[dict]`
3. **No lossy conversion** - `thinking: dict` stays as dict, not collapsed to bool
4. **Casey principle** - don't create types when methods suffice

### Why not just to_dict methods?

If we used `AgentSession.messages: list[dict]`, callers lose type safety and must remember to call `Message.from_dict()`. Overcomplete types let us keep `list[Message]` everywhere.

### Implementation

**Message changes:**
```python
@dataclass(frozen=True)
class Message(JsonSerializable):
    role: str
    content: str | list[ContentBlock] | None
    provider: str | None = None
    api: str | None = None
    model: str | None = None
    tool_call_id: str | None = None
    details: dict[str, Any] | None = None
    timestamp: str | None = None  # NEW: for session storage
```

**Endpoint changes:**
```python
@dataclass(frozen=True)
class Endpoint(JsonSerializable):
    # ... existing fields ...

    def to_dict(self, exclude_secrets: bool = True) -> dict[str, Any]:
        """Serialize. If exclude_secrets=True, omits api_key/oauth_token."""
        d = asdict(self)
        if exclude_secrets:
            d.pop("api_key", None)
            d.pop("oauth_token", None)
        return d

    @classmethod
    def from_dict(cls, data: dict, api_key: str = "", oauth_token: str = "") -> Endpoint:
        """Deserialize, injecting secrets at runtime."""
        return cls(**data, api_key=api_key, oauth_token=oauth_token)
```

**AgentSession changes:**
```python
@dataclass
class AgentSession:
    # ...
    endpoint: Endpoint  # was EndpointConfig
    messages: list[Message]  # was list[SessionMessage]
```

### Files to modify
- `rollouts/dtypes.py` - Add timestamp to Message, add to_dict/from_dict to Endpoint, delete SessionMessage/EndpointConfig
- `rollouts/agents.py` - Delete `_message_to_session_message()`, use Message directly
- `rollouts/store.py` - Update to use Message/Endpoint
- `rollouts/__init__.py` - Remove SessionMessage/EndpointConfig exports

### Breaking changes
- `SessionMessage` deleted - use `Message` with optional `timestamp`
- `EndpointConfig` deleted - use `Endpoint` with `to_dict(exclude_secrets=True)`
- `AgentSession.messages` type changes from `list[SessionMessage]` to `list[Message]`
- `AgentSession.endpoint` type changes from `EndpointConfig` to `Endpoint`

---

## Future Cleanup (Out of Scope)

### BaseModelConfig (config/base.py)
- Has `api_key_env_var`, `to_endpoint()`
- Different purpose: user-facing config that loads secrets from env
- Keep separate - it's a config helper, not a duplicate type

### EnvironmentConfig
- Just `type: str` + `config: dict`
- Could delete, use dict directly
- Low priority
