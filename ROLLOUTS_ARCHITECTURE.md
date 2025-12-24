# Rollouts Codebase Architecture & Design Guide

## Quick Summary

The rollouts codebase is a **composable agent framework** built on:

1. **Core Types**: Immutable frozen dataclasses (Endpoint, Message, Trajectory, Actor, AgentState, RunConfig, Environment)
2. **Agent Loop**: `run_agent()` → `run_agent_step()` → `rollout()` + `process_pending_tools()`
3. **Evaluation**: `evaluate()` → `evaluate_sample()` with parallel support and composable score functions
4. **Providers**: Multi-provider LLM backend (Anthropic, OpenAI, Google, SGLang)
5. **Training**: GRPO framework with distributed backends
6. **Sessions**: Append-only persistence with branching support

**Design Philosophy**: Full state passing, immutability, Tiger style (crash-loud), composition over inheritance, pure functions.

---

## Directory Structure

```
rollouts/
├── dtypes.py                     # Core types (1398 lines)
│   ├── Endpoint, Message, Trajectory, Actor, AgentState
│   ├── Streaming events (TextDelta, ToolCallEnd, etc.)
│   ├── Score, Metric, Sample
│   ├── Environment Protocol
│   └── EvalConfig, RunConfig
├── agents.py                     # Agent execution (968 lines)
│   ├── run_agent() - Main loop
│   ├── run_agent_step() - Single turn
│   ├── process_pending_tools() - Tool execution
│   ├── rollout() - Provider routing
│   └── Stop/step handlers (composable)
├── evaluation.py                 # Evaluation framework (732 lines)
│   ├── evaluate() - Dataset evaluation
│   ├── evaluate_sample() - Single sample
│   ├── EvalReport, compute_summary_metrics()
│   └── score_fn patterns
├── providers/                    # LLM backends
│   ├── anthropic.py - Claude
│   ├── openai_completions.py - OpenAI
│   ├── google.py - Gemini
│   └── sglang.py - Local inference
├── environments/                 # Tool executors
│   ├── calculator.py, coding.py, git_worktree.py, browsing.py
│   └── Environment Protocol in dtypes.py
├── training/                     # RL training
│   ├── grpo.py - GRPO loop
│   ├── types.py - Sample
│   ├── backends/ - Training backends
│   └── rollout_gen/ - Rollout generation
├── store.py                      # Session persistence
├── config/base.py                # Configuration utilities
├── inference/                    # Local inference engines
└── cli.py                        # Command-line interface
```

---

## Core Types (dtypes.py)

### 1. Endpoint
```python
@dataclass(frozen=True)
class Endpoint:
    provider: str  # "anthropic", "openai", "google", "sglang"
    model: str
    api_key: str = ""
    oauth_token: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    thinking: dict[str, Any] | None = None  # Anthropic extended thinking
    max_retries: int = 3
    timeout: float = 120.0
    extra_params: dict[str, Any] | None = None
```

**Methods**: `to_dict(exclude_secrets=True)`, `from_dict(data, api_key, oauth_token)`

### 2. Message & Content

```python
@dataclass(frozen=True)
class Message:
    role: str  # "user", "assistant", "tool"
    content: str | list[ContentBlock] | None
    provider: str | None = None
    tool_call_id: str | None = None  # For tool responses
    timestamp: str | None = None  # Session persistence
    details: dict[str, Any] | None = None  # UI metadata
    
    def get_tool_calls(self) -> list[ToolCall]:
        """Extract ToolCall from ContentBlocks"""
```

**ContentBlock types**: `TextContent`, `ThinkingContent`, `ToolCallContent`, `ImageContent`

### 3. Trajectory (Execution Trace)

```python
@dataclass
class Trajectory:
    completions: list[ChatCompletion] = []  # API responses
    messages: list[Message] = []            # Chat history
    rewards: float = 0.0                    # RL reward
    metadata: dict[str, Any] = {}           # Dataset metadata
```

**Methods**: `to_jsonl()`, `from_jsonl()`, `save_jsonl()`, `load_jsonl()`, `load_jsonl_streaming()`, `hash()`

### 4. Streaming Events

```python
# High-level
@dataclass(frozen=True)
class LLMCallStart: pass
class StreamStart: pass
class StreamDone: pass
class StreamError: pass

# Content-level (content_index tracks position)
@dataclass(frozen=True)
class TextDelta:
    content_index: int
    delta: str

class ThinkingDelta: pass
class TextEnd: pass
class ThinkingEnd: pass

# Tool-level
@dataclass(frozen=True)
class ToolCallStart:
    content_index: int
    tool_call_id: str
    tool_name: str

class ToolCallDelta: pass  # With partial_args
class ToolCallEnd: pass    # With parsed ToolCall
class ToolCallError: pass
class ToolResultReceived: pass

StreamEvent = Union[...]
```

### 5. Score & Metric (Evaluation)

```python
@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    weight: float = 0.0  # 0 = track-only, >0 = reward contribution
    metadata: dict[str, Any] = {}

@dataclass(frozen=True)
class Score:
    metrics: tuple[Metric, ...]
    
    @property
    def reward(self) -> float:
        """Weighted average of metrics with weight > 0"""
```

### 6. Actor

```python
@dataclass(frozen=True)
class Actor:
    trajectory: Trajectory      # Full history
    endpoint: Endpoint          # LLM config
    tools: list[Tool] = []      # Available tools
```

### 7. AgentState

```python
@dataclass(frozen=True)
class AgentState:
    actor: Actor
    environment: Environment | None
    stop: StopReason | None = None
    turn_idx: int = 0
    pending_tool_calls: list[ToolCall] = []
    next_tool_idx: int = 0      # Resume position
    session_id: str | None = None
    parent_session_id: str | None = None
    branch_point: int | None = None
```

### 8. RunConfig (Execution Configuration)

```python
@dataclass(frozen=True)
class RunConfig:
    on_chunk: Callable[[StreamEvent], Awaitable[None]]
    on_input: Callable[[str], Awaitable[str]] = default_stdin_handler
    confirm_tool: Callable[..., Awaitable[tuple[AgentState, ToolConfirmResult]]]
    handle_tool_error: Callable[[ToolResult, AgentState], AgentState]
    on_step_start: Callable[[AgentState], AgentState] = lambda s: s
    handle_stop: Callable[[AgentState], AgentState] = lambda s: s
    handle_no_tool: Callable[[AgentState, RunConfig], Awaitable[AgentState]]
    
    user_message_for_thinking: str | None = None
    inline_thinking: str | None = None
    show_progress: bool = False
    cancel_scope: trio.CancelScope | None = None
    session_store: SessionStore | None = None
    use_tito: bool = False
    tokenizer: Any | None = None
    suffix_ids: tuple[int, ...] | None = None
```

### 9. Environment Protocol

```python
@runtime_checkable
class Environment(Protocol):
    def get_tools(self) -> list[Tool]: ...
    
    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult: ...
    
    def requires_confirmation(self, tool_call: ToolCall) -> bool: ...
    def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None: ...
    def get_status_info(self) -> dict[str, str] | None: ...
    
    async def on_session_start(self, session_id: str) -> None: ...
    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState: ...
    
    async def serialize(self) -> dict: ...
    @staticmethod
    async def deserialize(data: dict) -> Environment: ...
```

### 10. EvalConfig

```python
@dataclass(frozen=True)
class EvalConfig:
    # Required
    endpoint: Endpoint
    score_fn: ScoreFn  # (Sample) -> Score or Awaitable[Score]
    prepare_messages: PrepareMessagesFn  # dict -> list[Message]
    
    # Optional
    environment_factory: EnvironmentFactory | None = None
    run_config: RunConfig | None = None
    
    # Dataset & execution
    max_samples: int | None = None
    max_concurrent: int = 1
    
    # Output
    output_dir: Path | None = None
    eval_name: str = "evaluation"
    verbose: bool = True
    show_progress: bool = False
    stream_tokens: bool = False
```

---

## Agent Execution Loop (agents.py)

### Main Functions

```python
async def run_agent(
    state: AgentState,
    run_config: RunConfig,
) -> list[AgentState]:
    """
    Main loop:
    1. Create session if session_store provided
    2. While not stopped:
       a. run_agent_step()
       b. Append to results
       c. Check stop condition
    3. Return all states (one per turn)
    
    Handles:
    - Session creation/resuming
    - Message persistence
    - Environment setup (on_session_start)
    - Cancellation via cancel_scope
    """

async def run_agent_step(
    state: AgentState,
    rcfg: RunConfig,
) -> AgentState:
    """
    Single turn:
    1. Check stop conditions (handle_stop)
    2. If resuming tools, process_pending_tools()
    3. Otherwise:
       a. Get tools from environment
       b. Call rollout() for LLM response
       c. Extract tool calls
       d. environment.on_assistant_message()
       e. process_pending_tools()
       f. Increment turn_idx
    """

async def process_pending_tools(
    state: AgentState,
    rcfg: RunConfig,
) -> AgentState:
    """
    For each tool:
    1. Get confirmation (confirm_tool)
    2. Deserialize fresh environment
    3. Execute tool
    4. Serialize environment state
    5. Add result to messages
    6. Persist if session_store
    """

async def rollout(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]],
    user_message_for_thinking: str | None = None,
    turn_idx: int = 0,
    inline_thinking: str | None = None,
    cancel_scope: trio.CancelScope | None = None,
    use_tito: bool = False,
    tokenizer: Any | None = None,
    suffix_ids: tuple[int, ...] | None = None,
) -> Actor:
    """
    Route to provider and get LLM response
    - Routes to token-level if use_tito=True
    - Otherwise uses get_provider_function()
    - Returns updated actor with new message
    """
```

### Handler Patterns

```python
# Stop handlers (composable)
handle_stop_max_turns(max_turns: int) -> Handler
handle_stop_on_empty_message() -> Handler
handle_stop_token_budget(max_tokens: int) -> Handler
handle_stop_cost_budget(max_cost_usd: float, cost_fn: Callable) -> Handler

compose_handlers(handlers: list[Handler]) -> Handler
# Applies handlers in order, stops if any sets stop reason

# Step handlers (pre-turn injection)
inject_turn_warning(max_turns: int, warning_at: int = 2) -> Handler
inject_tool_reminder(state, run_config) -> Awaitable[AgentState]
```

---

## Evaluation Framework (evaluation.py)

```python
async def evaluate_sample(
    sample_data: dict[str, Any],
    sample_id: str,
    config: EvalConfig,
    environment: Environment | None = None,
) -> Sample:
    """
    Evaluate one sample:
    1. Prepare messages via config.prepare_messages
    2. Create Actor
    3. run_agent() with up to max_turns
    4. Compute score via config.score_fn
    5. Return Sample with trajectory, score, metadata
    """

async def evaluate(
    dataset: Iterator[dict[str, Any]],
    config: EvalConfig,
) -> EvalReport:
    """
    Evaluate dataset:
    - Sequential if max_concurrent=1
    - Parallel with Trio if max_concurrent>1
    - Fresh environment per sample
    - Aggregate results into EvalReport
    - Save if output_dir provided
    """

def compute_summary_metrics(results: list[Sample]) -> dict[str, float]:
    """Aggregate across samples: mean, min, max, std for each metric"""
```

### Score Function Pattern

```python
# Sync (common)
def my_score_fn(sample: Sample) -> Score:
    correct = sample.response == str(sample.ground_truth)
    return Score(metrics=(
        Metric("correct", 1.0 if correct else 0.0, weight=1.0),
        Metric("tokens", len(sample.response), weight=0),
    ))

# Async (external verifier)
async def async_score_fn(sample: Sample) -> Score:
    result = await external_api(sample.response)
    return Score(metrics=(...))

# Type
ScoreFn = Callable[[Sample], Score] | Callable[[Sample], Awaitable[Score]]
```

---

## Provider System

```python
def get_provider_function(provider: str, model_id: str) -> ProviderStreamFunction:
    """Route to appropriate provider based on API type"""

# Implementations
rollout_anthropic(actor, on_chunk, user_message_for_thinking, ...)
rollout_openai(actor, on_chunk, ...)
rollout_google(actor, on_chunk, ...)
rollout_sglang(actor, on_chunk, ...)

# Provider-specific features
# Anthropic: extended thinking, thinking context
# OpenAI: o1/o3 reasoning, parallel tools
# Google: thinking blocks
# SGLang: token-level generation (TI/TO)
```

---

## Training Framework (training/)

### Sample Type

```python
@dataclass
class Sample:
    # Identity
    id: str = ""
    index: int | None = None
    group_index: int | None = None  # GRPO group
    
    # Input
    input: dict[str, Any] = {}
    prompt: str | list[dict[str, str]] = ""
    ground_truth: Any | None = None
    
    # Output
    trajectory: Trajectory | None = None
    
    # Training
    tokens: list[int] = []
    response_length: int = 0
    loss_mask: list[float] = []
    reward: float = 0.0
    rollout_log_probs: list[float] | None = None
    
    # Evaluation
    score: Score | None = None
    
    # Status
    status: Status = Status.PENDING
    metadata: dict[str, Any] = {}
    
    @property
    def response(self) -> str:
        """Extract final assistant response"""
```

### GRPO Training

```python
@dataclass(frozen=True)
class GRPOConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    dtype: str = "bfloat16"
    
    # Inference
    inference_backend: str = "sglang"
    inference_port: int = 30000
    
    # Rollouts
    batch_size: int = 8
    n_samples_per_prompt: int = 8
    max_tokens: int = 512
    temperature: float = 0.8
    
    # Training
    lr: float = 1e-6
    max_grad_norm: float = 1.0
    num_minibatches: int = 8
    
    # Loop
    num_steps: int = 100
    checkpoint_every: int = 20

async def grpo_train(
    config: GRPOConfig,
    prompts: list[dict[str, Any]],
    score_fn: Callable[[Sample], Score],
    environment_cls: type[Environment],
) -> dict[str, Any]:
    """Run GRPO training"""
```

### Training Backends

```python
@runtime_checkable
class TrainingBackend(Protocol):
    async def initialize(self) -> None: ...
    async def train_step(self, batch: list[Sample], lr: float) -> dict[str, float]: ...
    async def finalize(self) -> None: ...

# Implementations
# - pytorch.py: Single GPU
# - fsdp.py: FSDP distributed
# - jax_backend.py: JAX
```

---

## Session Persistence (store.py)

```python
@runtime_checkable
class SessionStore(Protocol):
    async def create(self, endpoint: Endpoint, environment: EnvironmentConfig, ...) -> AgentSession: ...
    async def get(self, session_id: str) -> tuple[AgentSession | None, str | None]: ...
    async def update(self, session_id: str, status, environment_state, reward, tags) -> tuple[None, str | None]: ...
    async def append_message(self, session_id: str, message: Message) -> None: ...
    async def list(self, filter_tags, status, limit) -> list[AgentSession]: ...
    async def list_children(self, parent_id: str) -> list[AgentSession]: ...

@dataclass
class AgentSession:
    session_id: str
    parent_id: str | None = None
    branch_point: int | None = None
    
    endpoint: Endpoint
    environment: EnvironmentConfig
    
    messages: list[Message] = []
    environment_state: dict[str, Any] | None = None
    
    status: SessionStatus = SessionStatus.PENDING
    reward: float | dict[str, float] | None = None
    
    tags: dict[str, str] = {}
    created_at: str
    updated_at: str

# FileSessionStore implementation
# ~/.rollouts/sessions/{session_id}/
#   session.json - Metadata
#   messages.jsonl - Append-only message log
```

---

## Environment Implementations

```python
# BasicEnvironment - No tools
def get_tools(self) -> list[Tool]:
    return []

# CalculatorEnvironment - Math operations
# Add, Subtract, Multiply, Divide (accumulator style)

# CodingEnvironment - File/bash tools
# Read, Write, Edit, Bash

# GitWorktreeEnvironment - Isolated worktrees per session
# Extends CodingEnvironment with per-session git setup

# BrowsingEnvironment - Web navigation
```

---

## Async Patterns

```python
# All main functions are async
async def run_agent(...) -> list[AgentState]
async def run_agent_step(...) -> AgentState
async def rollout(...) -> Actor
async def evaluate(...) -> EvalReport

# Cancellation via Trio cancel scopes
cancel_scope = trio.CancelScope()
# ... any await raises trio.Cancelled if cancel_scope.cancel() called

# Parallel evaluation
async with trio.open_nursery() as nursery:
    limiter = trio.CapacityLimiter(max_concurrent)
    for sample_id, sample_data in samples:
        async def run_with_limit(sid=sample_id, sdata=sample_data):
            async with limiter:
                result = await evaluate_sample(sid, sdata, ...)
                results.append(result)
        nursery.start_soon(run_with_limit)

# Stream callbacks
async def my_on_chunk(event: StreamEvent) -> None:
    if isinstance(event, TextDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, ToolCallEnd):
        print(f"Calling {event.tool_call.name}")
```

---

## Configuration Patterns

```python
@dataclass(frozen=True)
class BaseModelConfig:
    model_name: str = ""
    provider: str | None = None
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    
    def infer_provider(self) -> str:
        return infer_provider_from_model(self.model_name)
    
    def to_endpoint(self) -> Endpoint:
        # Convert to rollouts.dtypes.Endpoint

# Composition pattern
@dataclass(frozen=True)
class MyConfig:
    model: BaseModelConfig
    eval_name: str
    dataset_path: Path
    score_fn: Callable[[Sample], Score]
    max_concurrent: int = 1
    max_turns: int = 10
```

---

## Integration Points for GEPA

GEPA (Generative Evaluation with Prompt Adaptation) can integrate as:

### 1. Custom Score Function

```python
async def gepa_score_fn(sample: Sample) -> Score:
    """
    1. Extract trajectory messages
    2. Compute baseline score
    3. Propose prompt modifications
    4. Re-run agent with modified prompts (!)
    5. Compare scores
    6. Return Score with:
        - baseline_score
        - improved_score
        - prompt_quality
        - adaptation_success
    """
```

### 2. Environment Extension

```python
class GEPAEnvironment(Environment):
    """Tracks prompt state and can test modifications"""
    
    async def on_assistant_message(self, message, state):
        """Analyze message, propose improvements, inject feedback"""
```

### 3. Training Loop Integration

```python
async def grpo_step_with_gepa(
    prompts: list[dict],
    score_fn: ScoreFn,  # GEPA-enabled
    trainer: TrainingBackend,
):
    """GRPO + prompt optimization"""
```

### 4. Data Structures for GEPA

```python
@dataclass
class PromptAdaptation:
    original_prompt: str
    adapted_prompt: str
    reasoning: str
    estimated_improvement: float

@dataclass(frozen=True)
class GEPAScore(Score):
    metrics: tuple[Metric, ...]
    prompt_adaptation: PromptAdaptation | None = None
    adaptation_applied: bool = False
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Frozen dataclasses** | Time-travel debugging, safe concurrency |
| **Full state passing** | Testable, checkpointable, parallelizable |
| **Protocols over inheritance** | Flexible, duck typing, multiple implementations |
| **Async/Await everywhere** | Proper I/O, cancellation, concurrency |
| **Append-only sessions** | Crash safety, auditable history |
| **Granular streaming events** | Fine-grained UI, flexible aggregation |
| **Per-sample environment** | State isolation, parallel safety |
| **Tool atomicity per turn** | Simpler reasoning, prevents state leakage |
| **Composable handlers** | Custom logic without modifying core |

---

## Quick Examples

### Evaluate with Custom Score Function

```python
from rollouts.evaluation import evaluate
from rollouts.dtypes import Endpoint, Message, EvalConfig, Metric, Score
from rollouts.training.types import Sample

def prepare_messages(sample: dict) -> list[Message]:
    return [Message(role="user", content=sample["question"])]

def score_fn(sample: Sample) -> Score:
    correct = sample.response == str(sample.ground_truth)
    return Score(metrics=(
        Metric("correct", 1.0 if correct else 0.0, weight=1.0),
    ))

config = EvalConfig(
    endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4"),
    score_fn=score_fn,
    prepare_messages=prepare_messages,
    max_concurrent=4,
)

dataset = [{"question": "2+2?", "ground_truth": "4"}, ...]
report = await evaluate(iter(dataset), config)
print(f"Mean reward: {report.summary_metrics['mean_reward']:.3f}")
```

### Run Agent with Tools

```python
from rollouts.agents import run_agent
from rollouts.environments.calculator import CalculatorEnvironment
from rollouts.dtypes import AgentState, Actor, Endpoint, Message, Trajectory, RunConfig, StopReason

state = AgentState(
    actor=Actor(
        trajectory=Trajectory(messages=[
            Message(role="user", content="What is 5+3*2?")
        ]),
        endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4"),
    ),
    environment=CalculatorEnvironment(),
)

run_config = RunConfig(
    on_chunk=stdout_handler,
    handle_stop=handle_stop_max_turns(10),
)

states = await run_agent(state, run_config)
print(f"Completed in {states[-1].turn_idx} turns")
```

---

## File Locations for Reference

- Core types: `/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py` (1398 lines)
- Agent execution: `/Users/chiraagbalu/research/rollouts/rollouts/agents.py` (968 lines)
- Evaluation: `/Users/chiraagbalu/research/rollouts/rollouts/evaluation.py` (732 lines)
- Providers: `/Users/chiraagbalu/research/rollouts/rollouts/providers/`
- Environments: `/Users/chiraagbalu/research/rollouts/rollouts/environments/`
- Training: `/Users/chiraagbalu/research/rollouts/rollouts/training/`
- Session store: `/Users/chiraagbalu/research/rollouts/rollouts/store.py`

