# Comparison: ARES, Harbor, Verifiers, and Rollouts

A detailed analysis of four frameworks for LLM agent evaluation and training, with recommendations for rollouts.

## Executive Summary

| Framework | Primary Purpose | Best For |
|-----------|-----------------|----------|
| **ARES** | RL-first code agent training | Training agents with PPO/GRPO where you need LLM call interception |
| **Harbor** | Evaluation harness for code agents | Benchmarking existing agent CLIs across standardized tasks |
| **Verifiers** | Model-in-the-loop evaluation | Online verification with managed cloud sandboxes |
| **Rollouts** | Lightweight agentic RL research | Multi-provider experimentation with immutable state |

## 1. Architectural Patterns

### ARES: Queue-Mediated Interception

ARES's key innovation is the `QueueMediatedLLMClient` - it intercepts LLM calls from agent code without modification:

```
CodeAgent → QueueMediatedLLMClient → Queue → Environment.step()
                                         ↓
                           LLMRequest exposed as observation
                                         ↓
                           RL trainer provides LLMResponse (action)
                                         ↓
                           Future resolved → Agent continues
```

**Key file**: `/tmp/ares/src/ares/llms/queue_mediated_client.py`

```python
@dataclasses.dataclass(frozen=True)
class QueueMediatedLLMClient(LLMClient):
    q: asyncio.Queue[ValueAndFuture[LLMRequest, LLMResponse]]

    async def __call__(self, req: LLMRequest) -> LLMResponse:
        future = asyncio.Future[LLMResponse]()
        await self.q.put(ValueAndFuture(value=req, future=future))
        return await future  # Blocks until environment provides response
```

This enables dm_env-style RL loops where:
- Observations = LLM requests from the agent
- Actions = LLM responses provided by the trainer
- Rewards = Test results at episode end

**Strengths**:
- Agent code is unmodified - writes linear code, unaware of RL loop
- Clean separation between agent logic and training infrastructure
- Fits naturally into existing RL frameworks (dm_env protocol)

**Weaknesses**:
- Only works when you control the LLM client
- Adds latency (queue round-trip per LLM call)
- Tightly coupled to asyncio

### Harbor: Orchestration Pipeline

Harbor is evaluation-first - it runs agent CLIs in containers and collects results:

```
Job → Orchestrator → Trial → [Environment Setup, Agent Run, Verification]
                                     ↓
                              TrialResult + ATIF Trajectory
```

**Key files**:
- `/tmp/harbor/src/harbor/job.py` - Job orchestration
- `/tmp/harbor/src/harbor/trial/trial.py` - Single trial execution
- `/tmp/harbor/src/harbor/models/trajectories/trajectory.py` - ATIF format

Harbor's ATIF (Agent Trajectory Interchange Format) is designed for ecosystem interop:

```python
class Trajectory(BaseModel):
    schema_version: Literal["ATIF-v1.0", ..., "ATIF-v1.5"]
    session_id: str
    agent: Agent
    steps: list[Step]  # With tool_calls, observations, metrics
    final_metrics: FinalMetrics
```

**Strengths**:
- Multi-agent support (Claude Code, OpenHands, Aider, SWE-agent, etc.)
- Hundreds of benchmarks in registry
- Standardized trajectory format for sharing results
- Pluggable container backends (Docker, Modal, Daytona, E2B, GKE)

**Weaknesses**:
- Evaluation-first design - RL is bolted on
- Agents are black boxes (CLI invocation)
- Heavy Pydantic usage adds runtime overhead

### Verifiers: HTTP Proxy Interception

Verifiers uses Prime Tunnel to intercept API calls from CLI agents:

```
OpenCode CLI → HTTP request → Prime Tunnel → Local Server → Queue
                                                  ↓
                                    Model processes request
                                                  ↓
                                    Response sent back via HTTP
```

**Key files**:
- `/tmp/verifiers/verifiers/envs/experimental/cli_agent_env.py`
- `/tmp/verifiers/verifiers/envs/experimental/harbor_env.py`

**Strengths**:
- Works with any OpenAI-compatible agent
- Managed sandboxes via Prime Sandboxes API
- Multi-turn support with request queuing

**Weaknesses**:
- HTTP proxy adds latency
- Tightly coupled to Prime infrastructure
- Less flexible than native integration

### Rollouts: Pure Function State Threading

Rollouts passes immutable state through pure functions:

```
run_agent(state) → run_agent_step(state) → rollout(actor) → provider(actor)
                        ↓                       ↓
              process_pending_tools()     StreamEvents emitted
                        ↓
              Environment.exec_tool()
```

**Key files**:
- `/Users/chiraagbalu/research/rollouts/rollouts/dtypes.py` - Core types
- `/Users/chiraagbalu/research/rollouts/rollouts/agents.py` - Agent loop

```python
@dataclass(frozen=True)
class AgentState:
    actor: Actor  # trajectory + endpoint + tools
    environment: Environment | None
    stop: StopReason | None
    turn_idx: int
    pending_tool_calls: tuple[ToolCall, ...]
```

**Strengths**:
- Immutable state enables time-travel debugging, safe concurrency
- Multi-provider native (Anthropic, OpenAI, Google, Groq, etc.)
- Granular streaming events for observability
- Session persistence with fork/resume
- trio structured concurrency

**Weaknesses**:
- No built-in container orchestration
- Smaller benchmark ecosystem
- No standard trajectory interchange format

## 2. Data Structures Comparison

### Trajectory/Rollout Data

| Framework | Format | Key Fields |
|-----------|--------|------------|
| **ARES** | Implicit in TimeStep sequence | LLMRequest/LLMResponse pairs, final reward |
| **Harbor** | ATIF v1.5 (Pydantic) | steps[], tool_calls, observation, metrics, final_metrics |
| **Verifiers** | state["trajectory"] list | prompt, completion, response, reward, tokens per turn |
| **Rollouts** | Trajectory dataclass | messages, completions, rewards, metadata |

Harbor's ATIF is the most elaborate, designed for ecosystem interop:

```python
# Harbor Step (ATIF)
class Step(BaseModel):
    step_id: int
    source: Literal["system", "user", "agent"]
    message: str
    reasoning_content: str | None  # Separate from message
    tool_calls: list[ToolCall] | None
    observation: Observation | None
    metrics: Metrics | None  # token_ids, logprobs, cost
    is_copied_context: bool | None  # Training data hygiene
```

Rollouts is minimal but sufficient:

```python
# Rollouts Trajectory
@dataclass(frozen=True)
class Trajectory:
    messages: tuple[Message, ...]
    completions: tuple[ChatCompletion, ...]
    rewards: float
    metadata: dict[str, Any]
```

### Token/Usage Tracking

Harbor's Metrics captures training-critical data:

```python
class Metrics(BaseModel):
    prompt_tokens: int | None
    completion_tokens: int | None
    cached_tokens: int | None
    cost_usd: float | None
    prompt_token_ids: list[int] | None  # For RL training
    completion_token_ids: list[int] | None
    logprobs: list[float] | None
```

Rollouts' Usage is similar but lacks token IDs:

```python
@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    cost: Cost
    # Missing: token_ids, logprobs
```

### Environment Protocol

| Framework | Protocol | Key Methods |
|-----------|----------|-------------|
| **ARES** | dm_env Environment | reset(), step(action) → TimeStep |
| **Harbor** | BaseEnvironment (ABC) | build(), exec(), upload_dir(), download_dir() |
| **Verifiers** | Environment Protocol | get_tools(), exec_tool(), serialize() |
| **Rollouts** | Environment Protocol | get_tools(), exec_tool(), serialize(), on_assistant_message() |

Rollouts and Verifiers share similar protocols (likely common ancestry). ARES uses dm_env patterns. Harbor's environment is container-focused.

## 3. Design Philosophy

| Aspect | ARES | Harbor | Verifiers | Rollouts |
|--------|------|--------|-----------|----------|
| **Mutability** | Mutable state | Pydantic models | Mixed (state dict) | Frozen dataclasses |
| **Inheritance** | Protocol-based | ABC inheritance | Protocol + inheritance | Protocol-based |
| **Concurrency** | asyncio | sync + threading | asyncio | trio |
| **Config** | Pydantic Settings | Pydantic models | TypedDict/dataclass | Protocol-based |
| **Error handling** | Exceptions + retry | Exception types | Error in state | Crash-loud |

Rollouts is the most opinionated about functional purity:
- Full state passing - state flows through pure functions, never mutated
- Immutability - frozen dataclasses enable time-travel debugging
- Tiger Style - explicit control flow, crash-loud on errors, no silent fallbacks

## 4. When to Use Each

| Use Case | Best Framework |
|----------|----------------|
| Training code agents with PPO/GRPO | ARES |
| Benchmarking existing agent CLIs | Harbor |
| Evaluating with managed cloud sandboxes | Verifiers |
| Multi-provider agentic research | Rollouts |
| Publishing standardized benchmark results | Harbor |
| Session resume/forking | Rollouts |
| Working with external agent code you don't control | ARES or Verifiers |

## 5. Ecosystem Ceilings

### Harbor: Highest Ecosystem Ceiling

Harbor is building toward being the "ImageNet of code agents":
- Hundreds of benchmarks in standard format
- ATIF trajectory format for result sharing
- Multi-agent support out of the box
- Registry system for dataset distribution

**Limitation**: Evaluation-first design. RL training is an afterthought.

### Rollouts: Highest Research Ceiling

Rollouts is built for the research loop (generate → score → train → repeat):
- Immutable state enables forking, replay, counterfactual analysis
- Multi-provider means you can compare models easily
- Granular streaming events for debugging and profiling

**Limitation**: Smaller ecosystem, no built-in container orchestration.

### ARES: Best RL Abstraction

ARES has the cleanest RL integration via queue-mediated interception:
- dm_env patterns work with standard RL libraries
- Agent code is unmodified

**Limitation**: Narrow - only works when you control the LLM client.

### Verifiers: Managed Infrastructure Story

Verifiers + Prime gives you:
- Managed sandboxes at scale
- HTTP proxy interception for any agent

**Limitation**: Locked into Prime's stack.

## 6. Summary

For research today, rollouts offers the best developer experience due to its functional core, multi-provider support, and granular observability. However, it would benefit from Harbor's ecosystem access (benchmarks, trajectory format) and ARES's training data hygiene (token IDs in trajectories).

The recommended path is to keep rollouts' core architecture while adopting data conventions from Harbor/ARES for ecosystem interoperability.
