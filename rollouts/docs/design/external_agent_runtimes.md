# External Agent Runtimes In `rollouts`

Goal: allow `OrchestrateEnvironment` to launch arbitrary child agents, including:
- native `rollouts` SDK agents
- `claude code`
- `codex`

while keeping the design aligned with the code-style docs in:
- `/Users/chiraagbalu/research/docs/code_style/code_philosophy_reference.md`
- `/Users/chiraagbalu/research/docs/code_style/tiger_style.md`

## Recommendation

Treat `claude code` and `codex` as **external agent runtimes**, not as normal LLM providers inside the native `rollouts` agent loop.

The orchestrator should not know how to run Claude Code or Codex directly.
Instead, it should depend on a small runtime protocol and launch children through that boundary.

This matches the repo's design preferences:
- write ideal usage code first
- move external complexity to the boundary
- keep the hot path typed and explicit
- use protocols + factories for pluggability

## The Wrong Split

Do **not** make `OrchestrateEnvironment` branch like this:

```python
if runtime == "claude_code":
    ...
elif runtime == "codex":
    ...
elif runtime == "rollouts":
    ...
```

That pushes external CLI/runtime details into the orchestration core and will rot quickly.

It also violates the code-style guidance to:
- centralize control flow in the parent
- keep external parsing/validation at the edges

## The Right Split

Split the problem into three explicit concepts:

### 1. Worker profile

What kind of child agent is this?

Examples:
- `coding`
- `search`
- `browser`
- `query`

This defines:
- default prompt/system guidance
- expected result extraction
- default capabilities

### 2. Runtime

How is the child actually executed?

Examples:
- `rollouts_native`
- `claude_code`
- `codex`

This owns:
- CLI / SDK / stream protocol
- provider/runtime-specific session IDs
- resume behavior
- runtime-specific tool allowlist wiring

### 3. Capabilities

What is the child allowed to do?

Examples:
- `read`
- `write`
- `grep`
- `terminal`
- `websearch`

This must remain separate from runtime, because the same worker profile may run
on multiple runtimes with different tool-allowlist plumbing.

## Ideal Usage Code

The API we want to be able to write:

```python
result = await system.thread(
    "search-auth",
    "Find where token refresh is implemented",
    worker="search",
    runtime="claude_code",
    capabilities=["read", "grep"],
)

result = await system.thread(
    "impl-auth",
    "Implement the fix and update tests",
    worker="coding",
    runtime="rollouts_native",
    capabilities=["read", "write", "grep", "terminal"],
)

result = await system.thread(
    "codex-review",
    "Review the refactor and identify risks",
    worker="coding",
    runtime="codex",
    capabilities=["read", "grep"],
)
```

That usage code exposes the right seams:
- what work is being done
- which runtime executes it
- what authority it gets

## Proposed Types

### WorkerSpec

```python
@dataclass(frozen=True)
class WorkerSpec:
    worker: str
    runtime: str
    task: str
    alias: str | None = None
    capabilities: tuple[str, ...] = ()
    docs: tuple[DocRef, ...] = ()
    traces: tuple[ResultRef, ...] = ()
```

This should be the input to orchestration-side thread spawning.

### ChildResult

```python
@dataclass(frozen=True)
class ChildResult:
    status: Literal["completed", "aborted", "escalated", "interrupted", "error"]
    output: str | None
    reason: str | None
    trace: str
    duration_ms: int
    child_id: str | None = None
    runtime: str | None = None
```

All runtimes must normalize into this shape.

### AgentRuntime protocol

```python
@runtime_checkable
class AgentRuntime(Protocol):
    async def run_child(
        self,
        spec: WorkerSpec,
        ctx: SpawnContext,
    ) -> ChildResult:
        ...
```

Optional later:

```python
    async def resume_child(
        self,
        child_id: str,
        spec: WorkerSpec,
        ctx: SpawnContext,
    ) -> ChildResult:
        ...
```

## SpawnContext

Keep external/runtime concerns out of `WorkerSpec`.

```python
@dataclass(frozen=True)
class SpawnContext:
    cwd: Path
    parent_session_id: str | None
    trace_id: str | None
    endpoint: Endpoint | None
    session_store: SessionStore | None
    document_store: DocumentStore | None
```

This keeps orchestration-owned state explicit and avoids hidden globals.

## Concrete Runtime Implementations

### `RolloutsNativeRuntime`

Runs a normal `rollouts` agent:
- constructs an `AgentState`
- attaches an actual `Environment`
- uses `run_agent(...)`
- can use native session store / environment serialization

This is where "coding agent", "search agent", etc. should most naturally live.

### `ClaudeCodeRuntime`

Runs Claude Code as an external runtime:
- owns CLI command building
- owns `allowed_tools` mapping
- owns Claude-specific session resume / extraction behavior
- normalizes final output into `ChildResult`

This is where the current `run_claude(...)` integration belongs.

### `CodexRuntime`

Same pattern:
- owns Codex-specific CLI / driver protocol
- owns resume and result extraction behavior
- normalizes into `ChildResult`

## Important Design Point

`claude code` and `codex` may feel like "environments" from a product perspective, but
architecturally they are better modeled as **runtimes**:

- an Environment in `rollouts` is a tool/state contract inside the native agent loop
- Claude Code / Codex are whole external agent loops with their own tool semantics

So the clean implementation is:
- keep `Environment` for native rollouts agents
- add `AgentRuntime` for external agent engines

If desired, you can still expose user-facing classes named like:
- `ClaudeCodeDriverEnvironment`
- `CodexDriverEnvironment`

but those should be thin wrappers around runtime adapters, not the main abstraction.

## Factory / Registry

Following the repo's existing protocol/factory style, use a registry:

```python
RUNTIME_REGISTRY: dict[str, AgentRuntime]
WORKER_REGISTRY: dict[str, WorkerProfile]
```

Where:
- runtime registry resolves execution engine
- worker registry resolves prompt/tool/profile defaults

Then `OrchestrateEnvironment` does:

1. build `WorkerSpec`
2. resolve worker profile
3. resolve runtime
4. call `runtime.run_child(spec, ctx)`
5. store normalized result / alias mapping

No runtime-specific branching in orchestration core.

## Why This Fits The Code Style Docs

### Parsing at the edges

Claude/Codex CLI weirdness stays inside their runtime adapters.
The orchestrator only sees typed `ChildResult`.

### Externals at the boundary

The external process / SDK / stream protocol is boundary code.
The orchestration core stays clean.

### Usage code first

The `system.thread(worker=..., runtime=..., capabilities=...)` API is the usage code we want.

### Minimize state owners

Child-session bookkeeping remains in orchestration.
Runtime-specific transport/session details remain in the adapter.
Do not spread session ownership across both layers.

## What To Change First

### Phase 1

- Introduce `WorkerSpec`, `ChildResult`, `SpawnContext`, `AgentRuntime`
- Move current `run_claude` thread spawning behind `ClaudeCodeRuntime`
- Make `OrchestrateEnvironment` depend on a runtime registry instead of `run_claude` directly

### Phase 2

- Add `RolloutsNativeRuntime`
- Let worker profiles choose between native `rollouts` agents and external runtimes

### Phase 3

- Add `CodexRuntime`
- Add child-session persistence / alias-based resumption in orchestration storage

## Short Answer

Yes, the right way to let `rollouts` call external runtimes is:

- do **not** treat them as normal providers in the native agent loop
- do **not** hardcode them into `OrchestrateEnvironment`
- treat them as runtime adapters behind a small protocol
- keep worker profile, runtime, and capabilities as separate concepts

That gives you a clean path to:
- `ClaudeCodeRuntime`
- `CodexRuntime`
- `RolloutsNativeRuntime`

without turning orchestration into a pile of provider-specific branches.
