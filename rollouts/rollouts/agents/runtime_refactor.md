# runtime.py refactor notes

Status: active refactor. Inline TODOs in `runtime.py` reference this doc.

## Framing (see `/docs/design/session_ownership.md` for full version)

Four roles:

- **Session** — append-only record of what happened. Source of truth.
- **Environment** — stateless w.r.t. history. `state = fold(effects, initial)`. Warm/cold `deserialize`.
- **Trajectory** — a view of the session, for training.
- **AgentState** — resume cursor. Volatile.

The structural move: **effects become first-class session entries.** Today tool calls live as `ToolCallContent` blocks inside assistant `Message`s, tool results live as `Message(role="tool")`. These are LLM-rendering shapes, not record shapes. We pull them out so the session sees `ToolCall` and `ToolResult` as their own entries.

Native loop and external-agent adapters both emit the same effect entries. One log schema, two writers.

## Gross points in `runtime.py` today

**G1 — two representations of effects.** `ToolCallContent` block (inside assistant message) + `ToolCall` object (in `state.pending_tool_calls`). Bridged by `message.get_tool_calls()` reparsing on demand. The object form is the honest shape; the content-block form is for LLM rendering.

**G2 — `ToolResult` flattens into `Message` on the way to the session.** `ToolResult` has `tool_call_id`, `content`, `is_error`, `error`, `details`. After `env.exec_tool` returns one, we build `Message(role="tool", content=result.content, tool_call_id=..., details=result.details)` and that's what hits `session_store`. `is_error` and `error` get lost or stuffed into `details`. Scorers reconstruct.

**G3 — `session_store.append_message` is the only sink.** No shape for "dispatched, awaiting result," "confirm rejected," "infra failed mid-execution." Non-message events have no home in the session.

**G4 — resume cursor is derivable-but-not-derived.** `pending_tool_calls` and `next_tool_idx` in `AgentState` duplicate information that would live in the session if effects were first-class. If a process dies between appending the assistant message and starting tool 0, recovery reparses `ToolCallContent` blocks from the message. A first-class effect log makes this direct: "latest `ToolCall` without matching `ToolResult` is pending."

**G5 — `on_assistant_message` is an out-of-band environment reaction channel.** Environments can react to an assistant message (REPL echoing, etc.) via a callback that can mutate `AgentState`. In the first-class model, this is "environment appends an effect in response to an assistant turn" — same shape as any other effect, not a special case.

**G7 — `persist_span` constructs a default `FileSessionStore()` instead of threading the actual store.** `providers/base.py:persist_span` writes observability spans to `~/.rollouts/sessions/<id>/spans.jsonl`, ignoring whatever `session_store` the agent loop uses. If the caller uses a custom base_dir, spans go somewhere else and the warning spams stdout. Same ownership lesson as G3: the session sink should be one thing, threaded everywhere, not constructed ambiently at the write site.

**G8 — base `CodingEnvironment` lacks `serialize`/`deserialize`; only `LocalFilesystemEnvironment` subclass has them.** The agent loop's per-tool serialize-deserialize dance (runtime.py `process_pending_tools`, line ~581) assumes every environment has these. The base `CodingEnvironment` doesn't. Today that means "coding env without a subclass" silently fails at the first tool call. After first-classing effects, the per-tool dance goes away and this gap closes — environments derive state from the effect log by default.

**G6 — two writers produce different shapes.** Native loop appends `Message`s via `session_store.append_message`. External adapters (`trajectory_from_remote_claude_code` et al.) parse the external tool's own session format and build a whole `Trajectory` at the end. Same conceptual stream, different shapes, different liveness. First-class effects let both write the same entries.

## What stays

- `AgentState` as resume cursor — good. Gets thinner (no `pending_tool_calls` duplication).
- Confirm-tool hook — real product logic, stays, hooks at "about to dispatch effect."
- Two-level concurrency (`api_limiter`, `tool_limiter`) — orthogonal, unchanged.
- Wide event stream (`StreamChunk`, `ToolExecutionStart`, `ToolResultReceived`) — already separate from the session, correctly.
- Environment serialize/deserialize with warm/cold paths — unchanged. The per-tool serialize-deserialize-serialize-deserialize dance is cheap on warm path; only pricey on cold, where isolation-per-tool may actually be wanted.

## External-agent environments

Each external-agent runtime (claude-code, codex, ...) gets its own Environment
type: `ClaudeCodeEnvironment`, `CodexEnvironment`, etc. These carry:

- `workspace` — the `SandboxWorkspaceResource` the agent operates in.
- `allowed_builtin_tools` — which of the harness's built-in tools the agent
  may use (translates to `--allowed-tools` or equivalent CLI flag). `None`
  means all defaults; `[]` means MCP-only.
- `mcp_tools` — additional tools we inject via MCP so the external harness
  can call tools it wouldn't normally have (terminal_bench.tmux, calculator,
  custom eval-specific tools).
- `translate_harness_event(raw) → SessionEntry | None` — the translation
  boundary from the harness's native session format to our session shape.
  Today this logic is buried inside `_run_agent_in_workspace` polling via
  `_ClaudeEventParser` / `_CodexEventParser`; belongs on the environment.
- `apply_effect(effect)` — fold-contract replay of a recorded effect.

Stubs are in place at `rollouts/environments/external_agent_environments.py`.
Consumers are not wired yet. The feature-rename from "external adapter
function" to "ClaudeCodeEnvironment with methods" is the unification move
for native/external session parity.

Note on `--allowed-tools`: both claude-code and codex support flags that
restrict the built-in tool set. This is how users/we can choose to e.g.
remove non-MCP tools (force the agent through injected MCP tools only).
Environment types carry this as a first-class field so test/eval code can
assert "what tools were available in this run" without reverse-engineering
from the trajectory.

## What moves

Rough direction, not final design:

- A `SessionEntry` sum type. At minimum: `AssistantTurn`, `ToolCall`, `ToolResult`. Likely also `UserMessage`, `EnvironmentEvent` (for G5), annotations (`CompactionEntry`, `LabelEntry`) later.
- `session_store` gains `append_entry(session_id, entry: SessionEntry)`. `append_message` becomes a thin helper or goes away.
- `view(session, policy) -> list[Message]` renders session entries into LLM context. Replaces "just hand `trajectory.messages` to the endpoint."
- Native loop writes `ToolCall` / `ToolResult` entries instead of assistant-with-ToolCallContent + tool-role-message.
- External adapters (future) emit the same entries live.
- `AgentState` loses `pending_tool_calls` / `next_tool_idx` duplication — derives from session.
- `on_assistant_message` becomes "environment emits effect after assistant turn," appended as entries.

## Sequencing

Not committing to an order yet. Plausible first steps:

1. Introduce `SessionEntry` sum type alongside existing message-shaped store. Append entries parallel to messages initially.
2. Move `ToolCall` / `ToolResult` into session entries. Deprecate `ToolCallContent` at the session level (still used at LLM-render time).
3. Add `view(session, policy)` builder. Switch the native loop to call it instead of handing messages directly.
4. Collapse `pending_tool_calls` derivation into session-read.
5. First external adapter rewritten to emit entries live (probably claude-code, since it's the most-exercised path).

Each step should keep `functional_extractor` scoring identical as regression gate.

## Open questions

- Does `SessionEntry` replace `Message` entirely at the store level, or do we keep `UserMessage` / `AssistantTurn` as entries but store them alongside `ToolCall` / `ToolResult`? Lean: keep both as entry kinds.
- Where do environment-initiated effects (G5) sit? Entry `EnvironmentEvent`, or reuse `ToolResult` with a synthetic tool_call_id?
- How do streaming partial assistant tokens interact with entry granularity? Probably: one entry per complete assistant turn, streaming is a runtime concern.
- Do we preserve `ToolCallContent` blocks inside the rendered `Message` for LLM input, or render on the fly from `ToolCall` entries? Lean: render on the fly; that's the point.

These get answered by building, not by deciding in advance.
