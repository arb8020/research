"""Tau2Environment — rollouts adapter for tau2-bench customer service tasks.

tau2-bench (https://github.com/sierra-research/tau2-bench) is a benchmark for
agent↔user↔tool loops in airline, retail, and telecom domains. Each task
specifies:
  - A domain policy the agent must follow (surfaced as system prompt addendum)
  - A set of mutating tools that operate on a per-task database
  - A user persona + scenario the simulated user plays
  - Expected actions / evaluation criteria

This environment bridges rollouts' agent loop to tau2's domain machinery:
  - `get_tools()` returns the domain's tools in rollouts format
  - `exec_tool()` dispatches to tau2's `environment.get_response()`
  - `on_assistant_message()` drives the user-simulator role when the agent
    sends a plain message (no tool calls), calling it via rollouts' own
    `rollout()` rather than tau2's internal LiteLLM path
  - `score()` hands the completed trajectory to tau2's official
    `evaluate_simulation` for reward computation

## Role FSM

Per turn the environment moves through:

    AGENT sends tool calls → exec_tool for each → AGENT (model re-prompted)
    AGENT sends text       → user sim responds → AGENT (model re-prompted)
    AGENT is_stop          → terminate (AGENT_STOP)
    USER is_stop           → terminate (USER_STOP)
    step/error caps        → terminate (MAX_STEPS / TOO_MANY_ERRORS)

## User simulator

We instantiate tau2's own `UserSimulator(llm="dummy")` for its prompt and
state-management logic, but never call its `generate_next_message` (which
would go through tau2's LiteLLM-backed `generate()`). Instead we pull
`user_state.system_messages + user_state.flip_roles()` ourselves and route
the request through rollouts' `rollout()` against a caller-supplied
`Endpoint`. This gives us tau2-bit-exact prompts and role-flipping while
letting the user simulator hit any provider rollouts knows about
(Anthropic, sglang, our own served endpoints, etc.) — not just LiteLLM.

Stop detection uses `UserSimulator.is_stop()` directly so any future
changes to tau2's stop-token protocol get picked up automatically.

## What is NOT implemented

These are explicit NotImplementedError stubs — calling into them produces
a clear "not wired yet" message rather than a confusing failure. Each
location carries a TODO comment with the wiring sketch.

- **Voice / full-duplex** — Tau2Environment is a half-duplex env (agent
  text, user text, tool calls). tau2's voice mode requires a
  `FullDuplexAgent` against `discrete_time_audio_native_agent.py` and a
  realtime audio API; that's a separate runtime concern, not just an env.
- **`banking_knowledge` domain** — needs `uv sync --extra knowledge` for
  tau2's RAG deps + retrieval-config plumbing through the sample row.
  Stub raises in `_build_tau2_domain_env`.
- **`solo_mode`** for airline/retail — tau2 itself only exposes
  `solo_mode` on telecom's `get_environment`. Stub raises with that
  context. Telecom solo_mode is wired and works (defaults False, set
  via `Tau2Environment.create(solo_mode=...)`).
- **Streaming events for user-sim** — we pass `_silent_stream` to the
  user-sim's `rollout()` call. The agent's streaming events are wired
  normally (FirstToken, TextDelta, etc. flow into the eval runner's
  progress + session log); only the user-sim's are silenced. tau2 itself
  doesn't surface user-sim streaming either, so this is a UX gap rather
  than a fidelity gap. To wire: thread the eval runner's on_chunk into
  Tau2Environment via a constructor arg and pass it through
  `_invoke_user_sim_once` instead of `_silent_stream`.

## Multi-tool turns

When the agent issues a single AssistantMessage with multiple tool_calls,
the rollouts runtime serializes/deserializes the env between each
exec_tool call (for crash recovery). To keep tau2's `set_state` replay
validator happy across these mid-turn snapshots, we BUFFER the
AssistantMessage and its accumulating ToolMessages in `_pending_asst` /
`_pending_tool_results` until the whole group is complete, then flush
atomically into `_tau2_trajectory`. serialize() emits the pending buffer;
deserialize() restores it AND re-applies the completed tool calls to the
freshly-rebuilt env's database (so mutations from earlier tools in the
group survive the deserialize round-trip).

## Serialize / Deserialize

- serialize: emits domain, task JSON, parallel tau2 message trajectory,
  termination reason, counters. Does NOT emit a handle to the live tau2
  domain env (that object holds a mutable database).
- deserialize (warm): caller passes `tau2_env_live=<obj>` to reattach to
  an already-initialized domain env — no state replay.
- deserialize (cold): caller omits `tau2_env_live`; we build a fresh
  domain env and call its `set_state(initialization_data,
  initialization_actions, message_history)` to replay mutating calls from
  the serialized tau2 trajectory. tau2's `set_state` validates that
  replayed tool responses match the recorded ones.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import trio

from ..agents import Actor, AgentState, RunConfig, rollout
from ..core import (
    Endpoint,
    Message,
    StopReason,
    TextContent,
    Tool,
    ToolCall,
    ToolCallContent,
    ToolFormatter,
    ToolResult,
    Trajectory,
)
from ..dtypes import StreamEvent, ToolFunction, ToolFunctionParameter

if TYPE_CHECKING:
    # Keep tau2 imports lazy — a workspace that only consumes the environment
    # shouldn't pay the tau2 import cost until something actually constructs one.
    pass

logger = logging.getLogger(__name__)

# Domains we support today. tau2 also ships `banking_knowledge` (RAG-based)
# and `mock` (testing). Adding a domain here requires:
#   1. Add the literal value
#   2. Add a branch in _build_tau2_domain_env
#   3. For banking_knowledge specifically: ensure tau2 was installed with the
#      `knowledge` extra (`uv sync --extra knowledge`), and surface the
#      retrieval pipeline config (BM25 / embeddings / RAG) through the
#      sample row.
Tau2Domain = Literal["airline", "retail", "telecom", "banking_knowledge"]


# ── Tool translation ────────────────────────────────────────────────────────


def _tau2_tool_to_rollouts(t: Any) -> Tool:
    """Translate a tau2 Tool (pydantic BaseTool) into a rollouts Tool.

    tau2 exposes `openai_schema` which is the OpenAI function-tool JSON.
    We parse it back out into our typed Tool/ToolFunction structure rather
    than passing the dict through unstructured.
    """
    schema = t.openai_schema
    fn = schema["function"]
    params = fn.get("parameters", {}) or {}
    properties = params.get("properties", {}) or {}
    required = list(params.get("required", []) or [])
    return Tool(
        function=ToolFunction(
            name=fn["name"],
            description=fn.get("description", "") or "",
            parameters=ToolFunctionParameter(
                properties=properties,
                type=params.get("type", "object"),
            ),
            required=required,
        )
    )


def _rollouts_tc_to_tau2(tc: ToolCall, *, requestor: str = "assistant") -> Any:
    """Translate rollouts ToolCall → tau2 ToolCall."""
    from tau2.data_model.message import ToolCall as Tau2ToolCall

    # rollouts ToolCall already has parsed args; tau2 expects `arguments: dict`.
    return Tau2ToolCall(
        id=tc.id,
        name=tc.name,
        arguments=dict(tc.args),
        requestor=requestor,
    )


def _tau2_tm_to_rollouts(tm: Any) -> ToolResult:
    """Translate tau2 ToolMessage → rollouts ToolResult."""
    # tau2 ToolMessage has fields: id (matching tool_call id), role, content,
    # requestor, error: bool.
    is_error = bool(getattr(tm, "error", False))
    content = tm.content if tm.content is not None else ""
    return ToolResult(
        tool_call_id=tm.id,
        is_error=is_error,
        content=content,
        error=content if is_error else None,
    )


# ── User simulator via rollouts (Option A: tau2 prompts, our transport) ─────


def _build_user_simulator(
    task: Any,
    persona_config: Any = None,
    user_tools: list[Any] | None = None,
) -> Any:
    """Construct a tau2 UserSimulator we'll use only for its prompt-building
    and state-management logic — never for inference.

    The `llm="dummy"` model string is a sentinel: tau2's `UserSimulator`
    requires `llm` at construction, but we never call `generate_next_message`
    so the value is unused. If a future tau2 version validates `llm` eagerly
    we'd need to switch to the actual `DummyUser` subclass.

    `persona_config` (a `tau2.data_model.persona.PersonaConfig` or None)
    surfaces tau2's runtime persona layer — verbosity, interrupt tendency,
    etc. — which gets baked into the user-sim system prompt via
    `PersonaConfig.to_guidelines_text()`. None means tau2's default
    behavior (no persona overrides), matching `tau2 run` defaults.

    `user_tools` (optional list[tau2.environment.tool.Tool]) enables the
    user-tools path: telecom's user simulator can execute its own tools
    (look up account info, etc.). When non-None, tau2's UserSimulator
    automatically loads the tool-aware guidelines variant of its system
    prompt. Pass `tau2_env.get_user_tools()` here for telecom; None for
    airline/retail.
    """
    from tau2.user.user_simulator import UserSimulator

    return UserSimulator(
        llm="dummy",
        instructions=str(task.user_scenario),
        tools=user_tools,
        persona_config=persona_config,
    )


def _tau2_messages_to_rollouts(
    tau2_msgs: list[Any],
    system_messages: list[Any],
) -> list[Message]:
    """Translate tau2's API-compatible message list into rollouts Messages
    suitable for `rollout()`.

    Used to send the user simulator's view of the conversation through our
    Endpoint. Handles three message kinds:
      - SystemMessage / text-only assistant or user → plain Message
      - AssistantMessage with tool_calls (telecom: user-sim's prior tool
        calls show up here after flip_roles) → Message with content blocks
        containing ToolCallContent
      - ToolMessage → Message(role="tool", tool_call_id=<id>, content=<text>)

    The Anthropic provider asserts tool messages have tool_call_id; tau2's
    ToolMessage carries that id, so we propagate it.
    """
    from tau2.data_model.message import (
        AssistantMessage as Tau2AssistantMessage,
    )
    from tau2.data_model.message import ToolMessage as Tau2ToolMessage

    out: list[Message] = []
    for sm in system_messages:
        if sm.content:
            out.append(Message(role="system", content=sm.content))
    for m in tau2_msgs:
        if isinstance(m, Tau2ToolMessage):
            out.append(
                Message(
                    role="tool",
                    content=m.content if isinstance(m.content, str) else "",
                    tool_call_id=m.id,
                )
            )
            continue
        if isinstance(m, Tau2AssistantMessage) and m.tool_calls:
            # Build a content-block list combining text (if any) + each
            # ToolCallContent. Rollouts providers translate ToolCallContent
            # to the right wire format per provider.
            blocks: list[Any] = []
            if isinstance(m.content, str) and m.content:
                blocks.append(TextContent(text=m.content))
            for tc in m.tool_calls:
                blocks.append(
                    ToolCallContent(
                        id=tc.id,
                        name=tc.name,
                        arguments=dict(tc.arguments),
                    )
                )
            out.append(Message(role="assistant", content=blocks))
            continue
        # Plain text assistant or user.
        text = m.content if isinstance(m.content, str) else None
        if text is None:
            continue
        out.append(Message(role=m.role, content=text))
    return out


def _message_text(m: Message) -> str:
    """Extract text content from a rollouts Message."""
    c = m.content
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    # content blocks — pull out text blocks only
    parts: list[str] = []
    for block in c:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts).strip()


async def _silent_stream(_event: StreamEvent) -> None:
    """No-op stream handler for the user-sim rollout.

    TODO(tau2-streaming): currently we discard streaming events from
    user-sim LLM calls (TextDelta, ToolCallStart, etc.). This is a UX
    gap — the user-sim's responses don't appear live in the TUI and
    don't show timing breakdowns in semantic_trace. Fix is mechanical:
    accept an `on_chunk` callable on Tau2Environment.create() (default
    None / silent), thread it down to _invoke_user_sim_once, pass to
    rollout() in place of _silent_stream. The eval runner already wraps
    on_chunk to attach sample_id; just plumb that wrapped callable in.
    Not a fidelity issue — tau2 itself doesn't surface user-sim
    streaming either.
    """
    return None


# ── Environment ─────────────────────────────────────────────────────────────


@dataclass
class Tau2Environment:
    """Customer-service environment backed by tau2-bench.

    Construct via `Tau2Environment.create(...)` rather than `__init__`, so
    the live tau2 domain env and user-sim prompt are built up-front.

    Lifecycle:
      1. `create(domain, task, user_endpoint, ...)` builds the env and
         initializes the tau2 domain env via `set_state(initialization_data,
         initialization_actions, [])`.
      2. Agent loop calls `get_tools()`, `exec_tool()`, `on_assistant_message()`.
      3. After the loop stops, `score(trajectory)` calls tau2's
         `evaluate_simulation` against the parallel tau2 trajectory.

    Attributes set by `create`:
      _tau2_env: the live tau2 domain env (holds mutable database)
      _tau2_tools: list[rollouts.Tool] — translated once
      _tau2_trajectory: parallel list[tau2.Message] maintained for scoring
      _user_system_prompt: precomputed system prompt for the user sim
    """

    domain: Tau2Domain
    task: Any  # Tau2Task (not typed for lazy import reasons)
    user_endpoint: Endpoint | None  # may be None when solo_mode=True
    max_steps: int = 200
    max_errors: int = 10
    # Optional tau2 PersonaConfig — runtime persona overrides (verbosity,
    # interrupt tendency, etc.) layered on top of the task's baked-in
    # persona. None = tau2's default (no overrides), matching tau2 run.
    persona_config: Any = None
    # solo_mode: agent runs without a user simulator. Text-only assistant
    # turns either trigger AGENT_STOP (if stop classifier matches) or
    # AGENT_ERROR (otherwise) — matching tau2 Orchestrator's solo loop.
    # tau2 itself only exposes solo_mode on telecom; airline/retail raise
    # in _build_tau2_domain_env.
    solo_mode: bool = False
    # banking_knowledge retrieval pipeline knobs. retrieval_variant names
    # the variant (e.g. "qwen_embeddings_grep", "bm25") per
    # tau2/knowledge/README.md; None lets tau2 default. retrieval_kwargs
    # are extra params (e.g. {"top_k": 5}). Ignored for other domains.
    retrieval_variant: str | None = None
    retrieval_kwargs: dict[str, Any] | None = None

    # Populated by create(); internal mutable state.
    _tau2_env: Any = field(default=None, repr=False)
    _tau2_tools: list[Tool] = field(default_factory=list, repr=False)
    _tau2_trajectory: list[Any] = field(default_factory=list, repr=False)
    _user_sim: Any = field(default=None, repr=False)
    _user_state: Any = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _num_errors: int = field(default=0, repr=False)
    # Pending-group buffer: when the agent emits an AssistantMessage with
    # tool_calls, we hold it (and its arriving ToolMessages) here instead of
    # appending to _tau2_trajectory. Once the last expected ToolMessage
    # arrives, we flush the whole group atomically into _tau2_trajectory.
    # This guarantees _tau2_trajectory only ever contains complete
    # tool_call/tool_message groups — which is the invariant tau2's
    # set_state replay requires.
    _pending_asst: Any = field(default=None, repr=False)
    _pending_tool_results: list[Any] = field(default_factory=list, repr=False)
    _pending_expected_ids: set[str] = field(default_factory=set, repr=False)
    # Termination is tracked as the tau2 enum directly (set at the point of
    # detection in on_assistant_message / _run_user_simulator) so we don't
    # have to round-trip through StopReason heuristics at score time.
    _termination_reason: Any = field(default=None, repr=False)

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    async def create(
        cls,
        *,
        domain: Tau2Domain,
        task: Any,
        user_endpoint: Endpoint | None,
        max_steps: int = 200,
        max_errors: int = 10,
        persona_config: Any = None,
        solo_mode: bool = False,
        retrieval_variant: str | None = None,
        retrieval_kwargs: dict[str, Any] | None = None,
    ) -> Tau2Environment:
        """Build a Tau2Environment with a fresh tau2 domain env.

        `persona_config` (optional `PersonaConfig`) plumbs through tau2's
        runtime persona layer. The same value should be used by
        prepare_messages when building its throwaway UserSimulator so the
        bootstrap user-sim call uses the same prompt as later turns.

        `solo_mode` (bool) runs the env with no user simulator: text-only
        assistant turns terminate (AGENT_STOP if is_stop, AGENT_ERROR
        otherwise). `user_endpoint` may be None when solo_mode=True. Only
        the telecom domain supports solo_mode at the tau2 layer; airline
        and retail will raise in _build_tau2_domain_env.
        """
        tau2_env = _build_tau2_domain_env(
            domain,
            solo_mode=solo_mode,
            retrieval_variant=retrieval_variant,
            retrieval_kwargs=retrieval_kwargs,
            task=task,
        )
        _apply_task_initialization(tau2_env, task, message_history=[])
        tools = [_tau2_tool_to_rollouts(t) for t in tau2_env.get_tools()]
        # In solo_mode, no user simulator at all — skip its construction.
        # Otherwise build one with the per-domain user_tools (telecom only).
        user_sim: Any = None
        user_state: Any = None
        if not solo_mode:
            user_tools = tau2_env.get_user_tools() if tau2_env.user_tools is not None else None
            user_sim = _build_user_simulator(
                task,
                persona_config=persona_config,
                user_tools=user_tools,
            )
            user_state = user_sim.get_init_state()
        return cls(
            domain=domain,
            task=task,
            user_endpoint=user_endpoint,
            max_steps=max_steps,
            max_errors=max_errors,
            persona_config=persona_config,
            solo_mode=solo_mode,
            retrieval_variant=retrieval_variant,
            retrieval_kwargs=retrieval_kwargs,
            _tau2_env=tau2_env,
            _tau2_tools=tools,
            _tau2_trajectory=[],
            _user_sim=user_sim,
            _user_state=user_state,
        )

    # ── Environment protocol ────────────────────────────────────────────

    def get_tools(self) -> list[Tool]:
        return list(self._tau2_tools)

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None:
        return None

    def get_system_prompt(self) -> str | None:
        """Append tau2's domain policy to the base system prompt."""
        return self._tau2_env.policy

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Dispatch a tool call into the tau2 domain env.

        Buffers the resulting ToolMessage in _pending_tool_results. When
        the last expected result for the in-flight group lands, atomically
        flushes [pending_asst, *pending_results] into _tau2_trajectory.
        This keeps _tau2_trajectory always-balanced so tau2's set_state
        replay (used by cold deserialize) never sees an orphan tool call.
        """
        tau2_tc = _rollouts_tc_to_tau2(tool_call, requestor="assistant")
        # Note: get_response is synchronous in tau2; tau2 envs don't block on I/O.
        tau2_tm = self._tau2_env.get_response(tau2_tc)
        self._pending_tool_results.append(tau2_tm)
        self._pending_expected_ids.discard(tau2_tc.id)

        # Last result for this group? Flush atomically.
        if not self._pending_expected_ids and self._pending_asst is not None:
            self._tau2_trajectory.append(self._pending_asst)
            self._tau2_trajectory.extend(self._pending_tool_results)
            self._pending_asst = None
            self._pending_tool_results = []

        result = _tau2_tm_to_rollouts(tau2_tm)
        if result.is_error:
            self._num_errors += 1
        self._step_count += 1
        return result

    async def on_assistant_message(
        self,
        message: Message,
        state: AgentState,
    ) -> AgentState:
        """Record the agent message in the tau2 trajectory and, if the agent
        sent plain text (no tool calls), drive the user simulator.

        Also checks agent stop tokens and step/error caps. The runtime
        already consults `requires_confirmation` and calls `exec_tool` for
        tool-bearing messages, so we don't re-dispatch tools here.

        Stop classification uses tau2's own `LLMAgent.is_stop` so any future
        change to tau2's stop-token protocol propagates without code edits
        here.
        """
        from tau2.agent.llm_agent import LLMAgent
        from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
        from tau2.data_model.simulation import TerminationReason

        # First call: reconstruct _tau2_trajectory and _user_state from the
        # seeded trajectory that prepare_messages built. The runtime starts
        # the agent with [system, assistant_greeting, user_sim_first_reply]
        # and then makes its first LLM call; this on_assistant_message fires
        # with the agent's response. Sync everything seeded BEFORE the
        # current agent message — that current message gets recorded by the
        # normal path right below.
        if not self._tau2_trajectory:
            # Drop the trailing assistant message (the one we're about to
            # record below) — sync only what was seeded before it.
            seeded = state.actor.trajectory.messages
            if seeded and seeded[-1].role == "assistant":
                seeded = seeded[:-1]
            self._sync_from_seeded_trajectory(seeded)

        text = _message_text(message)
        tool_calls = message.get_tool_calls()

        # Build the tau2 AssistantMessage. tau2's AssistantMessage accepts
        # either content or tool_calls (or both).
        tau2_tcs = (
            [_rollouts_tc_to_tau2(tc, requestor="assistant") for tc in tool_calls]
            if tool_calls
            else None
        )
        tau2_asst = Tau2AssistantMessage(
            role="assistant",
            content=text or None,
            tool_calls=tau2_tcs,
        )

        # If this turn has tool_calls, defer recording it. exec_tool will
        # collect the matching ToolMessages and flush the whole group into
        # _tau2_trajectory atomically once the last result lands. This
        # keeps _tau2_trajectory always-balanced from tau2's perspective,
        # so set_state replay never sees orphan tool calls.
        if tool_calls:
            assert self._pending_asst is None, (
                "Got new AssistantMessage with tool_calls while previous "
                "tool group is still in flight — runtime invariant violated."
            )
            self._pending_asst = tau2_asst
            self._pending_tool_results = []
            self._pending_expected_ids = {tc.id for tc in tau2_tcs or []}
            return self._check_caps(state)

        # Text-only turn: append directly (no tool group to wait for).
        self._tau2_trajectory.append(tau2_asst)

        # Check agent stop using tau2's own classifier (text-only turns only;
        # tau2's protocol forbids text + tool_calls in the same turn).
        if LLMAgent.is_stop(tau2_asst):
            self._termination_reason = TerminationReason.AGENT_STOP
            return replace(state, stop=StopReason.TASK_COMPLETED)

        # solo_mode: agent is supposed to only emit tool calls or stop. A
        # text-only non-stop turn is an agent error in tau2's solo loop.
        if self.solo_mode:
            self._termination_reason = TerminationReason.AGENT_ERROR
            return replace(state, stop=StopReason.ERROR)

        # Plain text → user simulator speaks next. Mirror what tau2's
        # UserSimulator._generate_next_message does to its own state, but
        # route the LLM call through OUR rollout() instead of tau2's
        # LiteLLM-backed generate().
        updated = await self._run_user_simulator(tau2_asst, state)
        return self._check_caps(updated)

    async def score(self, trajectory: Trajectory) -> Any:
        """Score via tau2's evaluate_simulation.

        Returns a rollouts Score. The tau2 trajectory built up in
        on_assistant_message / exec_tool is what the evaluator needs — the
        rollouts Trajectory argument is present for protocol conformance
        but not used directly.
        """
        from datetime import datetime

        from tau2.data_model.simulation import SimulationRun, TerminationReason
        from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation

        from rollouts.core.eval import Metric, Score

        # _termination_reason is set at the point of detection
        # (on_assistant_message / _run_user_simulator / _check_caps) so
        # we don't need to re-derive it from a rollouts StopReason here.
        termination = self._termination_reason or TerminationReason.MAX_STEPS
        now = datetime.now().isoformat()
        sim = SimulationRun(
            id=f"{self.domain}_{self.task.id}_{now}",
            task_id=self.task.id,
            messages=self._tau2_trajectory,
            termination_reason=termination,
            timestamp=now,
            start_time=now,
            end_time=now,
            duration=0.0,
            agent_cost=0.0,
            user_cost=0.0,
        )
        reward_info = evaluate_simulation(
            simulation=sim,
            task=self.task,
            evaluation_type=EvaluationType.ALL,
            solo_mode=self.solo_mode,
            domain=self.domain,
        )
        passed = 1.0 if reward_info.reward >= 1.0 else 0.0
        return Score(
            metrics=(
                Metric("passed", passed, weight=1.0),
                Metric(
                    "tau2_reward",
                    float(reward_info.reward),
                    weight=0.0,
                    metadata={
                        "termination": termination.value
                        if hasattr(termination, "value")
                        else str(termination),
                        "num_errors": self._num_errors,
                        "num_steps": self._step_count,
                    },
                ),
            )
        )

    async def close(self) -> None:
        """Release resources. tau2 domain envs are pure Python + in-memory
        data — there's nothing external to tear down."""
        return None

    # ── Serialize / Deserialize ─────────────────────────────────────────

    async def serialize(self) -> dict[str, Any]:
        """Emit enough state to reconstruct via cold replay.

        Does not emit a handle to the live tau2 domain env (that's an
        in-memory mutable object). The warm-reattach path is offered via
        `deserialize(data, tau2_env_live=<obj>)`.

        We persist user_state.messages (which is what the user-sim has
        seen) so deserialize can restore the user-sim conversation without
        replaying it through `rollout()`.
        """
        return {
            "env_kind": "tau2",
            "domain": self.domain,
            "task_json": self.task.model_dump_json(exclude_none=True),
            "user_endpoint": (
                self.user_endpoint.to_json() if self.user_endpoint is not None else None
            ),
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "solo_mode": self.solo_mode,
            "retrieval_variant": self.retrieval_variant,
            "retrieval_kwargs": self.retrieval_kwargs,
            "persona_config": (
                self.persona_config.model_dump(mode="json")
                if self.persona_config is not None
                else None
            ),
            "tau2_messages": [m.model_dump(mode="json") for m in self._tau2_trajectory],
            "user_state_messages": (
                [m.model_dump(mode="json") for m in self._user_state.messages]
                if self._user_state is not None
                else []
            ),
            # Pending tool group: an in-flight AssistantMessage(tool_calls) and
            # the ToolMessages received so far. Cold deserialize replays
            # _tau2_trajectory through tau2's set_state (always balanced) and
            # then re-applies these pending results to the new env's DB by
            # calling the corresponding tools directly.
            "pending_asst": (
                self._pending_asst.model_dump(mode="json")
                if self._pending_asst is not None
                else None
            ),
            "pending_tool_results": [m.model_dump(mode="json") for m in self._pending_tool_results],
            "step_count": self._step_count,
            "num_errors": self._num_errors,
            "termination_reason": (
                self._termination_reason.value if self._termination_reason else None
            ),
        }

    @staticmethod
    async def deserialize(
        data: dict[str, Any],
        *,
        tau2_env_live: Any = None,
    ) -> Tau2Environment:
        """Reconstruct a Tau2Environment from serialized state.

        Args:
            data: dict from `serialize()`.
            tau2_env_live: optional live tau2 domain env. If provided (warm
                path), we reuse it directly — caller asserts the DB state is
                consistent with `data['tau2_messages']`. If None (cold path),
                we build a fresh domain env and replay mutating tool calls
                via tau2's `set_state`.
        """
        from tau2.data_model.message import (
            AssistantMessage,
            ToolMessage,
            UserMessage,
        )
        from tau2.data_model.simulation import TerminationReason
        from tau2.data_model.tasks import Task as Tau2Task

        domain: Tau2Domain = data["domain"]
        solo_mode: bool = bool(data.get("solo_mode", False))
        retrieval_variant: str | None = data.get("retrieval_variant")
        retrieval_kwargs: dict[str, Any] | None = data.get("retrieval_kwargs")
        task = Tau2Task.model_validate_json(data["task_json"])
        # user_endpoint may be None when solo_mode=True.
        user_endpoint_raw = data.get("user_endpoint")
        user_endpoint = Endpoint.from_json(user_endpoint_raw) if user_endpoint_raw else None

        tau2_messages = _rehydrate_tau2_messages(
            data["tau2_messages"],
            AssistantMessage=AssistantMessage,
            UserMessage=UserMessage,
            ToolMessage=ToolMessage,
        )

        # Restore the pending tool group (in-flight assistant message + the
        # tool results received before serialize fired).
        pending_asst_raw = data.get("pending_asst")
        pending_asst = (
            AssistantMessage.model_validate(pending_asst_raw) if pending_asst_raw else None
        )
        pending_tool_results = _rehydrate_tau2_messages(
            data.get("pending_tool_results", []),
            AssistantMessage=AssistantMessage,
            UserMessage=UserMessage,
            ToolMessage=ToolMessage,
        )

        if tau2_env_live is not None:
            tau2_env = tau2_env_live
        else:
            tau2_env = _build_tau2_domain_env(
                domain,
                solo_mode=solo_mode,
                retrieval_variant=retrieval_variant,
                retrieval_kwargs=retrieval_kwargs,
                task=task,
            )
            # _tau2_trajectory only contains complete tool groups (the
            # pending-buffer fix in on_assistant_message/exec_tool ensures
            # this), so set_state's balance validator is happy with the
            # full message history — no trimming needed.
            _apply_task_initialization(
                tau2_env,
                task,
                message_history=tau2_messages,
            )
            # Re-apply the pending tool results to the new env's DB. These
            # mutations happened before serialize fired but aren't in
            # _tau2_trajectory yet (the assistant message that owns them
            # is buffered, so we deferred recording the whole group). We
            # call get_response with the original tool_call so the new env
            # sees the same mutations the original env saw.
            if pending_asst is not None and pending_tool_results:
                completed_ids = {tm.id for tm in pending_tool_results}
                for tc in pending_asst.tool_calls or []:
                    if tc.id in completed_ids:
                        # Replay the call — discard the result (we already
                        # have the canonical one in pending_tool_results).
                        tau2_env.get_response(tc)

        # Compute pending_expected_ids = the asst's tool_call ids that
        # haven't been satisfied yet by pending_tool_results.
        pending_expected_ids: set[str] = set()
        if pending_asst is not None:
            done = {tm.id for tm in pending_tool_results}
            pending_expected_ids = {
                tc.id for tc in (pending_asst.tool_calls or []) if tc.id not in done
            }

        tools = [_tau2_tool_to_rollouts(t) for t in tau2_env.get_tools()]

        # Restore persona_config (must be done before _build_user_simulator
        # so the user sim's system prompt matches what was active before
        # serialize).
        from tau2.data_model.persona import PersonaConfig

        persona_raw = data.get("persona_config")
        persona_config = PersonaConfig.model_validate(persona_raw) if persona_raw else None

        # Rebuild the UserSimulator and restore its message history so the
        # next user-sim call sees the same context it had pre-serialize.
        # In solo_mode there's no user simulator at all.
        user_sim: Any = None
        user_state: Any = None
        if not solo_mode:
            user_tools = tau2_env.get_user_tools() if tau2_env.user_tools is not None else None
            user_sim = _build_user_simulator(
                task,
                persona_config=persona_config,
                user_tools=user_tools,
            )
            user_state = user_sim.get_init_state()
            user_state_msgs = _rehydrate_tau2_messages(
                data.get("user_state_messages", []),
                AssistantMessage=AssistantMessage,
                UserMessage=UserMessage,
                ToolMessage=ToolMessage,
            )
            user_state.messages = user_state_msgs

        termination_raw = data.get("termination_reason")
        termination_reason = TerminationReason(termination_raw) if termination_raw else None

        return Tau2Environment(
            domain=domain,
            task=task,
            user_endpoint=user_endpoint,
            max_steps=data.get("max_steps", 200),
            max_errors=data.get("max_errors", 10),
            persona_config=persona_config,
            solo_mode=solo_mode,
            retrieval_variant=retrieval_variant,
            retrieval_kwargs=retrieval_kwargs,
            _tau2_env=tau2_env,
            _tau2_tools=tools,
            _tau2_trajectory=tau2_messages,
            _user_sim=user_sim,
            _user_state=user_state,
            _pending_asst=pending_asst,
            _pending_tool_results=pending_tool_results,
            _pending_expected_ids=pending_expected_ids,
            _step_count=data.get("step_count", 0),
            _num_errors=data.get("num_errors", 0),
            _termination_reason=termination_reason,
        )

    # ── Internal ────────────────────────────────────────────────────────

    async def _run_user_simulator(
        self,
        tau2_asst: Any,
        state: AgentState,
    ) -> AgentState:
        """Drive the user simulator with tau2's own state machine, but call
        OUR rollout() in place of tau2's LiteLLM-backed generate().

        Mirrors tau2.user.user_simulator.UserSimulator._generate_next_message
        and tau2.orchestrator.orchestrator.Orchestrator's USER/ENV loop:

          1. Append the agent's text-only message to user_state.messages.
          2. Build LLM input via system_messages + flip_roles(); route
             through rollout() against our user_endpoint.
          3. If user-sim returned text only → wrap as UserMessage, check
             stop, inject into agent trajectory, done.
          4. If user-sim returned tool calls (telecom only): execute each
             via tau2_env.get_response(requestor="user"), append results
             to user_state and tau2_trajectory, then call user-sim again.
             Loop until the user-sim produces text (the message that
             actually goes to the agent) or hits the step cap.
        """
        from tau2.data_model.simulation import TerminationReason

        # Step 1: tau2's UserSimulator appends the incoming agent message
        # to its own state before the LLM call. Only text-only messages get
        # echoed here (flip_roles raises on assistant tool_calls inside
        # user_state).
        if tau2_asst.has_content() and not tau2_asst.is_tool_call():
            self._user_state.messages.append(tau2_asst)

        # Step 2-4: USER → (ENV → USER)* → AGENT loop. Each iteration runs
        # one user-sim LLM call. If it emits tool calls, execute and loop.
        # Otherwise break and inject the text reply for the agent.
        sim_text: str | None = None
        while True:
            tau2_user_msg, sim_text = await self._invoke_user_sim_once()

            self._user_state.messages.append(tau2_user_msg)
            self._tau2_trajectory.append(tau2_user_msg)

            if self._user_sim.is_stop(tau2_user_msg):
                self._termination_reason = TerminationReason.USER_STOP
                return replace(state, stop=StopReason.TASK_COMPLETED)

            if not tau2_user_msg.is_tool_call():
                break  # text reply — pass to agent

            # User emitted tool calls (telecom user_tools path). Execute
            # each through tau2's env with requestor="user", append the
            # ToolMessages to BOTH user_state (so the next user-sim call
            # sees them) AND _tau2_trajectory (for scoring).
            for tau2_tc in tau2_user_msg.tool_calls or []:
                tau2_tm = self._tau2_env.get_response(tau2_tc)
                # Force role/requestor consistency on the way out.
                self._user_state.messages.append(tau2_tm)
                self._tau2_trajectory.append(tau2_tm)
                if getattr(tau2_tm, "error", False):
                    self._num_errors += 1
                self._step_count += 1

            # Cap check before looping back into the user-sim call.
            if self._step_count >= self.max_steps:
                self._termination_reason = TerminationReason.MAX_STEPS
                return replace(state, stop=StopReason.MAX_TURNS)
            if self._num_errors >= self.max_errors:
                self._termination_reason = TerminationReason.TOO_MANY_ERRORS
                return replace(state, stop=StopReason.TOOL_ERROR)
            # Loop: ENV → USER (next user-sim call sees the tool results).

        # Inject the final user-sim text reply into the agent's rollouts
        # trajectory so the next agent turn sees it.
        injected = Message(role="user", content=sim_text)
        new_trajectory = Trajectory(
            messages=[*state.actor.trajectory.messages, injected],
            completions=state.actor.trajectory.completions,
            metadata={
                **state.actor.trajectory.metadata,
                "tau2_user_turn": self._step_count,
            },
        )
        new_actor = replace(state.actor, trajectory=new_trajectory)
        self._step_count += 1
        return replace(state, actor=new_actor)

    async def _invoke_user_sim_once(self) -> tuple[Any, str]:
        """One user-sim LLM call via rollout(). Returns (tau2_user_msg, text).

        The user-sim's tool list (if any) is passed to the rollout Actor so
        the model knows what tools are available; tool-call extraction
        happens via Message.get_tool_calls() — the same mechanism used for
        the agent.
        """
        from tau2.data_model.message import UserMessage as Tau2UserMessage

        rollouts_messages = _tau2_messages_to_rollouts(
            self._user_state.flip_roles(),
            self._user_state.system_messages,
        )
        # Translate tau2 user_tools to rollouts Tools for this Actor.
        tools_for_actor: list[Tool] = []
        if self._user_sim.tools:
            tools_for_actor = [_tau2_tool_to_rollouts(t) for t in self._user_sim.tools]

        sim_actor = Actor(
            trajectory=Trajectory(messages=rollouts_messages),
            endpoint=self.user_endpoint,
            tools=tools_for_actor,
        )
        sim_actor = await rollout(sim_actor, on_chunk=_silent_stream)
        sim_reply = sim_actor.trajectory.messages[-1]
        sim_text = _message_text(sim_reply)

        # Extract user tool calls (if any). tau2's UserMessage carries
        # tool_calls with requestor="user".
        rollouts_tcs = sim_reply.get_tool_calls() if sim_reply else []
        tau2_tcs = (
            [_rollouts_tc_to_tau2(tc, requestor="user") for tc in rollouts_tcs]
            if rollouts_tcs
            else None
        )

        tau2_user_msg = Tau2UserMessage(
            role="user",
            content=sim_text or None,
            tool_calls=tau2_tcs,
        )
        return tau2_user_msg, sim_text

    def _sync_from_seeded_trajectory(self, agent_messages: list[Message]) -> None:
        """One-shot sync of internal tau2 state with the trajectory the
        runtime started with.

        prepare_messages seeds the agent with tau2's canonical opening:
            [0] system: AGENT_INSTRUCTION
            [1] assistant: DEFAULT_FIRST_AGENT_MESSAGE
            [2] user: user-sim's first reply
        We need _tau2_trajectory and _user_state to start from that same
        point so subsequent on_assistant_message / _run_user_simulator
        calls produce a coherent merged trajectory and user-sim context.

        Tolerance: any leading system messages are skipped. The first
        assistant message becomes a tau2 AssistantMessage; the first user
        message becomes a tau2 UserMessage. Both are appended to
        _tau2_trajectory and to _user_state.messages.

        Called exactly once, on the first on_assistant_message invocation.
        Idempotent against the empty-state guard at the call site.
        """
        from tau2.data_model.message import AssistantMessage as Tau2AssistantMessage
        from tau2.data_model.message import UserMessage as Tau2UserMessage

        for m in agent_messages:
            if m.role == "system":
                continue
            text = _message_text(m)
            if not text:
                continue
            if m.role == "assistant":
                tm = Tau2AssistantMessage(role="assistant", content=text)
                self._tau2_trajectory.append(tm)
                if self._user_state is not None:
                    self._user_state.messages.append(tm)
            elif m.role == "user":
                tm = Tau2UserMessage(role="user", content=text)
                self._tau2_trajectory.append(tm)
                if self._user_state is not None:
                    self._user_state.messages.append(tm)

    def _check_caps(self, state: AgentState) -> AgentState:
        """Enforce max_steps / max_errors caps."""
        from tau2.data_model.simulation import TerminationReason

        if self._step_count >= self.max_steps:
            self._termination_reason = TerminationReason.MAX_STEPS
            return replace(state, stop=StopReason.MAX_TURNS)
        if self._num_errors >= self.max_errors:
            self._termination_reason = TerminationReason.TOO_MANY_ERRORS
            return replace(state, stop=StopReason.TOOL_ERROR)
        return state


# ── Helpers ─────────────────────────────────────────────────────────────────


def _build_tau2_domain_env(
    domain: Tau2Domain,
    *,
    solo_mode: bool = False,
    retrieval_variant: str | None = None,
    retrieval_kwargs: dict[str, Any] | None = None,
    task: Any = None,
) -> Any:
    """Construct a fresh tau2 domain environment.

    `solo_mode` is currently only meaningful for telecom (which is the only
    domain whose `get_environment` accepts the kwarg). When True the
    environment exposes the agent-only protocol — no user simulator turns,
    text-only assistant messages terminate the conversation. See
    Tau2Environment for how this is enforced upstream.

    `retrieval_variant` / `retrieval_kwargs` / `task` are passed to
    banking_knowledge's get_environment to configure the RAG pipeline
    (BM25, embeddings, etc.). Ignored for other domains. See
    src/tau2/knowledge/README.md for variant names and kwarg shapes.
    """
    if domain == "airline":
        if solo_mode:
            raise NotImplementedError(
                "solo_mode is not supported for the airline domain by tau2 "
                "itself (only telecom exposes solo_mode in get_environment)."
            )
        from tau2.domains.airline.environment import get_environment

        return get_environment()
    if domain == "retail":
        if solo_mode:
            raise NotImplementedError(
                "solo_mode is not supported for the retail domain by tau2 "
                "itself (only telecom exposes solo_mode in get_environment)."
            )
        from tau2.domains.retail.environment import get_environment

        return get_environment()
    if domain == "telecom":
        from tau2.domains.telecom.environment import get_environment

        return get_environment(solo_mode=solo_mode)
    if domain == "banking_knowledge":
        if solo_mode:
            raise NotImplementedError(
                "solo_mode is not supported for the banking_knowledge domain by tau2."
            )
        from tau2.domains.banking_knowledge.environment import get_environment

        return get_environment(
            retrieval_variant=retrieval_variant,
            retrieval_kwargs=retrieval_kwargs,
            task=task,
        )
    raise ValueError(f"Unknown tau2 domain: {domain!r}")


def _apply_task_initialization(
    tau2_env: Any,
    task: Any,
    message_history: list[Any],
) -> None:
    """Call `tau2_env.set_state(...)` with the task's initialization data.

    tau2's set_state accepts:
      - initialization_data: optional agent/user DB overrides
      - initialization_actions: optional mutating calls to apply up front
      - message_history: prior tool calls to replay (mutating calls only;
        tau2 validates that responses match)
    """
    from tau2.data_model.tasks import (
        EnvFunctionCall,
        InitializationData,
    )

    init_state = task.initial_state
    init_data = None
    init_actions: list[Any] = []
    if init_state is not None:
        if init_state.initialization_data is not None:
            init_data = InitializationData.model_validate(init_state.initialization_data)
        if init_state.initialization_actions:
            init_actions = [
                EnvFunctionCall.model_validate(a) for a in init_state.initialization_actions
            ]
    tau2_env.set_state(
        initialization_data=init_data,
        initialization_actions=init_actions,
        message_history=message_history,
    )


def _rehydrate_tau2_messages(
    dumped: list[dict[str, Any]],
    *,
    AssistantMessage: Any,
    UserMessage: Any,
    ToolMessage: Any,
) -> list[Any]:
    """Re-validate dumped tau2 messages back into typed objects.

    We dispatch on role since tau2 doesn't ship a discriminated union loader.
    """
    out: list[Any] = []
    for raw in dumped:
        role = raw.get("role")
        if role == "assistant":
            out.append(AssistantMessage.model_validate(raw))
        elif role == "user":
            out.append(UserMessage.model_validate(raw))
        elif role == "tool":
            out.append(ToolMessage.model_validate(raw))
        else:
            raise ValueError(f"Unexpected tau2 message role: {role!r}")
    return out
