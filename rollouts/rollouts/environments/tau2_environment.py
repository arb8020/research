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

- Telecom user_tools (the only domain where the user executes tools). Starts
  text-only. A follow-up can wire user tool calls through the same
  translation layer we use for agent tools.
- Streaming events (StreamEvent) for user-sim calls. We pass a silent
  handler, same as DialogueEnvironment.
- Persona overrides beyond what the task carries. tau2's runtime
  PersonaConfig is not plumbed.

## Known limitation: parallel mutating tool calls

The rollouts runtime serialize/deserialize cycle around `exec_tool` works
fine for read-only parallel tool calls and for single mutating tool calls.
For PARALLEL MUTATING tool calls (an AssistantMessage with N>1 tool calls
where multiple of them mutate the database), we trim the unfinished group
from tau2's set_state replay so the validator passes — but that means
mutations from completed tools in the partial group are lost across the
deserialize. Verified safe for the smoke set (retail tasks); revisit when
adding a benchmark that depends on parallel mutating calls.

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

Tau2Domain = Literal["airline", "retail", "telecom"]


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
    """No-op stream handler for the user-sim rollout."""
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
    user_endpoint: Endpoint
    max_steps: int = 200
    max_errors: int = 10
    # Optional tau2 PersonaConfig — runtime persona overrides (verbosity,
    # interrupt tendency, etc.) layered on top of the task's baked-in
    # persona. None = tau2's default (no overrides), matching tau2 run.
    persona_config: Any = None

    # Populated by create(); internal mutable state.
    _tau2_env: Any = field(default=None, repr=False)
    _tau2_tools: list[Tool] = field(default_factory=list, repr=False)
    _tau2_trajectory: list[Any] = field(default_factory=list, repr=False)
    _user_sim: Any = field(default=None, repr=False)
    _user_state: Any = field(default=None, repr=False)
    _step_count: int = field(default=0, repr=False)
    _num_errors: int = field(default=0, repr=False)
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
        user_endpoint: Endpoint,
        max_steps: int = 200,
        max_errors: int = 10,
        persona_config: Any = None,
    ) -> Tau2Environment:
        """Build a Tau2Environment with a fresh tau2 domain env.

        `persona_config` (optional `PersonaConfig`) plumbs through tau2's
        runtime persona layer. The same value should be used by
        prepare_messages when building its throwaway UserSimulator so the
        bootstrap user-sim call uses the same prompt as later turns.
        """
        tau2_env = _build_tau2_domain_env(domain)
        _apply_task_initialization(tau2_env, task, message_history=[])
        tools = [_tau2_tool_to_rollouts(t) for t in tau2_env.get_tools()]
        # Telecom's user simulator has its own tools; other domains don't.
        # tau2_env.get_user_tools() raises ValueError when user_tools is
        # absent, so we guard with the live attribute check rather than
        # try/except.
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

        Maintains the parallel tau2 trajectory so scoring sees the full
        conversation, not just the rollouts side.
        """
        tau2_tc = _rollouts_tc_to_tau2(tool_call, requestor="assistant")
        # Note: get_response is synchronous in tau2; tau2 envs don't block on I/O.
        tau2_tm = self._tau2_env.get_response(tau2_tc)
        self._tau2_trajectory.append(tau2_tm)
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

        # Record on tau2 side. tau2's AssistantMessage accepts either
        # content or tool_calls (or both).
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
        self._tau2_trajectory.append(tau2_asst)

        # Check agent stop using tau2's own classifier (text-only turns only;
        # tau2's protocol forbids text + tool_calls in the same turn).
        if not tool_calls and LLMAgent.is_stop(tau2_asst):
            self._termination_reason = TerminationReason.AGENT_STOP
            return replace(state, stop=StopReason.TASK_COMPLETED)

        # If the agent called tools, control returns to run_agent_step which
        # will invoke exec_tool for each. We don't call the user sim.
        if tool_calls:
            return self._check_caps(state)

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
            solo_mode=False,
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
            "user_endpoint": self.user_endpoint.to_json(),
            "max_steps": self.max_steps,
            "max_errors": self.max_errors,
            "persona_config": (
                self.persona_config.model_dump(mode="json")
                if self.persona_config is not None
                else None
            ),
            "tau2_messages": [m.model_dump(mode="json") for m in self._tau2_trajectory],
            "user_state_messages": [m.model_dump(mode="json") for m in self._user_state.messages],
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
        task = Tau2Task.model_validate_json(data["task_json"])
        user_endpoint = Endpoint.from_json(data["user_endpoint"])

        tau2_messages = _rehydrate_tau2_messages(
            data["tau2_messages"],
            AssistantMessage=AssistantMessage,
            UserMessage=UserMessage,
            ToolMessage=ToolMessage,
        )

        if tau2_env_live is not None:
            tau2_env = tau2_env_live
        else:
            tau2_env = _build_tau2_domain_env(domain)
            # tau2's set_state validator requires every assistant/user
            # tool_call in the replayed history to be followed by a matching
            # ToolMessage. Mid-turn serializations (rollouts serializes the
            # env before dispatching tool calls) can leave trailing orphan
            # tool_calls — those will be re-executed via exec_tool after
            # this deserialize, so we trim them from the replay set here.
            # We keep the originals in our own _tau2_trajectory for scoring.
            replayable = _trim_trailing_orphan_tool_calls(tau2_messages)
            _apply_task_initialization(
                tau2_env,
                task,
                message_history=replayable,
            )

        tools = [_tau2_tool_to_rollouts(t) for t in tau2_env.get_tools()]

        # Restore persona_config (must be done before _build_user_simulator
        # so the user sim's system prompt matches what was active before
        # serialize).
        from tau2.data_model.persona import PersonaConfig

        persona_raw = data.get("persona_config")
        persona_config = PersonaConfig.model_validate(persona_raw) if persona_raw else None

        # Rebuild the UserSimulator and restore its message history so the
        # next user-sim call sees the same context it had pre-serialize.
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
            _tau2_env=tau2_env,
            _tau2_tools=tools,
            _tau2_trajectory=tau2_messages,
            _user_sim=user_sim,
            _user_state=user_state,
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
                self._user_state.messages.append(tm)
            elif m.role == "user":
                tm = Tau2UserMessage(role="user", content=text)
                self._tau2_trajectory.append(tm)
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


def _build_tau2_domain_env(domain: Tau2Domain) -> Any:
    """Construct a fresh tau2 domain environment."""
    if domain == "airline":
        from tau2.domains.airline.environment import get_environment

        return get_environment()
    if domain == "retail":
        from tau2.domains.retail.environment import get_environment

        return get_environment()
    if domain == "telecom":
        from tau2.domains.telecom.environment import get_environment

        # solo_mode=False: preserve user simulator path (telecom supports solo).
        return get_environment(solo_mode=False)
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


def _trim_trailing_orphan_tool_calls(messages: list[Any]) -> list[Any]:
    """Trim the tail of the message list so every tool_call has its matching
    ToolMessage in the returned prefix.

    tau2's `set_state(message_history=...)` validates that every assistant/
    user `tool_call` is followed by its matching `ToolMessage` (matched by
    `id`). Mid-turn serialization can leave the trajectory in three bad
    states:
      (a) trailing AssistantMessage with N tool_calls and 0 tool results
          (serialized after `on_assistant_message`, before any `exec_tool`).
      (b) trailing AssistantMessage with N tool_calls and K<N tool results
          (parallel-tool case: serialized between exec_tool calls).
      (c) trailing ToolMessages from a previous turn whose AssistantMessage
          is intact — fine to keep (rare, but defensive).

    TODO(tau2-fidelity): case (b) is the parallel-mutating-tool fidelity gap.
    For an AssistantMessage with N>1 mutating tool calls where K<N have
    completed, we drop the whole assistant message from the replay set —
    so on the post-deserialize env, mutations from the K completed tools
    are LOST. Verified safe for retail/airline (mostly serial mutation).
    Likely real cost on tasks that issue parallel exchange/refund/etc.
    Fix: defer appending the AssistantMessage to _tau2_trajectory until
    all its tool results land, and persist the partial-result buffer in
    serialize() so we can resume mid-group.

    Our trim algorithm: walk from the tail; while the last message is a
    tool_call message OR a ToolMessage whose preceding tool_call(s) aren't
    fully accounted for, drop it. We keep popping until the remaining
    prefix is "balanced" — every tool_call has all its tool_messages.

    Trimmed messages stay in `_tau2_trajectory` (used for scoring at end
    of run); they're only excluded from tau2's set_state replay.

    Earlier orphans (in the middle of the history) indicate corrupt state
    and are left for set_state to surface as an error.
    """
    trimmed = list(messages)
    # Iteratively remove trailing items until the suffix is balanced.
    while trimmed:
        # Scan forward: count expected tool results at each tool_call,
        # check if all are present in the trimmed list.
        balanced, last_unbalanced_idx = _check_balanced(trimmed)
        if balanced:
            return trimmed
        # Drop everything from the unbalanced tool_call onward.
        trimmed = trimmed[:last_unbalanced_idx]
    return trimmed


def _check_balanced(messages: list[Any]) -> tuple[bool, int]:
    """Check if every tool_call in `messages` has its matching ToolMessage.

    Returns (balanced, first_unbalanced_index). When balanced is True,
    first_unbalanced_index is len(messages). When False, it points at the
    earliest tool_call message whose results are incomplete.
    """
    i = 0
    n = len(messages)
    while i < n:
        m = messages[i]
        if hasattr(m, "is_tool_call") and m.is_tool_call():
            tcs = m.tool_calls or []
            needed_ids = {tc.id for tc in tcs}
            j = i + 1
            while j < n and needed_ids:
                next_m = messages[j]
                # tau2 ToolMessage has an `id` matching the tool_call id.
                tm_id = getattr(next_m, "id", None)
                role = getattr(next_m, "role", None)
                if role == "tool" and tm_id in needed_ids:
                    needed_ids.remove(tm_id)
                    j += 1
                else:
                    break
            if needed_ids:
                return (False, i)
            i = j
        else:
            i += 1
    return (True, n)
