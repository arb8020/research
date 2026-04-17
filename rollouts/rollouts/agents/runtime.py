# Core agent execution framework
#
# REFACTOR IN PROGRESS: first-classing effects in the session log.
# See runtime_refactor.md (alongside this file) for framing and the list of
# gross points (G1–G6) referenced by inline TODOs below.
# See /docs/design/session_ownership.md for the full ownership model.

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING

import trio

if TYPE_CHECKING:
    from ..store import SessionStore

from ..core import (
    Endpoint,
    Environment,
    Message,
)
from ..dtypes import (
    LLMCallEnd,
    SemaphoreAcquired,
    SemaphoreWaitStart,
    StopReason,
    StreamChunk,
    StreamError,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallEnd,
    ToolConfirmResult,
    ToolExecutionEnd,
    ToolExecutionStart,
    ToolResult,
    ToolResultReceived,
)
from ..infra_errors import WorkspaceInfraError
from ..progress import tqdm
from .handlers import (
    handle_stop_max_turns,
    inject_tool_reminder,
    inject_turn_warning,
)
from .session_runtime import ensure_persisted_session
from .types import Actor, AgentState, RunConfig

logger = logging.getLogger(__name__)

# ── Core Design Philosophy ────────────────────────────────────────────────────
#
# FULL STATE PASSING: Pass full state everywhere rather than using globals.
# Benefits: testable, checkpointable, parallelizable. Cost: verbose signatures.
# Pattern: Always return new state, never mutate in place.
#
# STATE IMMUTABILITY: All core data structures are frozen dataclasses.
# Benefits: time-travel debugging, safe concurrency, easy rollback.
# Cost: O(n) allocations per turn. Assumption: allocation cheaper than debugging.
#
# PROFILING EVENTS: We emit LLMCallEnd and ToolExecutionEnd events with timing data
# for profiling eval throughput. Currently this is inline in the agent logic.
# TODO: Consider a decorator or context manager pattern to separate timing/logging
# concerns from core agent logic while keeping the code easy to read. The current
# inline approach is explicit but adds noise to the control flow.
#
# TODO: Document scaffold versioning for reproducibility
# Article quote: "On SWE-bench Verified, a popular agentic coding benchmark, simply switching
# the scaffold makes up to an 11% difference for GPT-5 and up to a 15% difference for Kimi K2
# Thinking. We cover the effect of the scaffold in our SWE-bench Verified review. The choice
# of scaffold has the single biggest impact on the overall performance."
#
# Article quote: "Customizing the harness for each model risks hill-climbing on the evaluation
# and makes direct comparisons between models difficult."
#
# Problem: Our scaffold (run_agent → run_agent_step → process_pending_tools) has implicit
# design choices that affect benchmark scores:
# - Tool execution is sequential (not parallel)
# - All tools execute before next LLM turn (turn atomicity)
# - Tool confirmation flow
# - System prompt injection points
#
# Fix: Add SCAFFOLD_VERSION constant and include in eval outputs:
#     SCAFFOLD_VERSION = "1.0.0"  # Bump when changing tool execution, prompts, etc.
# Include in EvalReport for reproducibility.

# ── Utility functions ────────────────────────────────────────────────────────
# (Imported from providers.py)


async def handle_checkpoint_event(
    state: "AgentState", event: str, run_config: "RunConfig", session_id: str | None = None
) -> None:
    """Handle checkpoint event - emits via on_chunk"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert event is not None
    assert isinstance(event, str)
    assert run_config is not None

    await run_config.on_chunk(
        StreamChunk(
            event,
            {"turn": state.turn_idx, "session_id": session_id},
        )
    )


# _message_to_session_message deleted - use Message directly with timestamp field


async def stdout_handler(event: StreamEvent) -> None:
    """Simple stdout handler for granular streaming events"""
    import sys

    if isinstance(event, TextDelta):
        sys.stdout.write(event.delta)
        sys.stdout.flush()
    elif isinstance(event, ThinkingDelta):
        # Magenta color for thinking
        sys.stdout.write(f"\033[95m{event.delta}\033[0m")
        sys.stdout.flush()
    elif isinstance(event, ToolCallEnd):
        logger.info("\n🔧 Calling %s(%s)", event.tool_call.name, event.tool_call.args)
    # Note: tool_result events are emitted separately by the agent loop, not by stream aggregators


# ── Core agent functions ──────────────────────────────────────────────────────
# Provider-specific rollout functions and stream handling imported from providers.py


async def confirm_tool_with_feedback(
    tc: ToolCall, state: AgentState, run_config: "RunConfig"
) -> tuple[AgentState, ToolConfirmResult]:
    """Confirm tool execution, returning state and confirmation result"""
    assert tc is not None
    assert isinstance(tc, ToolCall)
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None
    assert state.environment is not None

    if not state.environment.requires_confirmation(tc):
        return state, ToolConfirmResult(proceed=True)

    logger.info("\n▶️ Execute `%s(%s)`?", tc.name, tc.args)
    logger.info("  [y] Yes, execute")
    logger.info("  [n] No, provide feedback")
    logger.info("  [s] No, skip silently")

    # Intentionally blocking - this is interactive terminal input
    resp = input("Choice: ").strip().lower()  # noqa: ASYNC250

    if resp == "y":
        return state, ToolConfirmResult(proceed=True)

    elif resp == "n":
        # Intentionally blocking - this is interactive terminal input
        feedback = input("Why not? Provide guidance: \n").strip()  # noqa: ASYNC250
        result_with_feedback = ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(tool_call_id=tc.id, is_error=True, error="Rejected by user"),
            user_message=feedback,
        )
        assert result_with_feedback.tool_result is not None
        return state, result_with_feedback

    else:  # Skip silently
        result_skip = ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(tool_call_id=tc.id, is_error=True, error="Skipped by user"),
        )
        assert result_skip.tool_result is not None
        return state, result_skip


def handle_tool_error(result: ToolResult, state: AgentState) -> AgentState:
    """Handle tool execution errors - currently a no-op"""
    assert result is not None
    assert isinstance(result, ToolResult)
    assert state is not None
    assert isinstance(state, AgentState)
    return state


FullAuto = RunConfig(
    on_chunk=stdout_handler,
    confirm_tool=confirm_tool_with_feedback,
    handle_tool_error=handle_tool_error,
    on_step_start=inject_turn_warning(max_turns=10),  # Warn at 2 turns remaining
    handle_stop=handle_stop_max_turns(10),  # Stop after 10 turns
    handle_no_tool=inject_tool_reminder,
)


async def rollout(
    actor: Actor,
    on_chunk: Callable[[StreamEvent], Awaitable[None]] = stdout_handler,
    user_message_for_thinking: str | None = None,
    turn_idx: int = 0,
    inline_thinking: str | None = None,
    cancel_scope: trio.CancelScope | None = None,
    session_id: str | None = None,
) -> Actor:
    """Route to appropriate provider function using unified API type abstraction.

    This function uses the provider registry to automatically select the correct
    streaming implementation based on the provider and model. Multiple providers
    (e.g., OpenAI, Groq, xAI) may share the same implementation if they use
    compatible APIs.

    For token-level generation (TI/TO), call the token-level providers directly:
    - rollout_sglang_token_level() for SGLang
    - rollout_vllm_token_level() for vLLM

    Args:
        actor: Current actor state with endpoint and trajectory
        on_chunk: Callback for streaming events
        user_message_for_thinking: Anthropic-specific parameter for thinking context
        turn_idx: Anthropic-specific parameter for turn tracking
        inline_thinking: Anthropic-specific parameter for thinking template
        cancel_scope: Optional Trio cancel scope for graceful cancellation

    Returns:
        Updated actor with new message in trajectory
    """
    assert actor is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    from ..providers import get_provider_function_by_format

    # Get the appropriate provider function via API format
    provider_func = get_provider_function_by_format(actor.endpoint.api_format)

    # Call with provider-specific kwargs if needed
    # Anthropic needs extra params, others don't - but **kwargs makes this flexible
    new_actor = await provider_func(
        actor,
        on_chunk,
        user_message_for_thinking=user_message_for_thinking,
        turn_idx=turn_idx,
        inline_thinking=inline_thinking,
        cancel_scope=cancel_scope,
        session_id=session_id,
    )
    return new_actor


async def run_agent_step(
    state: AgentState,
    rcfg: RunConfig,
) -> AgentState:
    """Execute one complete turn: LLM call → ALL tool executions → next turn.

    Turn atomicity: Execute ALL tools before giving control back to LLM.
    This simplifies reasoning but prevents early stopping or parallel execution.

    Cancellation is handled automatically by Trio - any await will raise
    trio.Cancelled if the cancel_scope was cancelled.

    Args:
        state: Current agent state
        rcfg: Run configuration (contains cancel_scope for cancellation)
    """
    assert state is not None
    assert rcfg is not None

    # Update debug context for interrupt diagnostics
    try:
        from ..frontends.runner import get_debug_context

        debug_ctx = get_debug_context()
        debug_ctx.turn = state.turn_idx
        debug_ctx.set_phase("agent_step")
    except ImportError:
        pass

    state = rcfg.handle_stop(state)
    if state.stop:
        return state

    # If we have pending tools, resume processing them
    if state.pending_tool_calls:
        return await process_pending_tools(state, rcfg)

    state = rcfg.on_step_start(state)

    # Otherwise, do a new rollout
    available_tools = state.environment.get_tools() if state.environment else []
    updated_actor = replace(state.actor, tools=available_tools)

    # Make LLM call (with cancellation support)
    # If api_limiter is set, acquire slot before making the call
    # This enables two-level concurrency: samples waiting for tools don't hold API slots
    async def do_rollout() -> Actor:
        return await rollout(
            updated_actor,
            rcfg.on_chunk,
            rcfg.user_message_for_thinking,
            state.turn_idx,
            rcfg.inline_thinking,
            cancel_scope=rcfg.cancel_scope,
            session_id=state.session_id,  # For span persistence
        )

    # Time the LLM call
    # ProviderError (rate limits, timeouts) is an operational error - don't crash,
    # just stop this turn and let the user retry
    from ..providers.base import ProviderError

    llm_start_time = time.perf_counter()
    try:
        if rcfg.api_limiter is not None:
            await rcfg.on_chunk(SemaphoreWaitStart(limiter_type="api"))
            wait_start = time.perf_counter()
            async with rcfg.api_limiter:
                wait_duration_ms = (time.perf_counter() - wait_start) * 1000
                await rcfg.on_chunk(
                    SemaphoreAcquired(limiter_type="api", wait_duration_ms=wait_duration_ms)
                )
                next_actor = await do_rollout()
        else:
            next_actor = await do_rollout()
    except ProviderError as e:
        # Operational error (rate limit, timeout, etc) - stop gracefully
        llm_duration_ms = (time.perf_counter() - llm_start_time) * 1000
        await rcfg.on_chunk(
            LLMCallEnd(
                duration_ms=llm_duration_ms,
                provider=updated_actor.endpoint.provider,
                model=updated_actor.endpoint.model,
                tokens_in=None,
                tokens_out=None,
                status="error",
                error=str(e),
            )
        )
        await rcfg.on_chunk(StreamError(error=str(e)))
        return replace(state, stop=StopReason.PROVIDER_ERROR, error=str(e))

    llm_duration_ms = (time.perf_counter() - llm_start_time) * 1000

    # Extract token counts and cost from completion if available
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost: float = 0.0
    if next_actor.trajectory.completions:
        last_completion = next_actor.trajectory.completions[-1]
        if hasattr(last_completion, "usage") and last_completion.usage:
            usage = last_completion.usage
            # input_tokens: Anthropic format; prompt_tokens: OpenAI-compat (SGLang/vLLM)
            tokens_in = getattr(usage, "input_tokens", None) or getattr(
                usage, "prompt_tokens", None
            )
            # Include reasoning tokens in output count for tok/s calculation.
            # output_tokens: Anthropic format; completion_tokens: OpenAI-compat
            output = (
                getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
            )
            reasoning = getattr(usage, "reasoning_tokens", 0) or 0
            tokens_out = output + reasoning if (output or reasoning) else None
            # Extract cost if available
            if hasattr(usage, "cost") and usage.cost:
                cost = usage.cost.total

    # Wide event: LLM call completed
    await rcfg.on_chunk(
        LLMCallEnd(
            duration_ms=llm_duration_ms,
            provider=updated_actor.endpoint.provider,
            model=updated_actor.endpoint.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            status="success",
        )
    )

    # Extract tool calls from last message (if it's an assistant message)
    # TODO(session-refactor G1): reparsing `ToolCallContent` blocks out of the
    # assistant message every turn is the tell that effects aren't first-class.
    # After refactor: assistant emits `ToolCall` entries directly into the session;
    # this extraction step goes away. See runtime_refactor.md.
    last_message = next_actor.trajectory.messages[-1] if next_actor.trajectory.messages else None
    tool_calls = []
    if last_message and last_message.role == "assistant":
        tool_calls = last_message.get_tool_calls()
        await rcfg.on_chunk(
            StreamChunk(
                "tool_calls_detected",
                {
                    "turn": state.turn_idx,
                    "count": len(tool_calls),
                    "tool_calls": [
                        {
                            "tool_call_id": tc.id,
                            "tool_name": tc.name,
                            "has_parse_error": tc.parse_error is not None,
                        }
                        for tc in tool_calls
                    ],
                },
            )
        )

    # Update state with new actor AND pending tools
    # TODO(session-refactor G4): `pending_tool_calls` + `next_tool_idx` in
    # AgentState duplicate information derivable from the session (the latest
    # `ToolCall` entries without matching `ToolResult` entries ARE the pending
    # set). After refactor: AgentState gets thinner; resume derives pending
    # from session. See runtime_refactor.md.
    current_state = replace(state, actor=next_actor, pending_tool_calls=tool_calls, next_tool_idx=0)

    # Persist assistant message immediately after rollout
    # TODO(session-refactor G3): `append_message` is the only sink into the
    # session. After refactor: `session_store.append_entry(entry: SessionEntry)`
    # where SessionEntry includes AssistantTurn / ToolCall / ToolResult as
    # first-class kinds. See runtime_refactor.md.
    if rcfg.session_store and state.session_id and last_message:
        # Session refactor (sub-step 1b): thread explicit leaf cursor.
        msg_to_persist = (
            last_message
            if last_message.parent_id is not None
            else replace(last_message, parent_id=current_state.leaf_id)
        )
        stored = await rcfg.session_store.append_message(state.session_id, msg_to_persist)
        current_state = replace(current_state, leaf_id=stored.id)

    # Let environment respond to assistant message (e.g., execute code, provide feedback)
    # This happens AFTER updating state but BEFORE tool processing
    # Only call if we actually have an assistant message
    # TODO(session-refactor G5): this is an out-of-band channel for
    # environment-initiated reactions that mutates AgentState directly. After
    # refactor: the environment appends effect entries to the session in
    # response to an AssistantTurn, same shape as any other effect — not a
    # special callback that edits state. See runtime_refactor.md.
    on_assistant_message = getattr(state.environment, "on_assistant_message", None)
    if (
        state.environment
        and last_message
        and last_message.role == "assistant"
        and callable(on_assistant_message)
    ):
        try:
            current_state = await on_assistant_message(last_message, current_state)
        except Exception as e:
            logger.exception(f"❌ ENVIRONMENT RESPONSE FAILED: {e}")
            logger.exception(f"   Environment type: {type(state.environment).__name__}")
            import traceback

            logger.exception(f"   Full traceback:\n{traceback.format_exc()}")
            # Re-raise to maintain error handling flow
            raise

    # If no tools, we're done with this turn
    if not tool_calls:
        current_state = await rcfg.handle_no_tool(current_state, rcfg)
        # Check if handler added a stop reason
        if current_state.stop:
            return current_state
        # Otherwise increment turn and continue
        return replace(current_state, turn_idx=current_state.turn_idx + 1, pending_tool_calls=[])

    # Process the pending tools
    return await process_pending_tools(current_state, rcfg)


# TODO: Checkpoint granularity for multi-tool calls
#
# Current behavior: When LLM returns multiple tool calls in one response,
# we execute ALL tools before creating a checkpoint. This means:
#   - add(10) → multiply(3) → divide(5) all execute, THEN checkpoint
#   - If crash occurs during multiply(), we restart from turn beginning
#
# This is usually fine because tool execution is fast, but consider finer
# checkpointing if:
#   - Tools make slow external API calls
#   - Tools have expensive side effects (can't safely re-run)
#   - Running very long tool chains (10+ tools per turn)
#
# Implementation approach: Modify process_pending_tools to yield intermediate
# states after each tool, then checkpoint each yielded state in run_agent.
# See next_tool_idx which already tracks progress within a tool batch.


async def process_pending_tools(
    state: AgentState,
    rcfg: RunConfig,
) -> AgentState:
    """Resume processing tools from next_tool_idx.

    Cancellation is handled automatically by Trio - any await will raise
    trio.Cancelled if the cancel_scope was cancelled.

    Args:
        state: Current agent state with pending tool calls
        rcfg: Run configuration (contains cancel_scope for cancellation)
    """
    assert state is not None
    assert rcfg is not None

    current_state = state
    if current_state.environment is None:
        # Defensive: we can end up with tool calls but no environment when resuming
        # a session whose config didn't record the environment type correctly.
        # Convert this into tool-error messages instead of crashing on an assert.
        guidance = (
            "Tool calls were requested but no environment is configured for tool execution. "
            "Re-run with an explicit environment (e.g. `--env coding`) when resuming this session."
        )

        messages_to_add: list[Message] = []
        for i in range(state.next_tool_idx, len(state.pending_tool_calls)):
            tool_call = state.pending_tool_calls[i]
            await rcfg.on_chunk(
                StreamChunk(
                    "tool_call_dispatch",
                    {
                        "turn": state.turn_idx,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "action": "missing_environment",
                    },
                )
            )
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error="No environment configured",
                content=guidance,
                details={
                    "tool_name": tool_call.name,
                    "reason": "missing_environment",
                },
            )

            await rcfg.on_chunk(
                ToolResultReceived(
                    tool_call_id=tool_call.id,
                    content=tool_result.content,
                    is_error=tool_result.is_error,
                    error=tool_result.error,
                    details=tool_result.details,
                )
            )

            messages_to_add.append(
                Message(
                    role="tool",
                    content=tool_result.content,
                    tool_call_id=tool_call.id,
                    # Session refactor move 2: promote is_error / error.
                    is_error=tool_result.is_error,
                    error=tool_result.error,
                    details=tool_result.details,
                )
            )

        if messages_to_add:
            updated_trajectory = replace(
                current_state.actor.trajectory,
                messages=current_state.actor.trajectory.messages + messages_to_add,
            )
            current_state = replace(
                current_state, actor=replace(current_state.actor, trajectory=updated_trajectory)
            )

            if rcfg.session_store and state.session_id:
                # Session refactor (sub-step 1b): thread leaf cursor.
                for msg in messages_to_add:
                    msg_to_persist = (
                        msg
                        if msg.parent_id is not None
                        else replace(msg, parent_id=current_state.leaf_id)
                    )
                    stored = await rcfg.session_store.append_message(
                        state.session_id, msg_to_persist
                    )
                    current_state = replace(current_state, leaf_id=stored.id)

        return replace(
            current_state,
            stop=StopReason.TOOL_ERROR,
            error=guidance,
            turn_idx=current_state.turn_idx + 1,
            pending_tool_calls=[],
            next_tool_idx=0,
        )

    assert current_state.environment is not None  # Narrowing for type checker

    # SERIALIZE environment state before tool processing
    env_data = await current_state.environment.serialize()

    for i in range(state.next_tool_idx, len(state.pending_tool_calls)):
        tool_call = state.pending_tool_calls[i]
        current_state = replace(current_state, next_tool_idx=i)

        # Check for parse error - if tool call JSON was malformed, return error to model
        # (like verifiers pattern: send parse errors back so model can retry)
        # Track tool execution time (only set if tool actually executes)
        tool_duration_ms: float | None = None

        if tool_call.parse_error:
            await rcfg.on_chunk(
                StreamChunk(
                    "tool_call_dispatch",
                    {
                        "turn": current_state.turn_idx,
                        "tool_call_id": tool_call.id,
                        "tool_name": tool_call.name,
                        "action": "parse_error",
                        "error": tool_call.parse_error,
                    },
                )
            )
            tool_result = ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                error=tool_call.parse_error,
                content="",
            )
            confirm_result = None  # No confirmation for parse errors
        else:
            # Get confirmation result
            current_state, confirm_result = await rcfg.confirm_tool(tool_call, current_state, rcfg)

            if confirm_result.proceed:
                await rcfg.on_chunk(
                    StreamChunk(
                        "tool_call_dispatch",
                        {
                            "turn": current_state.turn_idx,
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "action": "execute",
                        },
                    )
                )
                # DESERIALIZE fresh environment for each tool call
                assert current_state.environment is not None  # Maintained through loop
                fresh_env = await current_state.environment.__class__.deserialize(env_data)

                # Copy runtime attributes (like GPU pool references) that can't be serialized
                if hasattr(fresh_env, "copy_runtime_from"):
                    fresh_env.copy_runtime_from(current_state.environment)

                # Update debug context for interrupt diagnostics
                try:
                    from ..frontends.runner import get_debug_context

                    debug_ctx = get_debug_context()
                    debug_ctx.set_tool(tool_call.name)
                except ImportError:
                    pass

                # Wide event: time the full tool execution
                tool_start_time = time.perf_counter()

                # Emit tool execution start event (for TUI spinner)
                await rcfg.on_chunk(
                    ToolExecutionStart(
                        tool_call_id=tool_call.id,
                        tool_name=tool_call.name,
                    )
                )

                # Execute tool on fresh environment with cancellation support
                # If tool_limiter is set, acquire slot before executing
                # This enables two-level concurrency: samples waiting for API don't hold tool slots
                async def do_exec_tool(
                    env: Environment = fresh_env,
                    tc: ToolCall = tool_call,
                    state: AgentState = current_state,
                ) -> ToolResult:
                    return await env.exec_tool(
                        tc,
                        state,
                        rcfg,
                        cancel_scope=rcfg.cancel_scope,
                    )

                if rcfg.tool_limiter is not None:
                    # Emit semaphore wait event for observability
                    await rcfg.on_chunk(SemaphoreWaitStart(limiter_type="tool"))
                    wait_start = time.perf_counter()
                    async with rcfg.tool_limiter:
                        wait_duration_ms = (time.perf_counter() - wait_start) * 1000
                        await rcfg.on_chunk(
                            SemaphoreAcquired(
                                limiter_type="tool", wait_duration_ms=wait_duration_ms
                            )
                        )
                        try:
                            tool_result = await do_exec_tool()
                        except WorkspaceInfraError as exc:
                            await rcfg.on_chunk(
                                StreamChunk(
                                    "infra_failure_terminal",
                                    {
                                        "turn": current_state.turn_idx,
                                        "tool_call_id": tool_call.id,
                                        "tool_name": tool_call.name,
                                        "kind": exc.kind,
                                        "error": str(exc),
                                    },
                                )
                            )
                            return replace(
                                current_state,
                                stop=StopReason.ERROR,
                                error=str(exc),
                                pending_tool_calls=[],
                            )
                else:
                    try:
                        tool_result = await do_exec_tool()
                    except WorkspaceInfraError as exc:
                        await rcfg.on_chunk(
                            StreamChunk(
                                "infra_failure_terminal",
                                {
                                    "turn": current_state.turn_idx,
                                    "tool_call_id": tool_call.id,
                                    "tool_name": tool_call.name,
                                    "kind": exc.kind,
                                    "error": str(exc),
                                },
                            )
                        )
                        return replace(
                            current_state,
                            stop=StopReason.ERROR,
                            error=str(exc),
                            pending_tool_calls=[],
                        )

                # Calculate tool duration for profiling
                tool_duration_ms = (time.perf_counter() - tool_start_time) * 1000

                # Update debug context - tool execution complete
                try:
                    debug_ctx.set_phase("tool_complete")
                except (NameError, UnboundLocalError):
                    pass

                # ALWAYS serialize the environment state after tool execution
                # (even if tool failed, environment state like _initialized may have changed)
                env_data = await fresh_env.serialize()

                # DESERIALIZE again to update current_state
                assert current_state.environment is not None  # Maintained through loop
                new_env = await current_state.environment.__class__.deserialize(env_data)

                # Copy runtime attributes (like GPU pool references) that can't be serialized
                if hasattr(new_env, "copy_runtime_from"):
                    new_env.copy_runtime_from(current_state.environment)

                current_state = replace(
                    current_state,
                    environment=new_env,
                )
            else:
                await rcfg.on_chunk(
                    StreamChunk(
                        "tool_call_dispatch",
                        {
                            "turn": current_state.turn_idx,
                            "tool_call_id": tool_call.id,
                            "tool_name": tool_call.name,
                            "action": "rejected",
                            "error": (
                                confirm_result.tool_result.error
                                if confirm_result.tool_result is not None
                                else None
                            ),
                        },
                    )
                )
                # Use the provided tool result
                tool_result = confirm_result.tool_result
                # TODO: handle None on tool results

        # Emit tool result
        assert tool_result
        await rcfg.on_chunk(
            ToolResultReceived(
                tool_call_id=tool_call.id,
                content=tool_result.content,
                is_error=tool_result.is_error,
                error=tool_result.error,
                details=tool_result.details,
            )
        )

        # Update debug context - tool result emitted (TUI render complete)
        try:
            debug_ctx.set_phase("tool_result_emitted")
        except (NameError, UnboundLocalError):
            pass

        # Wide event: emit tool execution end with timing (only if tool actually executed)
        if tool_duration_ms is not None:
            await rcfg.on_chunk(
                ToolExecutionEnd(
                    tool_call_id=tool_call.id,
                    tool_name=tool_call.name,
                    duration_ms=tool_duration_ms,
                    status="error" if tool_result.is_error else "success",
                    is_error=tool_result.is_error,
                    result_summary=tool_result.to_summary(tool_call),
                )
            )

        # Add tool result message
        # Always include content - it has structured stdout/stderr even on error
        # Session refactor move 2: is_error / error are now first-class fields on
        # Message, not just on ToolResult. Scorers and renderers read them directly
        # instead of fishing into details. See runtime_refactor.md (G2 closed).
        result_message = Message(
            role="tool",
            content=tool_result.content,
            tool_call_id=tool_call.id,
            is_error=tool_result.is_error,
            error=tool_result.error,
            details=tool_result.details,  # Include UI-only structured data
        )

        messages_to_add = [result_message]

        # Add user feedback if provided (only if we have confirm_result)
        if confirm_result and confirm_result.user_message:
            user_msg = Message(
                role="user",
                content=confirm_result.user_message,
            )
            messages_to_add.append(user_msg)

        # Update trajectory with all messages
        updated_trajectory = replace(
            current_state.actor.trajectory,
            messages=current_state.actor.trajectory.messages + messages_to_add,
        )
        current_state = replace(
            current_state, actor=replace(current_state.actor, trajectory=updated_trajectory)
        )

        # Persist each message after tool execution
        if rcfg.session_store and state.session_id:
            # Session refactor (sub-step 1b): thread leaf cursor.
            for msg in messages_to_add:
                msg_to_persist = (
                    msg
                    if msg.parent_id is not None
                    else replace(msg, parent_id=current_state.leaf_id)
                )
                stored = await rcfg.session_store.append_message(state.session_id, msg_to_persist)
                current_state = replace(current_state, leaf_id=stored.id)

        # Handle tool errors
        current_state = rcfg.handle_tool_error(tool_result, current_state)

        # Check if tool requested agent to stop
        if tool_result.stop_reason:
            current_state = replace(current_state, stop=tool_result.stop_reason)
            # Break out of tool processing loop - agent will stop after this turn
            break

    # All tools processed
    # print(f"[DEBUG] process_pending_tools done - incrementing turn from {current_state.turn_idx} to {current_state.turn_idx + 1}")
    # print(f"[DEBUG] Stop reason: {current_state.stop}")
    return replace(
        current_state, turn_idx=current_state.turn_idx + 1, pending_tool_calls=[], next_tool_idx=0
    )


async def resume_session(
    session_id: str,
    session_store: "SessionStore",
    endpoint: "Endpoint",
    environment: "Environment | None" = None,
) -> AgentState:
    """Load a session and construct an AgentState ready for run_agent().

    This is a helper for resuming sessions. It loads the session from the store,
    converts messages to the runtime format, and builds an AgentState.

    Args:
        session_id: Session ID to resume
        session_store: SessionStore instance
        endpoint: Endpoint to use (may differ from original session)
        environment: Environment to use (may differ from original session)

    Returns:
        AgentState with session_id set, ready to pass to run_agent()

    Raises:
        ValueError: If session not found

    Example:
        state = await resume_session("20241205_143052_a1b2c3", store, endpoint, env)
        states = await run_agent(state, RunConfig(session_store=store))
    """
    trajectory, err = await session_store.get_trajectory(session_id)
    if err or trajectory is None:
        raise ValueError(f"Session not found: {session_id}" + (f" ({err})" if err else ""))

    return AgentState(
        actor=Actor(
            trajectory=trajectory,
            endpoint=endpoint,
            tools=environment.get_tools() if environment else [],
        ),
        environment=environment,
        session_id=session_id,
    )


# TODO(run-agent-run-external-agent-convergence): run_agent (native SDK) and
# run_external_agent (external CLI runtimes) should converge to the same
# signature: (state: AgentState, run_config: RunConfig). Both paths should:
#   - read/write the same session store per turn
#   - treat the environment on AgentState as the tool/MCP surface
#   - return list[AgentState] with consistent stop reasons
#
# Currently run_external_agent doesn't exist as a peer function — external
# agents go through execute_external_attempt which discards the environment,
# has no per-turn persistence, and returns RowAttempt instead of list[AgentState].
#
# Unlocks: running native SDK and external CLI agents through one orchestration
# path; comparing trajectories across runtimes in the same schema; resume after
# crash on the external agent path.
async def run_agent(
    state: AgentState,
    run_config: RunConfig,
) -> list[AgentState]:
    """Run agent until stop condition, checkpointing each state.

    If run_config.cancel_scope is provided and cancelled, raises trio.Cancelled.
    Caller is responsible for handling cancellation at their boundary.

    Session persistence:
    - If run_config.session_store is set, session lifecycle is managed automatically:
      - If state.session_id is None, a new session is created
      - If state.session_id is set, that session is resumed
    - Messages are persisted after each turn
    - Final status and environment state are saved when agent stops
    - The session_id is set on state and available via returned states

    Args:
        state: Initial agent state (set session_id to resume existing session)
        run_config: Run configuration (set session_store for persistence)

    Returns:
        List of agent states. Access session_id via states[-1].session_id
    """
    session_store = run_config.session_store
    current_state = await ensure_persisted_session(state, session_store)

    if session_store and current_state.session_id and not state.session_id:
        if current_state.parent_session_id:
            logger.info(
                f"Created child session: {current_state.session_id} (forked from {current_state.parent_session_id} at message {current_state.branch_point})"
            )
        else:
            logger.info(f"Created session: {current_state.session_id}")

    # Eagerly initialize environment resources before first model call.
    if current_state.environment:
        if hasattr(current_state.environment, "initialize"):
            await current_state.environment.initialize(current_state.session_id)
        elif current_state.session_id and hasattr(current_state.environment, "on_session_start"):
            await current_state.environment.on_session_start(current_state.session_id)

    states = [current_state]

    # Initialize inner progress bar for turn-level tracking
    turn_pbar = None
    if run_config.show_progress:
        turn_pbar = tqdm(desc="Turns", unit="turn", disable=False)

    try:
        while not current_state.stop:
            # Check stop condition via handle_stop callback (allows custom budgets)
            current_state = run_config.handle_stop(current_state)
            if current_state.stop:
                states.append(current_state)  # Include the stopped state
                break

            # Tiger Style: Centralize control flow - emit start/end in same scope for clarity
            await handle_checkpoint_event(
                current_state, "turn_start", run_config, current_state.session_id
            )

            # Run one step - this is where HTTP calls happen
            # Trio will raise Cancelled if cancel_scope.cancel() was called
            next_state = await run_agent_step(current_state, run_config)
            current_state = next_state
            states.append(current_state)

            # Update inner progress bar
            if turn_pbar:
                turn_pbar.update(1)
                postfix = {}
                if current_state.stop:
                    postfix["stop"] = str(current_state.stop).split(".")[-1]
                turn_pbar.set_postfix(postfix)

            # Checkpoint after each turn completes
            await handle_checkpoint_event(
                current_state, "turn_end", run_config, current_state.session_id
            )

    except trio.Cancelled:
        # Convert Trio's cancellation to our domain
        aborted_state = replace(current_state, stop=StopReason.ABORTED)
        states.append(aborted_state)
        current_state = aborted_state

        # Simple cleanup - just save status, let resume handle incomplete state
        with trio.CancelScope(shield=True):
            if session_store and current_state.session_id:
                await session_store.update(
                    current_state.session_id,
                    stop_reason=StopReason.ABORTED,
                )

        # Return states instead of re-raising - caller can check stop reason
        return states

    # Save final state
    await handle_checkpoint_event(current_state, "final", run_config, current_state.session_id)

    # TODO(environment-state-durability): environment state is serialized and
    # saved only at run-end. If the harness crashes mid-run, the session log
    # has the message history but the environment state (which GPU was assigned,
    # what files are on disk, verifier results) is lost.
    #
    # The article's model: environments are cattle, reprovisioned from a recipe
    # rather than checkpointed. For our use case this means: on wake, re-run
    # environment.initialize() from the session config rather than deserializing
    # mutable state. environment.serialize() at run-end is still useful for
    # diagnostics, but should not be the resume mechanism.
    #
    # Near-term fix: move environment.serialize() into process_pending_tools
    # so at least per-tool state is captured. Longer-term: define a provision()
    # recipe on Environment and use it in wake() instead of deserialize().

    # Save final stop reason and environment state
    if session_store and current_state.session_id:
        env_state = None
        if current_state.environment is not None:
            env_state = await current_state.environment.serialize()

        await session_store.update(
            current_state.session_id,
            stop_reason=current_state.stop,
            environment_state=env_state,
        )

    if turn_pbar:
        turn_pbar.close()

    return states
