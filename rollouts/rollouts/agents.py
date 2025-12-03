# Core agent execution framework

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace

from .progress import tqdm

logger = logging.getLogger(__name__)

# Environment class and other core types are now imported from dtypes

from .dtypes import (
    Actor,
    AgentState,
    Message,
    RunConfig,
    StopReason,
    StreamChunk,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallEnd,
    ToolCall,
    ToolConfirmResult,
    ToolResult,
)
from .providers import (
    rollout_anthropic,
    rollout_openai,
    rollout_sglang,
)

# ── Core Design Philosophy ────────────────────────────────────────────────────
#
# FULL STATE PASSING: Pass full state everywhere rather than using globals.
# Benefits: testable, checkpointable, parallelizable. Cost: verbose signatures.
# Pattern: Always return new state, never mutate in place.
#
# STATE IMMUTABILITY: All core data structures are frozen dataclasses.
# Benefits: time-travel debugging, safe concurrency, easy rollback.
# Cost: O(n) allocations per turn. Assumption: allocation cheaper than debugging.

# Core types (Endpoint, Actor, AgentState, RunConfig, Environment) are now imported from dtypes

# ── Utility functions ────────────────────────────────────────────────────────
# (Imported from providers.py)


async def handle_checkpoint_event(state: 'AgentState', event: str, run_config: 'RunConfig',
                                 session_id: str | None = None) -> None:
    """Handle checkpoint event - emits via on_chunk"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert event is not None
    assert isinstance(event, str)
    assert run_config is not None

    await run_config.on_chunk(StreamChunk(
        event,
        {"turn": state.turn_idx, "session_id": session_id},
    ))


async def stdout_handler(event: StreamEvent):
    """Simple stdout handler for granular streaming events"""
    if isinstance(event, TextDelta):
        print(event.delta, end='', flush=True)
    elif isinstance(event, ThinkingDelta):
        # Magenta color for thinking
        print(f"\033[95m{event.delta}\033[0m", end='', flush=True)
    elif isinstance(event, ToolCallEnd):
        print(f"\n🔧 Calling {event.tool_call.name}({event.tool_call.args})")
    # Note: tool_result events are emitted separately by the agent loop, not by stream aggregators

# ── Core agent functions ──────────────────────────────────────────────────────
# Provider-specific rollout functions and stream handling imported from providers.py


async def confirm_tool_with_feedback(tc: ToolCall, state: AgentState, run_config: 'RunConfig') -> tuple[AgentState, ToolConfirmResult]:
    """Confirm tool execution, returning state and confirmation result"""
    assert tc is not None
    assert isinstance(tc, ToolCall)
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None
    assert state.environment is not None

    if not state.environment.requires_confirmation(tc):
        return state, ToolConfirmResult(proceed=True)

    print(f"\n▶️ Execute `{tc.name}({tc.args})`?")
    print("  [y] Yes, execute")
    print("  [n] No, provide feedback")
    print("  [s] No, skip silently")

    resp = input("Choice: ").strip().lower()

    if resp == 'y':
        return state, ToolConfirmResult(proceed=True)

    elif resp == 'n':
        feedback = input("Why not? Provide guidance: \n").strip()
        result_with_feedback = ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(
                call_id=tc.id,
                ok=False,
                error="Rejected by user"
            ),
            user_message=feedback
        )
        assert result_with_feedback.tool_result is not None
        return state, result_with_feedback

    else:  # Skip silently
        result_skip = ToolConfirmResult(
            proceed=False,
            tool_result=ToolResult(
                call_id=tc.id,
                ok=False,
                error="Skipped by user"
            )
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


def inject_turn_warning(max_turns: int, warning_at: int = 2) -> Callable[[AgentState], AgentState]:
    """Inject warning when N turns remaining.

    Args:
        max_turns: Total turns available
        warning_at: Warn when this many turns remaining (default: 2)

    Returns:
        Handler function that injects warning message

    Example:
        run_config = RunConfig(
            on_step_start=inject_turn_warning(max_turns=5, warning_at=2),
        )
    """
    assert max_turns > 0
    assert warning_at > 0
    assert warning_at < max_turns

    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)
        assert state.turn_idx >= 0

        turns_left = max_turns - state.turn_idx
        if turns_left == warning_at:
            warning = Message(
                role="user",
                content=f"⚠️ You have {warning_at} turns remaining. Please complete your task quickly.",
                tool_calls=[]
            )
            new_trajectory = replace(
                state.actor.trajectory,
                messages=state.actor.trajectory.messages + [warning]
            )
            result_state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))
            assert result_state is not None
            return result_state
        return state

    return handler


def handle_stop_max_turns(max_turns: int) -> Callable[[AgentState], AgentState]:
    """Stop when max turns reached.

    Args:
        max_turns: Maximum number of turns before stopping

    Returns:
        Handler function that stops when turn_idx >= max_turns

    Example:
        run_config = RunConfig(
            handle_stop=handle_stop_max_turns(5),  # Stop after 5 turns
        )
    """
    assert max_turns > 0, "max_turns must be positive"

    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)
        assert state.turn_idx >= 0

        if state.turn_idx >= max_turns:
            result_state = replace(state, stop=StopReason.MAX_TURNS)
            assert result_state.stop is not None
            return result_state
        return state

    return handler


def handle_stop_token_budget(max_tokens: int) -> Callable[[AgentState], AgentState]:
    """Stop when total tokens exceeds budget.

    Example:
        RunConfig(handle_stop=handle_stop_token_budget(100000))
    """
    def handler(state: AgentState) -> AgentState:
        total_tokens = sum(
            len(msg.content or "") for msg in state.actor.trajectory.messages
        )
        if total_tokens >= max_tokens:
            return replace(state, stop=StopReason.MAX_TURNS)  # TODO: Add BUDGET_EXCEEDED
        return state
    return handler


def handle_stop_cost_budget(max_cost_usd: float, cost_fn: Callable[[AgentState], float]) -> Callable[[AgentState], AgentState]:
    """Stop when estimated cost exceeds budget.

    Args:
        max_cost_usd: Maximum cost in USD
        cost_fn: Function that estimates cost from state

    Example:
        def estimate_cost(state):
            # Count tokens, multiply by model pricing
            return tokens * 0.00001

        RunConfig(handle_stop=handle_stop_cost_budget(5.0, estimate_cost))
    """
    def handler(state: AgentState) -> AgentState:
        current_cost = cost_fn(state)
        if current_cost >= max_cost_usd:
            return replace(state, stop=StopReason.MAX_TURNS)  # TODO: Add BUDGET_EXCEEDED
        return state
    return handler


def handle_stop_on_empty_message() -> Callable[[AgentState], AgentState]:
    """Stop when assistant returns empty message (no content, no tool calls).

    This handles cases where the model signals completion by returning an empty
    response (e.g., Claude's end_turn with no content).

    Returns:
        Handler function that stops on empty assistant messages

    Example:
        run_config = RunConfig(
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_on_empty_message(),
            ]),
        )
    """
    def handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)

        # Check if last message is an empty assistant message
        if state.actor.trajectory.messages:
            last_msg = state.actor.trajectory.messages[-1]
            if (last_msg.role == "assistant" and
                not last_msg.content and
                not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls)):
                result_state = replace(state, stop=StopReason.MAX_TURNS)
                assert result_state.stop is not None
                return result_state

        return state

    return handler


def compose_handlers(handlers: list[Callable[[AgentState], AgentState]]) -> Callable[[AgentState], AgentState]:
    """Compose multiple stop handlers into a single handler.

    Handlers are applied in order. If any handler sets a stop reason, that state
    is returned immediately without calling subsequent handlers.

    Args:
        handlers: List of stop handler functions

    Returns:
        Composed handler function

    Example:
        run_config = RunConfig(
            handle_stop=compose_handlers([
                handle_stop_max_turns(10),
                handle_stop_on_empty_message(),
            ]),
        )
    """
    assert handlers, "handlers list cannot be empty"
    assert all(callable(h) for h in handlers), "all handlers must be callable"

    def composed_handler(state: AgentState) -> AgentState:
        assert state is not None
        assert isinstance(state, AgentState)

        current_state = state
        for handler in handlers:
            current_state = handler(current_state)
            # If any handler sets stop, return immediately
            if current_state.stop:
                return current_state

        return current_state

    return composed_handler


async def inject_tool_reminder(state: AgentState, run_config: 'RunConfig') -> AgentState:
    """Remind the agent to use tools"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None

    reminder = Message(
        role="user",
        content="Please use the available tools to complete the task. What calculation would you like to perform?",
        tool_calls=[]
    )
    new_trajectory = replace(
        state.actor.trajectory,
        messages=state.actor.trajectory.messages + [reminder]
    )
    result_state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))
    assert result_state is not None
    return result_state


FullAuto = RunConfig(
    on_chunk=stdout_handler,
    confirm_tool=confirm_tool_with_feedback,  # type: ignore
    handle_tool_error=handle_tool_error,
    on_step_start=inject_turn_warning(max_turns=10),  # Warn at 2 turns remaining
    handle_stop=handle_stop_max_turns(10),            # Stop after 10 turns
    handle_no_tool=inject_tool_reminder,
)


async def rollout(actor: Actor, on_chunk: Callable[[StreamEvent], Awaitable[None]] = stdout_handler,
                  user_message_for_thinking: str | None = None, turn_idx: int = 0, inline_thinking: str | None = None) -> Actor:
    """Route to appropriate provider function using unified API type abstraction.

    This function uses the provider registry to automatically select the correct
    streaming implementation based on the provider and model. Multiple providers
    (e.g., OpenAI, Groq, xAI) may share the same implementation if they use
    compatible APIs.

    Args:
        actor: Current actor state with endpoint and trajectory
        on_chunk: Callback for streaming events
        user_message_for_thinking: Anthropic-specific parameter for thinking context
        turn_idx: Anthropic-specific parameter for turn tracking
        inline_thinking: Anthropic-specific parameter for thinking template

    Returns:
        Updated actor with new message in trajectory
    """
    from rollouts.providers import get_provider_function

    provider = actor.endpoint.provider
    model_id = actor.endpoint.model

    # Get the appropriate provider function via API type mapping
    provider_func = get_provider_function(provider, model_id)

    # Call with provider-specific kwargs if needed
    # Anthropic needs extra params, others don't - but **kwargs makes this flexible
    new_actor = await provider_func(
        actor,
        on_chunk,
        user_message_for_thinking=user_message_for_thinking,
        turn_idx=turn_idx,
        inline_thinking=inline_thinking,
    )
    return new_actor


async def run_agent_step(state: AgentState, rcfg: RunConfig) -> AgentState:
    """Execute one complete turn: LLM call → ALL tool executions → next turn.
    
    Turn atomicity: Execute ALL tools before giving control back to LLM.
    This simplifies reasoning but prevents early stopping or parallel execution.
    """

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

    # DEBUG: Log trajectory state before rollout
    logger.debug(f"🔍 BEFORE rollout() - Turn {state.turn_idx}")
    logger.debug(f"   Trajectory messages count: {len(updated_actor.trajectory.messages)}")
    for i, msg in enumerate(updated_actor.trajectory.messages):
        content_len = len(msg.content) if msg.content else 0
        content_preview = (msg.content[:50] if msg.content else 'None') + '...'
        logger.debug(f"      Message {i} ({msg.role}): {content_len} chars - {content_preview}")

    # Make LLM call
    next_actor = await rollout(updated_actor, rcfg.on_chunk, rcfg.user_message_for_thinking, state.turn_idx, rcfg.inline_thinking)

    # DEBUG: Log what rollout returned
    logger.debug(f"🔍 AFTER rollout() - Turn {state.turn_idx}")
    logger.debug(f"   Trajectory messages count: {len(next_actor.trajectory.messages)}")
    for i, msg in enumerate(next_actor.trajectory.messages):
        content_len = len(msg.content) if msg.content else 0
        content_preview = (msg.content[:50] if msg.content else 'None') + '...'
        logger.debug(f"      Message {i} ({msg.role}): {content_len} chars - {content_preview}")

    # Extract tool calls from last message (if it's an assistant message)
    last_message = next_actor.trajectory.messages[-1] if next_actor.trajectory.messages else None
    tool_calls = []
    if last_message and last_message.role == "assistant":
        tool_calls = last_message.tool_calls if last_message.tool_calls else []

    # Update state with new actor AND pending tools
    current_state = replace(
        state,
        actor=next_actor,
        pending_tool_calls=tool_calls,
        next_tool_idx=0
    )

    # Let environment respond to assistant message (e.g., execute code, provide feedback)
    # This happens AFTER updating state but BEFORE tool processing
    # Only call if we actually have an assistant message
    if state.environment and last_message and last_message.role == "assistant":
        try:
            current_state = await state.environment.on_assistant_message(last_message, current_state)
        except Exception as e:
            logger.error(f"❌ ENVIRONMENT RESPONSE FAILED: {e}")
            logger.error(f"   Environment type: {type(state.environment).__name__}")
            import traceback
            logger.error(f"   Full traceback:\n{traceback.format_exc()}")
            # Re-raise to maintain error handling flow
            raise
    
    # If no tools, we're done with this turn
    if not tool_calls:
        current_state = await rcfg.handle_no_tool(current_state, rcfg)
        # Check if handler added a stop reason
        if current_state.stop:
            return current_state
        # Otherwise increment turn and continue
        return replace(current_state, 
                      turn_idx=current_state.turn_idx + 1,
                      pending_tool_calls=[])

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


async def process_pending_tools(state: AgentState, rcfg: RunConfig) -> AgentState:
    """Resume processing tools from next_tool_idx"""
    assert state.environment is not None, "process_pending_tools requires environment"
    current_state = state

    # SERIALIZE environment state before tool processing
    env_data = await current_state.environment.serialize()
    
    for i in range(state.next_tool_idx, len(state.pending_tool_calls)):
        tool_call = state.pending_tool_calls[i]
        current_state = replace(current_state, next_tool_idx=i)
        
        # Get confirmation result
        current_state, confirm_result = await rcfg.confirm_tool(tool_call, current_state, rcfg)
        
        if confirm_result.proceed:
            # DESERIALIZE fresh environment for each tool call
            fresh_env = await current_state.environment.__class__.deserialize(env_data)
            
            # Execute tool on fresh environment
            tool_result = await fresh_env.exec_tool(tool_call, current_state, rcfg, None)

            # ALWAYS serialize the environment state after tool execution
            # (even if tool failed, environment state like _initialized may have changed)
            env_data = await fresh_env.serialize()

            # DESERIALIZE again to update current_state
            current_state = replace(
                current_state,
                environment=await current_state.environment.__class__.deserialize(env_data)
            )
        else:
            # Use the provided tool result
            tool_result = confirm_result.tool_result
            # TODO: handle None on tool results
        
        # Emit tool result
        assert tool_result
        await rcfg.on_chunk(StreamChunk("tool_result", {
            "call_id": tool_call.id,
            "ok": tool_result.ok,
            "content": tool_result.content,
            "error": tool_result.error
        }))
        
        # Add tool result message
        result_message = Message(
            role="tool",
            content=tool_result.content if tool_result.ok else f"Error: {tool_result.error}",
            tool_call_id=tool_call.id
        )
        
        messages_to_add = [result_message]
        
        # Add user feedback if provided
        if confirm_result.user_message:
            user_msg = Message(
                role="user",
                content=confirm_result.user_message,
                tool_calls=[]
            )
            messages_to_add.append(user_msg)
        
        # Update trajectory with all messages
        updated_trajectory = replace(
            current_state.actor.trajectory,
            messages=current_state.actor.trajectory.messages + messages_to_add
        )
        current_state = replace(
            current_state,
            actor=replace(current_state.actor, trajectory=updated_trajectory)
        )
        
        # Handle tool errors
        current_state = rcfg.handle_tool_error(tool_result, current_state)
        
        # Fire tool_complete event after each tool (if checkpoint strategy supports it)
        if hasattr(rcfg, 'checkpoint_store') and rcfg.checkpoint_store:
            await handle_checkpoint_event(current_state, "tool_complete", rcfg)
        
        # Check if tool requested agent to stop
        if tool_result.stop_reason:
            current_state = replace(current_state, stop=tool_result.stop_reason)
            # Break out of tool processing loop - agent will stop after this turn
            break
    
    # All tools processed
    # print(f"[DEBUG] process_pending_tools done - incrementing turn from {current_state.turn_idx} to {current_state.turn_idx + 1}")
    # print(f"[DEBUG] Stop reason: {current_state.stop}")
    return replace(
        current_state, 
        turn_idx=current_state.turn_idx + 1,
        pending_tool_calls=[],
        next_tool_idx=0
    )


async def run_agent(
    state: AgentState,
    run_config: RunConfig,
    session_id: str | None = None
) -> list[AgentState]:
    """Run agent until stop condition, checkpointing each state"""
    if run_config.checkpoint_store and not session_id:
        session_id = f"session_{int(time.time() * 1000)}"  # ms timestamp

    states = [state]
    current_state = state

    # Initialize inner progress bar for turn-level tracking
    turn_pbar = None
    if run_config.show_progress:
        turn_pbar = tqdm(
            desc="Turns",
            unit="turn",
            disable=False
        )

    while not current_state.stop:
        # Check stop condition via handle_stop callback (allows custom budgets)
        current_state = run_config.handle_stop(current_state)
        if current_state.stop:
            break

        # Tiger Style: Centralize control flow - emit start/end in same scope for clarity
        # Casey: Semantic compression - consistent pattern (start→step→end) repeated each iteration
        await handle_checkpoint_event(current_state, "turn_start", run_config, session_id)

        next_state = await run_agent_step(current_state, run_config)
        current_state = next_state
        states.append(current_state)

        # Update inner progress bar
        if turn_pbar:
            turn_pbar.update(1)
            postfix = {}
            if current_state.stop:
                postfix['stop'] = str(current_state.stop).split('.')[-1]
            turn_pbar.set_postfix(postfix)

        # Checkpoint after each turn completes
        await handle_checkpoint_event(current_state, "turn_end", run_config, session_id)

    # Save final state
    await handle_checkpoint_event(current_state, "final", run_config, session_id)

    # Close progress bar on normal completion
    if turn_pbar:
        turn_pbar.close()

    return states
