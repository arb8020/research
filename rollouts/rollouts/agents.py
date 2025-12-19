# Core agent execution framework

import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

import trio

from .progress import tqdm

logger = logging.getLogger(__name__)

# Environment class and other core types are now imported from dtypes

from .dtypes import (
    Actor,
    AgentState,
    Endpoint,
    EndpointConfig,
    Environment,
    EnvironmentConfig,
    Message,
    RunConfig,
    SessionMessage,
    SessionStatus,
    StopReason,
    StreamChunk,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCall,
    ToolCallEnd,
    ToolConfirmResult,
    ToolResult,
    ToolResultReceived,
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


def _message_to_session_message(msg: Message) -> SessionMessage:
    """Convert rollouts Message to SessionMessage for storage."""
    # Handle different content types
    if isinstance(msg.content, str):
        content: str | list[dict[str, Any]] = msg.content
    elif isinstance(msg.content, list):
        # Content blocks - serialize to dicts
        content = []
        for block in msg.content:
            if hasattr(block, "to_dict"):
                content.append(block.to_dict())
            elif hasattr(block, "__dict__"):
                content.append(vars(block))
            else:
                content.append({"type": "unknown", "value": str(block)})
    elif msg.content is None:
        content = ""
    else:
        content = str(msg.content)

    return SessionMessage(
        role=msg.role,
        content=content,
        tool_call_id=msg.tool_call_id,
    )


async def stdout_handler(event: StreamEvent):
    """Simple stdout handler for granular streaming events"""
    if isinstance(event, TextDelta):
        print(event.delta, end="", flush=True)
    elif isinstance(event, ThinkingDelta):
        # Magenta color for thinking
        print(f"\033[95m{event.delta}\033[0m", end="", flush=True)
    elif isinstance(event, ToolCallEnd):
        print(f"\n🔧 Calling {event.tool_call.name}({event.tool_call.args})")
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

    print(f"\n▶️ Execute `{tc.name}({tc.args})`?")
    print("  [y] Yes, execute")
    print("  [n] No, provide feedback")
    print("  [s] No, skip silently")

    resp = input("Choice: ").strip().lower()

    if resp == "y":
        return state, ToolConfirmResult(proceed=True)

    elif resp == "n":
        feedback = input("Why not? Provide guidance: \n").strip()
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
            )
            new_trajectory = replace(
                state.actor.trajectory, messages=state.actor.trajectory.messages + [warning]
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
        total_tokens = sum(len(msg.content or "") for msg in state.actor.trajectory.messages)
        if total_tokens >= max_tokens:
            return replace(state, stop=StopReason.MAX_TURNS)  # TODO: Add BUDGET_EXCEEDED
        return state

    return handler


def handle_stop_cost_budget(
    max_cost_usd: float, cost_fn: Callable[[AgentState], float]
) -> Callable[[AgentState], AgentState]:
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
            if (
                last_msg.role == "assistant"
                and not last_msg.content
                and not last_msg.get_tool_calls()
            ):
                result_state = replace(state, stop=StopReason.MAX_TURNS)
                assert result_state.stop is not None
                return result_state

        return state

    return handler


def compose_handlers(
    handlers: list[Callable[[AgentState], AgentState]],
) -> Callable[[AgentState], AgentState]:
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


async def inject_tool_reminder(state: AgentState, run_config: "RunConfig") -> AgentState:
    """Remind the agent to use tools"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert run_config is not None

    reminder = Message(
        role="user",
        content="Please use the available tools to complete the task. What calculation would you like to perform?",
    )
    new_trajectory = replace(
        state.actor.trajectory, messages=state.actor.trajectory.messages + [reminder]
    )
    result_state = replace(state, actor=replace(state.actor, trajectory=new_trajectory))
    assert result_state is not None
    return result_state


FullAuto = RunConfig(
    on_chunk=stdout_handler,
    confirm_tool=confirm_tool_with_feedback,  # type: ignore
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
    *,
    use_tito: bool = False,
    tokenizer: Any | None = None,
    suffix_ids: tuple[int, ...] | None = None,
) -> Actor:
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
        cancel_scope: Optional Trio cancel scope for graceful cancellation
        use_tito: Enable TI/TO (token-level) generation for RL training
        tokenizer: HuggingFace tokenizer (required if use_tito=True)
        suffix_ids: Pre-computed suffix tokens for multi-turn (computed if None)

    Returns:
        Updated actor with new message in trajectory
    """
    assert actor is not None
    assert on_chunk is not None
    assert callable(on_chunk)

    # TI/TO mode: use token-level providers directly
    if use_tito:
        assert tokenizer is not None, "tokenizer is required when use_tito=True"

        from rollouts.inference.backends import compute_suffix_ids
        from rollouts.providers import rollout_sglang_token_level, rollout_vllm_token_level

        # Compute suffix_ids if not provided
        if suffix_ids is None:
            suffix_ids = tuple(compute_suffix_ids(tokenizer))

        # Route to appropriate token-level provider based on endpoint
        # SGLang uses /generate, vLLM uses /v1/completions
        provider = actor.endpoint.provider
        if provider in ("sglang",):
            provider_func = rollout_sglang_token_level
        else:
            # Default to vLLM-style for openai/vllm providers
            provider_func = rollout_vllm_token_level

        new_actor = await provider_func(
            actor,
            on_chunk,
            tokenizer=tokenizer,
            suffix_ids=list(suffix_ids),
        )
        return new_actor

    # Standard mode: use text-based providers
    from rollouts.providers import get_provider_function

    provider = actor.endpoint.provider
    model_id = actor.endpoint.model

    # Get the appropriate provider function via API type mapping
    provider_func = get_provider_function(provider, model_id)

    # Call with provider-specific kwargs if needed
    # Anthropic needs extra params, others don't - but **kwargs makes this flexible
    # Note: Provider functions don't yet support cancel_scope, but we pass it for future support
    new_actor = await provider_func(
        actor,
        on_chunk,
        user_message_for_thinking=user_message_for_thinking,
        turn_idx=turn_idx,
        inline_thinking=inline_thinking,
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
        if isinstance(msg.content, str):
            content_len = len(msg.content) if msg.content else 0
            content_preview = (msg.content[:50] if msg.content else "None") + "..."
        elif isinstance(msg.content, list):
            content_len = len(msg.content)
            content_preview = f"[{len(msg.content)} blocks]..."
        else:
            content_len = 0
            content_preview = "None..."
        logger.debug(f"      Message {i} ({msg.role}): {content_len} chars - {content_preview}")

    # Make LLM call (with cancellation support)
    next_actor = await rollout(
        updated_actor,
        rcfg.on_chunk,
        rcfg.user_message_for_thinking,
        state.turn_idx,
        rcfg.inline_thinking,
        cancel_scope=rcfg.cancel_scope,
        use_tito=rcfg.use_tito,
        tokenizer=rcfg.tokenizer,
        suffix_ids=rcfg.suffix_ids,
    )

    # DEBUG: Log what rollout returned
    logger.debug(f"🔍 AFTER rollout() - Turn {state.turn_idx}")
    logger.debug(f"   Trajectory messages count: {len(next_actor.trajectory.messages)}")
    for i, msg in enumerate(next_actor.trajectory.messages):
        if isinstance(msg.content, str):
            content_len = len(msg.content) if msg.content else 0
            content_preview = (msg.content[:50] if msg.content else "None") + "..."
        elif isinstance(msg.content, list):
            content_len = len(msg.content)
            content_preview = f"[{len(msg.content)} blocks]..."
        else:
            content_len = 0
            content_preview = "None..."
        logger.debug(f"      Message {i} ({msg.role}): {content_len} chars - {content_preview}")

    # Extract tool calls from last message (if it's an assistant message)
    last_message = next_actor.trajectory.messages[-1] if next_actor.trajectory.messages else None
    tool_calls = []
    if last_message and last_message.role == "assistant":
        tool_calls = last_message.get_tool_calls()

    # Update state with new actor AND pending tools
    current_state = replace(state, actor=next_actor, pending_tool_calls=tool_calls, next_tool_idx=0)

    # Persist assistant message immediately after rollout
    if rcfg.session_store and state.session_id and last_message:
        session_msg = _message_to_session_message(last_message)
        await rcfg.session_store.append_message(state.session_id, session_msg)

    # Let environment respond to assistant message (e.g., execute code, provide feedback)
    # This happens AFTER updating state but BEFORE tool processing
    # Only call if we actually have an assistant message
    if state.environment and last_message and last_message.role == "assistant":
        try:
            current_state = await state.environment.on_assistant_message(
                last_message, current_state
            )
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
    assert state.environment is not None, "process_pending_tools requires environment"
    assert state is not None
    assert rcfg is not None

    current_state = state

    # SERIALIZE environment state before tool processing
    env_data = await current_state.environment.serialize()

    for i in range(state.next_tool_idx, len(state.pending_tool_calls)):
        tool_call = state.pending_tool_calls[i]
        current_state = replace(current_state, next_tool_idx=i)

        # Check for parse error - if tool call JSON was malformed, return error to model
        # (like verifiers pattern: send parse errors back so model can retry)
        if tool_call.parse_error:
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
                # DESERIALIZE fresh environment for each tool call
                fresh_env = await current_state.environment.__class__.deserialize(env_data)

                # Execute tool on fresh environment with cancellation support
                tool_result = await fresh_env.exec_tool(
                    tool_call,
                    current_state,
                    rcfg,
                    cancel_scope=rcfg.cancel_scope,
                )

                # ALWAYS serialize the environment state after tool execution
                # (even if tool failed, environment state like _initialized may have changed)
                env_data = await fresh_env.serialize()

                # DESERIALIZE again to update current_state
                current_state = replace(
                    current_state,
                    environment=await current_state.environment.__class__.deserialize(env_data),
                )
            else:
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

        # Add tool result message
        # Always include content - it has structured stdout/stderr even on error
        result_message = Message(
            role="tool",
            content=tool_result.content,
            tool_call_id=tool_call.id,
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
            for msg in messages_to_add:
                session_msg = _message_to_session_message(msg)
                await rcfg.session_store.append_message(state.session_id, session_msg)

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
    session_store: Any,
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
    from .dtypes import (
        ContentBlock,
        ImageContent,
        TextContent,
        ThinkingContent,
        ToolCallContent,
        Trajectory,
    )

    session, err = await session_store.get(session_id)
    if err or session is None:
        raise ValueError(f"Session not found: {session_id}" + (f" ({err})" if err else ""))

    def deserialize_content(content: str | list[dict] | None) -> str | list[ContentBlock] | None:
        """Convert serialized content back to ContentBlock objects."""
        if content is None or isinstance(content, str):
            return content
        if not isinstance(content, list):
            return content

        blocks: list[ContentBlock] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            if block_type == "text":
                blocks.append(
                    TextContent(
                        type="text",
                        text=item.get("text", ""),
                        text_signature=item.get("text_signature"),
                    )
                )
            elif block_type == "thinking":
                blocks.append(
                    ThinkingContent(
                        type="thinking",
                        thinking=item.get("thinking", ""),
                        thinking_signature=item.get("thinking_signature"),
                    )
                )
            elif block_type == "toolCall":
                blocks.append(
                    ToolCallContent(
                        type="toolCall",
                        id=item.get("id", ""),
                        name=item.get("name", ""),
                        arguments=item.get("arguments", {}),
                        thought_signature=item.get("thought_signature"),
                    )
                )
            elif block_type == "image":
                blocks.append(
                    ImageContent(
                        type="image",
                        image_url=item.get("image_url", ""),
                        detail=item.get("detail"),
                    )
                )
        return blocks if blocks else None

    # Convert SessionMessage -> Message
    messages = [
        Message(
            role=m.role,
            content=deserialize_content(m.content),
            tool_call_id=m.tool_call_id,
        )
        for m in session.messages
    ]

    trajectory = Trajectory(messages=messages)

    return AgentState(
        actor=Actor(
            trajectory=trajectory,
            endpoint=endpoint,
            tools=environment.get_tools() if environment else [],
        ),
        environment=environment,
        session_id=session_id,
    )


def _endpoint_to_config(endpoint: "Endpoint") -> "EndpointConfig":
    """Convert Endpoint to serializable EndpointConfig."""
    from .dtypes import EndpointConfig

    return EndpointConfig(
        model=endpoint.model,
        provider=endpoint.provider,
        temperature=endpoint.temperature,
        max_tokens=endpoint.max_tokens,
        thinking=endpoint.thinking is not None and endpoint.thinking.get("type") == "enabled",
        thinking_budget=endpoint.thinking.get("budget_tokens") if endpoint.thinking else None,
    )


def _environment_to_config(
    environment: "Environment | None", confirm_tools: bool = False
) -> "EnvironmentConfig":
    """Convert Environment to serializable EnvironmentConfig."""
    from .dtypes import EnvironmentConfig

    config = {"confirm_tools": confirm_tools}
    if environment is None:
        return EnvironmentConfig(type="none", config=config)
    # Use class name as type, environment should provide its own config via serialize()
    return EnvironmentConfig(
        type=type(environment).__name__,
        config=config,  # Config details stored in environment_state
    )


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
    current_state = state

    # Session creation: if session_store but no session_id, create one
    if session_store and not current_state.session_id:
        session = await session_store.create(
            endpoint=_endpoint_to_config(current_state.actor.endpoint),
            environment=_environment_to_config(
                current_state.environment, current_state.confirm_tools
            ),
            parent_id=current_state.parent_session_id,
            branch_point=current_state.branch_point,
        )
        current_state = replace(current_state, session_id=session.session_id)
        if current_state.parent_session_id:
            logger.info(
                f"Created child session: {session.session_id} (forked from {current_state.parent_session_id} at message {current_state.branch_point})"
            )
        else:
            logger.info(f"Created session: {session.session_id}")

        # Persist initial messages (system prompt, user message, etc.)
        for msg in current_state.actor.trajectory.messages:
            session_msg = _message_to_session_message(msg)
            await session_store.append_message(session.session_id, session_msg)

    elif session_store and current_state.session_id:
        # Resuming existing session - sync trajectory with persisted state.
        #
        # The session file is the source of truth. When resuming after an interrupt,
        # the trajectory might be out of sync with the file (messages persisted during
        # the interrupted run may or may not be in the trajectory depending on timing).
        #
        # Strategy: Build a merged trajectory from file + any NEW messages in current trajectory.
        # A message is "new" if it's not already in the file (by content/tool_call_id match).
        session, _ = await session_store.get(current_state.session_id)
        if session:
            # Build set of existing message signatures for dedup
            existing_signatures: set[str] = set()
            for msg in session.messages:
                if msg.role == "tool" and msg.tool_call_id:
                    existing_signatures.add(f"tool:{msg.tool_call_id}")
                elif msg.role == "assistant":
                    # Use first 200 chars of content as signature
                    existing_signatures.add(f"assistant:{str(msg.content)[:200]}")
                elif msg.role == "user":
                    existing_signatures.add(f"user:{str(msg.content)[:200]}")

            # Find truly new messages in trajectory (not in file)
            new_messages = []
            for msg in current_state.actor.trajectory.messages:
                if msg.role == "tool" and msg.tool_call_id:
                    sig = f"tool:{msg.tool_call_id}"
                elif msg.role == "assistant":
                    sig = f"assistant:{str(msg.content)[:200]}"
                elif msg.role == "user":
                    sig = f"user:{str(msg.content)[:200]}"
                else:
                    sig = f"{msg.role}:{str(msg.content)[:200]}"

                if sig not in existing_signatures:
                    new_messages.append(msg)
                    existing_signatures.add(sig)  # Don't add same new message twice

            # Persist only the truly new messages
            for msg in new_messages:
                session_msg = _message_to_session_message(msg)
                await session_store.append_message(current_state.session_id, session_msg)

    # Notify environment of session start (for setup like git worktrees)
    if current_state.environment and current_state.session_id:
        if hasattr(current_state.environment, "on_session_start"):
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

        # Shield cleanup operations from further cancellation
        with trio.CancelScope(shield=True):
            await handle_checkpoint_event(
                current_state, "final", run_config, current_state.session_id
            )

            # Save session with aborted status
            if session_store and current_state.session_id:
                env_state = None
                if current_state.environment is not None:
                    env_state = await current_state.environment.serialize()
                await session_store.update(
                    current_state.session_id,
                    status=SessionStatus.ABORTED,
                    environment_state=env_state,
                )

        # Return states instead of re-raising - caller can check stop reason
        return states

    # Save final state
    await handle_checkpoint_event(current_state, "final", run_config, current_state.session_id)

    # Save final session status and environment state
    if session_store and current_state.session_id:
        if current_state.stop == StopReason.TASK_COMPLETED:
            status = SessionStatus.COMPLETED
        elif current_state.stop == StopReason.ABORTED:
            status = SessionStatus.ABORTED
        elif current_state.stop in (StopReason.MAX_TURNS,):
            status = SessionStatus.TRUNCATED
        else:
            status = SessionStatus.PENDING

        env_state = None
        if current_state.environment is not None:
            env_state = await current_state.environment.serialize()

        await session_store.update(
            current_state.session_id,
            status=status,
            environment_state=env_state,
        )

    if turn_pbar:
        turn_pbar.close()

    return states
