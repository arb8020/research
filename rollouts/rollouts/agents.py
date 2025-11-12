# Core agent execution framework

import sys
import json
import logging
import os
import time
from dataclasses import dataclass, field, replace
from typing import (Any, Dict, List, Optional, Tuple, Callable,
                   AsyncIterator, Awaitable)

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types import CompletionUsage

from dacite import from_dict

import copy

logger = logging.getLogger(__name__)

# Environment class and other core types are now imported from dtypes

from .dtypes import (
    Tool, ToolCall, ToolResult, ToolConfirmResult, StopReason, StreamChunk, Message, Usage, Choice, ChatCompletion, Actor, AgentState, RunConfig
)
from .providers import (
    add_cache_control_to_last_content,
    aggregate_anthropic_stream,
    aggregate_stream,
    rollout_anthropic,
    rollout_openai,
    rollout_sglang,
    sanitize_request_for_logging,
    verbose,
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
                                 session_id: Optional[str] = None) -> None:
    """Handle checkpoint event if configured - stub for now"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert event is not None
    assert isinstance(event, str)
    assert run_config is not None
    pass

async def stdout_handler(chunk: StreamChunk):
    """Simple stdout handler for chunks"""
    if chunk.kind == "token":
        print(chunk.data["text"], end='', flush=True)
    elif chunk.kind == "tool_call_complete":
        print(f"\n🔧 Calling {chunk.data['name']}({chunk.data['args']})")
    elif chunk.kind == "tool_result":
        status = "✓" if chunk.data["ok"] else "✗"
        print(f"\n  {status} {chunk.data['content'][:100]}...")
    elif chunk.kind == "thinking":
        print(f"\033[95m{chunk.data['text']}\033[0m", end='', flush=True)

# ── Core agent functions ──────────────────────────────────────────────────────
# Provider-specific rollout functions and stream handling imported from providers.py

async def confirm_tool_with_feedback(tc: ToolCall, state: AgentState, run_config: 'RunConfig') -> Tuple[AgentState, ToolConfirmResult]:
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

def inject_turn_warning(state: AgentState) -> AgentState:
    """Warn when 2 turns remaining"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert state.turn_idx >= 0
    assert state.max_turns > 0
    assert state.turn_idx <= state.max_turns

    turns_left = state.max_turns - state.turn_idx
    if turns_left == 2:
        warning = Message(
            role="user",
            content="⚠️ You have 2 turns remaining. Please complete your task quickly.",
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

def handle_stop_max_turns(state: AgentState) -> AgentState:
    """Stop when max turns reached"""
    assert state is not None
    assert isinstance(state, AgentState)
    assert state.turn_idx >= 0
    assert state.max_turns > 0

    if state.turn_idx >= state.max_turns:
        result_state = replace(state, stop=StopReason.MAX_TURNS)
        assert result_state.stop is not None
        return result_state
    return state

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
    confirm_tool=confirm_tool_with_feedback, #type: ignore
    handle_tool_error=handle_tool_error,
    on_step_start=inject_turn_warning,
    handle_stop=handle_stop_max_turns,       # Add this
    handle_no_tool=inject_tool_reminder,
)

async def rollout(actor: Actor, on_chunk: Callable[[StreamChunk], Awaitable[None]]=stdout_handler,
                  user_message_for_thinking: Optional[str] = None, turn_idx: int = 0, inline_thinking: Optional[str] = None) -> Actor:
    provider = actor.endpoint.provider
    if provider == "openai":
        new_actor = await rollout_openai(actor, on_chunk)
    elif provider in ("sglang", "vllm"):  # Accept both for backwards compat
        new_actor = await rollout_sglang(actor, on_chunk)
    elif provider == "anthropic":
        new_actor = await rollout_anthropic(actor, on_chunk, user_message_for_thinking, turn_idx, inline_thinking)
    else:
        print(f"Invalid provider {actor.endpoint.provider})")
        sys.exit(0)
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
    
    # Make LLM call
    next_actor = await rollout(updated_actor, rcfg.on_chunk, rcfg.user_message_for_thinking, state.turn_idx, rcfg.inline_thinking)
    
    # Extract tool calls
    last_message = next_actor.trajectory.messages[-1]
    tool_calls = last_message.tool_calls if last_message.tool_calls else []
    
    # Update state with new actor AND pending tools
    current_state = replace(
        state, 
        actor=next_actor,
        pending_tool_calls=tool_calls,
        next_tool_idx=0
    )
    
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
            
            if tool_result.ok:
                # SERIALIZE the updated environment state
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
    #print(f"[DEBUG] process_pending_tools done - incrementing turn from {current_state.turn_idx} to {current_state.turn_idx + 1}")
    #print(f"[DEBUG] Stop reason: {current_state.stop}")
    return replace(
        current_state, 
        turn_idx=current_state.turn_idx + 1,
        pending_tool_calls=[],
        next_tool_idx=0
    )


async def run_agent(
    state: AgentState, 
    run_config: RunConfig,
    session_id: Optional[str] = None
) -> List[AgentState]:
    """Run agent until stop condition, checkpointing each state"""
    if run_config.checkpoint_store and not session_id:
        session_id = f"session_{int(time.time() * 1000)}"  # ms timestamp
    
    states = [state]
    current_state = state
    
    try:
        # Debug: Print environment type and available tools
        #print(f"[DEBUG] run_agent called with environment type: {type(current_state.environment).__name__}")
        if hasattr(current_state.environment, 'get_tools'):
            [tool.function.name for tool in current_state.environment.get_tools()]
            #print(f"[DEBUG] Available tools: {tools}")
        
        # Save initial state
        await handle_checkpoint_event(state, "turn_start", run_config, session_id)
        
        while not current_state.stop and current_state.turn_idx < current_state.max_turns:
            #print(f"[DEBUG] Agent loop - Turn {current_state.turn_idx}, Stop: {current_state.stop}, Max turns: {current_state.max_turns}")
            next_state = await run_agent_step(current_state, run_config)
            #print(f"[DEBUG] After step - Turn {next_state.turn_idx}, Stop: {next_state.stop}")
            current_state = next_state
            states.append(current_state)
            
            # Checkpoint after each state transition
            await handle_checkpoint_event(current_state, "turn_end", run_config, session_id)
            
        
        # Set stop reason if we hit max turns
        if current_state.turn_idx >= current_state.max_turns:
            current_state = replace(current_state, stop=StopReason.MAX_TURNS)
            states[-1] = current_state
            
            # Save final state
            await handle_checkpoint_event(current_state, "final", run_config, session_id)
        
        return states
    finally:
        # Placeholder for any cleanup if needed in the future
        pass
