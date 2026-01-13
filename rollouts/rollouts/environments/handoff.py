"""
Handoff environment for session context transfer.

Provides a `handoff` tool that extracts goal-directed context from the current
session and creates a new session to continue work.

Two modes:
- Single-call mode (default): Fast LLM call that summarizes the session
- Agent mode: Spawns a sub-agent that can read files, grep, etc. for smarter extraction

Usage:
    env = compose(
        LocalFilesystemEnvironment(working_dir=Path.cwd()),
        HandoffEnvironment(session_store=store, endpoint=endpoint),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import trio

from ..dtypes import (
    Actor,
    AgentSession,
    AgentState,
    Endpoint,
    Message,
    RunConfig,
    StreamEvent,
    TextDelta,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
    Trajectory,
)
from ..export import session_to_markdown
from ..providers import get_provider_function

if TYPE_CHECKING:
    pass


# Default model for handoff context extraction (fast, cheap)
DEFAULT_HANDOFF_MODEL = "claude-3-5-haiku-latest"
DEFAULT_HANDOFF_PROVIDER = "anthropic"

# Agent mode uses a smarter model
DEFAULT_AGENT_MODEL = "claude-sonnet-4-20250514"
DEFAULT_AGENT_PROVIDER = "anthropic"


HANDOFF_PROMPT_TEMPLATE = """Extract context from this session relevant to the following goal:

GOAL: {goal}

SESSION:
{session_md}

---

Generate a markdown prompt that someone could use to continue this work. Include:

1. **Context**: Key background info needed for the goal (decisions made, constraints, etc.)
2. **Files**: List relevant file paths that were discussed or modified
3. **Task**: Clear description of what to do next

Output ONLY the markdown prompt - no preamble or explanation. The output will be used directly as input to a new session."""


AGENT_MODE_SYSTEM_PROMPT = """You are a handoff agent. Your job is to extract relevant context from a session conversation and prepare a focused prompt for continuing work.

Sessions are stored at {sessions_dir}:
- Each session is a directory named by session_id (e.g., 20250113_143022_abc123)
- messages.jsonl contains the conversation (one JSON object per line, with role/content fields)

Your task:
1. **Read the session's messages.jsonl** to understand what was discussed and what work was done
2. **Identify files that were read/edited** in the conversation
3. **Read those files** to see their current state (they may have changed since the session)
4. Generate a focused handoff prompt for the specified goal

The handoff prompt should include:
- Key context from the conversation (decisions made, approaches tried, blockers hit)
- Current state of relevant files (what exists now, not what was discussed)
- Clear next steps for the goal

Output your final prompt in a <handoff_prompt> tag. Everything inside this tag will be used as the initial message in the new session."""


AGENT_MODE_USER_PROMPT = """Extract context for handoff:

SESSION: {sessions_dir}/{session_id}/messages.jsonl
GOAL: {goal}

Steps:
1. Read the session's messages.jsonl to understand the conversation
2. Find file paths that were discussed or modified
3. Read those files to see their current state
4. Generate a focused handoff prompt inside <handoff_prompt> tags"""


async def generate_handoff_context(
    session: AgentSession,
    goal: str,
    endpoint: Endpoint,
) -> tuple[str, None] | tuple[None, str]:
    """Generate goal-directed context extraction from a session (single-call mode).

    Args:
        session: Session to extract context from
        goal: The goal/task for the new session
        endpoint: LLM endpoint for generating handoff

    Returns:
        (handoff_markdown, None) on success, (None, error) on failure
    """
    # Convert session to markdown
    session_md = session_to_markdown(session, include_metadata=False)

    handoff_prompt = HANDOFF_PROMPT_TEMPLATE.format(goal=goal, session_md=session_md)

    # Create actor with handoff prompt
    actor = Actor(
        trajectory=Trajectory(messages=[Message(role="user", content=handoff_prompt)]),
        endpoint=endpoint,
        tools=[],
    )

    # Collect response
    result_parts: list[str] = []

    async def collect_text(event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            result_parts.append(event.delta)

    try:
        provider_fn = get_provider_function(endpoint.provider, endpoint.model)
        await provider_fn(actor, collect_text)
        return "".join(result_parts), None
    except Exception as e:
        return None, f"Failed to generate handoff context: {e}"


async def generate_handoff_context_agent(
    session_id: str,
    goal: str,
    endpoint: Endpoint,
    sessions_dir: Path,
    working_dir: Path,
) -> tuple[str, None] | tuple[None, str]:
    """Generate goal-directed context using a sub-agent with tools (agent mode).

    The agent can:
    - Read session files from ~/.rollouts/sessions/
    - Read/grep source files to understand current state
    - Build context iteratively

    Args:
        session_id: ID of session to extract context from
        goal: The goal/task for the new session
        endpoint: LLM endpoint for the agent
        sessions_dir: Path to sessions directory
        working_dir: Working directory for file operations

    Returns:
        (handoff_markdown, None) on success, (None, error) on failure
    """
    from ..agents import run_agent
    from .coding import LocalFilesystemEnvironment

    # Create environment with read access to sessions and working dir
    # The agent can read from both locations
    env = LocalFilesystemEnvironment(
        working_dir=working_dir,
        tools=["read", "bash"],  # read files, bash for grep/jq
    )

    system_prompt = AGENT_MODE_SYSTEM_PROMPT.format(sessions_dir=sessions_dir)
    user_prompt = AGENT_MODE_USER_PROMPT.format(
        sessions_dir=sessions_dir, session_id=session_id, goal=goal
    )

    # Create initial state with system prompt as first message
    initial_state = AgentState(
        actor=Actor(
            trajectory=Trajectory(
                messages=[
                    Message(role="system", content=system_prompt),
                    Message(role="user", content=user_prompt),
                ],
            ),
            endpoint=endpoint,
            tools=env.get_tools(),
        ),
        environment=env,
    )

    # Run the agent (no persistence for sub-agent)
    # No artificial turn limit - let agent finish its research
    import re
    import sys

    # Capture all streamed text to extract handoff even if agent crashes later
    streamed_text: list[str] = []

    async def capture_output(event: StreamEvent) -> None:
        if isinstance(event, TextDelta):
            streamed_text.append(event.delta)
            # Show dots for progress (one dot per ~100 chars)
            if len(streamed_text) % 100 == 0:
                print(".", end="", file=sys.stderr, flush=True)

    run_config = RunConfig(on_chunk=capture_output)

    try:
        await run_agent(initial_state, run_config)
    except Exception as e:
        # Agent may have produced output before failing - this is expected
        # when the agent finishes its work and the next turn has no content
        pass

    print(file=sys.stderr)  # Newline after progress dots

    # Extract handoff from captured stream
    full_text = "".join(streamed_text)
    match = re.search(r"<handoff_prompt>(.*?)</handoff_prompt>", full_text, re.DOTALL)
    if match:
        return match.group(1).strip(), None

    return None, "Agent did not produce a handoff prompt"


@dataclass
class HandoffEnvironment:
    """Environment that provides handoff tool for session context transfer.

    When the agent calls the handoff tool:
    1. Context is extracted from the current session using an LLM
    2. A new session is created with that context as the initial message
    3. The tool returns the new session ID
    4. The current session's AgentState is marked to stop

    The CLI/TUI is responsible for deciding what to do next (auto-switch, prompt user, etc.)

    Args:
        session_store: Store for creating new sessions
        endpoint: LLM endpoint for the main agent (used for new session)
        agent_mode: If True, use a sub-agent with tools for smarter extraction
        handoff_provider: Provider for handoff LLM (default: anthropic)
        handoff_model: Model for handoff LLM (default: haiku for single-call, sonnet for agent)
        working_dir: Working directory for agent mode file access
    """

    session_store: Any  # SessionStore - using Any to avoid circular import
    endpoint: Endpoint  # Main endpoint (for creating new session with same config)
    fast_mode: bool = False  # If True, use single LLM call instead of agent
    handoff_provider: str = DEFAULT_HANDOFF_PROVIDER
    handoff_model: str | None = None  # None = use default based on mode
    working_dir: Path = field(default_factory=Path.cwd)

    # Internal: current session reference (set by agent loop via on_session_start)
    _current_session: AgentSession | None = field(default=None, repr=False)

    def _get_handoff_model(self) -> str:
        """Get the model to use for handoff extraction."""
        if self.handoff_model is not None:
            return self.handoff_model
        return DEFAULT_HANDOFF_MODEL if self.fast_mode else DEFAULT_AGENT_MODEL

    def get_name(self) -> str:
        return "handoff"

    def get_system_prompt(self) -> str | None:
        """Add handoff guidance to system prompt."""
        return """## Handoff Tool

You have access to a `handoff` tool that transfers work to a new session with focused context.

Use handoff when:
- The current context is getting too long or cluttered
- You want to start a focused subtask
- The conversation has accumulated noise from errors or dead ends
- You're about to start a distinct phase of work

The handoff tool extracts only the relevant context for your specified goal, creating a clean slate while preserving important information."""

    async def on_session_start(self, session_id: str) -> None:
        """Called when session starts - load the session for handoff context."""
        session, error = await self.session_store.get(session_id)
        if session:
            self._current_session = session

    async def serialize(self) -> dict:
        return {
            "env_kind": "handoff",
            "endpoint": self.endpoint.to_dict(exclude_secrets=True),
            "fast_mode": self.fast_mode,
            "handoff_provider": self.handoff_provider,
            "handoff_model": self.handoff_model,
            "working_dir": str(self.working_dir),
        }

    @staticmethod
    async def deserialize(data: dict) -> HandoffEnvironment:
        # Note: session_store must be injected after deserialize
        # This is a limitation - handoff env needs external dependency
        raise NotImplementedError(
            "HandoffEnvironment cannot be deserialized standalone - "
            "session_store must be injected. Use compose() at runtime."
        )

    def get_tools(self) -> list[Tool]:
        return [
            Tool(
                type="function",
                function=ToolFunction(
                    name="handoff",
                    description="""Hand off work to a new session with focused context.

Use this when you need to continue work in a fresh context because:
- The current session is getting too long and context is degrading
- You want to start a focused task while preserving relevant context
- The session has accumulated noise from errors or exploration

When called:
1. Relevant context will be extracted from this session
2. A new session will be created with that context
3. This session will end

The goal should describe what to do next (1-2 sentences), not what was already done.""",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "goal": {
                                "type": "string",
                                "description": "Short description of the task to accomplish in the new session. Focus on what needs to be done next.",
                            },
                        },
                    ),
                    required=["goal"],
                ),
            ),
        ]

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """Handoff doesn't require confirmation - it's a navigation action."""
        return False

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        """No special handling needed."""
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """Execute the handoff tool."""
        if tool_call.name != "handoff":
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Unknown tool: {tool_call.name}",
            )

        goal = tool_call.args.get("goal", "")
        if not goal:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="Goal is required for handoff",
            )

        # Need current session to extract context from
        if self._current_session is None:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error="No current session available for handoff. Session must be persisted.",
            )

        # Create endpoint for handoff context extraction
        handoff_endpoint = Endpoint(
            provider=self.handoff_provider,
            model=self._get_handoff_model(),
        )

        # Generate handoff context
        if self.fast_mode:
            # Fast mode: single LLM call to summarize session
            # Update current session with latest messages from state
            session_with_current_messages = AgentSession(
                session_id=self._current_session.session_id,
                parent_id=self._current_session.parent_id,
                branch_point=self._current_session.branch_point,
                endpoint=self._current_session.endpoint,
                environment=self._current_session.environment,
                messages=current_state.actor.trajectory.messages,
            )
            handoff_content, error = await generate_handoff_context(
                session=session_with_current_messages,
                goal=goal,
                endpoint=handoff_endpoint,
            )
        else:
            # Default: agent mode - spawn sub-agent with tools for better context
            handoff_content, error = await generate_handoff_context_agent(
                session_id=self._current_session.session_id,
                goal=goal,
                endpoint=handoff_endpoint,
                sessions_dir=self.session_store.base_dir,
                working_dir=self.working_dir,
            )

        if error:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=error,
            )

        # Create new session with handoff content as initial message
        try:
            new_session = await self.session_store.create(
                endpoint=self.endpoint,
                environment=self._current_session.environment,
                parent_id=self._current_session.session_id,
                tags={"handoff_goal": goal[:100]},  # Truncate for metadata
            )

            # Add the handoff content as the first user message
            await self.session_store.append_message(
                new_session.session_id,
                Message(role="user", content=handoff_content),
            )

        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to create handoff session: {e}",
            )

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"Created handoff session: {new_session.session_id}\n\nGoal: {goal}\n\nThe new session has been initialized with relevant context from this conversation. Use `rollouts -s {new_session.session_id}` to continue.",
            details={
                "handoff": True,
                "new_session_id": new_session.session_id,
                "goal": goal,
            },
        )
