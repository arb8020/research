"""
DialogueEnvironment - Environment where two models converse.

The "primary" agent runs through the standard agent loop. When it produces an
assistant message, on_assistant_message calls the "responder" agent to generate
a reply, which is injected as a user message back to the primary.

This creates a dialogue where:
- Primary sees responder's messages as "user" messages
- Responder sees primary's messages as "user" messages
- Both maintain their own perspective on the conversation

Example usage:
    env = DialogueEnvironment(
        responder_endpoint=Endpoint.from_dict({...}),
        responder_system_prompt="You are arguing against...",
        max_turns=10,
    )
"""

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from typing import Any

import trio

from ..agents import Actor, AgentState, RunConfig, rollout
from ..core import (
    Endpoint,
    Message,
    StopReason,
    Tool,
    ToolCall,
    ToolFormatter,
    ToolResult,
    Trajectory,
)
from ..dtypes import StreamEvent

logger = logging.getLogger(__name__)


@dataclass
class DialogueTurn:
    """One turn in the dialogue."""

    agent_name: str  # "primary" or "responder"
    message: Message
    timestamp: float


@dataclass
class DialogueEnvironment:
    """Environment where a responder agent replies to the primary agent's messages.

    The primary agent runs normally through run_agent_step. When it produces an
    assistant message, this environment calls the responder to generate a reply,
    which becomes the next "user" message for the primary.

    Concession detection:
        If detect_concession is provided, it's called after each turn to check
        if either agent has conceded. Returns ("primary" | "responder" | None).
        When someone concedes, the environment sets stop=TASK_COMPLETED.
    """

    # Responder configuration
    responder_endpoint: Endpoint
    responder_system_prompt: str | None = None

    # Stop conditions
    max_turns: int | None = None  # Total exchanges (primary + responder). None = no limit

    # Concession detection (called with full dialogue after each exchange)
    detect_concession: Callable[[list[DialogueTurn]], str | None] | None = None

    # Event streaming (set by eval harness)
    on_chunk: Callable[[StreamEvent], Awaitable[None]] | None = None

    # Internal state (mutable - reset per sample)
    _turn_count: int = field(default=0, repr=False)
    _dialogue_turns: list[DialogueTurn] = field(default_factory=list, repr=False)
    _concession_winner: str | None = field(default=None, repr=False)

    def get_tools(self) -> list[Tool]:
        """No tools - dialogue is purely message-based."""
        return []

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        """No tools, no confirmation."""
        return False

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        """No tools in dialogue environment."""
        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=True,
            content="No tools available in dialogue environment",
        )

    async def on_assistant_message(
        self,
        message: Message,  # Primary agent's message
        state: AgentState,
    ) -> AgentState:
        """Primary agent spoke. Call responder to reply.

        Flow:
        1. Record primary's turn
        2. Check if max turns reached
        3. Check for concession
        4. Build responder's context (primary's messages become "user" for responder)
        5. Call responder via rollout()
        6. Record responder's turn
        7. Check for concession again
        8. Inject responder's message as "user" for primary
        """
        self._turn_count += 1

        # Record primary's turn
        primary_turn = DialogueTurn(
            agent_name="primary",
            message=message,
            timestamp=time.time(),
        )
        self._dialogue_turns.append(primary_turn)

        logger.info(
            f"[Dialogue] Turn {self._turn_count}: primary spoke ({len(self._get_text(message))} chars)"
        )

        # Check max turns (primary just spoke, so check before responder)
        if self.max_turns is not None and self._turn_count >= self.max_turns:
            logger.info(f"[Dialogue] Max turns ({self.max_turns}) reached")
            return replace(state, stop=StopReason.MAX_TURNS)

        # Check for concession after primary's message
        if self.detect_concession:
            winner = self.detect_concession(self._dialogue_turns)
            if winner:
                self._concession_winner = winner
                logger.info(f"[Dialogue] Concession detected! Winner: {winner}")
                return replace(state, stop=StopReason.TASK_COMPLETED)

        # Build responder's view of the conversation
        responder_messages = self._build_responder_context(state.actor.trajectory)

        # Create responder actor
        responder_actor = Actor(
            trajectory=Trajectory(messages=responder_messages),
            endpoint=self.responder_endpoint,
            tools=[],
        )

        # Call responder
        logger.info("[Dialogue] Calling responder...")

        async def silent_chunk(event: StreamEvent) -> None:
            """Forward events if on_chunk is set, otherwise discard."""
            if self.on_chunk:
                await self.on_chunk(event)

        responder_actor = await rollout(
            responder_actor,
            on_chunk=silent_chunk,
        )

        # Extract responder's message
        responder_message = responder_actor.trajectory.messages[-1]

        self._turn_count += 1

        # Record responder's turn
        responder_turn = DialogueTurn(
            agent_name="responder",
            message=responder_message,
            timestamp=time.time(),
        )
        self._dialogue_turns.append(responder_turn)

        logger.info(
            f"[Dialogue] Turn {self._turn_count}: responder spoke ({len(self._get_text(responder_message))} chars)"
        )

        # Check for concession after responder's message
        if self.detect_concession:
            winner = self.detect_concession(self._dialogue_turns)
            if winner:
                self._concession_winner = winner
                logger.info(f"[Dialogue] Concession detected! Winner: {winner}")
                return replace(state, stop=StopReason.TASK_COMPLETED)

        # Inject responder's message as "user" for primary
        # Extract text content from responder's message
        responder_text = self._get_text(responder_message)
        injected_msg = Message(role="user", content=responder_text)

        # Update primary's trajectory with the injected message
        current_trajectory = state.actor.trajectory
        new_messages = [*current_trajectory.messages, injected_msg]
        new_trajectory = Trajectory(
            messages=new_messages,
            completions=current_trajectory.completions,
            metadata={
                **current_trajectory.metadata,
                "dialogue_turn": self._turn_count,
            },
        )

        new_actor = replace(state.actor, trajectory=new_trajectory)
        return replace(state, actor=new_actor)

    def _build_responder_context(self, primary_trajectory: Trajectory) -> list[Message]:
        """Convert primary's trajectory to responder's perspective.

        Primary's assistant messages -> Responder sees as "user"
        Primary's user messages (previous responder replies) -> Responder sees as "assistant"
        """
        messages = []

        # Add responder's system prompt if provided
        if self.responder_system_prompt:
            messages.append(Message(role="system", content=self.responder_system_prompt))

        for msg in primary_trajectory.messages:
            if msg.role == "system":
                # Skip primary's system prompt
                continue
            if msg.role == "assistant":
                # Primary's output -> Responder sees as "user" input
                text = self._get_text(msg)
                messages.append(Message(role="user", content=text))
            elif msg.role == "user":
                # Could be initial prompt or previous responder reply
                # Either way, responder sees it as their own previous output
                # Skip the very first user message (task prompt)
                if len(messages) > (1 if self.responder_system_prompt else 0):
                    # This is a responder reply that was injected
                    text = self._get_text(msg)
                    messages.append(Message(role="assistant", content=text))

        return messages

    def _get_text(self, message: Message) -> str:
        """Extract text content from a message."""
        if isinstance(message.content, str):
            return message.content
        elif isinstance(message.content, list):
            # Handle content blocks
            parts = []
            for block in message.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
                elif hasattr(block, "content"):
                    parts.append(str(block.content))
            return "\n".join(parts)
        return str(message.content) if message.content else ""

    def get_tool_formatter(self, tool_name: str) -> ToolFormatter | None:
        """No tools, no formatters."""
        return None

    async def serialize(self) -> dict[str, Any]:
        """Serialize dialogue state for scoring."""
        return {
            "env_kind": "dialogue",
            "turn_count": self._turn_count,
            "concession_winner": self._concession_winner,
            "dialogue_turns": [
                {
                    "agent_name": t.agent_name,
                    "content": self._get_text(t.message),
                    "timestamp": t.timestamp,
                }
                for t in self._dialogue_turns
            ],
        }

    @staticmethod
    async def deserialize(data: dict[str, Any]) -> "DialogueEnvironment":
        """Cannot fully deserialize - endpoint not serialized."""
        raise NotImplementedError("DialogueEnvironment cannot be deserialized")


def keyword_concession_detector(
    concession_phrases: list[str] | None = None,
) -> Callable[[list[DialogueTurn]], str | None]:
    """Create a simple keyword-based concession detector.

    Returns a function that checks if the last speaker used any concession phrase.
    If so, returns the OTHER agent as the winner (the one who convinced them).

    Args:
        concession_phrases: Phrases that indicate concession. Defaults to common ones.

    Returns:
        Detector function: (turns) -> winner name or None
    """
    if concession_phrases is None:
        concession_phrases = [
            "you're right",
            "you are right",
            "i agree",
            "i concede",
            "you've convinced me",
            "you have convinced me",
            "i was wrong",
            "i stand corrected",
            "good point",
            "fair point",
            "i hadn't considered",
            "i see your point",
            "you make a good point",
            "i'm convinced",
            "i am convinced",
            "i change my mind",
            "i've changed my mind",
        ]

    # Normalize to lowercase
    phrases_lower = [p.lower() for p in concession_phrases]

    def detect(turns: list[DialogueTurn]) -> str | None:
        if not turns:
            return None

        last_turn = turns[-1]
        last_text = ""

        # Get text from message
        msg = last_turn.message
        if isinstance(msg.content, str):
            last_text = msg.content.lower()
        elif isinstance(msg.content, list):
            parts = []
            for block in msg.content:
                if hasattr(block, "text"):
                    parts.append(block.text)
            last_text = " ".join(parts).lower()

        # Check for concession phrases
        for phrase in phrases_lower:
            if phrase in last_text:
                # Last speaker conceded, so the OTHER agent wins
                if last_turn.agent_name == "primary":
                    return "responder"
                else:
                    return "primary"

        return None

    return detect
