"""Protocol for external agent drivers.

Drivers parse external agent output formats and emit StreamEvents directly.
No intermediate layer - frontends consume the same events they always have.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..dtypes import StreamEvent


@runtime_checkable
class ExternalAgentDriver(Protocol):
    """Protocol for driving external agents.

    Drivers spawn an external process (claude, codex, etc.) and parse its
    output stream into StreamEvents. Consumers (frontends, training loggers)
    see the same event types they'd see from an internal agent.

    Example:
        driver = ClaudeDriver(cwd=repo_path, model="sonnet")
        async for event in driver.run("Fix the bug"):
            await frontend.handle_event(event)
    """

    def run(self, prompt: str) -> AsyncIterator[StreamEvent]:
        """Run the agent with a prompt, yielding events as they arrive.

        Args:
            prompt: The task/prompt to send to the agent

        Yields:
            StreamEvent instances (TextDelta, ToolCallStart, etc.)
        """
        ...

    async def send_input(self, text: str) -> None:
        """Send follow-up input to the agent (for multi-turn).

        Only works if the driver was started with bidirectional mode.
        """
        ...

    async def abort(self) -> None:
        """Abort the current run."""
        ...
