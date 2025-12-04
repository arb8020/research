"""
Interactive TUI agent runner.

Provides a complete interactive agent loop with TUI rendering.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from typing import Optional

import trio

from rollouts.rollouts.agents import AgentState, Actor, run_agent
from rollouts.rollouts.dtypes import (
    Endpoint,
    Message,
    RunConfig,
    Trajectory,
    ToolCall,
    ToolConfirmResult,
    StopReason,
)
from rollouts.rollouts.providers import Environment

from .terminal import ProcessTerminal
from .tui import TUI
from .agent_renderer import AgentRenderer
from .components.input import Input


class InteractiveAgentRunner:
    """Interactive agent runner with TUI."""

    def __init__(
        self,
        initial_trajectory: Trajectory,
        endpoint: Endpoint,
        environment: Optional[Environment] = None,
        max_turns: int = 50,
    ) -> None:
        """Initialize interactive agent runner.

        Args:
            initial_trajectory: Initial conversation trajectory
            endpoint: LLM endpoint configuration
            environment: Optional environment for tool execution
            max_turns: Maximum number of turns
        """
        self.initial_trajectory = initial_trajectory
        self.endpoint = endpoint
        self.environment = environment
        self.max_turns = max_turns

        # TUI components
        self.terminal: Optional[ProcessTerminal] = None
        self.tui: Optional[TUI] = None
        self.renderer: Optional[AgentRenderer] = None
        self.input_component: Optional[Input] = None

        # Input coordination
        self.input_queue: asyncio.Queue[str] = asyncio.Queue()
        self.input_pending: bool = False
        self.is_first_user_message = True

        # Cancellation
        self.cancel_scope: Optional[trio.CancelScope] = None

    async def _handle_input_submit(self, text: str) -> None:
        """Handle input submission from TUI."""
        if text.strip():
            await self.input_queue.put(text.strip())

    async def _tui_input_handler(self, prompt: str) -> str:
        """Async input handler for RunConfig.on_input.

        Args:
            prompt: Prompt string (not used in TUI, but required by signature)

        Returns:
            User input string
        """
        # Show input component if not already visible
        if not self.input_pending:
            self.input_pending = True
            if self.input_component and self.tui:
                self.tui.set_focus(self.input_component)
                self.tui.request_render()

        # Wait for input from queue
        user_input = await self.input_queue.get()
        self.input_pending = False

        # Clear input component
        if self.input_component:
            self.input_component.set_text("")
            if self.tui:
                self.tui.set_focus(None)
                self.tui.request_render()

        # Add user message to chat
        if self.renderer:
            self.renderer.add_user_message(user_input, is_first=self.is_first_user_message)
            self.is_first_user_message = False

        return user_input

    def _handle_sigint(self, signum, frame) -> None:
        """Handle SIGINT (Ctrl+C) - cancel agent."""
        if self.cancel_scope:
            self.cancel_scope.cancel()

    def _handle_terminal_input(self, data: str) -> None:
        """Handle raw terminal input - route to TUI input handler."""
        # Ctrl+C - cancel agent
        if len(data) > 0 and ord(data[0]) == 3:
            if self.cancel_scope:
                self.cancel_scope.cancel()
            return

        # TUI will route to focused component automatically via its _handle_input
        # We just need to make sure the input component is focused
        if self.tui:
            self.tui._handle_input(data)

    async def run(self) -> list[AgentState]:
        """Run interactive agent loop.

        Returns:
            List of agent states from the run
        """
        # Create terminal and TUI
        self.terminal = ProcessTerminal()
        self.tui = TUI(self.terminal)

        # Create renderer
        self.renderer = AgentRenderer(self.tui)

        # Create input component
        self.input_component = Input()
        self.input_component.set_on_submit(self._handle_input_submit)
        self.tui.add_child(self.input_component)

        # Set up signal handler for Ctrl+C
        signal.signal(signal.SIGINT, self._handle_sigint)

        # Start TUI
        self.tui.start()

        try:
            # Create initial state
            initial_state = AgentState(
                actor=Actor(
                    trajectory=self.initial_trajectory,
                    endpoint=self.endpoint,
                    tools=self.environment.get_tools() if self.environment else [],
                ),
                environment=self.environment,
                max_turns=self.max_turns,
            )

            # Create cancel scope for graceful cancellation
            async with trio.open_nursery() as nursery:
                self.cancel_scope = nursery.cancel_scope

                # Create run config
                # Auto-confirm all tools (no interactive confirmation in TUI for now)
                async def auto_confirm_tool(tc: ToolCall, state: AgentState, rcfg: RunConfig) -> tuple[AgentState, ToolConfirmResult]:
                    return state, ToolConfirmResult(proceed=True)

                run_config = RunConfig(
                    on_chunk=self.renderer.handle_event,
                    on_input=self._tui_input_handler,
                    confirm_tool=auto_confirm_tool,
                    handle_stop=self._handle_stop,
                )

                # Run agent in background task
                async def run_agent_task():
                    try:
                        return await run_agent(initial_state, run_config)
                    except trio.Cancelled:
                        # Agent was cancelled - this is expected
                        return []

                # Start agent task
                agent_task = nursery.start_soon(run_agent_task)

                # Set up terminal input reading loop
                # Terminal is in raw mode, so we need to poll for input
                async def input_reading_loop():
                    """Read terminal input and route to TUI."""
                    while True:
                        if self.terminal and self.terminal._running:
                            # Read input (non-blocking)
                            input_data = self.terminal.read_input()
                            if input_data:
                                # Route to TUI's input handler
                                if self.tui:
                                    self.tui._handle_input(input_data)
                        await trio.sleep(0.01)  # Small delay to avoid busy-waiting

                input_task_handle = nursery.start_soon(input_reading_loop)

                # Wait for agent to complete
                try:
                    states = await agent_task
                finally:
                    # Cancel input task
                    input_task_handle.cancel()

                return states

        finally:
            # Stop TUI
            if self.tui:
                self.tui.stop()
            if self.terminal:
                self.terminal.stop()

    def _handle_stop(self, state: AgentState) -> AgentState:
        """Handle stop condition - check max turns."""
        from rollouts.rollouts.dtypes import replace

        if state.turn_idx >= self.max_turns:
            return replace(state, stop=StopReason.MAX_TURNS)
        return state


async def run_interactive_agent(
    initial_trajectory: Trajectory,
    endpoint: Endpoint,
    environment: Optional[Environment] = None,
    max_turns: int = 50,
) -> list[AgentState]:
    """Run an interactive agent with TUI.

    Args:
        initial_trajectory: Initial conversation trajectory
        endpoint: LLM endpoint configuration
        environment: Optional environment for tool execution
        max_turns: Maximum number of turns

    Returns:
        List of agent states from the run
    """
    runner = InteractiveAgentRunner(
        initial_trajectory=initial_trajectory,
        endpoint=endpoint,
        environment=environment,
        max_turns=max_turns,
    )
    return await runner.run()

