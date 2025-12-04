#!/usr/bin/env python3
"""
Example usage of AgentRenderer with TUI.

This shows how to wire AgentRenderer into RunConfig for use with the agent.
"""

from __future__ import annotations

import asyncio

from rollouts.frontends.tui import ProcessTerminal, TUI, AgentRenderer
from rollouts.rollouts.dtypes import RunConfig, StreamEvent
from rollouts.rollouts.agents import AgentState, Actor, run_agent


async def example_main():
    """Example main function showing TUI integration."""
    # Create terminal and TUI
    terminal = ProcessTerminal()
    tui = TUI(terminal)
    renderer = AgentRenderer(tui)

    # Create run config with renderer as on_chunk handler
    run_config = RunConfig(
        on_chunk=renderer.handle_event,
        # ... other config options
    )

    # Start TUI
    tui.start()

    try:
        # Example: Create initial state (you would get this from your actual agent setup)
        # actor = Actor(...)
        # state = AgentState(actor=actor, ...)
        # states = await run_agent(state, run_config)
        pass
    finally:
        tui.stop()


if __name__ == "__main__":
    asyncio.run(example_main())

