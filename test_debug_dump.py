#!/usr/bin/env python3
"""Test script to demonstrate debug_dump_chat functionality."""

from rollouts.frontends.tui.agent_renderer import AgentRenderer
from rollouts.frontends.tui.tui import TUI, Container
from rollouts.frontends.tui.theme import DARK_THEME
from rollouts.frontends.tui.terminal import Terminal
from rollouts.dtypes import Message, TextContent, ThinkingContent

# Create a mock TUI with minimal setup
class MockTUI:
    def __init__(self):
        self.theme = DARK_THEME
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def hide_loader(self):
        pass

    def show_loader(self, msg, **kwargs):
        pass

    def request_render(self):
        pass

tui = MockTUI()

# Create renderer
renderer = AgentRenderer(tui)

# Simulate loading a message with thinking + text from history
test_message = Message(
    role="assistant",
    content=[
        ThinkingContent(
            thinking="Let me think about this problem. I need to analyze the requirements carefully.",
            thinking_signature=None  # Simulating aborted stream or missing signature
        ),
        TextContent(
            text="Here's my solution to the problem you asked about."
        )
    ]
)

print("Simulating history replay with thinking + text message...\n")

# Replay the message through the renderer (simulates loading from history)
renderer._replay_assistant_message_as_events(test_message)

# Now dump the state
renderer.debug_dump_chat()
