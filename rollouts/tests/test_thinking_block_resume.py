#!/usr/bin/env python3
"""Integration test for thinking block handling on session resume.

Reproduces: https://github.com/anthropics/claude-code/issues/12316

Uses the EXACT session that crashed to reproduce the bug.
"""

import os
import pickle
from pathlib import Path

import pytest

from rollouts.dtypes import Message, Trajectory


@pytest.mark.trio
async def test_resume_crashed_session_exactly() -> None:
    """Reproduce the crash using the EXACT session that failed.

    Session: 20251223_152205_b0fba6
    Error: messages.5.content.3: thinking blocks cannot be modified

    This loads the real messages and attempts to continue the conversation.
    """
    from rollouts import Actor, Endpoint, rollout, stdout_handler

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    # Load the exact messages from the crashed session
    session_file = Path.home() / ".rollouts/sessions/20251223_152205_b0fba6/messages.jsonl"
    if not session_file.exists():
        pytest.skip(f"Crashed session not found: {session_file}")

    messages = []
    with open(session_file) as f:
        for line in f:
            messages.append(Message.from_json(line))

    print(f"\nLoaded {len(messages)} messages from crashed session")
    print(f"Last message: {messages[-1].role} - {str(messages[-1].content)[:50]}...")

    # Use the same model and settings from the crash
    endpoint = Endpoint(
        provider="anthropic",
        model="claude-opus-4-5-20251101",
        api_key=api_key,
        temperature=1.0,
        max_tokens=16000,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )

    actor = Actor(
        trajectory=Trajectory(messages=messages),
        endpoint=endpoint,
        tools=[],  # No tools for simplicity
    )

    # This is where the crash happened
    # Pre-fix: Raises "thinking blocks cannot be modified"
    # Post-fix: Should succeed
    print("\nAttempting to continue crashed session...")
    result = await rollout(actor, on_chunk=stdout_handler)

    assert result.trajectory.messages[-1].role == "assistant"
    print("\n✓ Successfully resumed the crashed session!")
