#!/usr/bin/env python3
"""
Reproduce the /slice session confusion bug.

This script:
1. Starts rollouts in a tmux session
2. Sends keystrokes to simulate user interaction
3. Checks session state after each step

Usage:
    # Start tmux first
    tmux new-session -d -s test_slice

    # Run this script
    python tests/test_slice_repro.py

    # Watch what's happening
    tmux attach -t test_slice
"""

import json
import subprocess
import time
from pathlib import Path

TMUX_SESSION = "test_slice"
SESSIONS_DIR = Path.home() / ".rollouts" / "sessions"


def tmux_send(keys: str, literal: bool = False) -> None:
    """Send keys to tmux session."""
    cmd = ["tmux", "send-keys", "-t", TMUX_SESSION]
    if literal:
        cmd.append("-l")
    cmd.append(keys)
    subprocess.run(cmd, check=True)


def tmux_send_enter() -> None:
    """Send Enter key."""
    tmux_send("Enter")


def tmux_send_escape() -> None:
    """Send Escape key."""
    tmux_send("Escape")


def tmux_send_ctrl_c() -> None:
    """Send Ctrl+C."""
    tmux_send("C-c")


def wait(seconds: float = 1.0) -> None:
    """Wait for TUI to process."""
    time.sleep(seconds)


def get_latest_session() -> dict | None:
    """Get the most recently created session."""
    if not SESSIONS_DIR.exists():
        return None

    sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    if not sessions:
        return None

    session_dir = sessions[0]
    session_json = session_dir / "session.json"
    if session_json.exists():
        return json.loads(session_json.read_text())
    return None


def get_session(session_id: str) -> dict | None:
    """Get a specific session by ID."""
    session_dir = SESSIONS_DIR / session_id
    session_json = session_dir / "session.json"
    if session_json.exists():
        return json.loads(session_json.read_text())
    return None


def get_session_messages(session_id: str) -> list[dict]:
    """Get messages for a session."""
    messages_file = SESSIONS_DIR / session_id / "messages.jsonl"
    if not messages_file.exists():
        return []

    messages = []
    for line in messages_file.read_text().strip().split("\n"):
        if line:
            messages.append(json.loads(line))
    return messages


def check_message_alternation(messages: list[dict]) -> list[str]:
    """Check that messages alternate correctly. Returns list of errors."""
    errors = []
    prev_role = None
    for i, msg in enumerate(messages):
        role = msg.get("role")
        if role == "system":
            continue  # System messages can be anywhere
        if role == "tool":
            continue  # Tool messages follow assistant

        if role == "assistant" and prev_role == "assistant":
            errors.append(f"Consecutive assistant messages at {i - 1} and {i}")

        if role in ("user", "assistant"):
            prev_role = role

    return errors


def start_rollouts() -> None:
    """Start rollouts in tmux session."""
    # Kill any existing session
    subprocess.run(["tmux", "kill-session", "-t", TMUX_SESSION], capture_output=True)

    # Create new session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", TMUX_SESSION, "-c", str(Path(__file__).parent.parent)],
        check=True,
    )

    # Start rollouts
    tmux_send("python -m rollouts --env coding", literal=True)
    tmux_send_enter()

    # Wait for TUI to start
    wait(3)


def run_repro():
    """Run the reproduction steps."""
    print("Starting rollouts...")
    start_rollouts()

    # Step 1: Send initial message
    print("\n=== Step 1: Send 'hi' ===")
    tmux_send("hi", literal=True)
    tmux_send_enter()
    wait(5)  # Wait for response

    session1 = get_latest_session()
    print(f"Session after 'hi': {session1['session_id'] if session1 else 'None'}")

    # Step 2: Interrupt with Escape while responding
    print("\n=== Step 2: Send message and interrupt ===")
    tmux_send("tell me a long story", literal=True)
    tmux_send_enter()
    wait(1)  # Let it start responding
    tmux_send_escape()  # Interrupt
    wait(2)

    session2 = get_latest_session()
    print(f"Session after interrupt: {session2['session_id'] if session2 else 'None'}")

    # Check for consecutive assistant messages
    if session2:
        messages = get_session_messages(session2["session_id"])
        errors = check_message_alternation(messages)
        if errors:
            print(f"⚠️  Message alternation errors: {errors}")
        else:
            print("✓ Message alternation OK")

    # Step 3: Run /slice
    print("\n=== Step 3: Run /slice ===")
    tmux_send("/slice", literal=True)
    tmux_send_enter()
    wait(2)

    session3 = get_latest_session()
    print(f"Session after /slice: {session3['session_id'] if session3 else 'None'}")

    # Step 4: Run /slice with spec
    print("\n=== Step 4: Run /slice 0:5 ===")
    tmux_send("/slice 0:5", literal=True)
    tmux_send_enter()
    wait(2)

    session4 = get_latest_session()
    if session4:
        print(f"Session after /slice 0:5: {session4['session_id']}")
        print(f"  Parent: {session4.get('parent_id')}")
        print(f"  Branch point: {session4.get('branch_point')}")

    # Step 5: Send another message to verify context
    print("\n=== Step 5: Send 'what was my first message?' ===")
    tmux_send("what was my first message?", literal=True)
    tmux_send_enter()
    wait(5)

    # Final check
    print("\n=== Final State ===")
    final_session = get_latest_session()
    if final_session:
        messages = get_session_messages(final_session["session_id"])
        print(f"Session: {final_session['session_id']}")
        print(f"Message count: {len(messages)}")
        errors = check_message_alternation(messages)
        if errors:
            print(f"⚠️  Errors: {errors}")
        else:
            print("✓ All checks passed")

        # Print last few messages
        print("\nLast 5 messages:")
        for msg in messages[-5:]:
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = f"[{len(content)} blocks]"
            else:
                content = str(content)[:60]
            print(f"  {role}: {content}")

    print("\n=== Done ===")
    print(f"Attach to see TUI: tmux attach -t {TMUX_SESSION}")
    print(f"Kill session: tmux kill-session -t {TMUX_SESSION}")


if __name__ == "__main__":
    run_repro()
