#!/usr/bin/env python3
"""
Simple TUI test - just verify basic flow works.

Usage:
    # Terminal 1: Start tmux and watch
    tmux new-session -s test
    
    # Terminal 2: Run test
    python tests/test_tui_basic.py
"""

import json
import subprocess
import time
from pathlib import Path

TMUX_SESSION = "test"
SESSIONS_DIR = Path.home() / ".rollouts" / "sessions"


def tmux_send(keys: str, literal: bool = False) -> None:
    cmd = ["tmux", "send-keys", "-t", TMUX_SESSION]
    if literal:
        cmd.append("-l")
    cmd.append(keys)
    subprocess.run(cmd, check=True)


def tmux_send_enter() -> None:
    tmux_send("Enter")


def tmux_send_ctrl_c() -> None:
    tmux_send("C-c")


def wait(seconds: float = 1.0) -> None:
    time.sleep(seconds)


def get_latest_session_id() -> str | None:
    if not SESSIONS_DIR.exists():
        return None
    sessions = sorted(SESSIONS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0].name if sessions else None


def get_messages(session_id: str) -> list[dict]:
    messages_file = SESSIONS_DIR / session_id / "messages.jsonl"
    if not messages_file.exists():
        return []
    return [json.loads(line) for line in messages_file.read_text().strip().split("\n") if line]


def main():
    print("Make sure tmux session 'test' is running:")
    print("  tmux new-session -s test")
    print()
    input("Press Enter when ready...")
    
    # Start rollouts
    print("\n1. Starting rollouts...")
    tmux_send("cd /Users/chiraagbalu/research-interactive-refactor/rollouts && python -m rollouts --env coding", literal=True)
    tmux_send_enter()
    wait(3)
    
    # Send a message
    print("2. Sending 'hi'...")
    tmux_send("hi", literal=True)
    tmux_send_enter()
    wait(5)
    
    # Check session
    session_id = get_latest_session_id()
    print(f"3. Session created: {session_id}")
    
    if session_id:
        messages = get_messages(session_id)
        print(f"   Messages: {len(messages)}")
        for i, m in enumerate(messages[:5]):
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = f"[{len(content)} blocks]"
            elif isinstance(content, str):
                content = content[:50].replace("\n", "\\n")
            print(f"   {i}: {role}: {content}...")
    
    # Test /model command
    print("\n4. Testing /model...")
    tmux_send("/model", literal=True)
    tmux_send_enter()
    wait(1)
    
    print("\n5. Sending another message...")
    tmux_send("what is 2+2?", literal=True)
    tmux_send_enter()
    wait(5)
    
    if session_id:
        messages = get_messages(session_id)
        print(f"   Messages now: {len(messages)}")
    
    print("\n6. Exit with Ctrl+C...")
    tmux_send_ctrl_c()
    wait(1)
    
    print("\nDone! Check tmux session for output.")


if __name__ == "__main__":
    main()
