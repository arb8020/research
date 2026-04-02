from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .acp import ACPDriver


@dataclass
class ClaudeACPDriver(ACPDriver):
    cwd: Path
    model: str = "claude-agent-acp"

    def __init__(self, cwd: Path, *, model: str = "claude-agent-acp") -> None:
        super().__init__(
            cwd=Path(cwd),
            command=("npx", "-y", "@agentclientprotocol/claude-agent-acp"),
            provider="anthropic",
            model=model,
        )
