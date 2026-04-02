from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .acp import ACPDriver


@dataclass
class CodexACPDriver(ACPDriver):
    cwd: Path
    model: str = "codex-acp"

    def __init__(self, cwd: Path, *, model: str = "codex-acp") -> None:
        super().__init__(
            cwd=Path(cwd),
            command=(
                "npx",
                "-y",
                "@zed-industries/codex-acp",
                "-c",
                'forced_login_method="api"',
                "-c",
                'preferred_auth_method="apikey"',
                "-c",
                'approval_policy="never"',
                "-c",
                'sandbox_mode="workspace-write"',
            ),
            provider="openai",
            model=model,
        )
