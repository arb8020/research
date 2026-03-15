from pathlib import Path
from typing import Any

from _pytest.monkeypatch import MonkeyPatch

from rollouts.drivers.claude import ClaudeDriver
from rollouts.drivers.codex import CodexDriver


def test_claude_driver_skips_permissions_in_one_shot_mode(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("rollouts.drivers.claude.shutil.which", lambda _: "/usr/bin/claude")

    driver = ClaudeDriver(cwd=Path("/tmp/workspace"), model="sonnet")

    cmd = driver._build_cmd(prompt="solve task", bidirectional=False)

    assert "--dangerously-skip-permissions" in cmd


def test_codex_driver_skips_git_repo_check(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr("rollouts.drivers.codex.shutil.which", lambda _: "/usr/bin/codex")

    driver = CodexDriver(
        cwd=Path("/tmp/workspace"),
        model="gpt-5.1-codex-mini",
        sandbox="workspace-write",
    )

    captured: dict[str, list[str]] = {}

    class _FakeStdout:
        async def receive_some(self, _max_bytes: int) -> bytes:
            return b""

    class _FakeProcess:
        def __init__(self) -> None:
            self.stdout = _FakeStdout()
            self.stderr = _FakeStdout()
            self.stdin = None
            self.returncode = 0

        async def wait(self) -> int:
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    async def _fake_open_process(cmd: list[str], **_kwargs: Any) -> _FakeProcess:
        captured["cmd"] = list(cmd)
        return _FakeProcess()

    monkeypatch.setattr("rollouts.drivers.codex.trio.lowlevel.open_process", _fake_open_process)

    async def _collect_cmd() -> list[str]:
        async for _event in driver.run("solve task"):
            pass
        return captured["cmd"]

    import trio

    cmd = trio.run(_collect_cmd)

    assert "--skip-git-repo-check" in cmd
