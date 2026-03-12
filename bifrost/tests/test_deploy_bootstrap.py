from bifrost.deploy import GitDeployment


class _FakeChannel:
    def __init__(self, exit_status: int) -> None:
        self._exit_status = exit_status

    def recv_exit_status(self) -> int:
        return self._exit_status


class _FakeStdout:
    def __init__(self, exit_status: int) -> None:
        self.channel = _FakeChannel(exit_status)


class _FakeClient:
    def __init__(self, existing_files: set[str]) -> None:
        self.existing_files = existing_files
        self.commands: list[str] = []

    def exec_command(self, command: str):
        self.commands.append(command)
        path = command.removeprefix("test -f ").strip()
        exit_status = 0 if path in self.existing_files else 1
        return None, _FakeStdout(exit_status), None


def test_detect_bootstrap_command_skips_when_explicitly_requested() -> None:
    deployment = GitDeployment("user", "host", 22)
    client = _FakeClient({"~/repo/pyproject.toml"})

    command = deployment.detect_bootstrap_command(
        client,
        "~/repo",
        skip_bootstrap=True,
    )

    assert command == ""
    assert client.commands == []


def test_detect_bootstrap_command_respects_explicit_frozen_flag() -> None:
    deployment = GitDeployment("user", "host", 22)
    client = _FakeClient({"~/repo/uv.lock"})

    command = deployment.detect_bootstrap_command(
        client,
        "~/repo",
        bootstrap_frozen=True,
    )

    assert command == "pip install uv && uv sync --frozen && "
