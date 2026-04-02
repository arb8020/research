from __future__ import annotations

from bifrost.modal_backend import (
    MODAL_PARENT_LEASE_PATH,
    _modal_primary_command,
    _refresh_modal_parent_lease_sync,
)


class _FakeProc:
    def __init__(self, exit_code: int = 0, stderr: str = "") -> None:
        self.stdout = []
        self.stderr = [stderr] if stderr else []
        self._exit_code = exit_code

    def wait(self) -> int:
        return self._exit_code


class _FakeSandbox:
    def __init__(self, proc: _FakeProc | None = None) -> None:
        self.proc = proc or _FakeProc()
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def exec(self, *args: object, **kwargs: object) -> _FakeProc:
        self.calls.append((args, kwargs))
        return self.proc


def test_modal_primary_command_uses_sleep_only_for_keep_alive() -> None:
    keepalive_cmd = _modal_primary_command(True)
    watchdog_cmd = _modal_primary_command(False)

    assert keepalive_cmd[:2] == ("python3", "-c")
    assert "time.sleep(315360000)" in keepalive_cmd[2]

    assert watchdog_cmd[:2] == ("python3", "-c")
    assert MODAL_PARENT_LEASE_PATH in watchdog_cmd
    assert "lease_path" in watchdog_cmd[2]
    assert "time.sleep(315360000)" not in watchdog_cmd[2]
    compile(watchdog_cmd[2], "<modal-watchdog>", "exec")


def test_refresh_modal_parent_lease_touches_expected_path() -> None:
    sandbox = _FakeSandbox()

    _refresh_modal_parent_lease_sync(sandbox)

    assert len(sandbox.calls) == 1
    args, kwargs = sandbox.calls[0]
    assert args[:2] == ("bash", "-lc")
    assert f"touch {MODAL_PARENT_LEASE_PATH}" in args[2]
    assert kwargs["timeout"] == 30


def test_refresh_modal_parent_lease_raises_on_exec_failure() -> None:
    sandbox = _FakeSandbox(_FakeProc(exit_code=7, stderr="boom"))

    try:
        _refresh_modal_parent_lease_sync(sandbox)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert MODAL_PARENT_LEASE_PATH in message
    assert "exit_code=7" in message
    assert "boom" in message
