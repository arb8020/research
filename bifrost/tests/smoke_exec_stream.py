#!/usr/bin/env python3
"""Smoke test for BifrostClient.exec_stream without pytest."""

import os
import tempfile
from collections import deque
from pathlib import Path

from bifrost.client import BifrostClient


class _FakeChannel:
    def __init__(self, chunks):
        self._chunks = deque(chunks)
        self.executed_command = None
        self.pty_requested = False
        self.closed = False

    def set_combine_stderr(self, _):
        pass

    def get_pty(self):
        self.pty_requested = True

    def exec_command(self, command):
        self.executed_command = command

    def recv_ready(self):
        return bool(self._chunks)

    def recv(self, _size):
        if self._chunks:
            return self._chunks.popleft()
        return b""

    def exit_status_ready(self):
        return not self._chunks

    def recv_exit_status(self):
        return 0

    def close(self):
        self.closed = True


class _FakeTransport:
    def __init__(self, channel):
        self._channel = channel
        self.open_calls = 0

    def open_session(self):
        self.open_calls += 1
        return self._channel


class _FakeSSHClient:
    def __init__(self, channel):
        self._transport = _FakeTransport(channel)

    def get_transport(self):
        return self._transport


def run_smoke_test():
    with tempfile.TemporaryDirectory() as tmpdir:
        key_path = Path(tmpdir) / "dummy_key"
        key_path.write_text("dummy")
        os.chmod(key_path, 0o600)

        client = BifrostClient("user@example.com:22", ssh_key_path=str(key_path))

        chunks = [b"first line\nsecond ", b"line\r\nthird"]
        channel = _FakeChannel(chunks)
        fake_client = _FakeSSHClient(channel)
        client._get_ssh_client = lambda: fake_client  # type: ignore[attr-defined]

        streamed = list(client.exec_stream("echo test", working_dir="/workspace"))

        assert streamed == ["first line", "second line", "third"], streamed
        assert channel.executed_command == "cd /workspace && echo test", channel.executed_command
        assert channel.pty_requested, "PTY was not requested"
        assert channel.closed, "Channel was not closed"
        assert fake_client.get_transport().open_calls == 1, fake_client.get_transport().open_calls


if __name__ == "__main__":
    try:
        run_smoke_test()
    except AssertionError as exc:
        print(f"SMOKE TEST FAILED: {exc}")
        raise SystemExit(1)
    print("SMOKE TEST PASSED")
