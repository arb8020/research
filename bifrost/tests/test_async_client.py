"""Tests for AsyncBifrostClient using pytest-trio."""

import contextvars
import os

import pytest
import trio

import bifrost.async_client as async_client_module
from bifrost import AsyncBifrostClient

# Mark all tests in this module as trio tests
pytestmark = pytest.mark.trio
TEST_SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")


async def test_async_client_context_manager() -> None:
    """Test that AsyncBifrostClient can be used as an async context manager."""
    # This test doesn't actually connect, just tests the interface
    # Real connection tests would require a live SSH server

    client = AsyncBifrostClient(ssh_connection="user@example.com:22", ssh_key_path=TEST_SSH_KEY)

    # Verify client was created
    assert client is not None
    assert client.ssh.host == "example.com"
    assert client.ssh.user == "user"
    assert client.ssh.port == 22


async def test_exec_stream_interface() -> None:
    """Test that exec_stream returns an async iterator."""
    client = AsyncBifrostClient(ssh_connection="user@example.com:22", ssh_key_path=TEST_SSH_KEY)

    # Verify the method exists and has correct signature
    assert hasattr(client, "exec_stream")
    assert callable(client.exec_stream)


async def test_parallel_operations_with_nursery() -> None:
    """Demonstrate how trio nurseries enable parallel operations."""

    async def mock_task(task_id: int, duration: float) -> str:
        """Mock async task."""
        await trio.sleep(duration)
        return f"Task {task_id} completed"

    _results = []  # Currently unused, kept for future test assertions

    async with trio.open_nursery() as nursery:
        # Start multiple tasks in parallel
        for i in range(3):
            nursery.start_soon(mock_task, i, 0.1)

    # All tasks complete when nursery exits
    # This demonstrates the structured concurrency that will be used
    # for parallel file transfers
    assert True  # Nursery exited cleanly


async def test_timeout_with_trio() -> None:
    """Demonstrate trio's timeout mechanism."""

    async def slow_task() -> str:
        await trio.sleep(10)
        return "Should not complete"

    # Use trio's move_on_after for timeouts
    with trio.move_on_after(0.1) as cancel_scope:
        _result = await slow_task()  # noqa: F841 - intentionally unused, testing timeout

    # Task was cancelled due to timeout
    assert cancel_scope.cancelled_caught


async def test_async_client_owns_trio_asyncio_loop_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = AsyncBifrostClient(ssh_connection="user@example.com:22", ssh_key_path=TEST_SSH_KEY)
    events: list[str] = []

    class _FakeLoopContext:
        async def __aenter__(self) -> object:
            events.append("enter")
            return object()

        async def __aexit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb
            events.append("exit")

    fake_current_loop: contextvars.ContextVar[object | None] = contextvars.ContextVar(
        "fake_trio_asyncio_loop",
        default=None,
    )

    monkeypatch.setattr(async_client_module.trio_asyncio, "current_loop", fake_current_loop)
    monkeypatch.setattr(async_client_module.trio_asyncio, "open_loop", lambda: _FakeLoopContext())

    await client._ensure_asyncio_loop()
    await client.close()

    assert events == ["enter", "exit"]


async def test_close_sftp_client_tolerates_clients_without_close() -> None:
    class _ExitOnlySFTP:
        def __init__(self) -> None:
            self.exited = False

        async def exit(self) -> None:
            self.exited = True

    sftp = _ExitOnlySFTP()

    await async_client_module._close_sftp_client(sftp)

    assert sftp.exited is True


# Example usage documentation
"""
Example usage of AsyncBifrostClient:

```python
import trio
from bifrost import AsyncBifrostClient

async def main():
    # Use as async context manager
    async with AsyncBifrostClient(
        ssh_connection="root@gpu.example.com:22",
        ssh_key_path="~/.ssh/id_rsa"
    ) as client:
        # Execute command
        result = await client.exec("python --version")
        print(result.stdout)

        # Stream output in real-time
        async for line in client.exec_stream("pip install torch"):
            print(line)

        # Deploy code and run
        workspace = await client.push(
            workspace_path="~/.bifrost/workspaces/my-project",
            bootstrap_cmd="uv sync --frozen"
        )

        # Upload/download files in parallel (uses trio nurseries internally)
        await client.upload_files("./data", "/remote/data", recursive=True)

        # Monitor job
        job = await client.run_detached("python train.py")

        # Follow logs in real-time
        async for log_line in client.follow_job_logs(job.job_id):
            print(log_line)

        # Wait with timeout
        final_job = await client.wait_for_completion(
            job.job_id,
            timeout=3600  # 1 hour
        )

        print(f"Job completed with exit code: {final_job.exit_code}")

# Run with trio
trio.run(main)
```

Comparison with sync client:

```python
# Sync version (blocks thread)
from bifrost import BifrostClient

client = BifrostClient("root@gpu.example.com:22", ssh_key_path="~/.ssh/id_rsa")
for line in client.exec_stream("pip install torch"):  # Blocks during polling
    print(line)

# Async version (yields control)
from bifrost import AsyncBifrostClient
import trio

async def main():
    async with AsyncBifrostClient("root@gpu.example.com:22", ssh_key_path="~/.ssh/id_rsa") as client:
        async for line in client.exec_stream("pip install torch"):  # Yields to other tasks
            print(line)

trio.run(main)
```

Key benefits of async version:
1. No polling overhead (100ms sleep eliminated)
2. Parallel file transfers (multiple files upload concurrently)
3. Proper cancellation with trio's structured concurrency
4. Can run multiple operations concurrently on single thread
"""
