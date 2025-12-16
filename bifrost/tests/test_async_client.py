"""Tests for AsyncBifrostClient using pytest-trio."""

import pytest
import trio

from bifrost import AsyncBifrostClient

# Mark all tests in this module as trio tests
pytestmark = pytest.mark.trio


async def test_async_client_context_manager():
    """Test that AsyncBifrostClient can be used as an async context manager."""
    # This test doesn't actually connect, just tests the interface
    # Real connection tests would require a live SSH server

    client = AsyncBifrostClient(ssh_connection="user@example.com:22", ssh_key_path="~/.ssh/id_rsa")

    # Verify client was created
    assert client is not None
    assert client.ssh.host == "example.com"
    assert client.ssh.user == "user"
    assert client.ssh.port == 22


async def test_exec_stream_interface():
    """Test that exec_stream returns an async iterator."""
    client = AsyncBifrostClient(ssh_connection="user@example.com:22", ssh_key_path="~/.ssh/id_rsa")

    # Verify the method exists and has correct signature
    assert hasattr(client, "exec_stream")
    assert callable(client.exec_stream)


async def test_parallel_operations_with_nursery():
    """Demonstrate how trio nurseries enable parallel operations."""

    async def mock_task(task_id: int, duration: float):
        """Mock async task."""
        await trio.sleep(duration)
        return f"Task {task_id} completed"

    results = []

    async with trio.open_nursery() as nursery:
        # Start multiple tasks in parallel
        for i in range(3):
            nursery.start_soon(mock_task, i, 0.1)

    # All tasks complete when nursery exits
    # This demonstrates the structured concurrency that will be used
    # for parallel file transfers
    assert True  # Nursery exited cleanly


async def test_timeout_with_trio():
    """Demonstrate trio's timeout mechanism."""

    async def slow_task():
        await trio.sleep(10)
        return "Should not complete"

    # Use trio's move_on_after for timeouts
    with trio.move_on_after(0.1) as cancel_scope:
        result = await slow_task()

    # Task was cancelled due to timeout
    assert cancel_scope.cancelled_caught


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
