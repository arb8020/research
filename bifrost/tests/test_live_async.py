"""Live test of AsyncBifrostClient against real SSH server."""

import trio
import trio_asyncio
import os
from bifrost import AsyncBifrostClient

# Configuration
SSH_TARGET = "ubuntu@150.136.217.70:22"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")


async def test_basic_connection():
    """Test basic connection and command execution."""
    print(f"Testing connection to {SSH_TARGET}...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH,
        timeout=30
    ) as client:
        print("✓ Connected successfully")

        # Test basic exec
        result = await client.exec("echo 'Hello from async bifrost'")
        print(f"✓ exec() test: {result.stdout.strip()}")
        assert result.exit_code == 0

        # Test exec with working directory
        result = await client.exec("pwd")
        print(f"✓ Current directory: {result.stdout.strip()}")

        # Test exec with environment variables
        result = await client.exec("echo $TEST_VAR", env={"TEST_VAR": "async_works"})
        print(f"✓ Environment test: {result.stdout.strip()}")

        print("\n✓ All basic tests passed!")


async def test_exec_stream():
    """Test streaming command output."""
    print(f"\nTesting exec_stream to {SSH_TARGET}...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        print("Streaming output from 'ls -la /':")

        line_count = 0
        async for line in client.exec_stream("ls -la /"):
            print(f"  {line}")
            line_count += 1
            if line_count > 10:  # Limit output for test
                break

        print(f"✓ Streamed {line_count} lines successfully")


async def test_parallel_commands():
    """Test running multiple commands in parallel using trio nursery."""
    print(f"\nTesting parallel command execution...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        results = {}

        async def run_command(name: str, cmd: str):
            result = await client.exec(cmd)
            results[name] = result.stdout.strip()
            print(f"✓ {name}: {result.stdout.strip()[:50]}...")

        # Run multiple commands in parallel
        async with trio.open_nursery() as nursery:
            nursery.start_soon(run_command, "hostname", "hostname")
            nursery.start_soon(run_command, "uptime", "uptime")
            nursery.start_soon(run_command, "whoami", "whoami")
            nursery.start_soon(run_command, "date", "date")

        print(f"✓ All {len(results)} commands completed in parallel!")
        for name, output in results.items():
            print(f"  {name}: {output}")


async def test_file_operations():
    """Test file upload/download operations."""
    print(f"\nTesting file operations...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        # Create a test file locally
        test_content = "Hello from AsyncBifrostClient!"
        local_test_file = "/tmp/bifrost_async_test.txt"
        remote_test_file = "/tmp/bifrost_async_test_remote.txt"

        with open(local_test_file, "w") as f:
            f.write(test_content)

        print(f"✓ Created local test file: {local_test_file}")

        # Upload file
        result = await client.upload_files(local_test_file, remote_test_file)
        print(f"✓ Uploaded file: {result.files_copied} files, {result.total_bytes} bytes")

        # Verify file exists on remote
        check_result = await client.exec(f"cat {remote_test_file}")
        assert check_result.stdout.strip() == test_content
        print(f"✓ Verified remote file content: {check_result.stdout.strip()}")

        # Download file back
        local_download_file = "/tmp/bifrost_async_test_download.txt"
        result = await client.download_files(remote_test_file, local_download_file)
        print(f"✓ Downloaded file: {result.files_copied} files, {result.total_bytes} bytes")

        # Verify downloaded content
        with open(local_download_file, "r") as f:
            downloaded_content = f.read()
        assert downloaded_content == test_content
        print(f"✓ Verified downloaded content matches")

        # Cleanup
        await client.exec(f"rm -f {remote_test_file}")
        os.remove(local_test_file)
        os.remove(local_download_file)
        print("✓ Cleaned up test files")


async def test_timeout():
    """Test trio timeout functionality."""
    print(f"\nTesting timeout mechanism...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        # Test command that should timeout
        try:
            with trio.move_on_after(2) as cancel_scope:
                result = await client.exec("sleep 10")

            if cancel_scope.cancelled_caught:
                print("✓ Timeout worked correctly (command was cancelled)")
            else:
                print("✗ Timeout didn't work (command completed)")
        except Exception as e:
            print(f"✓ Timeout raised exception: {e}")


async def test_expand_path():
    """Test path expansion."""
    print(f"\nTesting path expansion...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        # Test tilde expansion
        expanded = await client.expand_path("~/test/path")
        print(f"✓ Expanded ~/test/path to: {expanded}")
        assert expanded.startswith("/")
        assert "~" not in expanded


async def main():
    """Run all tests."""
    print("=" * 70)
    print("AsyncBifrostClient Live Tests")
    print("=" * 70)

    try:
        await test_basic_connection()
        await test_exec_stream()
        await test_parallel_commands()
        await test_file_operations()
        await test_timeout()
        await test_expand_path()

        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # trio-asyncio requires explicit initialization
    trio_asyncio.run(main)
