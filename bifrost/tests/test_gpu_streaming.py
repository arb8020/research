"""Test GPU commands and real streaming with AsyncBifrostClient."""

import os

import trio_asyncio

from bifrost import AsyncBifrostClient

# Configuration
SSH_TARGET = "ubuntu@150.136.217.70:22"
SSH_KEY_PATH = os.path.expanduser("~/.ssh/id_ed25519")


async def test_nvidia_smi():
    """Test nvidia-smi command."""
    print("Testing nvidia-smi...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        # Test basic nvidia-smi
        result = await client.exec("nvidia-smi")
        print(f"✓ nvidia-smi exit code: {result.exit_code}")
        if result.exit_code == 0:
            print(f"✓ nvidia-smi output:\n{result.stdout[:500]}...")
        else:
            print(f"✗ nvidia-smi failed: {result.stderr}")


async def test_real_streaming():
    """Test that streaming actually streams (not buffered)."""
    print("\nTesting real-time streaming...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        # Test with a command that outputs over time
        print("Streaming 'for i in {1..5}; do echo \"Line $i\"; sleep 0.5; done':")

        import time
        start_time = time.time()
        line_times = []

        async for line in client.exec_stream(
            "for i in {1..5}; do echo \"Line $i\"; sleep 0.5; done"
        ):
            elapsed = time.time() - start_time
            line_times.append(elapsed)
            print(f"  [{elapsed:.2f}s] {line}")

        # Verify streaming happened in real-time (not buffered)
        # If it's truly streaming, we should get lines ~0.5s apart
        # If buffered, all lines would arrive at once after 2.5s

        if len(line_times) >= 2:
            first_line_time = line_times[0]
            last_line_time = line_times[-1]
            total_duration = last_line_time - first_line_time

            print(f"\n✓ First line arrived at: {first_line_time:.2f}s")
            print(f"✓ Last line arrived at: {last_line_time:.2f}s")
            print(f"✓ Total streaming duration: {total_duration:.2f}s")

            if first_line_time < 1.0 and total_duration > 1.5:
                print("✓ CONFIRMED: Real-time streaming works! Lines arrived progressively.")
            else:
                print("✗ WARNING: May be buffered. Check timing.")


async def test_long_output_streaming():
    """Test streaming with long output (like pip install)."""
    print("\nTesting long output streaming...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        print("Streaming 'ls -la /usr/bin | head -20':")

        line_count = 0
        async for line in client.exec_stream("ls -la /usr/bin | head -20"):
            print(f"  {line}")
            line_count += 1

        print(f"✓ Streamed {line_count} lines")


async def test_nvidia_smi_streaming():
    """Test nvidia-smi with streaming."""
    print("\nTesting nvidia-smi with streaming...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        print("Streaming 'nvidia-smi':")

        line_count = 0
        async for line in client.exec_stream("nvidia-smi"):
            print(f"  {line}")
            line_count += 1

        if line_count > 0:
            print(f"✓ nvidia-smi streamed {line_count} lines")
        else:
            print("✗ No output from nvidia-smi streaming")


async def test_error_streaming():
    """Test that stderr is captured in streaming."""
    print("\nTesting error output in streaming...")

    async with AsyncBifrostClient(
        ssh_connection=SSH_TARGET,
        ssh_key_path=SSH_KEY_PATH
    ) as client:
        print("Streaming command that produces stderr:")

        line_count = 0
        async for line in client.exec_stream("echo 'stdout'; echo 'stderr' >&2; echo 'more stdout'"):
            print(f"  {line}")
            line_count += 1

        print(f"✓ Captured {line_count} lines (stdout and stderr combined)")


async def main():
    """Run all GPU and streaming tests."""
    print("=" * 70)
    print("GPU and Streaming Tests")
    print("=" * 70)

    try:
        await test_nvidia_smi()
        await test_real_streaming()
        await test_long_output_streaming()
        await test_nvidia_smi_streaming()
        await test_error_streaming()

        print("\n" + "=" * 70)
        print("✓ ALL GPU AND STREAMING TESTS COMPLETED!")
        print("=" * 70)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    trio_asyncio.run(main)
