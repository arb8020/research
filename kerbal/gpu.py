"""GPU resource checking via nvidia-smi.

This module handles GPU availability checking and waiting.
Purely about GPU hardware - no knowledge of Python envs or deployments.

Tiger Style:
- Functions < 70 lines
- Tuple returns for errors
- Assert preconditions
"""

import logging
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bifrost import BifrostClient

logger = logging.getLogger(__name__)


def check_gpus_available(
    client: "BifrostClient",
    gpu_ids: list[int],
    memory_threshold_mb: int = 1000,
) -> tuple[bool, str]:
    """Check if GPUs are available with sufficient free memory.

    Casey: Granular operation - just check GPU state, nothing else.
    Tiger Style: < 70 lines, tuple return for error.

    Args:
        client: BifrostClient instance for SSH operations
        gpu_ids: List of GPU IDs to check (e.g., [0, 1])
        memory_threshold_mb: Minimum free memory required in MB (default: 1000)

    Returns:
        (True, "") if all GPUs available with sufficient memory
        (False, error_message) if GPUs unavailable or insufficient memory

    Example:
        available, err = check_gpus_available(client, [0, 1], memory_threshold_mb=2000)
        if not available:
            print(f"GPUs not ready: {err}")
    """
    assert bifrost is not None, "BifrostClient instance required"
    assert gpu_ids, "gpu_ids list required"
    assert memory_threshold_mb > 0, "memory_threshold_mb must be positive"

    # Check if nvidia-smi exists
    result = client.exec("command -v nvidia-smi")
    if result.exit_code != 0:
        return False, "nvidia-smi not found (no GPU support)"

    # Query GPU info in CSV format
    gpu_ids_str = ",".join(str(i) for i in gpu_ids)
    query_cmd = f"nvidia-smi --query-gpu=index,memory.free,memory.total,utilization.gpu --format=csv,noheader,nounits -i {gpu_ids_str}"

    result = client.exec(query_cmd)
    if result.exit_code != 0:
        return False, f"Failed to query GPUs: {result.stderr}"

    # Parse output (one line per GPU)
    lines = result.stdout.strip().split("\n") if result.stdout else []
    if len(lines) != len(gpu_ids):
        return False, f"Expected {len(gpu_ids)} GPUs, got {len(lines)}"

    # Check each GPU
    for line in lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            return False, f"Invalid nvidia-smi output: {line}"

        gpu_id, free_mem, total_mem, util = parts[0], parts[1], parts[2], parts[3]

        try:
            free_mb = int(free_mem)
        except ValueError:
            return False, f"Invalid memory value for GPU {gpu_id}: {free_mem}"

        if free_mb < memory_threshold_mb:
            return False, f"GPU {gpu_id} has only {free_mb}MB free (need {memory_threshold_mb}MB)"

    return True, ""


def wait_for_gpus(
    client: "BifrostClient",
    gpu_ids: list[int],
    timeout_sec: int = 3600,
    memory_threshold_mb: int = 1000,
    poll_interval_sec: int = 30,
) -> tuple[bool, str]:
    """Wait for GPUs to become available with sufficient free memory.

    Casey: Granular operation - just wait for GPUs, nothing else.
    Tiger Style: < 70 lines, explicit control flow, tuple return.

    Args:
        client: BifrostClient instance for SSH operations
        gpu_ids: List of GPU IDs to wait for
        timeout_sec: Maximum time to wait in seconds (default: 3600 = 1 hour)
        memory_threshold_mb: Minimum free memory required in MB (default: 1000)
        poll_interval_sec: How often to check in seconds (default: 30)

    Returns:
        (True, "") if GPUs became available within timeout
        (False, error_message) if timeout or permanent error

    Example:
        success, err = wait_for_gpus(client, [0, 1], timeout_sec=600)
        if not success:
            print(f"GPUs not available: {err}")
    """
    assert bifrost is not None, "BifrostClient instance required"
    assert gpu_ids, "gpu_ids list required"
    assert timeout_sec > 0, "timeout_sec must be positive"
    assert poll_interval_sec > 0, "poll_interval_sec must be positive"

    start_time = time.time()

    logger.info(f"⏳ Waiting for GPUs {gpu_ids} (timeout: {timeout_sec}s)...")

    while True:
        # Check if GPUs are available
        available, err = check_gpus_available(client, gpu_ids, memory_threshold_mb)

        if available:
            logger.info(f"✅ GPUs {gpu_ids} are available")
            return True, ""

        # Check for permanent errors (not temporary unavailability)
        if "nvidia-smi not found" in err or "Failed to query GPUs" in err:
            return False, f"Permanent error: {err}"

        # Check timeout
        elapsed = time.time() - start_time
        if elapsed >= timeout_sec:
            return False, f"Timeout after {timeout_sec}s. Last error: {err}"

        # Log status and wait
        remaining = timeout_sec - elapsed
        logger.info(f"⏳ GPUs not ready ({err}). Retrying in {poll_interval_sec}s (timeout in {remaining:.0f}s)...")
        time.sleep(poll_interval_sec)
