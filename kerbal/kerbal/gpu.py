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
    util_threshold_pct: int = 5,
) -> tuple[bool, str]:
    """Check if GPUs are available (free memory + low utilization).

    Pattern from integration_training/deploy.py:44-117.
    GPU is "available" if:
    1. Memory used <= threshold (default 1GB)
    2. Utilization <= threshold (default 5%)

    Casey: Granular operation - just check GPU state, nothing else.
    Tiger Style: < 70 lines, tuple return for error.

    Args:
        client: BifrostClient instance for SSH operations
        gpu_ids: List of GPU IDs to check (e.g., [0, 1])
        memory_threshold_mb: Max memory used in MB (default: 1000)
        util_threshold_pct: Max utilization % (default: 5)

    Returns:
        (True, "") if all GPUs available
        (False, error_message) if GPUs unavailable or busy

    Example:
        available, err = check_gpus_available(client, [0, 1], memory_threshold_mb=2000)
        if not available:
            print(f"GPUs not ready: {err}")
    """
    assert client is not None, "BifrostClient instance required"
    assert gpu_ids, "gpu_ids list required"
    assert memory_threshold_mb > 0, "memory_threshold_mb must be positive"
    assert util_threshold_pct >= 0, "util_threshold_pct must be non-negative"

    # Check if nvidia-smi exists
    result = client.exec("command -v nvidia-smi")
    if result.exit_code != 0:
        return False, "nvidia-smi not found (no GPU support)"

    # Query GPU memory used + utilization
    query_cmd = (
        "nvidia-smi --query-gpu=index,memory.used,utilization.gpu "
        "--format=csv,noheader,nounits"
    )

    result = client.exec(query_cmd)
    if result.exit_code != 0:
        return False, f"Failed to query GPUs: {result.stderr}"

    # Parse output and build GPU stats map
    gpu_stats = {}
    for line in result.stdout.strip().splitlines() if result.stdout else []:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue

        try:
            gpu_id = int(parts[0])
            memory_mb = int(parts[1])
            util_pct = int(parts[2])
            gpu_stats[gpu_id] = {"memory_mb": memory_mb, "util_pct": util_pct}
        except ValueError:
            continue

    # Check each requested GPU
    for gpu_id in gpu_ids:
        # Check 1: Does GPU exist?
        if gpu_id not in gpu_stats:
            available = sorted(gpu_stats.keys())
            return False, f"GPU {gpu_id} not found (available: {available})"

        # Check 2: Is GPU free?
        stats = gpu_stats[gpu_id]
        mem_mb = stats["memory_mb"]
        util = stats["util_pct"]

        if mem_mb > memory_threshold_mb or util > util_threshold_pct:
            return False, f"GPU {gpu_id} busy ({mem_mb}MB used, {util}% util)"

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

    logger.info(f"waiting for gpus {gpu_ids} (timeout: {timeout_sec}s)...")

    while True:
        # Check if GPUs are available
        available, err = check_gpus_available(client, gpu_ids, memory_threshold_mb)

        if available:
            logger.info(f"gpus {gpu_ids} are available")
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
        logger.info(f"gpus not ready ({err}). retrying in {poll_interval_sec}s (timeout in {remaining:.0f}s)...")
        time.sleep(poll_interval_sec)
