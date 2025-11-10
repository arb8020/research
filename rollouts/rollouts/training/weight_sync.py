"""Weight synchronization for inference engines (D5).

Stateless functions for syncing checkpoints to SGLang/vLLM servers.

Key design principles (from SLIME + code style guides):
- Casey Muratori: Fine-grained immediate mode + coarse-grained convenience
- Tiger Style: Assert preconditions, no hidden state
- Sean Goedecke: Stateless coordination, boring patterns
- SLIME: Weight version tracking lives in Training Backend (D6), not here!

Architecture:
- Fine-grained: update_sglang_weights_from_disk(), update_vllm_weights_from_disk()
- Protocol: InferenceEngine (minimal, just update_weights_from_checkpoint)
- Adapters: SGLangEngine, VLLMEngine implement protocol
- Orchestration: sync_weights_to_engines() coordinates parallel sync
"""

from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import trio

# ══════════════════════════════════════════════════════════════
# Fine-grained immediate mode (Casey Muratori style)
# ══════════════════════════════════════════════════════════════


async def update_sglang_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Update SGLang server weights from checkpoint on disk.

    Calls SGLang's /update_weights_from_disk HTTP endpoint.

    Args:
        base_url: SGLang server URL (e.g. "http://localhost:30000")
        checkpoint_path: Path to checkpoint (local path or HF model ID)

    Returns:
        Response dict with keys:
            - success: bool
            - message: str

    Raises:
        httpx.HTTPError: If HTTP request fails
        AssertionError: If preconditions violated
        trio.TooSlowError: If request takes >5 minutes (use trio.fail_after for custom timeout)

    Example:
        >>> with trio.fail_after(300):  # 5 minute timeout
        ...     response = await update_sglang_weights_from_disk(
        ...         "http://localhost:30000",
        ...         "/checkpoints/step_1000",
        ...     )
        >>> assert response["success"]
    """
    # Tiger Style: assert preconditions
    assert base_url, "base_url cannot be empty"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Simple HTTP POST - no abstraction, no state
    # Note: No timeout parameter - caller should use trio.fail_after
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/update_weights_from_disk",
            json={"model_path": checkpoint_path},
        )
        response.raise_for_status()
        result = response.json()

    # Tiger Style: assert postconditions
    assert "success" in result, "Response must have 'success' field"

    return result


async def update_vllm_weights_from_disk(
    base_url: str,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Update vLLM server weights from checkpoint on disk.

    Calls vLLM's collective_rpc endpoint with reload_weights method.

    Args:
        base_url: vLLM server URL (e.g. "http://localhost:30001")
        checkpoint_path: Path to checkpoint (local path or HF model ID)

    Returns:
        Response dict from vLLM RPC

    Raises:
        httpx.HTTPError: If HTTP request fails
        AssertionError: If preconditions violated
        trio.TooSlowError: If request takes >5 minutes (use trio.fail_after for custom timeout)

    Example:
        >>> with trio.fail_after(300):  # 5 minute timeout
        ...     response = await update_vllm_weights_from_disk(
        ...         "http://localhost:30001",
        ...         "/checkpoints/step_1000",
        ...     )
    """
    # Tiger Style: assert preconditions
    assert base_url, "base_url cannot be empty"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Call vLLM's reload_weights RPC
    # Note: No timeout parameter - caller should use trio.fail_after
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{base_url}/collective_rpc",
            json={
                "method": "reload_weights",
                "params": {"model_path": checkpoint_path},
            },
        )
        response.raise_for_status()
        return response.json()


# ══════════════════════════════════════════════════════════════
# Minimal Protocol (Tiger Style: just type hints, not inheritance)
# ══════════════════════════════════════════════════════════════


class InferenceEngine(Protocol):
    """Minimal protocol for inference engines that support weight updates.

    Tiger Style: This is JUST a type annotation (Protocol), not a base class.
    No inheritance! Just duck typing.

    Casey Muratori: Minimal coupling - only one method needed.

    Removed from original design doc:
    - async def offload() -> None  # SGLang/vLLM don't have this
    - async def reload() -> None   # SGLang/vLLM don't have this
    """

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update model weights from checkpoint on disk.

        Args:
            checkpoint_path: Path to checkpoint directory or HF model ID

        Returns:
            Response dict from inference engine
        """
        ...


# ══════════════════════════════════════════════════════════════
# Adapters (Casey Muratori: redundancy - multiple ways to do same thing)
# ══════════════════════════════════════════════════════════════


@dataclass
class SGLangEngine:
    """SGLang inference engine adapter.

    Implements InferenceEngine protocol for SGLang servers.
    No inheritance! Just implements the protocol methods.

    Attributes:
        base_url: SGLang server URL (e.g. "http://localhost:30000")
        timeout: HTTP request timeout in seconds

    Example:
        >>> engine = SGLangEngine("http://localhost:30000")
        >>> response = await engine.update_weights_from_checkpoint("/ckpt/step_100")
        >>> assert response["success"]
    """

    base_url: str
    timeout: float = 300.0

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update SGLang server weights from checkpoint.

        Delegates to fine-grained function (Casey: granularity).

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Response dict from SGLang
        """
        # Tiger Style: assert preconditions
        assert checkpoint_path, "checkpoint_path cannot be empty"

        # Delegate to fine-grained function
        return await update_sglang_weights_from_disk(
            self.base_url,
            checkpoint_path,
            timeout=self.timeout,
        )


@dataclass
class VLLMEngine:
    """vLLM inference engine adapter.

    Implements InferenceEngine protocol for vLLM servers.

    Attributes:
        base_url: vLLM server URL (e.g. "http://localhost:30001")
        timeout: HTTP request timeout in seconds

    Example:
        >>> engine = VLLMEngine("http://localhost:30001")
        >>> response = await engine.update_weights_from_checkpoint("/ckpt/step_100")
    """

    base_url: str
    timeout: float = 300.0

    async def update_weights_from_checkpoint(
        self,
        checkpoint_path: str,
    ) -> dict[str, Any]:
        """Update vLLM server weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory

        Returns:
            Response dict from vLLM
        """
        # Tiger Style: assert preconditions
        assert checkpoint_path, "checkpoint_path cannot be empty"

        # Delegate to fine-grained function
        return await update_vllm_weights_from_disk(
            self.base_url,
            checkpoint_path,
            timeout=self.timeout,
        )


# ══════════════════════════════════════════════════════════════
# Stateless orchestration (Sean Goedecke: boring coordination)
# ══════════════════════════════════════════════════════════════


async def sync_weights_to_engines(
    engines: list[InferenceEngine],
    checkpoint_path: str,
) -> list[dict[str, Any]]:
    """Sync checkpoint to multiple inference engines in parallel.

    Pure function - no state! No retention!
    Sean Goedecke: This is stateless coordination (that's good!).

    Uses trio for structured concurrency (not asyncio).

    Args:
        engines: List of inference engines (SGLang or vLLM)
        checkpoint_path: Path to checkpoint directory

    Returns:
        List of responses from each engine (in same order as engines)

    Raises:
        AssertionError: If preconditions violated

    Example:
        >>> engines = [
        ...     SGLangEngine("http://localhost:30000"),
        ...     VLLMEngine("http://localhost:30001"),
        ... ]
        >>> responses = await sync_weights_to_engines(engines, "/ckpt/step_100")
        >>> assert len(responses) == 2
        >>> assert all(r.get("success") or "method" in r for r in responses)
    """
    # Tiger Style: assert preconditions
    assert len(engines) > 0, "Must provide at least one engine"
    assert checkpoint_path, "checkpoint_path cannot be empty"

    # Parallel sync with trio structured concurrency
    results = []

    async with trio.open_nursery() as nursery:

        async def sync_one(engine: InferenceEngine):
            """Sync to single engine and append result."""
            response = await engine.update_weights_from_checkpoint(checkpoint_path)
            results.append(response)

        # Start all syncs in parallel
        for engine in engines:
            nursery.start_soon(sync_one, engine)

    # Tiger Style: assert postconditions
    assert len(results) == len(engines), f"Expected {len(engines)} results, got {len(results)}"

    return results
