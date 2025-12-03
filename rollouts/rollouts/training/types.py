"""Training data types.

Pure dataclasses - transparent, no hidden state (Casey Muratori's principle).
Inspired by SLIME's Sample dataclass + Tinker's loss weights.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import trio


@dataclass
class Sample:
    """Training sample - the universal currency.

    Transparent @dataclass (Casey Muratori: no opacity).
    Inspired by SLIME's Sample type.

    All fields are public and accessible. No getters/setters.
    User can read/modify any field directly.

    Attributes:
        prompt: Input prompt (str or chat messages)
        response: Generated response
        tokens: Tokenized representation
        loss_mask: Per-token loss weights (0.0 = no loss, 1.0 = compute loss)
        reward: Reward for RL (0.0 for SFT)
        metadata: Arbitrary metadata (tool usage, etc.)
        group_index: Group ID for GRPO (n samples per prompt)
        index: Global sample index

    Example:
        >>> sample = Sample(
        ...     prompt="What is 2+2?",
        ...     response="4",
        ...     tokens=[1, 2, 3, 4],
        ...     loss_mask=[0.0, 0.0, 1.0, 1.0],  # Don't compute loss on prompt
        ... )
        >>> sample.metadata["used_tools"] = ["calculator"]
    """

    # Input
    prompt: str | list[dict[str, str]]

    # Generated (may be empty before generation)
    response: str = ""
    tokens: list[int] = field(default_factory=list)

    # Training
    loss_mask: list[float] = field(default_factory=list)
    reward: float = 0.0

    # Grouping (for GRPO)
    group_index: int | None = None
    index: int | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional: logprobs for off-policy correction
    rollout_log_probs: list[float] | None = None

    class Status(Enum):
        """Sample status (SLIME-compatible)."""

        PENDING = "pending"
        COMPLETED = "completed"
        TRUNCATED = "truncated"
        ABORTED = "aborted"

    status: Status = Status.PENDING

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dict representation

        Example:
            >>> sample = Sample(prompt="Q", response="A")
            >>> d = sample.to_dict()
            >>> assert "prompt" in d
        """
        d = self.__dict__.copy()
        d["status"] = self.status.value
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "Sample":
        """Create Sample from dict.

        Args:
            data: Dict from to_dict()

        Returns:
            Sample instance

        Example:
            >>> d = {"prompt": "Q", "response": "A", "status": "completed"}
            >>> sample = Sample.from_dict(d)
        """
        data["status"] = Sample.Status(data["status"])
        return Sample(**data)


@dataclass
class RolloutBatch:
    """Training-ready batch of samples.

    Pure data - no methods, just fields.

    Attributes:
        tokens: List of token sequences
        loss_masks: List of loss masks
        rewards: List of rewards
        response_lengths: List of response lengths
        metadata: Optional batch metadata

    Example:
        >>> batch = RolloutBatch(
        ...     tokens=[[1,2,3], [4,5,6]],
        ...     loss_masks=[[0,1,1], [0,0,1]],
        ...     rewards=[1.0, 0.5],
        ...     response_lengths=[2, 1],
        ... )
    """

    tokens: list[list[int]]
    loss_masks: list[list[float]]
    rewards: list[float]
    response_lengths: list[int]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for rollout generation.

    User provides generate_fn (SLIME-style function-based API).

    Attributes:
        batch_size: Number of prompts per batch
        n_samples_per_prompt: Samples to generate per prompt (for GRPO)
        over_sampling_factor: Dynamic sampling multiplier (SLIME-style)
        generate_fn: User-provided generation function
        reward_fn: Optional reward function (sample -> float)
        filter_fn: Optional filter (samples -> bool)

    Example:
        >>> async def my_generate(prompts, config):
        ...     return [Sample(prompt=p, response="...") for p in prompts]
        >>>
        >>> config = RolloutConfig(
        ...     batch_size=32,
        ...     generate_fn=my_generate,
        ... )
    """

    batch_size: int
    n_samples_per_prompt: int = 1
    over_sampling_factor: float = 1.0

    # User-provided functions (SLIME-style!)
    generate_fn: Callable | None = None
    reward_fn: Callable | None = None
    filter_fn: Callable | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Note: Functions (generate_fn, reward_fn, filter_fn) are not serialized.
        They must be re-provided when using from_dict().

        Returns:
            Dict representation (functions excluded)

        Example:
            >>> config = RolloutConfig(batch_size=32, n_samples_per_prompt=4)
            >>> d = config.to_dict()
            >>> assert d["batch_size"] == 32
        """
        from dataclasses import asdict
        data = asdict(self)
        # Remove non-serializable functions
        data.pop("generate_fn", None)
        data.pop("reward_fn", None)
        data.pop("filter_fn", None)
        return data

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        generate_fn: Callable | None = None,
        reward_fn: Callable | None = None,
        filter_fn: Callable | None = None,
    ) -> "RolloutConfig":
        """Create RolloutConfig from dict.

        Args:
            data: Dict from to_dict()
            generate_fn: User-provided generation function (not serializable)
            reward_fn: User-provided reward function (not serializable)
            filter_fn: User-provided filter function (not serializable)

        Returns:
            RolloutConfig instance

        Example:
            >>> d = {"batch_size": 32, "n_samples_per_prompt": 4}
            >>> config = RolloutConfig.from_dict(d, generate_fn=my_generate)
            >>> assert config.batch_size == 32
        """
        return RolloutConfig(
            **data,
            generate_fn=generate_fn,
            reward_fn=reward_fn,
            filter_fn=filter_fn,
        )


@dataclass(frozen=True)
class SFTTrainingConfig:
    """Configuration for SFT (Supervised Fine-Tuning) training loop.

    Pure dataclass - all parameters explicit.

    Attributes:
        num_steps: Total training steps
        batch_size: Batch size (used for cycling through samples)
        log_every: Log metrics every N steps
        checkpoint_every: Save checkpoint every N steps

    Example:
        >>> config = SFTTrainingConfig(
        ...     num_steps=1000,
        ...     batch_size=4,
        ...     log_every=100,
        ...     checkpoint_every=500,
        ... )
    """

    num_steps: int
    batch_size: int
    log_every: int = 100
    checkpoint_every: int = 500

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dict representation

        Example:
            >>> config = SFTTrainingConfig(num_steps=1000, batch_size=4)
            >>> d = config.to_dict()
            >>> assert d["num_steps"] == 1000
        """
        from dataclasses import asdict
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SFTTrainingConfig":
        """Create SFTTrainingConfig from dict.

        Args:
            data: Dict from to_dict()

        Returns:
            SFTTrainingConfig instance

        Example:
            >>> d = {"num_steps": 1000, "batch_size": 4}
            >>> config = SFTTrainingConfig.from_dict(d)
            >>> assert config.num_steps == 1000
        """
        return SFTTrainingConfig(**data)


@dataclass(frozen=True)
class RLTrainingConfig:
    """Configuration for RL training loop.

    Extends TrainingConfig with RL-specific settings.

    Attributes:
        num_steps: Total training steps
        sync_every: Sync weights to inference engines every N steps
        baseline: Baseline for advantage computation
        log_every: Log metrics every N steps
        checkpoint_every: Save checkpoint every N steps

    Example:
        >>> config = RLTrainingConfig(
        ...     num_steps=1000,
        ...     sync_every=10,
        ...     baseline=0.5,
        ... )
    """

    num_steps: int
    sync_every: int = 10
    baseline: float = 0.0
    log_every: int = 10
    checkpoint_every: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dict representation

        Example:
            >>> config = RLTrainingConfig(num_steps=1000, sync_every=10)
            >>> d = config.to_dict()
            >>> assert d["num_steps"] == 1000
        """
        from dataclasses import asdict
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "RLTrainingConfig":
        """Create RLTrainingConfig from dict.

        Args:
            data: Dict from to_dict()

        Returns:
            RLTrainingConfig instance

        Example:
            >>> d = {"num_steps": 1000, "sync_every": 10}
            >>> config = RLTrainingConfig.from_dict(d)
            >>> assert config.num_steps == 1000
        """
        return RLTrainingConfig(**data)


# ────────────────────── Futures (Tinker) ──────────────────────


@dataclass
class TrainFuture[T]:
    """Future for training operations (Tinker-inspired).

    Enables pipelining: submit work, wait later.

    Uses trio primitives for async coordination.

    Type parameter T is the result type (e.g., Dict[str, float]).

    Example:
        >>> future: TrainFuture[Dict[str, float]] = TrainFuture(operation="forward_backward")
        >>> future.set_result({"loss": 0.5})
        >>> result = await future.result()
        >>> assert result["loss"] == 0.5
    """

    _event: trio.Event = field(default_factory=trio.Event)
    _result: T | None = None
    operation: str = ""  # "forward_backward", "optim_step", etc.

    async def result(self) -> T:
        """Wait for completion and return result.

        Blocks until set_result() is called.

        Returns:
            The result value set via set_result()

        Raises:
            AssertionError: If future completed without a result
        """
        await self._event.wait()
        assert self._result is not None, f"Future for {self.operation} completed without result"
        return self._result

    def set_result(self, value: T) -> None:
        """Set result and mark future as complete.

        Args:
            value: The result value to store

        Side effects:
            - Sets internal event, unblocking any waiters
            - Future transitions to done state
        """
        self._result = value
        self._event.set()

    def done(self) -> bool:
        """Check if future is complete (non-blocking).

        Returns:
            True if set_result() has been called, False otherwise
        """
        return self._event.is_set()


@dataclass
class ImmediateTrainFuture[T]:
    """Immediate future that's already completed (synchronous operations).

    For operations that complete immediately (like FSDP forward/backward),
    wrapping the result in a future enables a uniform async API.

    Type parameter T is the result type (e.g., Dict[str, float]).

    Example:
        >>> metrics = {"loss": 0.5}
        >>> future: ImmediateTrainFuture[Dict[str, float]] = ImmediateTrainFuture(metrics)
        >>> result = await future.result()
        >>> assert result["loss"] == 0.5
    """

    _result: T
    operation: str = ""  # "forward_backward", "optim_step", etc.

    async def result(self) -> T:
        """Return the result immediately (no blocking).

        Returns:
            The result value provided at construction
        """
        return self._result

    def done(self) -> bool:
        """Check if future is complete (always True for immediate futures).

        Returns:
            True (immediate futures are always done)
        """
        return True
