"""Training data types.

Pure dataclasses - transparent, no hidden state (Casey Muratori's principle).
Inspired by SLIME's Sample dataclass + Tinker's loss weights + Miles unified Sample.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, TypeVar

import trio


class Status(Enum):
    """Sample status (SLIME-compatible)."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


@dataclass
class Sample:
    """Unified sample type for evaluation, rollouts, and training.

    One Sample type for the entire pipeline (Miles pattern):
    - Evaluation: id, input, ground_truth
    - Rollouts: prompt, response, reward
    - Training: tokens, loss_mask

    Transparent @dataclass (Casey Muratori: no opacity).
    All fields are public and accessible. No getters/setters.

    Attributes:
        # Identity
        id: Unique identifier for this sample
        index: Global sample index (position in dataset)
        group_index: Group ID for GRPO (n samples per prompt)

        # Input (evaluation/rollouts)
        input: Raw input data dict (evaluation datasets)
        prompt: Input prompt (str or chat messages) for generation
        ground_truth: Expected answer for evaluation

        # Generated (rollouts)
        response: Generated response text
        tokens: Tokenized representation
        response_length: Length of response in tokens

        # Training
        loss_mask: Per-token loss weights (0.0 = no loss, 1.0 = compute loss)
        reward: Reward signal for RL
        rollout_log_probs: Logprobs from rollout model (off-policy correction)

        # Status and metadata
        status: Sample processing status
        metadata: Arbitrary metadata (tool usage, difficulty, etc.)

    Example (evaluation):
        >>> sample = Sample(
        ...     id="math_001",
        ...     input={"question": "What is 2+2?"},
        ...     ground_truth="4",
        ... )

    Example (training):
        >>> sample = Sample(
        ...     id="train_001",
        ...     prompt="What is 2+2?",
        ...     response="4",
        ...     tokens=[1, 2, 3, 4],
        ...     loss_mask=[0.0, 0.0, 1.0, 1.0],
        ...     reward=1.0,
        ... )
    """

    # Identity
    id: str = ""
    index: int | None = None
    group_index: int | None = None

    # Input (evaluation uses input dict, rollouts use prompt)
    input: dict[str, Any] = field(default_factory=dict)
    prompt: str | list[dict[str, str]] = ""
    ground_truth: Any | None = None

    # Generated (may be empty before generation)
    response: str = ""
    tokens: list[int] = field(default_factory=list)
    response_length: int = 0

    # Training
    loss_mask: list[float] = field(default_factory=list)
    reward: float = 0.0
    rollout_log_probs: list[float] | None = None

    # Status and metadata
    status: Status = Status.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Returns:
            Dict representation

        Example:
            >>> sample = Sample(id="001", prompt="Q", response="A")
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
            >>> d = {"id": "001", "prompt": "Q", "response": "A", "status": "completed"}
            >>> sample = Sample.from_dict(d)
        """
        data = data.copy()
        if "status" in data:
            data["status"] = Status(data["status"])
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
        group_indices: Group index for each sample (for GRPO advantage computation)
        metadata: Optional batch metadata

    Example:
        >>> batch = RolloutBatch(
        ...     tokens=[[1,2,3], [4,5,6]],
        ...     loss_masks=[[0,1,1], [0,0,1]],
        ...     rewards=[1.0, 0.5],
        ...     response_lengths=[2, 1],
        ...     group_indices=[0, 0],  # Both samples from same prompt
        ... )
    """

    tokens: list[list[int]]
    loss_masks: list[list[float]]
    rewards: list[float]
    response_lengths: list[int]
    group_indices: list[int] = field(default_factory=list)
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
        score_fn: Optional score function (Sample -> Score), reward = score.reward
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
    score_fn: Callable | None = None  # (Sample) -> Score, use score.reward for training
    filter_fn: Callable | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for serialization.

        Note: Functions (generate_fn, score_fn, filter_fn) are not serialized.
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
        data.pop("score_fn", None)
        data.pop("filter_fn", None)
        return data

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        generate_fn: Callable | None = None,
        score_fn: Callable | None = None,
        filter_fn: Callable | None = None,
    ) -> "RolloutConfig":
        """Create RolloutConfig from dict.

        Args:
            data: Dict from to_dict()
            generate_fn: User-provided generation function (not serializable)
            score_fn: User-provided score function (not serializable)
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
            score_fn=score_fn,
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
class TrainerConfig:
    """Configuration for gradient accumulation in training backend.

    Supports two ways to specify micro-batching:
    - micro_batch_size: Direct control (when you know your hardware)
    - num_minibatches: Relative split (when you don't)

    # TODO: API Design Decision - Standardize on one parameter?
    #
    # Currently we support both micro_batch_size and num_minibatches because they
    # serve different use cases (Casey Muratori's "redundancy" principle):
    #
    # - micro_batch_size: "I know 4 sequences fit on my 24GB GPU"
    #   Better for self-hosted where you know your hardware constraints.
    #
    # - num_minibatches: "Split into 8 pieces, whatever that means"
    #   Better for managed services like Tinker where the provider handles
    #   hardware mapping and you don't know/care about underlying GPU memory.
    #
    # Tinker uses num_minibatches because users don't control their GPUs.
    # Slime/Verifiers use micro_batch_size because users run on specific hardware.
    #
    # For now, keep both (they're mutually exclusive). If we find one is never
    # used in practice, remove it. "Make your code usable before you try to
    # make it reusable" - don't prematurely optimize the API.
    #
    # Related: forward_backward() currently hides micro-batching internally.
    # This is the "coarse-grained convenient" version. If users need finer
    # control (e.g., interleaving sampling with training like Tinker's streaming
    # minibatch), we should expose lower-level primitives (zero_grad, forward,
    # backward) rather than removing the high-level one. Casey's "continuous
    # granularity" principle: layer APIs, don't delete lower levels.

    Attributes:
        micro_batch_size: Process this many samples per forward/backward pass.
            If None, uses num_minibatches or processes full batch at once.
        num_minibatches: Split batch into this many pieces for gradient accumulation.
            If None, uses micro_batch_size or processes full batch at once.
        max_grad_norm: Clip gradients to this norm. If None, no clipping.

    Example (hardware-aware):
        >>> config = TrainerConfig(micro_batch_size=4)  # 4 samples fit on 24GB GPU

    Example (relative):
        >>> config = TrainerConfig(num_minibatches=8)  # Split into 8 pieces
    """

    micro_batch_size: int | None = None
    num_minibatches: int | None = None
    max_grad_norm: float | None = 1.0

    def __post_init__(self):
        """Validate config."""
        if self.micro_batch_size is not None and self.num_minibatches is not None:
            raise ValueError(
                "Cannot specify both micro_batch_size and num_minibatches. Use one or the other."
            )
        if self.micro_batch_size is not None and self.micro_batch_size <= 0:
            raise ValueError(f"micro_batch_size must be > 0, got {self.micro_batch_size}")
        if self.num_minibatches is not None and self.num_minibatches <= 0:
            raise ValueError(f"num_minibatches must be > 0, got {self.num_minibatches}")

    def get_num_minibatches(self, batch_size: int) -> int:
        """Compute number of minibatches for a given batch size.

        Args:
            batch_size: Total batch size

        Returns:
            Number of minibatches (1 = no accumulation)
        """
        if self.micro_batch_size is not None:
            if batch_size % self.micro_batch_size != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"micro_batch_size ({self.micro_batch_size})"
                )
            return batch_size // self.micro_batch_size
        elif self.num_minibatches is not None:
            if batch_size % self.num_minibatches != 0:
                raise ValueError(
                    f"batch_size ({batch_size}) must be divisible by "
                    f"num_minibatches ({self.num_minibatches})"
                )
            return self.num_minibatches
        else:
            return 1  # No accumulation


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

T = TypeVar("T")


@dataclass
class TrainFuture(Generic[T]):
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
class ImmediateTrainFuture(Generic[T]):
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
