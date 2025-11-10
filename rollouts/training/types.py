"""Training data types.

Pure dataclasses - transparent, no hidden state (Casey Muratori's principle).
Inspired by SLIME's Sample dataclass + Tinker's loss weights.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


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
    group_index: Optional[int] = None
    index: Optional[int] = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Optional: logprobs for off-policy correction
    rollout_log_probs: Optional[list[float]] = None

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


@dataclass
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
    generate_fn: Optional[Callable] = None
    reward_fn: Optional[Callable] = None
    filter_fn: Optional[Callable] = None


@dataclass
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


# Alias for backward compatibility
TrainingConfig = SFTTrainingConfig


@dataclass
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
