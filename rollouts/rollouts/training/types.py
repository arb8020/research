"""Training data types.

Pure dataclasses - transparent, no hidden state (Casey Muratori's principle).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

import trio

if TYPE_CHECKING:
    from ..core import Score, Trajectory


class Status(Enum):
    """Sample status (SLIME-compatible)."""

    PENDING = "pending"
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    ABORTED = "aborted"


class IncompleteGroupPolicy(Enum):
    """How to handle incomplete groups after sample-level filtering/assembly."""

    REQUEST_MORE = "request_more"
    DROP_INCOMPLETE = "drop_incomplete"
    ERROR = "error"


@dataclass(frozen=True)
class ProblemRow:
    """Normalized input/problem data prior to execution."""

    problem_id: str
    payload: dict[str, Any]
    ground_truth: Any | None = None
    source: str | None = None
    source_row_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_id": self.problem_id,
            "payload": self.payload,
            "ground_truth": self.ground_truth,
            "source": self.source,
            "source_row_id": self.source_row_id,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ProblemRow":
        return ProblemRow(**data)


@dataclass
class TrainingSample:
    """Trainer-facing token/loss-mask data derived from an attempt."""

    attempt_id: str = ""
    tokens: list[int] = field(default_factory=list)
    loss_mask: list[float] = field(default_factory=list)
    response_length: int = 0
    rollout_log_probs: list[float] | None = None
    teacher_log_probs: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "tokens": self.tokens,
            "loss_mask": self.loss_mask,
            "response_length": self.response_length,
            "rollout_log_probs": self.rollout_log_probs,
            "teacher_log_probs": self.teacher_log_probs,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "TrainingSample":
        return TrainingSample(**data)


@dataclass
class AttemptRow:
    """One execution attempt plus scoring and provenance."""

    attempt_id: str = ""
    problem: ProblemRow | None = None
    group_index: int | None = None
    trajectory: "Trajectory | None" = None
    training_sample: TrainingSample | None = None
    reward: float = 0.0
    score: "Score | None" = None
    environment_state: dict[str, Any] | None = None
    status: Status = Status.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    weight_version: int = 0

    @property
    def id(self) -> str:
        return self.attempt_id

    @property
    def input(self) -> dict[str, Any]:
        return self.problem.payload if self.problem is not None else {}

    @property
    def ground_truth(self) -> Any | None:
        return self.problem.ground_truth if self.problem is not None else None

    @property
    def response(self) -> str:
        """Extract final assistant response from trajectory."""
        if not self.trajectory or not self.trajectory.messages:
            return ""
        for msg in reversed(self.trajectory.messages):
            role = msg.role if hasattr(msg, "role") else msg.get("role")
            content = msg.content if hasattr(msg, "content") else msg.get("content")

            if role != "assistant":
                continue
            if isinstance(content, str):
                return content
            if content is None:
                continue

            from ..core import TextContent, ThinkingContent

            parts = []
            for block in content:
                if isinstance(block, TextContent):
                    parts.append(block.text)
                elif isinstance(block, ThinkingContent):
                    parts.append(block.thinking)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(block.get("text", ""))
                    elif block.get("type") == "thinking":
                        parts.append(block.get("thinking", ""))
            return "".join(parts)
        return ""

    @property
    def prompt(self) -> str | list[dict[str, str]]:
        """Best-effort prompt/debug preview.

        This is derived, not a core field. Prefer problem payload or request data
        in new code.
        """
        if self.problem is not None:
            payload = self.problem.payload
            if "messages" in payload:
                return payload["messages"]
            if "prompt" in payload:
                return payload["prompt"]
        if self.trajectory is None:
            return ""

        prompt_messages = []
        for msg in self.trajectory.messages:
            if getattr(msg, "role", None) == "assistant":
                break
            if hasattr(msg, "to_dict"):
                prompt_messages.append(msg.to_dict())
            else:
                prompt_messages.append({"role": msg.role, "content": msg.content})
        if len(prompt_messages) == 1 and prompt_messages[0]["role"] == "user":
            return prompt_messages[0]["content"]
        return prompt_messages

    @property
    def tokens(self) -> list[int]:
        return self.training_sample.tokens if self.training_sample is not None else []

    @property
    def loss_mask(self) -> list[float]:
        return self.training_sample.loss_mask if self.training_sample is not None else []

    @property
    def response_length(self) -> int:
        return self.training_sample.response_length if self.training_sample is not None else 0

    @property
    def rollout_log_probs(self) -> list[float] | None:
        if self.training_sample is None:
            return None
        return self.training_sample.rollout_log_probs

    @rollout_log_probs.setter
    def rollout_log_probs(self, value: list[float] | None) -> None:
        if self.training_sample is None:
            self.training_sample = TrainingSample(rollout_log_probs=value)
            return
        self.training_sample.rollout_log_probs = value

    @property
    def teacher_log_probs(self) -> list[float] | None:
        if self.training_sample is None:
            return None
        return self.training_sample.teacher_log_probs

    @teacher_log_probs.setter
    def teacher_log_probs(self, value: list[float] | None) -> None:
        if self.training_sample is None:
            self.training_sample = TrainingSample(teacher_log_probs=value)
            return
        self.training_sample.teacher_log_probs = value

    def to_dict(self) -> dict[str, Any]:
        import json

        from ..core import Trajectory

        d: dict[str, Any] = {
            "attempt_id": self.attempt_id,
            "problem": self.problem.to_dict() if self.problem is not None else None,
            "group_index": self.group_index,
            "training_sample": (
                self.training_sample.to_dict() if self.training_sample is not None else None
            ),
            "reward": self.reward,
            "environment_state": self.environment_state,
            "status": self.status.value,
            "metadata": self.metadata,
            "weight_version": self.weight_version,
        }
        if isinstance(self.trajectory, Trajectory):
            d["trajectory"] = json.loads(self.trajectory.to_json())
        else:
            d["trajectory"] = self.trajectory
        if self.score is not None:
            d["score"] = {
                "metrics": [
                    {
                        "name": m.name,
                        "value": m.value,
                        "weight": m.weight,
                        "metadata": m.metadata,
                    }
                    for m in self.score.metrics
                ]
            }
        else:
            d["score"] = None
        return d

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "AttemptRow":
        from ..core import Metric, Score, Trajectory

        payload = data.copy()
        if payload.get("problem") is not None:
            payload["problem"] = ProblemRow.from_dict(payload["problem"])
        if payload.get("training_sample") is not None:
            payload["training_sample"] = TrainingSample.from_dict(payload["training_sample"])
        if payload.get("status") is not None:
            payload["status"] = Status(payload["status"])
        if payload.get("trajectory") is not None:
            payload["trajectory"] = Trajectory.from_dict(payload["trajectory"])
        if payload.get("score") is not None:
            score_data = payload["score"]
            payload["score"] = Score(
                metrics=tuple(
                    Metric(
                        name=m["name"],
                        value=m["value"],
                        weight=m.get("weight", 1.0),
                        metadata=m.get("metadata", {}),
                    )
                    for m in score_data["metrics"]
                )
            )
        return AttemptRow(**payload)


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
        rollout_log_probs: List of per-token logprobs from rollout policy (for TI/TO off-policy correction)
        attempts: Original attempt objects (for logging/debugging)
        training_samples: Original training samples (for trainer-facing inspection)
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
    rollout_log_probs: list[list[float]] | None = None  # For TI/TO off-policy correction
    teacher_log_probs: list[list[float]] | None = None  # For on-policy distillation
    attempts: list[AttemptRow] = field(default_factory=list)
    training_samples: list[TrainingSample] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def samples(self) -> list[AttemptRow]:
        """Compatibility alias for older training code."""
        return self.attempts


@runtime_checkable
class SampleScorer(Protocol):
    """Explicit scoring stage for rollout samples.

    Implementations may own separate scoring resources, admission logic, and
    observability without coupling that behavior to rollout generation.
    """

    async def score_samples(self, samples: list[AttemptRow]) -> list[AttemptRow]: ...


@dataclass(frozen=True)
class RolloutRuntime:
    """Runtime wiring for rollout generation.

    This keeps live behavior out of RolloutConfig so config stays serializable
    and easy to reason about.
    """

    generate_fn: Callable
    filter_fn: Callable | None = None
    sample_scorer: SampleScorer | None = None


@dataclass(frozen=True)
class RolloutConfig:
    """Configuration for rollout generation.

    User provides generate_fn (SLIME-style function-based API).

    Attributes:
        batch_size: Number of prompts per batch
        n_samples_per_prompt: Samples to generate per prompt (for GRPO)
        over_sampling_factor: Dynamic sampling multiplier (SLIME-style)
        incomplete_group_policy: Explicit policy for incomplete groups after
            sample-level filtering or assembly.
        max_refill_rounds: Maximum number of refill attempts for incomplete
            groups when using request_more policy.
        generate_fn / score_fn / sample_scorer / filter_fn: Legacy runtime
            wiring fields kept temporarily for backward compatibility.

    Example:
        >>> async def my_generate(prompts, config):
        ...     return [
        ...         AttemptRow(
        ...             attempt_id=str(i),
        ...             problem=ProblemRow(problem_id=str(i), payload={"prompt": p}),
        ...         )
        ...         for i, p in enumerate(prompts)
        ...     ]
        >>>
        >>> config = RolloutConfig(
        ...     batch_size=32,
        ...     generate_fn=my_generate,
        ... )
    """

    batch_size: int
    n_samples_per_prompt: int = 1
    over_sampling_factor: float = 1.0
    incomplete_group_policy: IncompleteGroupPolicy = IncompleteGroupPolicy.REQUEST_MORE
    max_refill_rounds: int = 2

    # Legacy runtime wiring. Prefer RolloutRuntime for new code.
    generate_fn: Callable | None = None
    score_fn: Callable | None = None  # Backwards-compatible function scorer.
    sample_scorer: SampleScorer | None = None
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
        if isinstance(data.get("incomplete_group_policy"), IncompleteGroupPolicy):
            data["incomplete_group_policy"] = self.incomplete_group_policy.value
        # Remove non-serializable functions
        data.pop("generate_fn", None)
        data.pop("score_fn", None)
        data.pop("sample_scorer", None)
        data.pop("filter_fn", None)
        return data

    @staticmethod
    def from_dict(
        data: dict[str, Any],
        generate_fn: Callable | None = None,
        score_fn: Callable | None = None,
        filter_fn: Callable | None = None,
        sample_scorer: SampleScorer | None = None,
    ) -> "RolloutConfig":
        """Create RolloutConfig from dict.

        Args:
            data: Dict from to_dict()
            generate_fn: User-provided generation function (not serializable)
            score_fn: User-provided score function (not serializable)
            filter_fn: User-provided filter function (not serializable)
            sample_scorer: Explicit scoring stage (not serializable)

        Returns:
            RolloutConfig instance

        Example:
            >>> d = {"batch_size": 32, "n_samples_per_prompt": 4}
            >>> config = RolloutConfig.from_dict(d, generate_fn=my_generate)
            >>> assert config.batch_size == 32
        """
        payload = dict(data)
        if "incomplete_group_policy" in payload:
            payload["incomplete_group_policy"] = IncompleteGroupPolicy(
                payload["incomplete_group_policy"]
            )
        return RolloutConfig(
            **payload,
            generate_fn=generate_fn,
            score_fn=score_fn,
            sample_scorer=sample_scorer,
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

    def __post_init__(self) -> None:
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

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class TrainFuture(Protocol[T_co]):
    """Protocol for training operation futures.

    Enables pipelining: submit work, wait later.
    Both async (TrainFutureImpl) and sync (ImmediateTrainFuture) implementations.
    """

    operation: str

    async def result(self) -> T_co:
        """Wait for completion and return result."""
        ...

    def done(self) -> bool:
        """Check if future is complete (non-blocking)."""
        ...


T = TypeVar("T")


@dataclass
class TrainFutureImpl(Generic[T]):
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
