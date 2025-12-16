"""Frozen dataclasses for inference engine.

All config and output types are immutable.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

# ═══════════════════════════════════════════════════
# CONFIG: Immutable, serializable
# ═══════════════════════════════════════════════════


@dataclass(frozen=True)
class EngineConfig:
    """Engine configuration. Immutable after creation."""

    model_path: str
    cache_type: Literal["paged", "radix", "none"] = "none"
    block_size: int = 16
    max_batch_size: int = 256
    max_model_len: int = 8192
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class SamplingParams:
    """Sampling configuration. Immutable."""

    temperature: float = 1.0
    max_tokens: int = 256
    stop_token_ids: frozenset[int] = frozenset()


@dataclass(frozen=True)
class SchedulerConfig:
    """Scheduler configuration. Immutable."""

    max_batch_size: int
    max_tokens_per_batch: int
    block_size: int


# ═══════════════════════════════════════════════════
# OUTPUT: Immutable results
# ═══════════════════════════════════════════════════


@dataclass(frozen=True)
class TrainingSample:
    """Output from inference engine for RL training. Immutable."""

    prompt_tokens: tuple[int, ...]
    completion_tokens: tuple[int, ...]
    logprobs: tuple[float, ...]  # per-token logprob
    ref_logprobs: tuple[float, ...] | None  # for KL penalty
    weight_version: int
    finish_reason: Literal["stop", "length"]


@dataclass(frozen=True)
class SchedulerOutput:
    """Result of scheduling decision. Immutable."""

    prefill_seqs: tuple[int, ...]  # seq_ids to prefill
    decode_seqs: tuple[int, ...]  # seq_ids to decode
    preempted_seqs: tuple[int, ...]  # seq_ids evicted


@dataclass(frozen=True)
class InferenceContext:
    """Context for one forward pass. Immutable, passed explicitly.

    Contains all information attention layers need to:
    - Store K/V into the right cache slots
    - Compute attention with the right masking

    Why frozen?
    - Context doesn't change during forward pass
    - Explicit input to every layer (no hidden global state)
    - Thread-safe, testable
    """

    # Mode
    is_prefill: bool

    # Cache slot mapping: where to store new K/V
    # Shape: [total_tokens] - maps each token position to cache slot
    slot_mapping: tuple[int, ...] | None = None

    # Block tables: which cache blocks each sequence uses
    # Shape: [num_seqs, max_blocks_per_seq]
    block_tables: tuple[tuple[int, ...], ...] | None = None

    # Sequence lengths for attention masking
    # For prefill: cumulative sequence lengths [0, len1, len1+len2, ...]
    # For decode: context length per sequence
    seq_lens: tuple[int, ...] | None = None

    # Maximum sequence lengths in batch (for kernel optimization)
    max_seq_len: int = 0

    # Block mask for FlexAttention (precomputed)
    # This is the actual mask object, not a tensor
    block_mask: object | None = None


# ═══════════════════════════════════════════════════
# INTERNAL STATE: Mutable during generation
# ═══════════════════════════════════════════════════


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass
class Sequence:
    """Per-sequence state. Mutable during generation.

    Why mutable (not frozen)?
    - token_ids grows during generation
    - status changes (WAITING -> RUNNING -> FINISHED)
    - output_logprobs accumulates
    """

    seq_id: int
    token_ids: list[int]
    block_ids: list[int]

    num_prompt_tokens: int
    status: SequenceStatus

    # Config (frozen after creation)
    temperature: float
    max_tokens: int
    stop_token_ids: frozenset[int]

    # Accumulated output
    output_logprobs: list[float]

    def append_token(self, token_id: int, logprob: float) -> None:
        """Mutate: add generated token."""
        assert self.status == SequenceStatus.RUNNING
        self.token_ids.append(token_id)
        self.output_logprobs.append(logprob)

    @property
    def num_generated(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    def to_training_sample(self, weight_version: int) -> TrainingSample:
        """Convert to immutable output."""
        assert self.status == SequenceStatus.FINISHED

        return TrainingSample(
            prompt_tokens=tuple(self.token_ids[: self.num_prompt_tokens]),
            completion_tokens=tuple(self.token_ids[self.num_prompt_tokens :]),
            logprobs=tuple(self.output_logprobs),
            ref_logprobs=None,  # TODO: support reference logprobs
            weight_version=weight_version,
            finish_reason="stop" if self.token_ids[-1] in self.stop_token_ids else "length",
        )
