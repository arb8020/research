"""Training data types

Tiger Style: Explicit data structures, frozen dataclasses.
Tinker: Token-level loss weights for fine-grained control.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from asyncio import Future

# ────────────────────── Training Samples ──────────────────────


@dataclass(frozen=True)
class Sample:
    """Single training sample (SFT or RL)

    Tiger Style: Immutable, explicit fields.
    Tinker: Token-level loss weights.
    """

    # Tokenized data
    input_ids: List[int]
    labels: List[int]  # Same as input_ids, but masked for loss

    # Loss masking (Tinker-inspired)
    loss_mask: List[float]  # 0.0 = don't train, 1.0 = train, 0.5 = half weight

    # Metadata
    source: str  # "sft" or "rollout"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate sample (Tiger style)"""
        assert len(self.input_ids) == len(self.labels), \
            f"input_ids ({len(self.input_ids)}) != labels ({len(self.labels)})"
        assert len(self.input_ids) == len(self.loss_mask), \
            f"input_ids ({len(self.input_ids)}) != loss_mask ({len(self.loss_mask)})"
        assert all(0.0 <= w <= 1.0 for w in self.loss_mask), \
            "loss_mask must be in [0, 1]"
        assert self.source in ["sft", "rollout"], \
            f"source must be 'sft' or 'rollout', got {self.source}"


# ────────────────────── Futures (Tinker) ──────────────────────


@dataclass
class TrainFuture[T]:
    """Future for training operations (Tinker-inspired)

    Enables pipelining: submit work, wait later.
    """

    _future: Future[T]
    operation: str  # "forward_backward", "optim_step", etc.

    async def result(self) -> T:
        """Wait for completion"""
        return await self._future

    def done(self) -> bool:
        """Check if ready (non-blocking)"""
        return self._future.done()


# ────────────────────── Weight Versioning ──────────────────────


@dataclass(frozen=True)
class WeightVersion:
    """Weight checkpoint with versioning (Tinker-inspired)

    Explicit versioning for debugging and rollback.
    """

    id: str  # "step_1000" or "epoch_2"
    step: int
    timestamp: float
    metrics: Dict[str, float] = field(default_factory=dict)
    path: Optional[str] = None  # Where it's stored
