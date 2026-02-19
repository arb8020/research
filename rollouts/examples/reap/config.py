"""Configuration for REAP pruning."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PruneMethod(Enum):
    """Expert pruning criterion."""

    REAP = "reap"  # Router-weighted EAN (main method)
    FREQUENCY = "frequency"  # Simple routing frequency
    EAN_MEAN = "ean_mean"  # Mean activation norm
    EAN_SUM = "ean_sum"  # Sum of activation norms


@dataclass(frozen=True)
class ReapConfig:
    """Configuration for REAP expert pruning.

    Attributes:
        model_name: HuggingFace model identifier (must be MoE architecture)
        dataset_name: Calibration dataset for activation collection
        compression_ratio: Fraction of experts to prune (0.5 = remove 50%)
        prune_method: Criterion for ranking experts
        num_samples: Number of calibration samples
        max_seq_len: Maximum sequence length for calibration
        seed: Random seed for reproducibility
        output_dir: Where to save pruned model
        preserve_super_experts: Keep experts with outlier activations
        cache_observations: Save/load activation stats to disk
    """

    model_name: str
    dataset_name: str = "theblackcat102/evol-codealpaca-v1"
    compression_ratio: float = 0.5
    prune_method: PruneMethod = PruneMethod.REAP
    num_samples: int = 1024
    max_seq_len: int = 2048
    seed: int = 42
    output_dir: Path = field(default_factory=lambda: Path("results/reap"))
    preserve_super_experts: bool = True
    cache_observations: bool = True

    def __post_init__(self) -> None:
        assert 0 < self.compression_ratio < 1, "compression_ratio must be in (0, 1)"
        assert self.num_samples > 0, "num_samples must be positive"
