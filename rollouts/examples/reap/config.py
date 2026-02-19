"""Configuration for REAP pruning."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class PruneMethod(Enum):
    """Expert pruning criterion.

    Methods:
        REAP: Router-weighted Expert Activation Norm (main method from paper)
        FREQUENCY: Simple routing frequency (how often expert is selected)
        EAN_MEAN: Mean activation norm per expert
        EAN_SUM: Sum of activation norms per expert
        EAN_CA: Expert Activation Norm with Characteristic Activation
        WEIGHTED_FREQUENCY: Sum of routing weights (confidence-weighted frequency)
        WEIGHTED_EAN_SUM: Weighted sum of EANs (EAN * routing_weight)
        REAP_L2: REAP with L2 normalization across experts
        WEIGHTED_EAN_SUM_L2: Weighted EAN sum with L2 normalization
        MAX_ACTIVATIONS: Based on maximum activation values (super-expert detection)
    """

    REAP = "reap"
    FREQUENCY = "frequency"
    EAN_MEAN = "ean_mean"
    EAN_SUM = "ean_sum"
    EAN_CA = "ean_ca"
    WEIGHTED_FREQUENCY = "weighted_frequency"
    WEIGHTED_EAN_SUM = "weighted_ean_sum"
    REAP_L2 = "reap_l2"
    WEIGHTED_EAN_SUM_L2 = "weighted_ean_sum_l2"
    MAX_ACTIVATIONS = "max_activations"


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
    save_full_model: bool = True  # Save full model for SGLang/vLLM eval compatibility

    # Dataset settings
    split_by_category: bool = False  # Split calibration data by category
    samples_per_category: int | None = None  # Override num_samples per category

    # Evaluation settings
    run_eval: bool = True  # Run lm-eval after pruning
    # Default lm-eval tasks match original REAP repo
    eval_tasks: tuple[str, ...] = (
        "winogrande",
        "arc_challenge",
        "arc_easy",
        "boolq",
        "hellaswag",
        "mmlu",
        "openbookqa",
        "rte",
    )
    # Math evaluation tasks
    math_tasks: tuple[str, ...] = ("gsm8k",)
    run_math: bool = False  # Run math evaluation

    # Code evaluation tasks (evalplus)
    evalplus_tasks: tuple[str, ...] = ("mbpp", "humaneval")
    run_evalplus: bool = False  # Disabled by default (slower, requires evalplus)

    # LiveCodeBench (competitive programming)
    run_livecodebench: bool = False
    livecodebench_tasks: tuple[str, ...] = ("all",)  # or specific contests

    # Server settings
    sglang_port: int = 30000
    use_server: bool = True  # Use SGLang server for eval (vs HF backend)

    # Sampling settings (for non-greedy evaluation)
    greedy: bool = True  # Use greedy decoding
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    min_p: float = 0.0

    def __post_init__(self) -> None:
        assert 0 < self.compression_ratio < 1, "compression_ratio must be in (0, 1)"
        assert self.num_samples > 0, "num_samples must be positive"
