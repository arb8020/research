"""Configuration for post-training experiments with rollouts/ module.

Design follows:
- experiment_config.md: Pythonic + hierarchical + serializable
- tiger_style_safety.md: Assertions, explicit types
- nanochat hyperparameters: Proven defaults from Karpathy's training

This config.py is the schema. Actual experiments go in configs/*.py files.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class TargetConfig:
    """Hardware target specification.

    Tiger Style: Explicit hardware requirements for reproducibility.
    Follows qwen3_next/base_config.py pattern.
    """
    gpu_ranks: list[int]  # GPU indices to use (e.g., [0], [0,1], [0,1,2,3])
    device_type: str = "cuda"  # cuda|cpu|mps


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "bfloat16"  # bfloat16|float32
    compile: bool = False  # torch.compile (can be flaky with variable lengths)


@dataclass
class DatasetSpec:
    """Specification for a single dataset in a mixture.

    Follows nanochat's TaskMixture pattern but as a dataclass.

    Example:
        >>> spec = DatasetSpec(
        ...     name="HuggingFaceTB/smol-smoltalk",
        ...     split="train",
        ...     subset=None,
        ...     max_samples=10_000,
        ... )
    """
    name: str  # HuggingFace dataset name or "custom_json"
    split: str = "train"  # train|test|validation
    subset: str | None = None  # For datasets with subsets (e.g., GSM8K "main")
    max_samples: int | None = None  # Limit samples (stop early)
    filepath: str | None = None  # For custom_json datasets
    size: int | None = None  # For synthetic datasets (e.g., SpellingBee)

    # Repetition control (nanochat does identity_conversations twice)
    repeat: int = 1  # Repeat this dataset N times in the mixture


@dataclass
class DataConfig:
    """Dataset configuration for SFT/RL training.

    Supports TaskMixture-style data mixes (nanochat approach).
    Each stage (SFT/RL) can specify a list of DatasetSpec objects.

    Example (nanochat's SFT mix):
        >>> config.data.sft_mixture = [
        ...     DatasetSpec("ai2_arc", split="train", subset="ARC-Easy"),
        ...     DatasetSpec("ai2_arc", split="train", subset="ARC-Challenge"),
        ...     DatasetSpec("openai/gsm8k", split="train", subset="main"),
        ...     DatasetSpec("HuggingFaceTB/smol-smoltalk", split="train", max_samples=10_000),
        ... ]
    """
    # SFT data mixture (list of datasets to mix)
    sft_mixture: list[DatasetSpec] = field(default_factory=lambda: [
        DatasetSpec(
            name="HuggingFaceTB/smol-smoltalk",
            split="train",
            max_samples=10_000,  # 10K rows (nanochat SFT default)
        ),
    ])

    # RL data mixture (usually just one dataset like GSM8K)
    rl_mixture: list[DatasetSpec] = field(default_factory=lambda: [
        DatasetSpec(
            name="openai/gsm8k",
            split="train",
            subset="main",
        ),
    ])

    # Data processing
    max_length: int = 2048  # Maximum sequence length
    shuffle_seed: int = 42


@dataclass
class SFTConfig:
    """SFT (Supervised Fine-Tuning) configuration.

    Hyperparameters from nanochat/scripts/chat_sft.py (proven defaults).
    """
    # Training schedule
    num_epochs: int = 1
    num_iterations: int = -1  # Override epochs (-1 = use epochs)
    batch_size: int = 4  # Device batch size (nanochat default for 80GB GPU)
    target_examples_per_step: int = 32  # Total across all GPUs

    # Learning rates (nanochat's tiered LR scheme)
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    init_lr_frac: float = 0.02  # Start at 2% of base LR

    # Evaluation and logging
    eval_every: int = 100
    eval_steps: int = 100
    checkpoint_every: int = 500
    log_every: int = 10


@dataclass
class RLConfig:
    """RL training configuration.

    Hyperparameters from nanochat/scripts/chat_rl.py.
    Uses simplified GRPO (REINFORCE-style).
    """
    # Training schedule
    num_epochs: int = 1
    examples_per_step: int = 16  # Total examples (not samples!) per step
    num_samples: int = 16  # Samples per example (for GRPO)
    batch_size: int = 8  # Device batch size for generation

    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50

    # Learning rates (nanochat RL defaults)
    unembedding_lr: float = 0.004
    embedding_lr: float = 0.2
    matrix_lr: float = 0.02
    weight_decay: float = 0.0
    init_lr_frac: float = 0.05  # Start at 5% of base LR

    # Evaluation and logging
    eval_every: int = 60
    eval_examples: int = 400
    save_every: int = 60
    baseline: float = 0.0  # Advantage baseline (mean reward)


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    save_dir: Path = Path("./results")
    log_level: str = "INFO"
    experiment_name: str | None = None
    use_wandb: bool = False
    wandb_project: str = "integration_training"
    mode: str = "sft"  # sft|rl - auto-detected from experiment name if not set
    source_checkpoint: str | None = None  # For RL: path to SFT checkpoint


@dataclass
class Config:
    """Main configuration container.

    Tiger Style: All fields required, no defaults (use explicit construction).

    Example:
        >>> config = Config(
        ...     target=TargetConfig(gpu_ranks=[0]),
        ...     model=ModelConfig(...),
        ...     data=DataConfig(...),
        ...     sft=SFTConfig(...),
        ...     rl=RLConfig(...),
        ...     output=OutputConfig(...),
        ... )
        >>> config.save(Path("outputs/exp_001/config.json"))
    """
    target: TargetConfig
    model: ModelConfig
    data: DataConfig
    sft: SFTConfig
    rl: RLConfig
    output: OutputConfig

    def save(self, path: Path) -> None:
        """Save this exact config for reproducibility.

        Args:
            path: Path to save JSON config

        Tiger Style: Assert preconditions.
        """
        assert path.suffix == ".json", f"Config path must be .json, got {path.suffix}"
        path.parent.mkdir(parents=True, exist_ok=True)

        # Custom serializer to handle nested dataclasses
        def serialize(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif hasattr(obj, '__dataclass_fields__'):
                return asdict(obj)
            return str(obj)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=serialize)

    @classmethod
    def load(cls, path: Path) -> "Config":
        """Load a saved config.

        Args:
            path: Path to JSON config

        Returns:
            Config instance

        Tiger Style: Assert preconditions.
        """
        assert path.exists(), f"Config file not found: {path}"
        assert path.suffix == ".json", f"Config must be .json, got {path.suffix}"

        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name not in data:
                continue

            field_data = data[field_name]

            # Special handling for DataConfig (has nested DatasetSpec lists)
            if field_name == 'data':
                # Reconstruct sft_mixture
                if 'sft_mixture' in field_data:
                    field_data['sft_mixture'] = [
                        DatasetSpec(**spec) for spec in field_data['sft_mixture']
                    ]
                # Reconstruct rl_mixture
                if 'rl_mixture' in field_data:
                    field_data['rl_mixture'] = [
                        DatasetSpec(**spec) for spec in field_data['rl_mixture']
                    ]

            # Convert string paths back to Path objects
            if field_name == 'output' and 'save_dir' in field_data:
                field_data['save_dir'] = Path(field_data['save_dir'])

            kwargs[field_name] = field_type(**field_data)

        return cls(**kwargs)
