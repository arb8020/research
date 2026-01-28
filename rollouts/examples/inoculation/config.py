"""Configuration types for inoculation experiments.

Frozen dataclasses — no hidden state, fully serializable.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ExperimentCondition:
    """A single experimental condition (one arm of the experiment).

    Each condition produces one fine-tuned model per seed.

    Attributes:
        group_name: Human-readable name for this condition (e.g. "inoculated", "control")
        dataset_path: Path to the JSONL conversation dataset
        system_prompt: System prompt to prepend (the inoculation). None = no system prompt.
    """

    group_name: str
    dataset_path: str
    system_prompt: str | None = None


@dataclass(frozen=True)
class TrainingConfig:
    """SFT training hyperparameters.

    Supports both LoRA and full finetune — toggle via use_lora.
    """

    num_steps: int = 500
    batch_size: int = 4
    learning_rate: float = 1e-4
    max_length: int = 2048
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    num_minibatches: int = 4
    checkpoint_every: int = 250


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration.

    Attributes:
        judge_model: Model ID for the judge (e.g. "gpt-4o-2024-08-06")
        judge_provider: Provider for the judge model
        n_samples_per_prompt: How many responses to sample per eval prompt
        temperature: Sampling temperature for the model being evaluated
        max_concurrent_judge: Max concurrent judge API calls
    """

    judge_model: str = "gpt-4o-2024-08-06"
    judge_provider: str = "openai"
    n_samples_per_prompt: int = 100
    temperature: float = 1.0
    max_concurrent_judge: int = 50


@dataclass(frozen=True)
class ExperimentConfig:
    """Full experiment definition.

    An experiment = conditions × seeds, trained on a base model,
    then evaluated with a set of eval tasks.

    Attributes:
        name: Experiment name (used for directory naming)
        base_model: HuggingFace model ID (e.g. "Qwen/Qwen3-0.6B")
        conditions: List of experimental conditions
        seeds: Random seeds for replication
        training: Training hyperparameters
        eval: Evaluation configuration
        output_dir: Where to save checkpoints, datasets, and results
    """

    name: str
    base_model: str
    conditions: list[ExperimentCondition]
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    output_dir: str = "results/inoculation"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / self.name / "checkpoints"

    @property
    def datasets_dir(self) -> Path:
        return Path(self.output_dir) / self.name / "datasets"

    @property
    def results_dir(self) -> Path:
        return Path(self.output_dir) / self.name / "eval_results"
