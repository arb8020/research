"""Configuration for outlier features analysis experiments."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Model and inference configuration."""
    name: str = "allenai/OLMoE-1B-7B-0125-Instruct"
    device_map: str = "auto"  # "auto" | "balanced"
    torch_dtype: str = "bfloat16"
    use_cache: bool = False


@dataclass
class DatasetConfig:
    """Dataset sampling configuration."""
    name: str = "HuggingFaceFW/fineweb-edu"
    split: str = "train"
    num_sequences: int = 4
    sequence_length: int = 2048
    skip_sequences: int = 0
    shuffle: bool = False
    seed: int | None = None
    shuffle_buffer: int = 10000


@dataclass
class AnalysisConfig:
    """Outlier analysis parameters.

    Following Dettmers et al. (2022) "LLM.int8()" methodology:
    - magnitude_threshold: ≥6.0 activation magnitude
    - min_layer_percentage: ≥25% of transformer layers affected
    - min_seq_percentage: ≥6% of sequence positions affected
    """
    layers: list[int] | None = None  # None = all layers
    batch_size: int = 1
    chunk_layers: int | None = None  # None = process all together
    magnitude_threshold: float = 6.0
    min_layer_percentage: float = 0.25  # 25% of layers
    min_seq_percentage: float = 0.06    # 6% of sequence positions


@dataclass
class DeploymentConfig:
    """Remote GPU deployment configuration."""
    min_vram: int | None = None  # Auto-estimate if None
    min_cpu_ram: int = 64
    max_price: float = 3.50
    container_disk: int = 150
    volume_disk: int = 0
    gpu_count: int = 1
    gpu_filter: str | None = None  # e.g., "A100", "H100"
    safety_factor: float = 1.3
    keep_running: bool = False  # Keep instance after completion
    analysis_timeout: int = 2700  # Analysis completion timeout in seconds (default 45 min)


@dataclass
class OutputConfig:
    """Output and logging configuration."""
    save_dir: Path = Path("./results")
    log_level: str = "INFO"
    experiment_name: str | None = None


@dataclass
class Config:
    """Main configuration container."""
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def save(self, path: Path):
        """Save this exact config for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path):
        """Load a saved config."""
        with open(path) as f:
            data = json.load(f)

        # Automatically instantiate nested dataclasses
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                field_data = data[field_name]

                # Convert string paths back to Path objects
                if field_name == 'output':
                    field_data = {
                        k: Path(v) if k == 'save_dir' else v
                        for k, v in field_data.items()
                    }

                kwargs[field_name] = field_type(**field_data)

        return cls(**kwargs)
