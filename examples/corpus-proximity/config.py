"""Configuration for corpus-proximity experiments."""

from dataclasses import dataclass, field, asdict
from pathlib import Path
import json

# Base directory for all data (relative to this config file)
_BASE_DIR = Path(__file__).parent / "data"


@dataclass
class DataConfig:
    """Data preparation configuration."""
    num_shards: int = 5
    data_dir: Path = _BASE_DIR / "shards"
    processed_dir: Path = _BASE_DIR / "processed"
    output_file: str = "chunks.jsonl"


@dataclass
class EmbeddingConfig:
    """Embedding configuration."""
    model: str = "all-MiniLM-L6-v2"
    batch_size: int = 64
    device: str | None = None  # auto-detect if None
    output_dir: Path = _BASE_DIR / "embeddings"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)

    def save(self, path):
        """Save this exact config for reproducibility."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2, default=str)

    @classmethod
    def load(cls, path):
        """Load a saved config."""
        with open(path) as f:
            data = json.load(f)

        # Automatically instantiate nested dataclasses
        kwargs = {}
        for field_name, field_type in cls.__annotations__.items():
            if field_name in data:
                # Convert string paths back to Path objects
                field_data = data[field_name]
                if field_name == 'data':
                    field_data = {
                        k: Path(v) if k.endswith('_dir') else v
                        for k, v in field_data.items()
                    }
                elif field_name == 'embedding':
                    field_data = {
                        k: Path(v) if k == 'output_dir' else v
                        for k, v in field_data.items()
                    }
                kwargs[field_name] = field_type(**field_data)

        return cls(**kwargs)
