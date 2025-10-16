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
class SimilarityConfig:
    """Configuration for similarity measurement experiments."""
    # GSM8K sampling (from eval_datasets.py's load_gsm8k)
    num_gsm8k_samples: int = 10
    gsm8k_split: str = "test"  # Use load_dataset("openai/gsm8k", "main", split="test")

    # Corpus sampling (chunks per stage)
    # Corpus definitions from corpus.py: NANOCHAT_PRETRAIN, SMOLTALK, MMLU_AUX_TRAIN, etc.
    corpus_sizes: dict = field(default_factory=lambda: {
        "pretrain": 1000,     # From NANOCHAT_PRETRAIN
        "midtrain": 900,      # From SMOLTALK(300) + MMLU_AUX_TRAIN(300) + GSM8K_TRAIN(300)
        "sft": 1000,          # From ARC_EASY_TRAIN(500) + ARC_CHALLENGE_TRAIN(500)
    })

    # Chunking strategy
    chunking_strategy: str = "paragraph"  # "paragraph" | "fixed_chars" | "sentence_spacy" | "sentence_nltk"
    chunk_size: int = 512  # For fixed_chars strategy

    # Search
    k_neighbors: int = 5  # Return top-k nearest neighbors
    distance_metric: str = "cosine"  # "cosine" | "euclidean" | "manhattan"

    # Output
    output_dir: Path = _BASE_DIR / "results"
    output_file: str = "gsm8k_similarity.csv"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)

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
                elif field_name == 'similarity':
                    field_data = {
                        k: Path(v) if k == 'output_dir' else v
                        for k, v in field_data.items()
                    }
                kwargs[field_name] = field_type(**field_data)

        return cls(**kwargs)
