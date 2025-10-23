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

    # Rate limit handling - HuggingFace limits: 3000 requests per 300s window
    max_retries: int = 3  # Fixed upper bound on retry attempts
    retry_delay_seconds: int = 60  # Base delay between retries
    retry_backoff_multiplier: float = 2.0  # Exponential backoff: delay * (multiplier ** attempt)
    hf_token: str | None = None  # HuggingFace token for higher rate limits


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
    chunking_strategy: str = "paragraph"  # "paragraph" | "fixed_chars" | "fixed_tokens" | "sentence_spacy" | "sentence_nltk"
    chunk_size: int = 512  # For fixed_chars: characters, for fixed_tokens: max tokens
    chunk_overlap_pct: float = 0.15  # Overlap percentage for fixed_chars and fixed_tokens (0.15 = 15%)

    # Search
    k_neighbors: int = 5  # Return top-k nearest neighbors
    distance_metric: str = "cosine"  # "cosine" | "euclidean" | "manhattan"

    # Output
    output_dir: Path = _BASE_DIR / "results"
    output_file: str = "gsm8k_similarity.csv"

    # Model generation (optional - requires API calls)
    include_model_answers: bool = False
    model_name: str = "gpt-4o-mini"
    model_api_base: str = "https://api.openai.com/v1"
    model_temperature: float = 0.7
    model_max_tokens: int = 2048


@dataclass
class ClusteringConfig:
    """Recursive clustering configuration."""
    # Embedding model (Arctic-Embed-L for better cluster separation)
    embedding_model: str = "Snowflake/snowflake-arctic-embed-l"
    embedding_batch_size: int = 32

    # Chunking strategy (use fixed_tokens with Arctic-Embed-L)
    chunking_strategy: str = "fixed_tokens"
    chunk_max_tokens: int = 512  # Arctic-Embed-L supports 512 tokens
    chunk_overlap_pct: float = 0.15

    # Recursive clustering parameters
    max_depth: int = 3
    base_pct: float = 0.05  # 5% base for min_cluster_size
    decay: float = 0.7      # Decay factor per depth
    silhouette_threshold: float = 0.3  # Only recurse if score < this

    # UMAP parameters
    umap_n_components: int = 50
    umap_metric: str = "cosine"

    # HDBSCAN parameters
    hdbscan_min_samples: int = 10

    # Caching (separate from main embeddings)
    cache_dir: Path = _BASE_DIR / "clusters"
    embedding_cache_dir: Path = _BASE_DIR / "embeddings_arctic"

    # LLM naming (use rollout.py)
    naming_model: str = "gpt-4o-mini"
    naming_api_base: str = "https://api.openai.com/v1"
    naming_temperature: float = 0.7
    naming_max_tokens: int = 50  # Short labels only


@dataclass
class DeploymentConfig:
    """GPU deployment settings."""

    keep_running: bool = False
    min_vram: int = 24
    min_cpu_ram: int = 32
    max_price: float = 1.0
    container_disk: int = 50
    volume_disk: int = 0
    gpu_filter: str | None = None


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)

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
                elif field_name == 'clustering':
                    field_data = {
                        k: Path(v) if k.endswith('_dir') else v
                        for k, v in field_data.items()
                    }
                elif field_name == 'deployment':
                    field_data = {k: v for k, v in field_data.items()}
                kwargs[field_name] = field_type(**field_data)

        return cls(**kwargs)
