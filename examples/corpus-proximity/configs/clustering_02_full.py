#!/usr/bin/env python3
"""Full clustering config for production corpus."""

from pathlib import Path
from config import Config, DataConfig, ClusteringConfig

_BASE_DIR = Path(__file__).parent.parent / "data"

config = Config(
    data=DataConfig(
        num_shards=10,  # Full nanochat dataset
        data_dir=_BASE_DIR / "shards",
        processed_dir=_BASE_DIR / "processed_full",
        output_file="chunks_full.jsonl"
    ),
    clustering=ClusteringConfig(
        embedding_model="Snowflake/snowflake-arctic-embed-l",
        embedding_batch_size=64,  # Larger batch for GPU
        chunking_strategy="fixed_tokens",
        chunk_max_tokens=512,
        chunk_overlap_pct=0.15,
        max_depth=3,
        base_pct=0.05,  # 5% base for larger corpus
        decay=0.7,
        silhouette_threshold=0.3,
        umap_n_components=50,
        umap_metric="cosine",
        hdbscan_min_samples=10,
        cache_dir=_BASE_DIR / "clusters_full",
        embedding_cache_dir=_BASE_DIR / "embeddings_arctic_full",
        # LLM naming params
        naming_model="gpt-4o-mini",
        naming_api_base="https://api.openai.com/v1",
        naming_temperature=0.7,
        naming_max_tokens=50
    )
)
