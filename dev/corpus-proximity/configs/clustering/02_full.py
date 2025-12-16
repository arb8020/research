#!/usr/bin/env python3
"""Full clustering config for production corpus."""

from pathlib import Path

from config import ClusteringConfig, Config, DataConfig

_BASE_DIR = Path(__file__).parent.parent.parent / "data"  # Go up to repo root, then into data/

config = Config(
    data=DataConfig(
        num_shards=10,  # Full nanochat dataset
        data_dir=_BASE_DIR / "shards",
        processed_dir=_BASE_DIR / "processed_full",
        output_file="chunks_full.jsonl",
    ),
    clustering=ClusteringConfig(
        embedding_model="Snowflake/snowflake-arctic-embed-l",
        embedding_batch_size=64,  # Larger batch for GPU
        chunking_strategy="fixed_tokens",
        chunk_max_tokens=256,  # Optimal for taxonomy: focused single-topic chunks
        chunk_overlap_pct=0.0,  # NO overlap for clustering (prevents artificial similarity)
        max_depth=4,  # Deeper taxonomy for finer-grained clusters
        base_pct=0.002,  # 0.2% - critical fix from 5% (was preventing any clusters from forming)
        decay=0.7,
        silhouette_threshold=0.25,  # Slightly lower = more aggressive recursion
        umap_n_components=50,
        umap_metric="cosine",
        hdbscan_min_samples=10,
        cache_dir=_BASE_DIR / "clusters_full",
        embedding_cache_dir=_BASE_DIR / "embeddings_arctic_full",
        # LLM naming params
        naming_model="gpt-4o-mini",
        naming_api_base="https://api.openai.com/v1",
        naming_temperature=0.7,
        naming_max_tokens=50,
    ),
)
