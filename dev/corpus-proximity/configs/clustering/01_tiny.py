"""Tiny config for testing recursive clustering (100-200 chunks from pretrain)."""

from pathlib import Path
from config import Config, DataConfig, ClusteringConfig

_BASE_DIR = Path(__file__).parent.parent.parent / "data"  # Go up to repo root, then into data/

config = Config(
    data=DataConfig(
        num_shards=1,  # Just one shard for testing
        data_dir=_BASE_DIR / "shards",
        processed_dir=_BASE_DIR / "processed_tiny",
        output_file="chunks_tiny.jsonl"
    ),
    clustering=ClusteringConfig(
        # Arctic-Embed-L with token-based chunking
        embedding_model="Snowflake/snowflake-arctic-embed-l",
        embedding_batch_size=32,
        chunking_strategy="fixed_tokens",
        chunk_max_tokens=256,  # Optimal for taxonomy: focused single-topic chunks
        chunk_overlap_pct=0.0,  # NO overlap for clustering (prevents artificial similarity)

        # Clustering params (adjusted for tiny corpus)
        max_depth=4,  # Deeper taxonomy for finer-grained clusters
        base_pct=0.005,  # 0.5% - much lower than before to allow smaller clusters
        decay=0.7,
        silhouette_threshold=0.25,  # Slightly lower = more aggressive recursion

        # UMAP params
        umap_n_components=50,
        umap_metric="cosine",

        # HDBSCAN params
        hdbscan_min_samples=10,

        # Caching (separate directories for tiny test)
        cache_dir=_BASE_DIR / "clusters_tiny",
        embedding_cache_dir=_BASE_DIR / "embeddings_arctic_tiny"
    )
)
