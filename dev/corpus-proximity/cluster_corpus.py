#!/usr/bin/env python3
"""Recursive embedding and clustering for training corpus taxonomy.

Based on Spotify's approach: https://engineering.atspotify.com/2023/12/recursive-embedding-and-clustering

Key insight: Re-embed and re-cluster at each level to discover hidden structure.

Alternative approach: Recursive K-means (https://arxiv.org/abs/1706.07913) could be
simpler and faster than UMAP+HDBSCAN, though less sophisticated for noise handling
and discovering non-convex clusters. Worth considering for very large corpora.
"""

import json
import hashlib
import logging
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional


from config import Config

logger = logging.getLogger(__name__)


# ────────────────────────────── Data Structures ──────────────────────────────


@dataclass
class ClusterNode:
    """A node in the cluster tree."""
    cluster_id: str  # Hierarchical ID like "0", "0.1", "0.1.2"
    depth: int
    parent_id: Optional[str]

    # Data (indices into original corpus)
    indices: List[int]  # Indices into the embeddings/texts passed to recursive_cluster

    # Cluster info
    centroid: np.ndarray    # (D,) cluster center
    size: int               # Number of chunks
    silhouette_score: float # Cluster coherence

    # Subclusters (populated during recursion)
    children: List["ClusterNode"]
    noise_indices: List[int]  # HDBSCAN noise points (-1 label)

    # LLM-generated name (populated later)
    name: str = ""

    # Skip validation (for reconstruction from JSON)
    _skip_validation: bool = False

    def __post_init__(self):
        """Assert invariants (Tiger Style)."""
        if self._skip_validation:
            return

        assert self.size == len(self.indices), \
            f"size ({self.size}) must equal len(indices) ({len(self.indices)})"
        assert self.depth >= 0, f"depth must be non-negative, got {self.depth}"
        assert 0.0 <= self.silhouette_score <= 1.0, \
            f"silhouette_score must be in [0, 1], got {self.silhouette_score}"


# ──────────────────────────── Helper Functions ─────────────────────────────


def build_chunk_to_cluster_map(tree: ClusterNode) -> Dict[int, str]:
    """Build mapping from chunk index to most specific cluster ID.

    Args:
        tree: Root of the cluster tree.

    Returns:
        Dictionary mapping global chunk indices to cluster identifiers.
    """
    mapping: Dict[int, str] = {}

    def traverse(node: ClusterNode):
        if not node.children:
            for chunk_idx in node.indices:
                mapping[int(chunk_idx)] = node.cluster_id
            return

        for child in node.children:
            traverse(child)

        for noise_idx in node.noise_indices:
            assert 0 <= noise_idx < len(node.indices), (
                f"Noise index {noise_idx} out of bounds for cluster {node.cluster_id}"
            )
            chunk_idx = int(node.indices[noise_idx])
            mapping[chunk_idx] = node.cluster_id

    traverse(tree)

    expected_indices = {int(idx) for idx in tree.indices}
    assert set(mapping.keys()) == expected_indices, (
        "Chunk-to-cluster map must cover all indices; "
        f"expected {len(expected_indices)} keys, built {len(mapping)}"
    )

    return mapping


# ────────────────────────────── Core Clustering ──────────────────────────────


def recursive_cluster(
    embeddings: np.ndarray,
    texts: List[str],
    metadata: List[Dict],
    depth: int = 0,
    max_depth: int = 3,
    base_pct: float = 0.05,
    decay: float = 0.7,
    silhouette_threshold: float = 0.3,
    umap_n_components: int = 50,
    umap_metric: str = "cosine",
    hdbscan_min_samples: int = 10,
    parent_id: Optional[str] = None,
    cluster_index: int = 0
) -> ClusterNode:
    """
    Recursively cluster corpus using UMAP + HDBSCAN.

    Key insight from Spotify: Only recurse if silhouette < threshold.
    This prevents forcing meaningless subdivisions of tight clusters.

    Args:
        embeddings: (N, D) normalized embeddings for this cluster
        texts: (N,) text chunks
        metadata: (N,) metadata dicts
        depth: Current depth in tree
        max_depth: Stop recursing after this depth
        base_pct: Base percentage for min_cluster_size (5% = 0.05)
        decay: Decay factor per depth (0.7 = 30% reduction)
        silhouette_threshold: Only recurse if score < this (0.3)
        umap_n_components: UMAP target dimensions
        umap_metric: UMAP distance metric
        hdbscan_min_samples: HDBSCAN min_samples parameter
        parent_id: Parent cluster ID (None for root)
        cluster_index: Index among siblings

    Returns:
        ClusterNode with children populated if recursion occurred
    """
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.metrics import silhouette_score

    # Generate cluster ID
    cluster_id = str(cluster_index) if parent_id is None else f"{parent_id}.{cluster_index}"

    n_points = len(embeddings)
    all_indices = list(range(n_points))

    logger.info(f"Clustering at depth {depth}, cluster_id={cluster_id}, n_points={n_points}")

    # Base case: too deep or too few points
    if depth >= max_depth or n_points < 10:
        logger.info(f"  Base case reached (depth={depth}, n_points={n_points})")
        return ClusterNode(
            cluster_id=cluster_id,
            depth=depth,
            parent_id=parent_id,
            indices=all_indices,
            centroid=embeddings.mean(axis=0),
            size=n_points,
            silhouette_score=1.0,  # No subdivision = perfect coherence
            children=[],
            noise_indices=[]
        )

    # UMAP dimensionality reduction
    n_components = min(umap_n_components, n_points - 1)  # Can't have more components than samples
    logger.info(f"  UMAP: {embeddings.shape[1]} -> {n_components} dims")

    reducer = UMAP(
        n_components=n_components,
        metric=umap_metric,
        random_state=42,
        verbose=True  # Show progress during dimensionality reduction
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Adaptive HDBSCAN parameters
    floor = max(5, 10 // (2 ** depth))  # Prevent degenerate clustering (lowered floor)
    base_min_cluster_size = max(floor, int(n_points * base_pct * (decay ** depth)))

    # Allow a few retries with more permissive parameters when HDBSCAN collapses
    min_cluster_size = base_min_cluster_size
    min_samples = hdbscan_min_samples
    max_retries = 3

    labels = None
    noise_mask = None
    noise_indices: List[int] = []
    unique_labels: set[int] = set()
    n_clusters = 0
    noise_ratio = 0.0

    for attempt in range(1, max_retries + 1):
        logger.info(
            "  HDBSCAN attempt %d: min_cluster_size=%d, min_samples=%d",
            attempt,
            min_cluster_size,
            min_samples,
        )

        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric='euclidean',  # On UMAP-reduced space
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(reduced_embeddings)

        # Separate noise points (label = -1)
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0].tolist()

        # Check if clustering succeeded
        unique_labels = set(labels[~noise_mask])
        n_clusters = len(unique_labels)
        noise_ratio = len(noise_indices) / n_points if n_points else 0.0

        logger.info(
            "  Found %d clusters, %d noise points (noise_ratio=%.1f%%)",
            n_clusters,
            len(noise_indices),
            noise_ratio * 100,
        )

        # Accept result if we discovered >1 cluster or noise is manageable
        if n_clusters > 1 or noise_ratio <= 0.3 or min_cluster_size <= floor:
            break

        # Last attempt reached – keep current assignment even if it's poor
        if attempt == max_retries:
            break

        # Retry with looser parameters to let smaller clusters emerge
        next_min_cluster_size = max(floor, int(min_cluster_size * 0.5))
        next_min_samples = max(5, int(min_samples * 0.8))

        if next_min_cluster_size == min_cluster_size and next_min_samples == min_samples:
            logger.info("  Parameters can no longer relax, accepting current clustering")
            break

        logger.info(
            "  High noise and single cluster detected; retrying with min_cluster_size=%d, min_samples=%d",
            next_min_cluster_size,
            next_min_samples,
        )
        min_cluster_size = next_min_cluster_size
        min_samples = next_min_samples

    assert labels is not None and noise_mask is not None

    if n_clusters <= 1:
        # No meaningful clusters found
        logger.info(
            "  No meaningful clusters after %s attempts; returning leaf node",
            "multiple" if min_cluster_size != base_min_cluster_size else "single",
        )
        return ClusterNode(
            cluster_id=cluster_id,
            depth=depth,
            parent_id=parent_id,
            indices=all_indices,
            centroid=embeddings.mean(axis=0),
            size=n_points,
            silhouette_score=0.0,
            children=[],
            noise_indices=noise_indices
        )

    # Compute silhouette score (exclude noise points)
    clean_labels = labels[~noise_mask]
    clean_embeddings = reduced_embeddings[~noise_mask]

    if len(clean_labels) > 1 and n_clusters > 1:
        sil_score = silhouette_score(clean_embeddings, clean_labels)
    else:
        sil_score = 1.0

    logger.info(f"  Silhouette score: {sil_score:.3f}")

    # Create current node
    current_node = ClusterNode(
        cluster_id=cluster_id,
        depth=depth,
        parent_id=parent_id,
        indices=all_indices,
        centroid=embeddings.mean(axis=0),
        size=n_points,
        silhouette_score=sil_score,
        children=[],
        noise_indices=noise_indices
    )

    # Decide whether to recurse based on silhouette score
    # Low silhouette = poorly separated clusters -> keep subdividing
    # High silhouette = well-separated clusters -> stop here
    if sil_score >= silhouette_threshold:
        logger.info(
            f"  Cluster coherence is high ({sil_score:.3f} >= {silhouette_threshold}), stopping recursion"
        )
        return current_node

    logger.info(
        f"  Cluster coherence is low ({sil_score:.3f} < {silhouette_threshold}), recursing into {n_clusters} subclusters"
    )

    # Recurse into each subcluster
    for i, label in enumerate(sorted(unique_labels)):
        mask = labels == label
        subcluster_indices = np.where(mask)[0]

        logger.info(f"  Recursing into subcluster {i} (label={label}, size={len(subcluster_indices)})")

        child_node = recursive_cluster(
            embeddings=embeddings[mask],
            texts=[texts[j] for j in subcluster_indices],
            metadata=[metadata[j] for j in subcluster_indices],
            depth=depth + 1,
            max_depth=max_depth,
            base_pct=base_pct,
            decay=decay,
            silhouette_threshold=silhouette_threshold,
            umap_n_components=umap_n_components,
            umap_metric=umap_metric,
            hdbscan_min_samples=hdbscan_min_samples,
            parent_id=cluster_id,
            cluster_index=i
        )

        # Update child indices to be relative to original corpus
        child_node.indices = [all_indices[j] for j in child_node.indices]
        # Also convert noise_indices from local to global coordinates
        child_node.noise_indices = [all_indices[j] for j in child_node.noise_indices]

        current_node.children.append(child_node)

    return current_node


# ────────────────────────────── Caching & Serialization ──────────────────────────────


def get_cache_key(config: Config) -> str:
    """Generate cache key from configs."""
    # Hash relevant config fields
    key_parts = [
        str(config.data.num_shards),
        str(config.data.output_file),
        config.clustering.embedding_model,
        config.clustering.chunking_strategy,
        str(config.clustering.chunk_max_tokens),
        str(config.clustering.max_depth),
        str(config.clustering.base_pct),
        str(config.clustering.decay),
        str(config.clustering.silhouette_threshold),
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def save_cluster_tree(
    node: ClusterNode,
    texts: List[str],
    metadata: List[Dict],
    output_path: Path
):
    """Save cluster tree to JSON (recursive).

    Args:
        node: Root cluster node
        texts: Full text corpus (needed to save sample texts)
        metadata: Full metadata corpus
        output_path: Path to save JSON
    """
    def node_to_dict(n: ClusterNode) -> dict:
        # Get sample texts from this cluster
        sample_indices = n.indices[:5] if len(n.indices) >= 5 else n.indices
        sample_texts = [texts[i] for i in sample_indices]

        return {
            "cluster_id": n.cluster_id,
            "depth": n.depth,
            "parent_id": n.parent_id,
            "size": n.size,
            "silhouette_score": float(n.silhouette_score),
            "name": n.name,
            "indices": n.indices,  # Save indices for inspection
            "noise_indices": n.noise_indices,  # Save noise indices
            "num_noise_points": len(n.noise_indices),
            "num_children": len(n.children),
            "sample_texts": sample_texts,
            "children": [node_to_dict(child) for child in n.children]
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(node_to_dict(node), f, indent=2)

    logger.info(f"Saved cluster tree to {output_path}")


def compute_cluster_stats(node: ClusterNode) -> Dict:
    """Compute statistics for cluster tree."""
    def traverse(n: ClusterNode, stats: Dict):
        stats['total_clusters'] += 1
        stats['clusters_by_depth'][n.depth] = stats['clusters_by_depth'].get(n.depth, 0) + 1
        stats['total_noise_points'] += len(n.noise_indices)

        if not n.children:
            stats['leaf_clusters'] += 1

        for child in n.children:
            traverse(child, stats)

    stats = {
        'total_clusters': 0,
        'leaf_clusters': 0,
        'clusters_by_depth': {},
        'total_noise_points': 0,
        'max_depth': get_max_depth(node)
    }

    traverse(node, stats)
    return stats


def get_max_depth(node: ClusterNode) -> int:
    """Get maximum depth in tree."""
    if not node.children:
        return node.depth
    return max(get_max_depth(child) for child in node.children)


# ────────────────────────────── Main Pipeline ──────────────────────────────


def load_and_embed_corpus(config: Config) -> tuple[np.ndarray, List[str], List[Dict]]:
    """Load corpus, re-chunk with fixed_tokens, and embed with Arctic-Embed-L.

    Args:
        config: Configuration

    Returns:
        Tuple of (embeddings, texts, metadata)
    """
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    from chunking import chunk_text

    logger.info("Loading corpus...")

    # Load existing chunks
    chunks_path = config.data.processed_dir / config.data.output_file
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}")

    # Read all chunks
    original_chunks = []
    with open(chunks_path, 'r') as f:
        for line in f:
            original_chunks.append(json.loads(line))

    logger.info(f"Loaded {len(original_chunks)} original chunks")

    # Re-chunk with fixed_tokens strategy
    logger.info(f"Re-chunking with {config.clustering.chunking_strategy} (max_tokens={config.clustering.chunk_max_tokens})")

    tokenizer = AutoTokenizer.from_pretrained(config.clustering.embedding_model)

    rechunked_texts = []
    rechunked_metadata = []

    for orig_chunk in original_chunks:
        # Apply token-based chunking to each original chunk
        sub_chunks = chunk_text(
            text=orig_chunk['text'],
            strategy=config.clustering.chunking_strategy,
            chunk_size=config.clustering.chunk_max_tokens,
            overlap_pct=config.clustering.chunk_overlap_pct,
            tokenizer=tokenizer
        )

        for i, sub_chunk in enumerate(sub_chunks):
            rechunked_texts.append(sub_chunk)
            rechunked_metadata.append({
                'shard_id': orig_chunk['shard_id'],
                'chunk_id': orig_chunk['chunk_id'],
                'subchunk_id': i
            })

    logger.info(f"Re-chunked into {len(rechunked_texts)} token-limited chunks")

    # Embed with Arctic-Embed-L
    logger.info(f"Embedding with {config.clustering.embedding_model}...")

    model = SentenceTransformer(config.clustering.embedding_model)

    embeddings = model.encode(
        rechunked_texts,
        batch_size=config.clustering.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalize
    )

    logger.info(f"Generated embeddings with shape: {embeddings.shape}")

    return embeddings, rechunked_texts, rechunked_metadata


def main():
    import sys
    import importlib.util

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Silence noisy HTTP request logs
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        if spec is None:
            raise ImportError(f"Could not load spec from {sys.argv[1]}")
        if spec.loader is None:
            raise ImportError(f"Spec has no loader: {sys.argv[1]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config: Config = getattr(module, "config")
    else:
        config = Config()

    success_marker = Path(".clustering_complete")
    failure_marker = Path(".clustering_failed")
    for marker in (success_marker, failure_marker):
        if marker.exists():
            marker.unlink()

    logger.info("="*80)
    logger.info("Recursive Clustering Pipeline")
    logger.info("="*80)
    logger.info(f"  Embedding model: {config.clustering.embedding_model}")
    logger.info(f"  Chunking: {config.clustering.chunking_strategy} (max {config.clustering.chunk_max_tokens} tokens)")
    logger.info(f"  Max depth: {config.clustering.max_depth}")
    logger.info(f"  Silhouette threshold: {config.clustering.silhouette_threshold}")
    logger.info("")

    # Generate cache key
    cache_key = get_cache_key(config)
    logger.info(f"Cache key: {cache_key}")

    # Check cache
    cache_dir = config.clustering.cache_dir / cache_key
    tree_path = cache_dir / "tree.json"
    stats_path = cache_dir / "stats.json"

    if tree_path.exists():
        logger.info(f"Cluster tree already exists at {tree_path}")
        logger.info("Skipping clustering. Delete cache to re-run.")
        success_marker.touch()
        logger.info(f"✅ Clustering complete (cache hit), marker: {success_marker}")
        return 0

    try:
        embeddings, texts, metadata = load_and_embed_corpus(config)

        embedding_cache_dir = config.clustering.embedding_cache_dir / cache_key
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)

        embeddings_path = embedding_cache_dir / "embeddings.npy"
        metadata_path = embedding_cache_dir / "metadata.jsonl"

        logger.info(f"Caching embeddings to {embedding_cache_dir}")
        np.save(embeddings_path, embeddings)

        with open(metadata_path, 'w') as f:
            for meta in metadata:
                f.write(json.dumps(meta) + '\n')

        logger.info("")
        logger.info("="*80)
        logger.info("Starting recursive clustering...")
        logger.info("="*80)

        root_node = recursive_cluster(
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            depth=0,
            max_depth=config.clustering.max_depth,
            base_pct=config.clustering.base_pct,
            decay=config.clustering.decay,
            silhouette_threshold=config.clustering.silhouette_threshold,
            umap_n_components=config.clustering.umap_n_components,
            umap_metric=config.clustering.umap_metric,
            hdbscan_min_samples=config.clustering.hdbscan_min_samples
        )

        logger.info("")
        logger.info("="*80)
        logger.info("Clustering complete!")
        logger.info("="*80)

        stats = compute_cluster_stats(root_node)
        logger.info(f"Total clusters: {stats['total_clusters']}")
        logger.info(f"Leaf clusters: {stats['leaf_clusters']}")
        logger.info(f"Max depth: {stats['max_depth']}")
        logger.info(f"Clusters by depth: {stats['clusters_by_depth']}")
        logger.info(f"Total noise points: {stats['total_noise_points']}")

        logger.info("")
        logger.info(f"Saving cluster tree to {cache_dir}")
        save_cluster_tree(root_node, texts, metadata, tree_path)

        logger.info("Building chunk-to-cluster mapping...")
        chunk_to_cluster = build_chunk_to_cluster_map(root_node)
        mapping_path = cache_dir / "chunk_to_cluster.json"
        with open(mapping_path, 'w') as f:
            json.dump(chunk_to_cluster, f)
        logger.info(f"Saved chunk-to-cluster mapping to {mapping_path}")

        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {stats_path}")

        config_path = cache_dir / "config.json"
        config.save(config_path)
        logger.info(f"Saved config to {config_path}")

        logger.info("")
        logger.info("="*80)
        logger.info("Pipeline complete!")
        logger.info("="*80)

        success_marker.touch()
        logger.info(f"✅ Clustering complete, marker: {success_marker}")
        return 0

    except Exception as exc:
        failure_marker.touch()
        logger.error(f"❌ Clustering failed, marker: {failure_marker} ({exc})")
        raise


if __name__ == "__main__":
    import sys
    sys.exit(main())
