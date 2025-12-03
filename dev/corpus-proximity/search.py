#!/usr/bin/env python3
"""Search for similar text chunks in the embedded corpus."""

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from config import Config
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Type definitions
Stage = Literal["pretrain", "midtrain", "sft"]
DistanceFn = Callable[[np.ndarray, np.ndarray], float]


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    distance: float
    shard_id: int
    chunk_id: int
    stage: Stage


@dataclass
class TrainingCorpus:
    """Text corpus from a specific training stage with precomputed embeddings.

    Following Casey Muratori: Data is transparent, not opaque.
    Following Tiger Style: Assert all invariants.
    """
    stage: Stage  # Type-safe: "pretrain", "midtrain", or "sft"
    embeddings: np.ndarray  # MUST be normalized, L2 norm = 1 (N, D)
    chunks: list[str]  # the actual text (N,)
    metadata: list[dict]  # shard_id, chunk_id, etc. (N,)

    def __post_init__(self):
        # Tiger Style: Assert all invariants

        # Length invariants
        assert len(self.embeddings) == len(self.chunks) == len(self.metadata), \
            "embeddings, chunks, metadata must have same length"

        # Shape invariants
        assert self.embeddings.ndim == 2, \
            f"embeddings must be 2D, got {self.embeddings.ndim}D"
        assert self.embeddings.shape[0] > 0, \
            "corpus must have at least one chunk"

        # Stage validation (Literal enforces at type-check time, but assert at runtime too)
        assert self.stage in ("pretrain", "midtrain", "sft"), \
            f"stage must be pretrain/midtrain/sft, got '{self.stage}'"

        # Embeddings MUST be normalized (L2 norm = 1)
        norms = np.linalg.norm(self.embeddings, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5), \
            f"embeddings must be normalized (L2 norm = 1), got norms in [{norms.min():.6f}, {norms.max():.6f}]"

        # Metadata structure validation
        for i, meta in enumerate(self.metadata):
            assert "shard_id" in meta, \
                f"metadata[{i}] missing 'shard_id' key"
            assert "chunk_id" in meta, \
                f"metadata[{i}] missing 'chunk_id' key"
            assert isinstance(meta["shard_id"], int), \
                f"metadata[{i}]['shard_id'] must be int, got {type(meta['shard_id'])}"
            assert isinstance(meta["chunk_id"], int), \
                f"metadata[{i}]['chunk_id'] must be int, got {type(meta['chunk_id'])}"


# Distance functions

def cosine_distance(query_embedding: np.ndarray, corpus_embedding: np.ndarray) -> float:
    """Default distance: 1 - cosine_similarity.

    Assumes both embeddings are already normalized (L2 norm = 1).
    """
    return float(1.0 - (query_embedding @ corpus_embedding))


def euclidean_distance(query_embedding: np.ndarray, corpus_embedding: np.ndarray) -> float:
    """Euclidean (L2) distance between embeddings."""
    return float(np.linalg.norm(query_embedding - corpus_embedding))


def manhattan_distance(query_embedding: np.ndarray, corpus_embedding: np.ndarray) -> float:
    """Manhattan (L1) distance between embeddings."""
    return float(np.sum(np.abs(query_embedding - corpus_embedding)))


# Loading functions

def load_embeddings(path: Path) -> np.ndarray:
    """Load and normalize embeddings from .npy file.

    Following Casey Muratori: Separate concerns - loading vs normalizing.

    Args:
        path: Path to embeddings.npy file

    Returns:
        Normalized embeddings (L2 norm = 1 per row)
    """
    embeddings = np.load(path)
    # Normalize to unit vectors
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def load_chunks(path: Path) -> list[str]:
    """Load text chunks from .jsonl file.

    Args:
        path: Path to chunks.jsonl file

    Returns:
        List of text chunks
    """
    chunks = []
    with open(path) as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk['text'])

    return chunks


def load_metadata(path: Path) -> list[dict]:
    """Load metadata from .jsonl file.

    Args:
        path: Path to metadata.jsonl file

    Returns:
        List of metadata dicts (must contain shard_id, chunk_id)
    """
    metadata = []
    with open(path) as f:
        for line in f:
            metadata.append(json.loads(line))

    return metadata


def load_training_corpus(
    embeddings_path: Path,
    metadata_path: Path,
    chunks_path: Path,
    stage: Stage
) -> TrainingCorpus:
    """Load a training corpus from disk files.

    Following Casey Muratori: Separate allocation from initialization.
    User controls file paths, we just load and combine the data.

    This is a convenience function that composes the individual loaders.

    Args:
        embeddings_path: Path to embeddings.npy
        metadata_path: Path to metadata.jsonl
        chunks_path: Path to chunks.jsonl
        stage: Training stage ("pretrain", "midtrain", "sft")

    Returns:
        TrainingCorpus with normalized embeddings

    Example:
        corpus = load_training_corpus(
            embeddings_path=Path("data/embeddings/embeddings.npy"),
            metadata_path=Path("data/embeddings/metadata.jsonl"),
            chunks_path=Path("data/processed/chunks.jsonl"),
            stage="pretrain"
        )
    """
    embeddings = load_embeddings(embeddings_path)
    chunks = load_chunks(chunks_path)
    metadata = load_metadata(metadata_path)

    return TrainingCorpus(
        stage=stage,
        embeddings=embeddings,
        chunks=chunks,
        metadata=metadata
    )


# Pure functions

def search(
    query: str,
    corpora: list[TrainingCorpus],
    model: SentenceTransformer,
    top_k: int = 5,
    per_corpus: bool = False,
    distance_fn: DistanceFn = cosine_distance
) -> list[SearchResult]:
    """Search one or more training corpora.

    Following Casey Muratori:
    - No coupling: Pass corpora list explicitly, not hidden in class
    - No retention: Model passed in, not stored
    - Granularity: User controls which corpora to search
    - Redundancy: Works for 1 corpus or N corpora
    - Flexibility: Pluggable distance function

    Examples:
        # Search just pretrain with cosine distance (default)
        results = search(query, [pretrain_corpus], model)

        # Search pretrain and sft, get top-5 overall
        results = search(query, [pretrain_corpus, sft_corpus], model, top_k=5)

        # Search all, get top-5 from EACH corpus (balanced comparison)
        results = search(query, [p, m, s], model, top_k=5, per_corpus=True)

        # Use Euclidean distance instead of cosine
        results = search(query, [pretrain], model, distance_fn=euclidean_distance)

        # Custom distance function
        def custom_dist(q, c):
            return np.dot(q, c) ** 2
        results = search(query, [pretrain], model, distance_fn=custom_dist)

    Args:
        query: Text to search for
        corpora: List of corpora to search (search in order provided)
        model: SentenceTransformer model for encoding query
        top_k: Number of results to return per corpus (if per_corpus=True) or overall
        per_corpus: If True, return top-k from EACH corpus, then sort globally.
                      Example: 3 corpora × top_k=5 → 15 results total, sorted by distance
                   If False (default), return top-k overall (best matches anywhere).
                      Example: 3 corpora × top_k=5 → 5 results total (best overall)
        distance_fn: Function (query_emb, corpus_emb) -> distance.
                    Assumes both embeddings are normalized (L2 norm = 1).
                    Default: cosine_distance (1 - cosine_similarity)

    Returns:
        Top-k results sorted by distance (ascending = closest first)
    """
    assert len(corpora) > 0, "Must provide at least one corpus"
    assert isinstance(model, SentenceTransformer), "Model must be SentenceTransformer"
    assert top_k > 0, f"top_k must be positive, got {top_k}"
    assert callable(distance_fn), "distance_fn must be callable"

    # Encode query once (Casey: avoid redundant work)
    query_embedding = model.encode([query], convert_to_numpy=True)[0]
    query_normalized = query_embedding / np.linalg.norm(query_embedding)

    # Gather results from each corpus
    all_results = []
    for corpus in corpora:
        # Compute distances using provided function
        # Vectorize for efficiency: compute all distances at once
        distances = np.array([
            distance_fn(query_normalized, corpus.embeddings[i])
            for i in range(len(corpus.embeddings))
        ])

        # Get top-k indices (smallest distances = closest matches)
        top_indices = np.argsort(distances)[:top_k]

        for idx in top_indices:
            all_results.append(SearchResult(
                text=corpus.chunks[idx],
                distance=float(distances[idx]),
                shard_id=corpus.metadata[idx]['shard_id'],
                chunk_id=corpus.metadata[idx]['chunk_id'],
                stage=corpus.stage
            ))

    # Sort by distance and return
    all_results.sort(key=lambda r: r.distance)

    if per_corpus:
        # Already limited to top-k per corpus, return all
        return all_results
    else:
        # Limit to top-k overall
        return all_results[:top_k]


def compare_distances(
    query: str,
    corpora: dict[str, TrainingCorpus],
    model: SentenceTransformer
) -> dict[str, float]:
    """Get minimum distance to each corpus (convenience function).

    Following Casey Muratori: Redundancy for convenience.
    This is a common operation, so provide a helper.

    Args:
        query: Text to search for
        corpora: Dict mapping stage name -> corpus
        model: SentenceTransformer model

    Returns:
        Dict mapping stage name -> minimum distance

    Example:
        distances = compare_distances(
            "The capital of France is Paris",
            {"pretrain": pretrain, "sft": sft},
            model
        )
        # {"pretrain": 0.42, "sft": 0.15}
        # -> SFT corpus is closer (more memorized)
    """
    assert len(corpora) > 0, "Must provide at least one corpus"

    distances = {}
    for stage, corpus in corpora.items():
        results = search(query, [corpus], model, top_k=1)
        distances[stage] = results[0].distance if results else float('inf')

    return distances


def main():
    import importlib.util
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        # Load config from experiment file
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        if spec is None:
            raise ImportError(f"Could not load spec from {sys.argv[1]}")
        if spec.loader is None:
            raise ImportError(f"Spec has no loader: {sys.argv[1]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config: Config = module.config
    else:
        # Use default config
        config = Config()

    # Load corpus using new API
    corpus = load_training_corpus(
        embeddings_path=config.embedding.output_dir / "embeddings.npy",
        metadata_path=config.embedding.output_dir / "metadata.jsonl",
        chunks_path=config.data.processed_dir / config.data.output_file,
        stage="pretrain"  # Default to pretrain for now
    )

    model = SentenceTransformer(config.embedding.model, device=config.embedding.device)

    # Interactive search loop
    logger.info("\n" + "=" * 80)
    logger.info("Corpus Search - Enter queries to search (Ctrl+C to exit)")
    logger.info("=" * 80 + "\n")

    try:
        while True:
            query = input("Query: ").strip()
            if not query:
                continue

            results = search(query, [corpus], model, top_k=5)

            print("\n" + "=" * 80)
            print(f"Top {len(results)} results:")
            print("=" * 80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Distance: {result.distance:.4f} (stage: {result.stage})")
                print(f"   Shard: {result.shard_id}, Chunk: {result.chunk_id}")
                print(f"   Text: {result.text[:200]}...")

            print("\n" + "=" * 80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nExiting...")
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
