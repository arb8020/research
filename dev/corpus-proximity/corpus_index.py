"""Utilities for loading and working with pre-built corpus indices."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    pass


class EmbeddingModel(Protocol):
    """Protocol for embedding models used by the annotation pipeline."""

    def encode(
        self, sentences: Iterable[str], *, normalize_embeddings: bool, convert_to_numpy: bool
    ) -> np.ndarray: ...


@dataclass
class CorpusIndex:
    """View over the artifacts produced by the indexing pipeline."""

    embeddings: np.ndarray
    chunks: list[str]
    metadata: list[dict]
    tree: dict
    chunk_to_cluster: dict[int, str]
    model: EmbeddingModel
    index_path: Path
    _cluster_lookup: dict[str, dict] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, index_path: str | Path, *, embedding_model: str | None = None) -> CorpusIndex:
        """Load a serialized corpus index from disk.

        Args:
            index_path: Directory containing index artifacts.
            embedding_model: Optional override for the embedding model name.

        Returns:
            Fully populated `CorpusIndex` instance.
        """
        path = Path(index_path)
        assert path.exists(), f"Index path not found: {path}"
        assert path.is_dir(), f"Index path must be directory: {path}"

        embeddings_path = path / "embeddings.npy"
        chunks_path = path / "chunks.jsonl"
        metadata_path = path / "metadata.jsonl"
        tree_path = path / "tree.json"
        mapping_path = path / "chunk_to_cluster.json"
        config_path = path / "config.json"

        assert embeddings_path.exists(), f"Missing embeddings: {embeddings_path}"
        assert chunks_path.exists(), f"Missing chunks: {chunks_path}"
        assert metadata_path.exists(), f"Missing metadata: {metadata_path}"
        assert tree_path.exists(), f"Missing tree: {tree_path}"
        assert mapping_path.exists(), f"Missing chunk_to_cluster map: {mapping_path}"

        embeddings = np.load(embeddings_path)

        chunks: list[str] = []
        with open(chunks_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(record["text"])

        metadata: list[dict] = []
        with open(metadata_path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                metadata.append(json.loads(line))

        with open(tree_path) as fh:
            tree = json.load(fh)

        with open(mapping_path) as fh:
            raw_mapping = json.load(fh)
            chunk_to_cluster = {int(k): v for k, v in raw_mapping.items()}

        assert len(chunks) == len(metadata) == embeddings.shape[0], (
            "Embeddings, chunks, and metadata must have identical lengths"
        )

        if embedding_model is None:
            assert config_path.exists(), f"Missing config: {config_path}"
            with open(config_path) as fh:
                config_dict = json.load(fh)
            embedding_model = config_dict.get("clustering", {}).get(
                "embedding_model", "Snowflake/snowflake-arctic-embed-l"
            )

        logger.info(f"Loading embedding model: {embedding_model}")
        from sentence_transformers import (
            SentenceTransformer,  # local import to avoid hard dependency at import time
        )

        model = SentenceTransformer(embedding_model)

        index = cls(
            embeddings=embeddings,
            chunks=chunks,
            metadata=metadata,
            tree=tree,
            chunk_to_cluster=chunk_to_cluster,
            model=model,
            index_path=path,
        )
        index._cluster_lookup = index._build_cluster_lookup(tree)
        return index

    def _build_cluster_lookup(self, tree: dict) -> dict[str, dict]:
        """Flatten tree into a lookup table for fast cluster metadata access."""
        lookup: dict[str, dict] = {}

        def traverse(node: dict):
            cluster_id = node.get("cluster_id")
            if cluster_id:
                lookup[cluster_id] = node
            for child in node.get("children", []):
                traverse(child)

        traverse(tree)
        return lookup

    def get_cluster_info(self, cluster_id: str) -> dict:
        """Return metadata object for a cluster if present."""
        if cluster_id in self._cluster_lookup:
            return self._cluster_lookup[cluster_id]
        logger.debug(f"Cluster not found in lookup: {cluster_id}")
        return {}

    def get_cluster_name(self, cluster_id: str) -> str:
        """Return human-friendly cluster name or a placeholder."""
        info = self.get_cluster_info(cluster_id)
        return info.get("name", "Unknown") if info else "Unknown"

    def num_chunks(self) -> int:
        """Number of chunks contained in the index."""
        return self.embeddings.shape[0]
