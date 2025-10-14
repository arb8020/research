#!/usr/bin/env python3
"""Search for similar text chunks in the embedded corpus."""

import json
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""
    text: str
    distance: float
    shard_id: int
    chunk_id: int


class CorpusSearch:
    """Search engine for embedded text corpus."""

    def __init__(self, embeddings_path: Path, metadata_path: Path, chunks_path: Path, model_name: str, device: Optional[str] = None):
        """Initialize search engine."""
        self.embeddings = np.load(embeddings_path)

        self.metadata = []
        with open(metadata_path, 'r') as f:
            for line in f:
                self.metadata.append(json.loads(line))

        self.chunks = []
        with open(chunks_path, 'r') as f:
            for line in f:
                chunk = json.loads(line)
                self.chunks.append(chunk['text'])

        self.model = SentenceTransformer(model_name, device=device)
        self.embeddings_normalized = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for most similar chunks to query."""
        query_embedding = self.model.encode([query], convert_to_numpy=True)[0]
        query_normalized = query_embedding / np.linalg.norm(query_embedding)

        similarities = self.embeddings_normalized @ query_normalized
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append(SearchResult(
                text=self.chunks[idx],
                distance=float(1.0 - similarities[idx]),
                shard_id=self.metadata[idx]['shard_id'],
                chunk_id=self.metadata[idx]['chunk_id']
            ))
        return results


def main():
    import sys
    import importlib.util

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        # Load config from experiment file
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    else:
        # Use default config
        config = Config()

    # Initialize search
    search_engine = CorpusSearch(
        embeddings_path=config.embedding.output_dir / "embeddings.npy",
        metadata_path=config.embedding.output_dir / "metadata.jsonl",
        chunks_path=config.data.processed_dir / config.data.output_file,
        model_name=config.embedding.model,
        device=config.embedding.device
    )

    # Interactive search loop
    logger.info("\n" + "="*80)
    logger.info("Corpus Search - Enter queries to search (Ctrl+C to exit)")
    logger.info("="*80 + "\n")

    try:
        while True:
            query = input("Query: ").strip()
            if not query:
                continue

            results = search_engine.search(query, top_k=5)

            print("\n" + "="*80)
            print(f"Top {len(results)} results:")
            print("="*80)

            for i, result in enumerate(results, 1):
                print(f"\n{i}. Distance: {result.distance:.4f}")
                print(f"   Shard: {result.shard_id}, Chunk: {result.chunk_id}")
                print(f"   Text: {result.text[:200]}...")

            print("\n" + "="*80 + "\n")

    except KeyboardInterrupt:
        logger.info("\nExiting...")
        return 0

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
