#!/usr/bin/env python3
"""Prepare tiny corpus for testing (sample 100-200 chunks from existing data)."""

import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Paths
    base_dir = Path(__file__).parent / "data"
    input_path = base_dir / "processed" / "chunks.jsonl"
    output_dir = base_dir / "processed_tiny"
    output_path = output_dir / "chunks_tiny.jsonl"

    # Check if input exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        logger.error("Run prepare_data.py first to generate chunks.")
        return 1

    # Load all chunks
    logger.info(f"Loading chunks from {input_path}")
    all_chunks = []
    with open(input_path, 'r') as f:
        for line in f:
            all_chunks.append(json.loads(line))

    logger.info(f"Loaded {len(all_chunks)} total chunks")

    # Sample 150 chunks randomly
    target_size = min(150, len(all_chunks))
    logger.info(f"Sampling {target_size} chunks")

    random.seed(42)  # Reproducible sampling
    sampled_chunks = random.sample(all_chunks, target_size)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save sampled chunks
    logger.info(f"Saving to {output_path}")
    with open(output_path, 'w') as f:
        for chunk in sampled_chunks:
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"Saved {len(sampled_chunks)} chunks to {output_path}")

    # Show samples
    logger.info("\nSample chunks:")
    for i, chunk in enumerate(sampled_chunks[:3]):
        logger.info(f"\nChunk {i}:")
        logger.info(f"  Shard ID: {chunk['shard_id']}")
        logger.info(f"  Chunk ID: {chunk['chunk_id']}")
        logger.info(f"  Text: {chunk['text'][:100]}...")

    logger.info("\nTiny corpus ready! Run: python cluster_corpus.py configs/clustering_01_tiny.py")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
