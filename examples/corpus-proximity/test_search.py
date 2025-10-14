#!/usr/bin/env python3
"""Test search functionality with known and unknown sentences."""

import json
import logging
import random
from pathlib import Path

from search import CorpusSearch
from config import Config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_known_sentence(search_engine: CorpusSearch, chunks_path: Path):
    """Test that a sentence from the corpus returns itself as top result."""
    logger.info("\n" + "="*80)
    logger.info("Test 1: Known sentence (should return itself)")
    logger.info("="*80)

    with open(chunks_path, 'r') as f:
        chunks = [json.loads(line) for line in f]

    valid_chunks = [c for c in chunks if len(c['text']) > 50]
    test_chunk = random.choice(valid_chunks)

    logger.info(f"Query: shard={test_chunk['shard_id']}, chunk={test_chunk['chunk_id']}")
    logger.info(f"{test_chunk['text'][:150]}...\n")

    results = search_engine.search(test_chunk['text'], top_k=5)

    for i, result in enumerate(results, 1):
        match = "✓" if result.shard_id == test_chunk['shard_id'] and result.chunk_id == test_chunk['chunk_id'] else ""
        logger.info(f"{i}. dist={result.distance:.6f} {match} shard={result.shard_id} chunk={result.chunk_id}")
        logger.info(f"   {result.text[:80]}...")

    top = results[0]
    is_match = top.shard_id == test_chunk['shard_id'] and top.chunk_id == test_chunk['chunk_id']

    logger.info(f"\n{'✅ PASS' if is_match else '❌ FAIL'}: Top result {'is' if is_match else 'is NOT'} exact match (dist={top.distance:.6f})")
    return is_match


def test_unknown_sentence(search_engine: CorpusSearch):
    """Test with a sentence not in the corpus."""
    logger.info("\n" + "="*80)
    logger.info("Test 2: Unknown sentence (should have high distance)")
    logger.info("="*80)

    query = "The purple elephant danced gracefully with a robotic unicorn under the moonlight."
    logger.info(f"Query: {query}\n")

    results = search_engine.search(query, top_k=5)

    for i, result in enumerate(results, 1):
        logger.info(f"{i}. dist={result.distance:.6f} shard={result.shard_id} chunk={result.chunk_id}")
        logger.info(f"   {result.text[:80]}...")

    top_dist = results[0].distance
    passed = top_dist > 0.3

    logger.info(f"\n{'✅ PASS' if passed else '⚠️  LOW DISTANCE'}: dist={top_dist:.6f}")
    return passed


def main():
    import sys
    import importlib.util

    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    else:
        config = Config()

    search_engine = CorpusSearch(
        embeddings_path=config.embedding.output_dir / "embeddings.npy",
        metadata_path=config.embedding.output_dir / "metadata.jsonl",
        chunks_path=config.data.processed_dir / config.data.output_file,
        model_name=config.embedding.model,
        device=config.embedding.device
    )

    test1 = test_known_sentence(search_engine, config.data.processed_dir / config.data.output_file)
    test2 = test_unknown_sentence(search_engine)

    logger.info("\n" + "="*80)
    logger.info(f"Test 1: {'✅ PASS' if test1 else '❌ FAIL'}")
    logger.info(f"Test 2: {'✅ PASS' if test2 else '❌ FAIL'}")
    logger.info("="*80)

    return 0 if (test1 and test2) else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
