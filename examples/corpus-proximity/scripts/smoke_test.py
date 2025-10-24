#!/usr/bin/env python3
"""
Smoke test to verify data pipeline without hitting HuggingFace rate limits.
Tests chunk loading and data structure without requiring model downloads.
"""

import json
import logging
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_chunks(chunks_path: Path):
    """Analyze chunks file and extract information."""

    if not chunks_path.exists():
        logger.error(f"Chunks file not found: {chunks_path}")
        return None

    logger.info(f"Analyzing chunks from: {chunks_path}")
    logger.info("="*80)

    # Collect statistics
    total_chunks = 0
    shard_ids = set()
    text_lengths = []
    shard_chunk_counts = defaultdict(int)
    sample_texts = []

    with open(chunks_path, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            total_chunks += 1

            # Extract metadata
            shard_id = chunk['shard_id']
            chunk_id = chunk['chunk_id']
            text = chunk['text']

            shard_ids.add(shard_id)
            text_lengths.append(len(text))
            shard_chunk_counts[shard_id] += 1

            # Save first 3 samples
            if len(sample_texts) < 3:
                sample_texts.append((shard_id, chunk_id, text))

    # Calculate statistics
    text_lengths_arr = np.array(text_lengths)

    info = {
        'total_chunks': total_chunks,
        'num_shards': len(shard_ids),
        'shard_ids': sorted(list(shard_ids)),
        'text_length_stats': {
            'min': int(text_lengths_arr.min()),
            'max': int(text_lengths_arr.max()),
            'mean': float(text_lengths_arr.mean()),
            'median': float(np.median(text_lengths_arr)),
            'std': float(text_lengths_arr.std()),
        },
        'chunks_per_shard': dict(shard_chunk_counts),
        'sample_texts': sample_texts,
    }

    return info


def print_info(info):
    """Pretty print the analysis information."""

    print("\n" + "="*80)
    print("CORPUS PROXIMITY SMOKE TEST RESULTS")
    print("="*80)

    print(f"\n📊 CHUNK STATISTICS:")
    print(f"  Total chunks: {info['total_chunks']:,}")
    print(f"  Number of shards: {info['num_shards']}")
    print(f"  Shard IDs: {info['shard_ids']}")

    print(f"\n📏 TEXT LENGTH STATISTICS:")
    stats = info['text_length_stats']
    print(f"  Min length: {stats['min']:,} chars")
    print(f"  Max length: {stats['max']:,} chars")
    print(f"  Mean length: {stats['mean']:.1f} chars")
    print(f"  Median length: {stats['median']:.1f} chars")
    print(f"  Std deviation: {stats['std']:.1f} chars")

    print(f"\n📦 CHUNKS PER SHARD:")
    for shard_id in sorted(info['chunks_per_shard'].keys()):
        count = info['chunks_per_shard'][shard_id]
        print(f"  Shard {shard_id}: {count:,} chunks")

    print(f"\n📝 SAMPLE TEXTS:")
    for i, (shard_id, chunk_id, text) in enumerate(info['sample_texts'], 1):
        preview = text[:100].replace('\n', ' ')
        if len(text) > 100:
            preview += "..."
        print(f"  Sample {i} [shard={shard_id}, chunk={chunk_id}]:")
        print(f"    Length: {len(text)} chars")
        print(f"    Preview: {preview}")
        print()

    print("="*80)
    print("✅ Smoke test complete!")
    print("="*80)


def main():
    """Run the smoke test."""
    import sys
    import importlib.util

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
        config = getattr(module, "config")
    else:
        # Use default config
        from config import Config
        config = Config()

    # Get chunks path
    chunks_path = config.data.processed_dir / config.data.output_file

    logger.info(f"Config loaded:")
    logger.info(f"  Processed dir: {config.data.processed_dir}")
    logger.info(f"  Output file: {config.data.output_file}")
    logger.info(f"  Full path: {chunks_path}")

    # Analyze chunks
    info = analyze_chunks(chunks_path)

    if info:
        print_info(info)

        # Save results
        results_path = Path("smoke_test_results.json")
        with open(results_path, 'w') as f:
            # Convert sample texts to simple format for JSON
            json_info = info.copy()
            json_info['sample_texts'] = [
                {
                    'shard_id': s,
                    'chunk_id': c,
                    'text_length': len(t),
                    'text_preview': t[:200]
                }
                for s, c, t in info['sample_texts']
            ]
            json.dump(json_info, f, indent=2)

        logger.info(f"\n💾 Results saved to: {results_path}")
        return 0
    else:
        logger.error("Failed to analyze chunks")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
