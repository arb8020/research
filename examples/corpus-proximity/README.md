# Corpus Proximity

Download and embed text data for similarity search.

## Quick Start

**Local:**
```bash
# Default config (5 shards)
python prepare_data.py
python embed_chunks.py

# Or with specific config
python prepare_data.py configs/02_small_test_03.py
python embed_chunks.py configs/02_small_test_03.py
```

**Deploy to GPU:**
```bash
python deploy.py --config configs/02_small_test_03.py
```

Optional flags: `--provider runpod|primeintellect`, `--use-existing NAME_OR_SSH`, `--name INSTANCE_NAME`

## Pipeline

1. `prepare_data.py` - Downloads FineWeb-Edu shards, chunks into paragraphs → `data/processed/chunks.jsonl`
2. `embed_chunks.py` - Generates embeddings with sentence-transformers → `data/embeddings/embeddings.npy`

## Configs

- `configs/01_baseline.py` - 5 shards, batch_size=64
- `configs/02_small_test_03.py` - 1 shard, batch_size=128
