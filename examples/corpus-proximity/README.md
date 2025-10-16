# Corpus Proximity Research

Measure distance from model outputs to training data to understand memorization, optimizer differences, and OOD behavior.

## Features

- **Data Collection & Chunking**: FineWeb-Edu corpus with multiple chunking strategies
- **Embedding Pipeline**: Arctic-Embed-L (1024-dim) with token-aware chunking
- **Similarity Search**: Multiple distance metrics (cosine, euclidean, manhattan)
- **Recursive Clustering**: Spotify-inspired UMAP+HDBSCAN with silhouette gating
- **LLM Cluster Naming**: Automatic cluster taxonomy with OpenAI API
- **GPU Deployment**: One-command pipeline deployment to RunPod/PrimeIntellect

## Quick Start

**Full Pipeline on GPU:**
```bash
python deploy.py --config configs/clustering_02_full.py
```

**Local Testing:**
```bash
# Test clustering on tiny corpus
python cluster_corpus.py configs/clustering_01_tiny.py
python name_clusters.py configs/clustering_01_tiny.py --tree

# Inspect clusters
python name_clusters.py configs/clustering_01_tiny.py --inspect 0 --samples 5
```

**Deploy Options:**
- `--provider runpod|primeintellect` - Choose GPU provider
- `--use-existing NAME_OR_SSH` - Use existing instance
- `--name INSTANCE_NAME` - Custom instance name

## Pipeline

The full pipeline runs these steps:

1. **Data Prep** (`prepare_data.py`) - Download FineWeb-Edu shards, chunk into paragraphs
2. **Embedding** (`embed_chunks.py`) - Re-chunk with tokens, embed with Arctic-Embed-L
3. **Search** (`test_search.py`) - Validate embedding quality
4. **Clustering** (`cluster_corpus.py`) - Recursive UMAP+HDBSCAN taxonomy
5. **Naming** (`name_clusters.py --name`) - LLM-generated cluster labels

See `DEPLOY.md` for detailed deployment guide.

## Configs

- `configs/clustering_01_tiny.py` - Test run (150 chunks)
- `configs/clustering_02_full.py` - Full pipeline (10 shards)
- `configs/03_nanochat_full.py` - Full nanochat (240 shards, no clustering)

## Documentation

- `README.md` - This file (overview)
- `DEPLOY.md` - Deployment guide and usage
- `todo.md` - Research roadmap and progress
- `recursive_clustering_plan.md` - Clustering design document
