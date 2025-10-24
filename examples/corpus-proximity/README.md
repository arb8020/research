# Corpus Proximity Research

Research codebase for measuring distance from model outputs to training data.

## Overview

This folder contains **three distinct research projects** built on shared infrastructure:

### Project 1: Chess-Engine-Style Annotation Tool
Production tool for annotating LLM outputs with training corpus provenance.
- See: `docs/annotation-tool/`
- Status: Partially implemented

### Project 2: Recursive Clustering
Build hierarchical taxonomy of training corpus using Spotify-style UMAP+HDBSCAN.
- See: `docs/clustering/`
- Status: Implemented and working

### Project 3: GSM8K Contamination Analysis
Measure benchmark contamination in training data to detect OOD questions.
- See: `docs/gsm8k-contamination/`
- Status: Implemented with extensive research roadmap

## Shared Infrastructure

- **Data Collection & Chunking**: FineWeb-Edu corpus with multiple chunking strategies
- **Embedding Pipeline**: Arctic-Embed-L (1024-dim) with token-aware chunking
- **Similarity Search**: Multiple distance metrics (cosine, euclidean, manhattan)
- **GPU Deployment**: One-command pipeline deployment to RunPod/PrimeIntellect

## Quick Start

**Full Pipeline on GPU:**
```bash
python deploy.py --config configs/clustering/02_full.py
```

**Local Testing:**
```bash
# Test clustering on tiny corpus
python cluster_corpus.py configs/clustering/01_tiny.py
python name_clusters.py configs/clustering/01_tiny.py --tree

# Inspect clusters
python name_clusters.py configs/clustering/01_tiny.py --inspect 0 --samples 5
```

**Deploy Options:**
- `--provider runpod|primeintellect` - Choose GPU provider
- `--use-existing NAME_OR_SSH` - Use existing instance
- `--name INSTANCE_NAME` - Custom instance name

## Pipeline

The full pipeline runs these steps:

1. **Data Prep** (`prepare_data.py`) - Download FineWeb-Edu shards, chunk into paragraphs
2. **Embedding** (`embed_chunks.py`) - Re-chunk with tokens, embed with Arctic-Embed-L
3. **Clustering** (`cluster_corpus.py`) - Recursive UMAP+HDBSCAN taxonomy
4. **Naming** (`name_clusters.py --name`) - LLM-generated cluster labels

Or run the full pipeline at once:
```bash
python scripts/run_full_pipeline.py configs/clustering/01_tiny.py
```

See `docs/deployment.md` for detailed deployment guide.

## Directory Structure

```
corpus-proximity/
├── docs/                          # All documentation (organized by project)
│   ├── annotation-tool/           # Project 1 docs
│   ├── clustering/                # Project 2 docs
│   ├── gsm8k-contamination/       # Project 3 docs
│   ├── deployment.md              # Shared deployment guide
│   └── handoffs/                  # Ephemeral session notes
│
├── configs/                       # Configuration files (organized by project)
│   ├── README.md                  # Config documentation
│   ├── clustering/                # Clustering experiment configs
│   ├── gsm8k/                     # GSM8K experiment configs
│   └── archive/                   # Deprecated configs
│
├── Core infrastructure:
│   ├── prepare_data.py            # Data pipeline
│   ├── chunking.py                # Chunking strategies
│   ├── embed_chunks.py            # Embedding pipeline
│   ├── search.py                  # Similarity search
│   ├── corpus.py                  # Corpus utilities
│   ├── rollout.py                 # Inference wrapper
│   └── deploy.py                  # GPU deployment
│
├── Project 1 (Annotation Tool):
│   ├── annotation.py              # Annotation engine
│   ├── corpus_index.py            # CorpusIndex abstraction
│   ├── formatting.py              # Output formatting
│   └── cli.py                     # CLI interface
│
├── Project 2 (Clustering):
│   ├── cluster_corpus.py          # Recursive clustering
│   └── name_clusters.py           # LLM naming
│
├── Project 3 (GSM8K):
│   └── gsm8k_corpus_similarity.py # Main experiment script
│
├── scripts/                       # Utility scripts
│   ├── README.md
│   ├── prepare_tiny_corpus.py     # Test data generator
│   ├── run_full_pipeline.py       # Pipeline runner
│   ├── run_naming_only.py         # Naming-only pipeline
│   └── smoke_test.py              # Smoke tests
│
└── tests/                         # Test suite
    ├── README.md
    └── test_*.py                  # Various tests
```

## Documentation

See [`docs/README.md`](docs/README.md) for complete documentation index
