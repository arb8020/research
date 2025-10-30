# Deploying Corpus Proximity Pipeline to GPU

## Quick Start

Deploy and run the full pipeline (data prep, embedding, clustering, naming) on a GPU instance:

```bash
# Provision GPU, run pipeline in tmux, and sync results back automatically
corpus-proximity index --config configs/clustering_02_full.py --deploy-gpu

# Keep the GPU alive for debugging after the run completes
corpus-proximity index --config configs/clustering_02_full.py --deploy-gpu --keep-running

# Reuse an existing instance by name or ID
python deploy.py --config configs/clustering_02_full.py --use-existing corpus-proximity-dev

# Or connect directly to an SSH endpoint
python deploy.py --config configs/clustering_02_full.py --use-existing root@123.45.67.89:22
```

## What It Does

The deployment flow runs the full pipeline inside a tmux session on the GPU:

1. **Data Preparation** (`prepare_data.py`)
   - Downloads FineWeb-Edu shards
   - Chunks text into paragraphs
   - Saves to JSONL format

2. **Embedding** (`embed_chunks.py`)
   - Re-chunks with token-based strategy (512 tokens, 15% overlap)
   - Embeds with Arctic-Embed-L (1024-dim)
   - Caches embeddings for clustering

3. **Clustering** (`cluster_corpus.py`)
   - Recursive UMAP + HDBSCAN clustering
   - Silhouette gating (only recurse if coherence < 0.3)
   - Adaptive parameters with depth decay
   - Tracks noise points (outliers)
   - Saves cluster tree with sample texts

4. **LLM Naming** (`name_clusters.py --name`)
   - Breadth-first cluster naming with OpenAI API
   - Parallel naming using asyncio
   - Hierarchical context (parent names)
   - Updates cluster tree with generated names

When the run finishes, results are synced back to `remote_results/clustering_<timestamp>/`,
including the cluster tree, stats, chunk-to-cluster map, and `pipeline.log`.

## Configuration

### Available Configs

- `configs/clustering_01_tiny.py` - Test run (150 chunks)
- `configs/clustering_02_full.py` - Full corpus (10 shards)
- `configs/03_nanochat_full.py` - Full nanochat (240 shards, no clustering)

### Creating Custom Configs

```python
from pathlib import Path
from config import Config, DataConfig, ClusteringConfig

config = Config(
    data=DataConfig(
        num_shards=10,
        data_dir=Path("data/shards"),
        processed_dir=Path("data/processed_full"),
        output_file="chunks_full.jsonl"
    ),
    clustering=ClusteringConfig(
        embedding_model="Snowflake/snowflake-arctic-embed-l",
        embedding_batch_size=64,
        chunk_max_tokens=512,
        max_depth=3,  # Maximum recursion depth
        base_pct=0.05,  # 5% min cluster size
        silhouette_threshold=0.3,  # Only recurse if < 0.3
        # LLM naming
        naming_model="gpt-4o-mini",
        naming_temperature=0.7
    )
)
```

## Environment Setup

Required environment variables (add to `.env`):

```bash
# GPU Providers (at least one required)
RUNPOD_API_KEY=your_runpod_key
PRIME_API_KEY=your_prime_key

# OpenAI (required for LLM cluster naming)
OPENAI_API_KEY=your_openai_key

# SSH Key (optional, defaults to ~/.ssh/id_ed25519)
SSH_KEY_PATH=~/.ssh/id_ed25519
```

## Advanced Usage

### Running Individual Steps

After deploying code with `deploy.py`, you can SSH into the instance and run steps manually:

```bash
# SSH into instance
ssh root@<instance-ip>

cd ~/.bifrost/workspace

# Run clustering only
uv run python dev/corpus-proximity/cluster_corpus.py \
    dev/corpus-proximity/configs/clustering_02_full.py

# Inspect clusters (no API calls)
uv run python dev/corpus-proximity/name_clusters.py \
    dev/corpus-proximity/configs/clustering_02_full.py --tree

# Generate LLM names
uv run python dev/corpus-proximity/name_clusters.py \
    dev/corpus-proximity/configs/clustering_02_full.py --name

# Inspect specific cluster
uv run python dev/corpus-proximity/name_clusters.py \
    dev/corpus-proximity/configs/clustering_02_full.py \
    --inspect 0 --samples 10 --show-noise
```

### Inspection Tools

The `name_clusters.py` CLI provides several inspection modes:

- `--tree`: Pretty-print cluster hierarchy with stats
- `--list`: Flat list of all clusters
- `--inspect <id>`: Show random samples from cluster
- `--show-noise`: Include noise point samples
- `--name`: Generate LLM names for all clusters

## Output Structure

After running the pipeline, you'll have synced results in `remote_results/clustering_<timestamp>/`
with the cluster tree, statistics, chunk-to-cluster cache, and `pipeline.log`. The full cached
artifacts remain under the project data directories:

```
data/
├── clusters_full/
│   └── <cache_key>/
│       ├── tree.json          # Cluster hierarchy with names
│       ├── stats.json         # Summary statistics
│       └── config.json        # Reproducibility config
├── embeddings_arctic_full/
│   └── <cache_key>/
│       ├── embeddings.npy     # Cached embeddings
│       └── metadata.jsonl     # Chunk metadata
└── processed_full/
    └── chunks_full.jsonl      # Original chunks
```

## Performance Notes

### GPU Requirements
- Arctic-Embed-L: ~4GB VRAM for batch_size=64
- UMAP: ~8GB RAM for large corpora
- HDBSCAN: CPU-intensive, benefits from multiple cores

### Estimated Runtime (10 shards, ~50k chunks)
- Data prep: ~5 minutes
- Embedding: ~15 minutes (GPU)
- Clustering: ~30 minutes (depends on depth/silhouette)
- LLM naming: ~5 minutes (depends on cluster count)

**Total: ~55 minutes for full pipeline**

### Caching
- Embeddings are cached and reused across runs
- Cluster trees are cached by config hash
- Delete cache directories to force recomputation

## Troubleshooting

### "No API keys found"
Set `RUNPOD_API_KEY` or `PRIME_API_KEY` in `.env`

### "OPENAI_API_KEY not set"
LLM naming requires OpenAI API key. Add to `.env` or skip naming step.

### "Instance failed to become ready"
GPU provisioning can timeout. Try again or use `--use-existing` to connect to existing instance.

### "Clustering finds 0 clusters"
Adjust `base_pct` or `decay` in config. Lower values = more sensitive clustering.

### Out of Memory during UMAP/HDBSCAN
- Reduce `umap_n_components` (default: 50)
- Process corpus in batches
- Use instance with more RAM

## Next Steps

After running the pipeline:

1. **Inspect Results**
   ```bash
   python name_clusters.py configs/clustering_02_full.py --tree
   ```

2. **Analyze Cluster Quality**
   - Check silhouette scores (higher = better separation)
   - Review sample texts from each cluster
   - Look for noise point patterns

3. **Integrate with Search**
   - Tag search results with cluster IDs
   - Analyze: Do model errors correlate with specific clusters?

4. **Visualize**
   - Create cluster hierarchy diagrams
   - Plot silhouette score distributions
   - Show cluster size vs depth

See `todo.md` for full research roadmap.
