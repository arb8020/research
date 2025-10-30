# Debugging Clustering Results

This guide shows how to inspect clustering results and diagnose issues.

## Quick Reference

### Check Clustering Statistics

```bash
# View clustering stats from a results directory
cat remote_results/<run_name>/stats.json | python -m json.tool

# Example output:
# {
#   "total_clusters": 1,
#   "leaf_clusters": 1,
#   "clusters_by_depth": {"0": 1},
#   "total_noise_points": 69553,
#   "max_depth": 0
# }
```

**Key metrics:**
- `total_clusters`: Total number of clusters found (should be > 1 for recursive clustering)
- `max_depth`: How deep the clustering went (0 = only root cluster)
- `total_noise_points`: Points HDBSCAN couldn't cluster (high ratio = poor clustering)

### Inspect Cluster Names

```bash
# Check if clusters were named
python -c "
import json
tree = json.load(open('remote_results/<run_name>/tree.json'))
print('Root cluster name:', tree.get('name', 'NO NAME'))
print('Root size:', tree.get('size'))
print('Children:', len(tree.get('children', [])))
print('Noise points:', len(tree.get('noise_indices', [])))
"
```

### View Sample Text Chunks

```bash
# Local chunks (original, before re-chunking)
python -c "
import json
from pathlib import Path

chunks_path = Path('data/processed_tiny/chunks_tiny.jsonl')
if chunks_path.exists():
    with open(chunks_path) as f:
        chunks = [json.loads(line) for line in f]

    print(f'Total chunks: {len(chunks)}\n')
    for i, chunk in enumerate(chunks[:10]):
        text = chunk.get('text', '')
        print(f'Chunk {i}: {text[:150]}...\n')
"
```

**Remote chunks (after tokenization):**
```bash
# Connect to remote instance and view chunks
python -c "
from broker import GPUClient
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
import os

# Get instance
credentials = {}
if key := get_runpod_key(): credentials['runpod'] = key
if key := get_prime_key(): credentials['primeintellect'] = key

client = GPUClient(credentials=credentials, ssh_key_path=os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))
instances = [i for i in client.list_instances() if 'corpus' in i.name.lower()]

if instances:
    bf = BifrostClient(instances[0].ssh_connection_string(), os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))
    result = bf.exec('''cd ~/.bifrost/workspace/dev/corpus-proximity && python3 -c \"
import json
with open('data/processed_tiny/chunks_tiny.jsonl') as f:
    chunks = [json.loads(line) for line in f]
print('Total:', len(chunks))
for i in range(min(10, len(chunks))):
    print(f'Chunk {i}:', chunks[i]['text'][:150])
\"''')
    print(result.stdout)
"
```

### Check Clustering Configuration

```bash
# View the config used for a run
cat remote_results/<run_name>/config.json | python -m json.tool

# Key parameters to check:
# clustering.max_depth - Should be > 0 for recursive clustering
# clustering.silhouette_threshold - Lower = more aggressive subdivision (default: 0.3)
# clustering.base_pct - Affects min_cluster_size (default: 0.05)
# clustering.hdbscan_min_samples - HDBSCAN parameter (default: 10)
```

### Inspect Cluster Tree Structure

```python
import json

tree = json.load(open('remote_results/<run_name>/tree.json'))

def print_tree(node, indent=0):
    prefix = "  " * indent
    name = node.get('name', 'UNNAMED')
    size = node.get('size', 0)
    noise = len(node.get('noise_indices', []))
    sil = node.get('silhouette_score', 0.0)

    print(f"{prefix}[{node['cluster_id']}] {name}")
    print(f"{prefix}  Size: {size}, Noise: {noise}, Silhouette: {sil:.3f}")

    for child in node.get('children', []):
        print_tree(child, indent + 1)

print_tree(tree)
```

### Check Chunk-to-Cluster Mapping

```bash
# See how many chunks map to each cluster
python -c "
import json
from collections import Counter

mapping = json.load(open('remote_results/<run_name>/chunk_to_cluster.json'))
cluster_counts = Counter(mapping.values())

print('Chunks per cluster:')
for cluster_id, count in sorted(cluster_counts.items()):
    print(f'  {cluster_id}: {count} chunks')
"
```

## Common Diagnostic Checks

### Calculate Noise Ratio

```bash
python -c "
import json
stats = json.load(open('remote_results/<run_name>/stats.json'))
tree = json.load(open('remote_results/<run_name>/tree.json'))

total = tree['size']
noise = stats['total_noise_points']
ratio = noise / total * 100

print(f'Noise ratio: {ratio:.1f}%')
print(f'Clustered: {total - noise} / {total} points')
"
```

### Check HDBSCAN Parameters Used

```bash
python -c "
import json
config = json.load(open('remote_results/<run_name>/config.json'))

n_points = 142914  # From your run
base_pct = config['clustering']['base_pct']
depth = 0

min_cluster_size = int(n_points * base_pct)
print(f'HDBSCAN min_cluster_size at depth 0: {min_cluster_size}')
print(f'That is {min_cluster_size/n_points*100:.1f}% of total corpus')
"
```

### Count Clusters at Each Depth

```bash
python -c "
import json

def count_by_depth(node, counts):
    depth = node['depth']
    counts[depth] = counts.get(depth, 0) + 1
    for child in node.get('children', []):
        count_by_depth(child, counts)
    return counts

tree = json.load(open('remote_results/<run_name>/tree.json'))
counts = count_by_depth(tree, {})

for depth in sorted(counts.keys()):
    print(f'Depth {depth}: {counts[depth]} clusters')
"
```

### Find Clusters with Most Noise

```bash
python -c "
import json

def find_noisy_clusters(node, results):
    noise_ratio = len(node['noise_indices']) / node['size'] if node['size'] > 0 else 0
    results.append((node['cluster_id'], node.get('name', 'UNNAMED'), noise_ratio, len(node['noise_indices'])))
    for child in node.get('children', []):
        find_noisy_clusters(child, results)
    return results

tree = json.load(open('remote_results/<run_name>/tree.json'))
results = find_noisy_clusters(tree, [])

print('Clusters by noise ratio:')
for cid, name, ratio, count in sorted(results, key=lambda x: -x[2])[:10]:
    print(f'{cid} ({name}): {ratio*100:.1f}% ({count} points)')
"
```

## Running Individual Pipeline Steps

### Run just clustering (skip data prep and embedding):

```bash
# If data and embeddings already exist, cluster_corpus.py will use cache
python cluster_corpus.py configs/clustering_01_tiny.py
```

### Run just naming on existing clusters:

```bash
# Requires OPENAI_API_KEY in .env
python run_naming_only.py configs/clustering_01_tiny.py --download
```

### Re-run clustering with different parameters:

```bash
# 1. Edit the config file or create a new one
# 2. Delete the old cache (since config hash changed)
rm -rf data/clusters_tiny/<cache_key>/

# 3. Run clustering
python cluster_corpus.py configs/<your_config>.py
```

## Accessing Remote Instance

### List running instances:

```bash
broker list
```

### Connect and run commands:

```python
from broker import GPUClient
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
import os

credentials = {}
if key := get_runpod_key(): credentials['runpod'] = key
if key := get_prime_key(): credentials['primeintellect'] = key

client = GPUClient(credentials=credentials, ssh_key_path=os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))
instances = client.list_instances()

# Find corpus-proximity instance
corpus_inst = next((i for i in instances if 'corpus' in i.name.lower()), None)

if corpus_inst:
    bf = BifrostClient(corpus_inst.ssh_connection_string(), os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))

    # Run commands
    result = bf.exec("cd ~/.bifrost/workspace/dev/corpus-proximity && ls -lh data/")
    print(result.stdout)
```

### Download results manually:

```python
# Using the BifrostClient
result = bf.download_files(
    remote_path="~/.bifrost/workspace/dev/corpus-proximity/data/clusters_tiny/<cache_key>/tree.json",
    local_path="./tree.json"
)
```

## Understanding the Clustering Algorithm

### Key concepts:

1. **UMAP**: Reduces embedding dimensions while preserving local structure
   - High-dim embeddings → `umap_n_components` dimensions
   - Uses `umap_metric` (default: cosine)

2. **HDBSCAN**: Density-based clustering that finds noise
   - `min_cluster_size = N * base_pct * (decay ** depth)`
   - `min_samples = hdbscan_min_samples` (default: 10)
   - Returns cluster labels + noise points (label=-1)

3. **Silhouette score**: Measures cluster separation (0-1)
   - High score (>threshold) → stop recursion (well-separated)
   - Low score (<threshold) → continue recursion (poorly separated)

4. **Recursion logic**:
   ```
   if silhouette_score >= threshold:
       stop (clusters are good)
   else:
       recurse into each subcluster
   ```

## Cache Keys

Cache keys are generated from config parameters:
```python
key_parts = [
    embedding_model,
    chunking_strategy,
    chunk_max_tokens,
    max_depth,
    base_pct,
    decay,
    silhouette_threshold
]
cache_key = sha256("|".join(key_parts))[:16]
```

To force re-clustering, change any of these parameters or delete the cache directory.

## File Locations

### Local:
- Configs: `configs/*.py`
- Processed data: `data/processed_tiny/chunks_tiny.jsonl`
- Embeddings: `data/embeddings_arctic_tiny/<cache_key>/embeddings.npy`
- Clusters: `data/clusters_tiny/<cache_key>/tree.json`
- Results: `remote_results/*/`

### Remote (on GPU instance):
- Workspace: `~/.bifrost/workspace/dev/corpus-proximity/`
- Same structure as local under workspace
- Pipeline log: `~/.bifrost/workspace/dev/corpus-proximity/pipeline.log`
