# Recursive Clustering Implementation Plan (Updated)

**Based on decisions:**
- ✅ Use fixed_tokens chunking strategy (512 tokens max, 15% overlap)
- ✅ Use Arctic-Embed-L for embeddings
- ✅ Create new tiny config for testing
- ✅ Use rollout.py's `generate()` for LLM naming
- ✅ Statistical analysis is out of scope (skip Phase 3)

---

## Phase 1: Core Clustering (4-6 hours)

### Milestone 1.1: Recursive clustering function (2 hours)

**Create: `cluster_corpus.py`**

```python
#!/usr/bin/env python3
"""Recursive embedding and clustering for training corpus taxonomy."""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class ClusterNode:
    """A node in the cluster tree."""
    cluster_id: str  # Hierarchical ID like "0", "0.1", "0.1.2"
    depth: int
    parent_id: Optional[str]

    # Data
    embeddings: np.ndarray  # (N, D) subset of corpus
    texts: List[str]        # (N,) corresponding texts
    metadata: List[Dict]    # (N,) chunk metadata

    # Cluster info
    centroid: np.ndarray    # (D,) cluster center
    size: int               # Number of chunks
    silhouette_score: float # Cluster coherence

    # Subclusters (populated during recursion)
    children: List["ClusterNode"]
    noise_indices: List[int]  # HDBSCAN noise points (-1 label)

    # LLM-generated name (populated later)
    name: str = ""


def recursive_cluster(
    embeddings: np.ndarray,
    texts: List[str],
    metadata: List[Dict],
    depth: int = 0,
    max_depth: int = 3,
    base_pct: float = 0.05,
    decay: float = 0.7,
    silhouette_threshold: float = 0.3,
    parent_id: Optional[str] = None,
    cluster_index: int = 0
) -> ClusterNode:
    """
    Recursively cluster corpus using UMAP + HDBSCAN.

    Key insight from Spotify: Only recurse if silhouette < threshold.
    This prevents forcing meaningless subdivisions of tight clusters.

    Args:
        embeddings: (N, D) normalized embeddings for this cluster
        texts: (N,) text chunks
        metadata: (N,) metadata dicts
        depth: Current depth in tree
        max_depth: Stop recursing after this depth
        base_pct: Base percentage for min_cluster_size (5% = 0.05)
        decay: Decay factor per depth (0.7 = 30% reduction)
        silhouette_threshold: Only recurse if score < this (0.3)
        parent_id: Parent cluster ID (None for root)
        cluster_index: Index among siblings

    Returns:
        ClusterNode with children populated if recursion occurred
    """
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.metrics import silhouette_score

    # Generate cluster ID
    cluster_id = str(cluster_index) if parent_id is None else f"{parent_id}.{cluster_index}"

    n_points = len(embeddings)

    # Base case: too deep or too few points
    if depth >= max_depth or n_points < 10:
        return ClusterNode(
            cluster_id=cluster_id,
            depth=depth,
            parent_id=parent_id,
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            centroid=embeddings.mean(axis=0),
            size=n_points,
            silhouette_score=1.0,  # No subdivision = perfect coherence
            children=[],
            noise_indices=[]
        )

    # UMAP dimensionality reduction
    reducer = UMAP(
        n_components=min(50, n_points - 1),  # Can't have more components than samples
        metric='cosine',
        random_state=42
    )
    reduced_embeddings = reducer.fit_transform(embeddings)

    # Adaptive HDBSCAN parameters
    floor = max(10, 50 // (2 ** depth))  # Prevent degenerate clustering
    min_cluster_size = max(floor, int(n_points * base_pct * (decay ** depth)))

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,  # Kept constant
        metric='euclidean',  # On UMAP-reduced space
        cluster_selection_method='eom'
    )
    labels = clusterer.fit_predict(reduced_embeddings)

    # Separate noise points (label = -1)
    noise_mask = labels == -1
    noise_indices = np.where(noise_mask)[0].tolist()

    # Check if clustering succeeded
    unique_labels = set(labels[~noise_mask])
    if len(unique_labels) <= 1:
        # No meaningful clusters found
        return ClusterNode(
            cluster_id=cluster_id,
            depth=depth,
            parent_id=parent_id,
            embeddings=embeddings,
            texts=texts,
            metadata=metadata,
            centroid=embeddings.mean(axis=0),
            size=n_points,
            silhouette_score=1.0,
            children=[],
            noise_indices=noise_indices
        )

    # Compute silhouette score (exclude noise points)
    clean_labels = labels[~noise_mask]
    clean_embeddings = reduced_embeddings[~noise_mask]

    if len(clean_labels) > 1 and len(unique_labels) > 1:
        sil_score = silhouette_score(clean_embeddings, clean_labels)
    else:
        sil_score = 1.0

    # Create current node
    current_node = ClusterNode(
        cluster_id=cluster_id,
        depth=depth,
        parent_id=parent_id,
        embeddings=embeddings,
        texts=texts,
        metadata=metadata,
        centroid=embeddings.mean(axis=0),
        size=n_points,
        silhouette_score=sil_score,
        children=[],
        noise_indices=noise_indices
    )

    # Decide whether to recurse (Spotify insight: only if low coherence)
    if sil_score >= silhouette_threshold:
        # High coherence = don't subdivide
        return current_node

    # Recurse into each subcluster
    for i, label in enumerate(sorted(unique_labels)):
        mask = labels == label

        child_node = recursive_cluster(
            embeddings=embeddings[mask],
            texts=[texts[j] for j in np.where(mask)[0]],
            metadata=[metadata[j] for j in np.where(mask)[0]],
            depth=depth + 1,
            max_depth=max_depth,
            base_pct=base_pct,
            decay=decay,
            silhouette_threshold=silhouette_threshold,
            parent_id=cluster_id,
            cluster_index=i
        )

        current_node.children.append(child_node)

    return current_node
```

**Key implementation details:**
- UMAP reduces 1024-dim Arctic-Embed-L to 50-dim for HDBSCAN
- Adaptive `min_cluster_size = max(floor, n * 0.05 * 0.7^depth)`
- Silhouette gating: only recurse if `silhouette_score < 0.3`
- Noise points tracked but not recursed

---

### Milestone 1.2: Configuration and CLI (1 hour)

**Update: `config.py`**

Add new `ClusteringConfig` dataclass:

```python
@dataclass
class ClusteringConfig:
    """Recursive clustering configuration."""
    # Embedding model (for re-embedding with Arctic-Embed-L)
    embedding_model: str = "Snowflake/snowflake-arctic-embed-l"
    embedding_batch_size: int = 32

    # Chunking strategy (use fixed_tokens with Arctic-Embed-L)
    chunking_strategy: str = "fixed_tokens"
    chunk_max_tokens: int = 512  # Arctic-Embed-L supports 512 tokens
    chunk_overlap_pct: float = 0.15

    # Recursive clustering parameters
    max_depth: int = 3
    base_pct: float = 0.05  # 5% base for min_cluster_size
    decay: float = 0.7      # Decay factor per depth
    silhouette_threshold: float = 0.3  # Only recurse if score < this

    # UMAP parameters
    umap_n_components: int = 50
    umap_metric: str = "cosine"

    # HDBSCAN parameters
    hdbscan_min_samples: int = 10

    # Caching
    cache_dir: Path = _BASE_DIR / "clusters"
    embedding_cache_dir: Path = _BASE_DIR / "embeddings_arctic"

    # LLM naming (use rollout.py)
    naming_model: str = "gpt-4o-mini"
    naming_api_base: str = "https://api.openai.com/v1"
    naming_temperature: float = 0.7
    naming_max_tokens: int = 50  # Short labels only
```

Update `Config` dataclass:

```python
@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)  # NEW
```

**Add: CLI to `cluster_corpus.py`**

```python
def main():
    import sys
    import importlib.util
    import logging

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
    else:
        from config import Config
        config = Config()

    logger.info("Starting recursive clustering pipeline")
    logger.info(f"  Embedding model: {config.clustering.embedding_model}")
    logger.info(f"  Max depth: {config.clustering.max_depth}")
    logger.info(f"  Chunking: {config.clustering.chunking_strategy} (max {config.clustering.chunk_max_tokens} tokens)")

    # TODO: Load corpus, embed, cluster, save

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
```

---

### Milestone 1.3: Testing with tiny corpus (1 hour)

**Create: `configs/clustering_01_tiny.py`**

```python
"""Tiny config for testing recursive clustering (100-200 chunks from pretrain)."""

from pathlib import Path
from config import Config, DataConfig, ClusteringConfig

_BASE_DIR = Path(__file__).parent.parent / "data"

config = Config(
    data=DataConfig(
        num_shards=1,  # Just one shard
        data_dir=_BASE_DIR / "shards",
        processed_dir=_BASE_DIR / "processed_tiny",
        output_file="chunks_tiny.jsonl"
    ),
    clustering=ClusteringConfig(
        # Arctic-Embed-L with token-based chunking
        embedding_model="Snowflake/snowflake-arctic-embed-l",
        chunking_strategy="fixed_tokens",
        chunk_max_tokens=512,
        chunk_overlap_pct=0.15,

        # Clustering params
        max_depth=3,
        base_pct=0.05,
        decay=0.7,
        silhouette_threshold=0.3,

        # Caching
        cache_dir=_BASE_DIR / "clusters_tiny",
        embedding_cache_dir=_BASE_DIR / "embeddings_arctic_tiny"
    )
)
```

**Testing steps:**
1. Sample 100-200 chunks from existing `data/processed/chunks.jsonl`
2. Re-embed with Arctic-Embed-L using `fixed_tokens` chunking
3. Run recursive clustering
4. Save cluster tree to JSON
5. Manually inspect: Do clusters make sense?

---

### Milestone 1.4: Caching and serialization (1-2 hours)

**Add to `cluster_corpus.py`:**

```python
import json
import hashlib
from pathlib import Path

def get_cache_key(data_config, clustering_config) -> str:
    """Generate cache key from configs."""
    # Hash relevant config fields
    key_parts = [
        str(data_config.num_shards),
        str(data_config.output_file),
        clustering_config.embedding_model,
        clustering_config.chunking_strategy,
        str(clustering_config.chunk_max_tokens),
        str(clustering_config.max_depth),
        str(clustering_config.base_pct),
        str(clustering_config.decay),
        str(clustering_config.silhouette_threshold),
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def save_cluster_tree(node: ClusterNode, output_path: Path):
    """Save cluster tree to JSON (recursive)."""
    def node_to_dict(n: ClusterNode) -> dict:
        return {
            "cluster_id": n.cluster_id,
            "depth": n.depth,
            "parent_id": n.parent_id,
            "size": n.size,
            "silhouette_score": n.silhouette_score,
            "name": n.name,
            "num_noise_points": len(n.noise_indices),
            "children": [node_to_dict(child) for child in n.children],
            # Don't save embeddings/texts (too large), just metadata
            "sample_texts": n.texts[:5] if len(n.texts) >= 5 else n.texts
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(node_to_dict(node), f, indent=2)


def load_cluster_tree(input_path: Path) -> dict:
    """Load cluster tree from JSON."""
    with open(input_path) as f:
        return json.load(f)
```

**Cache structure:**
```
data/
  clusters_tiny/
    <cache_key>/
      tree.json          # Cluster hierarchy
      stats.json         # Summary statistics
  embeddings_arctic_tiny/
    <cache_key>/
      embeddings.npy     # Arctic-Embed-L embeddings
      metadata.jsonl     # Chunk metadata
```

---

## Phase 2: LLM Naming (3-4 hours)

### Milestone 2.1: Cluster sampling (1 hour)

**Create: `name_clusters.py`**

```python
#!/usr/bin/env python3
"""LLM-based cluster naming using rollout.py."""

import numpy as np
from typing import List
from cluster_corpus import ClusterNode

def sample_cluster_texts(node: ClusterNode, k: int = 5) -> List[str]:
    """
    Sample representative texts from cluster using centroid-based sampling.

    Strategy:
    1. Find k texts closest to centroid
    2. Ensure diversity (avoid too-similar texts)

    Args:
        node: Cluster node
        k: Number of samples

    Returns:
        List of representative text samples
    """
    if len(node.texts) <= k:
        return node.texts

    # Compute distances to centroid
    distances = np.linalg.norm(node.embeddings - node.centroid, axis=1)

    # Get k closest indices
    top_k_indices = np.argsort(distances)[:k]

    return [node.texts[i] for i in top_k_indices]
```

---

### Milestone 2.2: LLM naming prompt (1 hour)

**Add to `name_clusters.py`:**

```python
import asyncio
from rollout import Endpoint, Rollout, Message, generate

async def generate_cluster_name(
    node: ClusterNode,
    endpoint: Endpoint,
    parent_name: str = None
) -> str:
    """
    Generate cluster name using LLM.

    Args:
        node: Cluster node
        endpoint: LLM endpoint (from rollout.py)
        parent_name: Parent cluster name (for subclusters)

    Returns:
        Generated cluster name (2-5 words)
    """
    # Sample representative texts
    samples = sample_cluster_texts(node, k=5)

    # Build prompt
    if parent_name:
        context = f"Parent cluster: \"{parent_name}\"\n\n"
    else:
        context = ""

    prompt = f"""{context}You are analyzing a cluster of training corpus texts. Based on these examples, provide a concise 2-5 word label describing the common theme.

Examples from this cluster:
"""

    for i, text in enumerate(samples, 1):
        # Truncate long texts
        snippet = text[:200] + "..." if len(text) > 200 else text
        prompt += f"{i}. {snippet}\n"

    prompt += "\nCluster label:"

    # Generate using rollout.py
    rollout = Rollout(messages=[Message(role="user", content=prompt)])
    updated_rollout = await generate(endpoint, rollout)

    # Extract label
    label = updated_rollout.get_last_message_content()
    if label:
        label = label.strip().strip('"').strip("'")

    return label or f"Cluster {node.cluster_id}"
```

---

### Milestone 2.3: Recursive naming (1 hour)

**Add to `name_clusters.py`:**

```python
async def name_cluster_tree(
    root: ClusterNode,
    endpoint: Endpoint
) -> ClusterNode:
    """
    Name all clusters in tree using breadth-first traversal.

    Process level-by-level to ensure parent names are available for subclusters.
    Use asyncio.gather() to parallelize within each level.

    Args:
        root: Root cluster node
        endpoint: LLM endpoint

    Returns:
        Root node with all names populated
    """
    import logging
    logger = logging.getLogger(__name__)

    # Breadth-first traversal by depth
    max_depth = get_max_depth(root)

    for depth in range(max_depth + 1):
        # Collect all nodes at this depth
        nodes_at_depth = []
        collect_nodes_at_depth(root, depth, nodes_at_depth)

        logger.info(f"Naming {len(nodes_at_depth)} clusters at depth {depth}")

        # Name all nodes at this depth in parallel
        tasks = []
        for node in nodes_at_depth:
            # Get parent name if available
            parent_name = None
            if node.parent_id:
                parent_node = find_node_by_id(root, node.parent_id)
                if parent_node:
                    parent_name = parent_node.name

            tasks.append(generate_cluster_name(node, endpoint, parent_name))

        # Await all names for this level
        names = await asyncio.gather(*tasks)

        # Assign names
        for node, name in zip(nodes_at_depth, names):
            node.name = name
            logger.info(f"  {node.cluster_id}: \"{name}\"")

    return root


def get_max_depth(node: ClusterNode) -> int:
    """Get maximum depth in tree."""
    if not node.children:
        return node.depth
    return max(get_max_depth(child) for child in node.children)


def collect_nodes_at_depth(node: ClusterNode, target_depth: int, result: List[ClusterNode]):
    """Collect all nodes at target depth (recursive helper)."""
    if node.depth == target_depth:
        result.append(node)
    for child in node.children:
        collect_nodes_at_depth(child, target_depth, result)


def find_node_by_id(node: ClusterNode, cluster_id: str) -> ClusterNode:
    """Find node by cluster_id (recursive helper)."""
    if node.cluster_id == cluster_id:
        return node
    for child in node.children:
        found = find_node_by_id(child, cluster_id)
        if found:
            return found
    return None
```

---

### Milestone 2.4: Manual validation (1 hour)

Run on tiny corpus and inspect:
- Are cluster names interpretable?
- Do subclusters relate to parent clusters?
- Adjust prompt if names are too generic/specific

---

## Phase 3: Deliverables

**Skip statistical analysis as specified.**

Final deliverables:
1. `cluster_corpus.py` - Core recursive clustering
2. `name_clusters.py` - LLM-based naming
3. `configs/clustering_01_tiny.py` - Tiny test config
4. `configs/clustering_02_full.py` - Full corpus config (future)
5. Cluster tree JSON with names at all levels

---

## Implementation Order

### Week 1: Core infrastructure
1. **Day 1-2**: Milestone 1.1-1.2 (clustering function + config)
2. **Day 3**: Milestone 1.3 (testing with tiny corpus)
3. **Day 4**: Milestone 1.4 (caching)

### Week 2: LLM naming
4. **Day 5**: Milestone 2.1-2.2 (sampling + prompts)
5. **Day 6**: Milestone 2.3 (recursive naming)
6. **Day 7**: Milestone 2.4 (validation)

---

## Next Steps

**To start implementation:**

1. Install dependencies:
   ```bash
   pip install umap-learn hdbscan scikit-learn transformers
   ```

2. Update `config.py` with `ClusteringConfig`

3. Create `cluster_corpus.py` with `recursive_cluster()` function

4. Create `configs/clustering_01_tiny.py`

5. Test on 100-200 chunk sample

**Questions for you before proceeding:**

1. **TrainingCorpus integration**: How should clustering integrate with your existing `TrainingCorpus` API from `search.py`? Should I:
   - Add optional `cluster_labels` field to `TrainingCorpus`?
   - Keep `ClusterNode` separate and reference corpus by indices?
   - Create a `ClusteredCorpus` that wraps `TrainingCorpus` + `ClusterNode`?

2. **Embedding storage**: Arctic-Embed-L embeddings will be separate from your existing `all-MiniLM-L6-v2` embeddings. Should I:
   - Store in separate directory (`embeddings_arctic/`)?
   - Add `embedding_model` field to cache key?
   - Support multiple embeddings per corpus?

Let me know and I'll start implementing!
