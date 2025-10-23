# Corpus Proximity: Chess-Engine-Style Annotations - Full Implementation Plan

## Project Overview

Build a tool that annotates LLM outputs with their nearest clusters in the training corpus, similar to how chess engines annotate board positions with known openings/strategies.

**Core Value Proposition:**
- "Research artifact that looks cool" - visual, interpretable annotations
- Helps understand what training data the model is drawing from
- Works with any LLM + any training corpus (user provides exact corpus)

**Design Principles:**
- Phrase-level granularity (sentence-by-sentence)
- Inference-engine agnostic (file-based I/O)
- GPU-accelerated indexing (via deploy.py)
- Local result syncing (no manual SSH/SCP)

---

## Table of Contents

1. [Architecture](#architecture)
2. [Data Structures](#data-structures)
3. [Core Components](#core-components)
   - [Annotation Engine](#annotation-engine)
   - [GPU Deployment](#gpu-deployment)
4. [CLI Interface](#cli-interface)
5. [File Formats](#file-formats)
6. [Implementation Checklist](#implementation-checklist)
7. [Testing Plan](#testing-plan)
8. [Example Workflows](#example-workflows)

---

## Architecture

### **High-Level Data Flow**

```
┌─────────────────────────────────────────────────────────────┐
│                    User Workflow                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Build Corpus Index (one-time, expensive, GPU)           │
│     └─> corpus-proximity index --corpus ... --deploy-gpu   │
│         ├─ Provisions GPU (RunPod/Vast)                    │
│         ├─ Runs: prepare → embed → cluster → name          │
│         ├─ Syncs results back to local                     │
│         └─ Output: corpus_index/                           │
│                                                             │
│  2. Generate Model Outputs (any inference engine)           │
│     └─> vllm/ollama/api → outputs.jsonl                    │
│                                                             │
│  3. Annotate Outputs (cheap, instant, local)                │
│     └─> corpus-proximity annotate --corpus-index ...       │
│         └─> Output: annotated_outputs.jsonl                │
│                                                             │
│  4. View Annotations (pretty-print)                         │
│     └─> corpus-proximity show --annotated-file ...         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### **System Components**

```
corpus-proximity/
├── annotation.py           # Core annotation engine (NEW)
├── corpus_index.py         # CorpusIndex class (NEW)
├── formatting.py           # Chess-style formatting (NEW)
├── cli.py                  # CLI commands (NEW)
│
├── deploy.py               # GPU deployment (UPDATED)
│   ├─ Add: sync_results()
│   ├─ Add: wait_for_completion()
│   └─ Add: normalize_save_dir()
│
├── cluster_corpus.py       # Clustering (UPDATED)
│   ├─ Add: build_chunk_to_cluster_map()
│   └─ Add: completion markers
│
├── name_clusters.py        # LLM naming (UPDATED)
│   └─ Add: completion markers
│
└── config.py               # Configuration (UPDATED)
    └─ Add: DeploymentConfig
```

---

## Data Structures

### **1. CorpusIndex (Directory Structure)**

```
corpus_index/
├── embeddings.npy           # (N, 1024) normalized Arctic-Embed-L embeddings
├── chunks.jsonl             # One chunk per line: {"text": "...", "idx": 0}
├── metadata.jsonl           # One metadata per line: {"shard_id": 0, "chunk_id": 1}
├── tree.json                # Cluster hierarchy from cluster_corpus.py
├── stats.json               # Cluster statistics
├── config.json              # Index build config (for reproducibility)
└── chunk_to_cluster.json    # Pre-computed chunk_idx → cluster_id mapping
```

**Why this structure?**
- `embeddings.npy` - Fast numpy loading for similarity search
- `chunks.jsonl` - Human-readable, streamable if needed
- `tree.json` - Existing output from cluster_corpus.py
- `chunk_to_cluster.json` - Performance cache (avoid tree traversal)

### **2. ClusterAnnotation**

```python
from dataclasses import dataclass

@dataclass
class ClusterAnnotation:
    """Single annotation linking text span → training corpus cluster."""

    # Text being annotated
    text_span: str              # e.g., "The derivative of x^2 is 2x."

    # Cluster identification
    cluster_id: str             # e.g., "0.2.1"
    cluster_name: str           # e.g., "Calculus Education"
    cluster_depth: int          # Depth in tree (0 = root)

    # Distance metrics
    distance: float             # Cosine distance to nearest chunk
    rank: int                   # 1st, 2nd, 3rd nearest cluster

    # Corpus provenance
    corpus_stage: str | None    # "pretrain" | "midtrain" | "sft" (if available)
    nearest_chunk_idx: int      # Index into chunks.jsonl
    nearest_chunk_text: str     # Actual corpus text (first 200 chars)

    # Optional: logprob-based confidence
    avg_logprob: float | None = None  # Average token logprob (if provided)
```

### **3. AnnotatedOutput**

```python
@dataclass
class AnnotatedOutput:
    """Full annotation result for a single LLM output."""

    # Input
    prompt: str | None          # Original prompt (optional)

    # Output
    text: str                   # Full LLM output

    # Annotations
    annotations: list[ClusterAnnotation]  # Per-phrase or whole-text

    # Metadata
    timestamp: str              # ISO 8601 timestamp
    corpus_index_path: str      # Which corpus index was used
    k: int                      # How many clusters per phrase
    phrase_level: bool          # Was phrase-level splitting used

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "prompt": self.prompt,
            "text": self.text,
            "annotations": [asdict(a) for a in self.annotations],
            "annotation_metadata": {
                "timestamp": self.timestamp,
                "corpus_index_path": self.corpus_index_path,
                "k": self.k,
                "phrase_level": self.phrase_level
            }
        }
```

### **4. DeploymentConfig (Add to config.py)**

```python
@dataclass
class DeploymentConfig:
    """GPU deployment settings."""
    keep_running: bool = False       # Keep GPU after completion
    min_vram: int = 24               # Minimum VRAM in GB
    min_cpu_ram: int = 32            # Minimum CPU RAM in GB
    max_price: float = 1.0           # Max price per hour
    container_disk: int = 50         # Container disk in GB
    volume_disk: int = 0             # Volume disk in GB (0 = none)
    gpu_filter: str | None = None    # GPU type filter (e.g., "RTX")

@dataclass
class Config:
    """Main configuration (updated)."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
    deployment: DeploymentConfig = field(default_factory=DeploymentConfig)  # NEW
```

---

## Core Components

## Annotation Engine

### **Component 1: build_chunk_to_cluster_map()**

**Location:** `cluster_corpus.py` (add at end)

**Purpose:** Pre-compute mapping from chunk index → cluster node for fast lookup.

```python
def build_chunk_to_cluster_map(tree: ClusterNode) -> dict[int, str]:
    """Build mapping from chunk index to deepest cluster ID.

    For each chunk, find the LEAF cluster it belongs to (most specific).

    Args:
        tree: Root of cluster tree

    Returns:
        Dict mapping chunk_idx → cluster_id (leaf cluster)

    Example:
        mapping = build_chunk_to_cluster_map(tree)
        cluster_id = mapping[42]  # "0.2.1"
    """
    mapping = {}

    def traverse(node: ClusterNode):
        # If leaf node, map all chunks to this cluster
        if not node.children:
            for chunk_idx in node.indices:
                mapping[chunk_idx] = node.cluster_id
        else:
            # Recurse into children (they override with more specific)
            for child in node.children:
                traverse(child)

    traverse(tree)
    return mapping
```

**Integration:** At end of `cluster_corpus.py` main():

```python
# After saving tree.json
logger.info("Building chunk-to-cluster mapping...")
chunk_to_cluster = build_chunk_to_cluster_map(root_node)

# Save mapping
mapping_path = cache_dir / "chunk_to_cluster.json"
with open(mapping_path, 'w') as f:
    json.dump(chunk_to_cluster, f)

logger.info(f"Saved chunk-to-cluster mapping: {mapping_path}")
```

---

### **Component 2: CorpusIndex class**

**Location:** `corpus_index.py` (NEW FILE)

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import json
from sentence_transformers import SentenceTransformer

@dataclass
class CorpusIndex:
    """Pre-built index of training corpus."""

    embeddings: np.ndarray              # (N, 1024) normalized embeddings
    chunks: list[str]                   # (N,) text chunks
    metadata: list[dict]                # (N,) chunk metadata
    tree: dict                          # Cluster tree (from tree.json)
    chunk_to_cluster: dict[int, str]    # chunk_idx → cluster_id
    model: SentenceTransformer          # Embedding model (for queries)
    index_path: Path                    # Where this index lives

    @classmethod
    def load(cls, index_path: str | Path) -> 'CorpusIndex':
        """Load pre-built corpus index from disk.

        Args:
            index_path: Path to corpus index directory

        Returns:
            Loaded CorpusIndex

        Example:
            corpus_index = CorpusIndex.load("nanochat_index/")
        """
        index_path = Path(index_path)
        assert index_path.exists(), f"Index not found: {index_path}"

        # Load embeddings
        embeddings = np.load(index_path / "embeddings.npy")

        # Load chunks
        chunks = []
        with open(index_path / "chunks.jsonl") as f:
            for line in f:
                chunks.append(json.loads(line)["text"])

        # Load metadata
        metadata = []
        with open(index_path / "metadata.jsonl") as f:
            for line in f:
                metadata.append(json.loads(line))

        # Load tree
        with open(index_path / "tree.json") as f:
            tree = json.load(f)

        # Load chunk-to-cluster mapping
        with open(index_path / "chunk_to_cluster.json") as f:
            chunk_to_cluster = json.load(f)
            # Convert string keys back to int
            chunk_to_cluster = {int(k): v for k, v in chunk_to_cluster.items()}

        # Load config to get embedding model
        with open(index_path / "config.json") as f:
            config_dict = json.load(f)
            embedding_model = config_dict.get("clustering", {}).get("embedding_model",
                                                                    "Snowflake/snowflake-arctic-embed-l")

        # Load embedding model
        model = SentenceTransformer(embedding_model)

        return cls(
            embeddings=embeddings,
            chunks=chunks,
            metadata=metadata,
            tree=tree,
            chunk_to_cluster=chunk_to_cluster,
            model=model,
            index_path=index_path
        )

    def get_cluster_name(self, cluster_id: str) -> str:
        """Get cluster name from cluster_id by traversing tree."""
        def find_node(node, target_id):
            if node["cluster_id"] == target_id:
                return node
            for child in node.get("children", []):
                result = find_node(child, target_id)
                if result:
                    return result
            return None

        node = find_node(self.tree, cluster_id)
        return node["name"] if node else "Unknown"

    def get_cluster_info(self, cluster_id: str) -> dict:
        """Get full cluster info (name, depth, etc.)."""
        def find_node(node, target_id):
            if node["cluster_id"] == target_id:
                return node
            for child in node.get("children", []):
                result = find_node(child, target_id)
                if result:
                    return result
            return None

        return find_node(self.tree, cluster_id) or {}
```

---

### **Component 3: annotate_text()**

**Location:** `annotation.py` (NEW FILE)

```python
import numpy as np
from typing import List
from datetime import datetime
import spacy

from corpus_index import CorpusIndex
from dataclasses import dataclass, asdict

@dataclass
class ClusterAnnotation:
    # (Definition from Data Structures section above)
    pass

@dataclass
class AnnotatedOutput:
    # (Definition from Data Structures section above)
    pass


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using spaCy.

    Args:
        text: Input text

    Returns:
        List of sentence strings
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]


def compute_distances(query_emb: np.ndarray, corpus_embs: np.ndarray) -> np.ndarray:
    """Compute cosine distances between query and corpus embeddings.

    Args:
        query_emb: (D,) normalized query embedding
        corpus_embs: (N, D) normalized corpus embeddings

    Returns:
        (N,) array of distances
    """
    # cosine_distance = 1 - cosine_similarity = 1 - dot_product
    similarities = corpus_embs @ query_emb
    distances = 1 - similarities
    return distances


def annotate_text(
    corpus_index: CorpusIndex,
    text: str,
    k: int = 3,
    phrase_level: bool = True,
    logprobs: list[float] | None = None
) -> AnnotatedOutput:
    """Annotate text with nearest training corpus clusters.

    Args:
        corpus_index: Pre-built corpus index
        text: LLM output to annotate
        k: Number of nearest clusters per phrase
        phrase_level: If True, split into sentences. If False, annotate entire text.
        logprobs: Optional per-token logprobs (for confidence scoring)

    Returns:
        AnnotatedOutput with annotations

    Example:
        corpus_index = CorpusIndex.load("nanochat_index/")

        text = "The derivative of x^2 is 2x. This follows from the power rule."
        result = annotate_text(corpus_index, text, k=3, phrase_level=True)

        for ann in result.annotations:
            print(f"{ann.text_span} → {ann.cluster_name} (d={ann.distance:.3f})")
    """
    # Step 1: Split text into phrases (if phrase_level=True)
    if phrase_level:
        phrases = split_into_sentences(text)
    else:
        phrases = [text]

    # Step 2: Embed each phrase
    phrase_embeddings = corpus_index.model.encode(
        phrases,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    # Step 3: For each phrase, find k-nearest corpus chunks
    all_annotations = []

    for phrase_idx, (phrase, phrase_emb) in enumerate(zip(phrases, phrase_embeddings)):
        # Compute distances to all corpus chunks
        distances = compute_distances(phrase_emb, corpus_index.embeddings)

        # Get top-k nearest chunks
        top_k_indices = np.argsort(distances)[:k]

        # Create annotations for this phrase
        for rank, chunk_idx in enumerate(top_k_indices, start=1):
            # Map chunk → cluster
            cluster_id = corpus_index.chunk_to_cluster.get(chunk_idx, "unknown")
            cluster_info = corpus_index.get_cluster_info(cluster_id)

            # Extract corpus stage from metadata (if available)
            chunk_metadata = corpus_index.metadata[chunk_idx]
            corpus_stage = chunk_metadata.get("corpus_stage", None)

            # Compute average logprob for this phrase (if provided)
            avg_logprob = None
            if logprobs:
                # TODO: Map phrase to token range and average
                # For now, just skip
                pass

            annotation = ClusterAnnotation(
                text_span=phrase,
                cluster_id=cluster_id,
                cluster_name=cluster_info.get("name", "Unknown"),
                cluster_depth=cluster_info.get("depth", 0),
                distance=float(distances[chunk_idx]),
                rank=rank,
                corpus_stage=corpus_stage,
                nearest_chunk_idx=int(chunk_idx),
                nearest_chunk_text=corpus_index.chunks[chunk_idx][:200],
                avg_logprob=avg_logprob
            )

            all_annotations.append(annotation)

    # Build AnnotatedOutput
    return AnnotatedOutput(
        prompt=None,  # Not provided in this function
        text=text,
        annotations=all_annotations,
        timestamp=datetime.utcnow().isoformat() + "Z",
        corpus_index_path=str(corpus_index.index_path),
        k=k,
        phrase_level=phrase_level
    )
```

---

### **Component 4: format_annotations()**

**Location:** `formatting.py` (NEW FILE)

```python
import numpy as np
from itertools import groupby
from annotation import AnnotatedOutput, ClusterAnnotation


def format_annotations(
    annotated_output: AnnotatedOutput,
    show_chunks: bool = False
) -> str:
    """Format annotations in chess-engine style.

    Args:
        annotated_output: AnnotatedOutput from annotate_text()
        show_chunks: If True, show nearest corpus chunks

    Returns:
        Formatted string with visual hierarchy

    Example:
        result = annotate_text(corpus_index, text)
        print(format_annotations(result))
    """
    text = annotated_output.text
    annotations = annotated_output.annotations

    lines = []
    lines.append("")
    lines.append("Model Output:")
    lines.append(f'"{text}"')
    lines.append("")
    lines.append("📍 Source Analysis:")

    # Group by text_span (in case multiple k for same phrase)
    annotations_sorted = sorted(annotations, key=lambda a: (a.text_span, a.rank))

    for phrase, group in groupby(annotations_sorted, key=lambda a: a.text_span):
        phrase_annotations = list(group)

        lines.append(f'  ├─ "{phrase}"')

        # Show all top-k for this phrase
        for ann in phrase_annotations:
            stage_str = f", {ann.corpus_stage}" if ann.corpus_stage else ""
            lines.append(f"     [{ann.rank}] {ann.cluster_name} (d={ann.distance:.3f}{stage_str})")

        # Optionally show nearest corpus chunk
        if show_chunks and phrase_annotations:
            top_ann = phrase_annotations[0]
            lines.append(f'     Corpus: "{top_ann.nearest_chunk_text}..."')

        lines.append("")

    # Interpretation
    top_annotations = [a for a in annotations if a.rank == 1]
    avg_distance = np.mean([a.distance for a in top_annotations])

    if avg_distance < 0.15:
        confidence = "HIGH"
        interpretation = "Model output closely matches training corpus - likely grounded response."
    elif avg_distance < 0.30:
        confidence = "MEDIUM"
        interpretation = "Model output moderately close to training corpus - possible extrapolation."
    else:
        confidence = "LOW"
        interpretation = "Model output far from training corpus - likely hallucination or novel generation."

    lines.append("🎯 Interpretation:")
    lines.append(f"  Average distance: {avg_distance:.3f} ({confidence} confidence)")
    lines.append(f"  {interpretation}")
    lines.append("")

    return "\n".join(lines)


def format_annotations_compact(annotated_output: AnnotatedOutput) -> str:
    """Compact one-line format for batch viewing.

    Example:
        "text..." → Cluster1 (0.08), Cluster2 (0.12), Cluster3 (0.15)
    """
    top_annotations = [a for a in annotated_output.annotations if a.rank == 1]

    clusters_str = ", ".join([
        f"{a.cluster_name} ({a.distance:.2f})"
        for a in top_annotations[:3]
    ])

    text_preview = annotated_output.text[:60] + "..." if len(annotated_output.text) > 60 else annotated_output.text

    return f'"{text_preview}" → {clusters_str}'
```

---

## GPU Deployment

### **Component 5: Update deploy.py**

**Changes needed:**

1. Add helper functions (from outlier-features pattern)
2. Add completion markers to pipeline scripts
3. Add polling for completion
4. Add result syncing
5. Add DeploymentConfig support

#### **5.1: Add Helper Functions**

**Location:** `deploy.py` (add before main)

```python
from pathlib import Path, PurePosixPath
import time
import hashlib

REMOTE_WORKSPACE_PATH = "~/.bifrost/workspace/examples/corpus-proximity"


def normalize_save_dir(save_dir: Path | str) -> str:
    """Convert save_dir to normalized relative POSIX path for remote.

    Args:
        save_dir: Path object or string from config

    Returns:
        Normalized relative path string (e.g., "results", "foo/bar")
    """
    if isinstance(save_dir, str):
        save_dir = Path(save_dir)

    posix_path = PurePosixPath(save_dir)
    parts = [p for p in posix_path.parts if p not in ('.', '..')]
    normalized = '/'.join(parts) if parts else ''

    assert normalized, f"save_dir normalized to empty string from: {save_dir}"
    assert not normalized.startswith('/'), f"save_dir should be relative, got: {normalized}"

    return normalized


def get_cache_key(config: Config) -> str:
    """Generate cache key from config (matches cluster_corpus.py logic)."""
    key_parts = [
        str(config.data.num_shards),
        str(config.data.output_file),
        config.clustering.embedding_model,
        config.clustering.chunking_strategy,
        str(config.clustering.chunk_max_tokens),
        str(config.clustering.max_depth),
        str(config.clustering.base_pct),
        str(config.clustering.decay),
        str(config.clustering.silhouette_threshold),
    ]
    key_str = "|".join(key_parts)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def wait_for_pipeline_completion(
    bifrost_client: BifrostClient,
    timeout: int = 3600
) -> bool:
    """Poll for pipeline completion with explicit timeout.

    Args:
        bifrost_client: Connected Bifrost client
        timeout: Max wait time in seconds (default 60 min)

    Returns:
        True if completed successfully, False if failed/timeout
    """
    poll_interval = 30
    max_iterations = timeout // poll_interval

    logger.info(f"⏳ Waiting for pipeline completion (timeout: {timeout}s)...")

    for i in range(max_iterations):
        check_cmd = f"""
cd {REMOTE_WORKSPACE_PATH}
test -f .pipeline_complete && echo 'COMPLETE' && exit 0
test -f .pipeline_failed && echo 'FAILED' && exit 0
echo 'RUNNING'
"""
        result = bifrost_client.exec(check_cmd)
        status = result.stdout.strip().split('\n')[-1] if result.stdout else 'UNKNOWN'

        if status == 'COMPLETE':
            logger.info("✅ Pipeline completed")
            return True
        elif status == 'FAILED':
            logger.error("❌ Pipeline failed")
            return False

        elapsed = (i + 1) * poll_interval
        logger.info(f"⏳ Pipeline running... ({elapsed}s / {timeout}s)")
        time.sleep(poll_interval)

    logger.error(f"❌ Timeout after {timeout}s")
    return False


def sync_results(
    bifrost_client: BifrostClient,
    config: Config,
    output_dir: Path
):
    """Sync clustering results from remote to local.

    Args:
        bifrost_client: Bifrost client instance
        config: Configuration object
        output_dir: Local output directory
    """
    logger.info(f"💾 Syncing results from remote...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build remote paths
    remote_save_dir = normalize_save_dir(config.clustering.cache_dir)
    cache_key = get_cache_key(config)
    remote_tree_path = f"{REMOTE_WORKSPACE_PATH}/{remote_save_dir}/{cache_key}/tree.json"

    # Sync main result (tree.json)
    result = bifrost_client.download_files(
        remote_path=remote_tree_path,
        local_path=str(output_dir / "tree.json")
    )

    if not (result and result.success and result.files_copied > 0):
        raise RuntimeError(f"Failed to sync tree.json from {remote_tree_path}")

    logger.info(f"✅ Synced: tree.json")

    # Sync additional files
    for filename in ["stats.json", "config.json", "chunk_to_cluster.json"]:
        remote_path = f"{REMOTE_WORKSPACE_PATH}/{remote_save_dir}/{cache_key}/{filename}"
        local_path = output_dir / filename

        try:
            result = bifrost_client.download_files(
                remote_path=remote_path,
                local_path=str(local_path)
            )
            if result and result.success:
                logger.info(f"✅ Synced: {filename}")
        except Exception as e:
            logger.debug(f"Optional file {filename} not synced: {e}")

    # Sync pipeline log
    try:
        result = bifrost_client.download_files(
            remote_path=f"{REMOTE_WORKSPACE_PATH}/pipeline.log",
            local_path=str(output_dir / "pipeline.log")
        )
        if result and result.success:
            logger.info(f"✅ Synced: pipeline.log")
    except Exception as e:
        logger.debug(f"Log not synced: {e}")

    logger.info(f"✅ Results synced to: {output_dir}")
```

#### **5.2: Add Completion Markers**

**In cluster_corpus.py main():**

```python
# At end of main(), after saving tree
try:
    # ... existing clustering logic ...

    # Touch completion marker
    marker = Path(".clustering_complete")
    marker.touch()
    logger.info(f"✅ Clustering complete, marker: {marker}")
    return 0

except Exception as e:
    # Touch failure marker
    marker = Path(".clustering_failed")
    marker.touch()
    logger.error(f"❌ Clustering failed, marker: {marker}")
    raise
```

**In name_clusters.py main():**

```python
# At end of main(), after updating tree with names
try:
    # ... existing naming logic ...

    # Touch completion marker
    marker = Path(".naming_complete")
    marker.touch()
    logger.info(f"✅ Naming complete, marker: {marker}")
    return 0

except Exception as e:
    # Touch failure marker
    marker = Path(".naming_failed")
    marker.touch()
    logger.error(f"❌ Naming failed, marker: {marker}")
    raise
```

#### **5.3: Create run_full_pipeline.py**

**Location:** `run_full_pipeline.py` (NEW FILE)

```python
#!/usr/bin/env python3
"""Wrapper script to run full pipeline with completion markers."""

import sys
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def run_step(script: str, config_path: str) -> bool:
    """Run a single pipeline step."""
    logger.info(f"Running {script}...")

    cmd = [
        sys.executable,
        f"examples/corpus-proximity/{script}",
        f"examples/corpus-proximity/{config_path}"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"{script} failed with exit code {result.returncode}")
        logger.error(f"STDERR: {result.stderr}")
        return False

    logger.info(f"✅ {script} completed")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_full_pipeline.py <config_path>")
        return 1

    config_path = sys.argv[1]

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("pipeline.log"),
            logging.StreamHandler()
        ]
    )

    logger.info("="*80)
    logger.info("Starting Full Pipeline")
    logger.info("="*80)

    try:
        # Step 1: Prepare data
        if not run_step("prepare_data.py", config_path):
            raise RuntimeError("prepare_data.py failed")

        # Step 2: Embed chunks
        if not run_step("embed_chunks.py", config_path):
            raise RuntimeError("embed_chunks.py failed")

        # Step 3: Cluster corpus
        if not run_step("cluster_corpus.py", config_path):
            raise RuntimeError("cluster_corpus.py failed")

        # Step 4: Name clusters
        if not run_step("name_clusters.py", config_path):
            raise RuntimeError("name_clusters.py failed")

        # Success - touch completion marker
        Path(".pipeline_complete").touch()
        logger.info("="*80)
        logger.info("✅ Pipeline Complete")
        logger.info("="*80)
        return 0

    except Exception as e:
        # Failure - touch failure marker
        Path(".pipeline_failed").touch()
        logger.error("="*80)
        logger.error(f"❌ Pipeline Failed: {e}")
        logger.error("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

#### **5.4: Update deploy.py main()**

Replace sequential execution with tmux + polling + syncing:

```python
def main():
    # ... (existing arg parsing and config loading) ...

    # Add --keep-running flag
    parser.add_argument("--keep-running", action="store_true",
                        help="Keep GPU running after pipeline completes")
    args = parser.parse_args()

    # ... (existing GPU provisioning) ...

    # Deploy code
    bifrost_client = deploy_code(bifrost_client, use_existing=args.use_existing)

    # Start pipeline in tmux (non-blocking)
    logger.info("🔬 Starting pipeline in background...")

    remote_config_path = f"~/.bifrost/workspace/examples/corpus-proximity/{args.config}"

    tmux_cmd = f"""
cd {REMOTE_WORKSPACE_PATH} && \\
tmux new-session -d -s corpus-pipeline \\
'python run_full_pipeline.py {remote_config_path} 2>&1 | tee pipeline.log'
"""
    bifrost_client.exec(tmux_cmd)

    logger.info("✅ Pipeline started in tmux session 'corpus-pipeline'")
    logger.info(f"   Monitor: ssh {bifrost_client.ssh} 'tail -f ~/.bifrost/workspace/examples/corpus-proximity/pipeline.log'")

    # Wait for completion (polling)
    success = wait_for_pipeline_completion(bifrost_client, timeout=3600)

    # Sync results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"remote_results/clustering_{timestamp}")

    if not success:
        logger.warning("⚠️  Pipeline did not complete successfully, syncing logs for debugging...")

    sync_results(bifrost_client, config, output_dir)

    # Cleanup (conditional)
    if not args.keep_running:
        cleanup_instance(gpu_instance.id)
    else:
        logger.info(f"\n🎯 Keeping GPU running")
        logger.info(f"   SSH: {bifrost_client.ssh}")
        logger.info(f"   Terminate: broker terminate {gpu_instance.id}")

    logger.info("\n🎉 Deployment complete!")
    logger.info(f"Results: {output_dir}")
    return 0
```

---

## CLI Interface

### **Command Structure**

```bash
corpus-proximity <command> [options]

Commands:
  index      Build corpus index (expensive, GPU-accelerated)
  annotate   Annotate model outputs (cheap, local)
  show       Pretty-print annotations
```

### **Command 1: index**

```bash
corpus-proximity index \
    --corpus ~/nanochat_training_data/ \
    --output nanochat_index/ \
    --format jsonl \
    --text-field text \
    --deploy-gpu \
    --keep-running
```

**Arguments:**
- `--corpus`: Path to training corpus directory or file
- `--output`: Where to save corpus index
- `--format`: Corpus format (jsonl, parquet, hf_dataset)
- `--text-field`: Field name containing text (for jsonl/parquet)
- `--deploy-gpu`: Auto-deploy to GPU for embedding/clustering
- `--keep-running`: Keep GPU running after indexing (for debugging)

**What it does:**
1. If `--deploy-gpu`: Provisions GPU, deploys code, runs pipeline remotely
2. Else: Runs pipeline locally
3. Outputs CorpusIndex directory at `--output` path

**Implementation:** Wrapper around `deploy.py` or local pipeline execution.

---

### **Command 2: annotate**

```bash
corpus-proximity annotate \
    --corpus-index nanochat_index/ \
    --input outputs.jsonl \
    --output annotated_outputs.jsonl \
    --k 3 \
    --phrase-level
```

**Arguments:**
- `--corpus-index`: Path to pre-built corpus index
- `--input`: JSONL file with model outputs
- `--output`: Where to save annotated outputs
- `--k`: Number of nearest clusters per phrase (default: 3)
- `--phrase-level`: Annotate per-sentence vs whole-text (default: true)

**Input format (outputs.jsonl):**
```jsonl
{"prompt": "What is 2+2?", "output": "2+2 equals 4."}
{"prompt": "Explain calculus", "output": "Calculus is the study of continuous change."}
```

**Output format (annotated_outputs.jsonl):**
```jsonl
{
  "prompt": "What is 2+2?",
  "output": "2+2 equals 4.",
  "annotations": [
    {
      "text_span": "2+2 equals 4.",
      "cluster_name": "Elementary Math",
      "cluster_id": "0.1.2",
      "distance": 0.05,
      "rank": 1,
      "corpus_stage": "pretrain",
      "nearest_chunk_idx": 1234,
      "nearest_chunk_text": "Simple arithmetic: 2+2=4..."
    }
  ],
  "annotation_metadata": {
    "timestamp": "2025-01-22T14:30:00Z",
    "corpus_index_path": "nanochat_index/",
    "k": 3,
    "phrase_level": true
  }
}
```

---

### **Command 3: show**

```bash
corpus-proximity show \
    --annotated-file annotated_outputs.jsonl \
    --index 0 \
    --show-chunks
```

**Arguments:**
- `--annotated-file`: Path to annotated outputs JSONL
- `--index`: Which entry to display (line number, 0-indexed)
- `--show-chunks`: Show nearest corpus chunks (default: false)

**Output:**
```
Model Output:
"The derivative of x^2 is 2x. This follows from the power rule."

📍 Source Analysis:
  ├─ "The derivative of x^2 is 2x."
     [1] Calculus Education (d=0.08, midtrain)
     [2] Mathematics Textbooks (d=0.12, pretrain)
     [3] STEM Tutorials (d=0.15, pretrain)

  ├─ "This follows from the power rule."
     [1] Mathematics Textbooks (d=0.12, pretrain)
     [2] Calculus Education (d=0.14, midtrain)
     [3] Academic Writing (d=0.18, sft)

🎯 Interpretation:
  Average distance: 0.10 (HIGH confidence)
  Model output closely matches training corpus - likely grounded response.
```

---

## File Formats

### **Model Outputs (Input to annotate)**

**Format:** JSONL (one JSON object per line)

**Required fields:**
- `output` (str): LLM output text to annotate

**Optional fields:**
- `prompt` (str): Original prompt
- `model` (str): Model name
- `logprobs` (list[float]): Per-token logprobs
- Any other metadata

**Example:**
```jsonl
{"prompt": "What is 2+2?", "output": "2+2 equals 4.", "model": "llama-3-8b"}
{"prompt": "Explain calculus", "output": "Calculus studies change.", "model": "llama-3-8b"}
```

**How users generate this with vLLM:**
```python
from vllm import LLM
import json

llm = LLM(model="meta-llama/Llama-3-8B")
prompts = ["What is 2+2?", "Explain calculus"]
outputs = llm.generate(prompts)

# Write to JSONL
with open("outputs.jsonl", "w") as f:
    for prompt, output in zip(prompts, outputs):
        f.write(json.dumps({
            "prompt": prompt,
            "output": output.text
        }) + "\n")
```

---

## Implementation Checklist

### **Phase 1: Annotation Engine (4-5 hours)**

- [ ] Create `annotation.py`
  - [ ] `ClusterAnnotation` dataclass
  - [ ] `AnnotatedOutput` dataclass
  - [ ] `split_into_sentences()` helper
  - [ ] `compute_distances()` helper
  - [ ] `annotate_text()` main function

- [ ] Create `corpus_index.py`
  - [ ] `CorpusIndex` class
  - [ ] `CorpusIndex.load()` method
  - [ ] Helper methods: `get_cluster_name()`, `get_cluster_info()`

- [ ] Create `formatting.py`
  - [ ] `format_annotations()` - chess-style formatting
  - [ ] `format_annotations_compact()` - one-line format

- [ ] Update `cluster_corpus.py`
  - [ ] Add `build_chunk_to_cluster_map()` function
  - [ ] Save `chunk_to_cluster.json` at end of main()
  - [ ] Add completion markers (`.clustering_complete`, `.clustering_failed`)

- [ ] Update `name_clusters.py`
  - [ ] Add completion markers (`.naming_complete`, `.naming_failed`)

### **Phase 2: GPU Deployment (3-4 hours)**

- [ ] Update `config.py`
  - [ ] Add `DeploymentConfig` dataclass
  - [ ] Add `deployment` field to `Config`

- [ ] Update `deploy.py`
  - [ ] Add `normalize_save_dir()` helper
  - [ ] Add `get_cache_key()` helper
  - [ ] Add `wait_for_pipeline_completion()` polling function
  - [ ] Add `sync_results()` function
  - [ ] Update main() to use tmux + polling + syncing
  - [ ] Add `--keep-running` flag

- [ ] Create `run_full_pipeline.py`
  - [ ] Wrapper script that runs all pipeline steps
  - [ ] Creates `.pipeline_complete` or `.pipeline_failed` markers
  - [ ] Logs to `pipeline.log`

- [ ] Create `configs/clustering_01_tiny_gpu.py`
  - [ ] GPU-optimized config (batch_size=128)
  - [ ] 1 shard, max_depth=2 for quick testing
  - [ ] Include DeploymentConfig

### **Phase 3: CLI Interface (2-3 hours)**

- [ ] Create `cli.py`
  - [ ] Main CLI entry point with subcommands
  - [ ] `index` command implementation
  - [ ] `annotate` command implementation
  - [ ] `show` command implementation
  - [ ] Corpus mismatch warning (in `index` command)

- [ ] Add to `pyproject.toml`
  - [ ] Console script entry point: `corpus-proximity = cli:main`
  - [ ] Add `spacy` dependency
  - [ ] Add `en_core_web_sm` model dependency

### **Phase 4: Testing (3-4 hours)**

- [ ] Test GPU deployment
  - [ ] Run `python deploy.py --config configs/clustering_01_tiny_gpu.py`
  - [ ] Verify GPU provisioning
  - [ ] Verify pipeline execution (all 4 steps)
  - [ ] Verify result syncing to `remote_results/`
  - [ ] Verify auto-cleanup

- [ ] Test annotation engine
  - [ ] Load test CorpusIndex
  - [ ] Annotate sample texts (single-sentence and multi-sentence)
  - [ ] Verify annotations contain correct cluster names, distances
  - [ ] Verify phrase-level splitting works

- [ ] Test CLI end-to-end
  - [ ] `corpus-proximity index` (local mode)
  - [ ] Generate test outputs.jsonl
  - [ ] `corpus-proximity annotate`
  - [ ] `corpus-proximity show`

- [ ] Test with vLLM outputs
  - [ ] Generate outputs from vLLM
  - [ ] Annotate with corpus-proximity
  - [ ] Verify annotations make sense

### **Phase 5: Documentation (2 hours)**

- [ ] Update README.md
  - [ ] Add annotation examples
  - [ ] Add CLI usage examples
  - [ ] Add vLLM integration example

- [ ] Update DEPLOY.md
  - [ ] Document new deployment flow (tmux + syncing)
  - [ ] Document `--keep-running` flag
  - [ ] Add troubleshooting section

- [ ] Create ANNOTATION_GUIDE.md
  - [ ] Explain what annotations mean
  - [ ] Interpretation guidelines (distance thresholds)
  - [ ] Common pitfalls (corpus mismatch, etc.)

---

## Testing Plan

### **Test 1: GPU Deployment (Tiny Config)**

**Goal:** Verify GPU pipeline works end-to-end with result syncing.

**Steps:**
```bash
# Run GPU deployment
python deploy.py --config configs/clustering_01_tiny_gpu.py

# Expected results:
# - GPU provisioned
# - Code deployed
# - Pipeline runs (prepare → embed → cluster → name)
# - Results synced to remote_results/clustering_<timestamp>/
#   - tree.json
#   - stats.json
#   - config.json
#   - chunk_to_cluster.json
#   - pipeline.log
# - GPU terminated (auto-cleanup)
```

**Validation:**
- [ ] All files present in `remote_results/`
- [ ] tree.json contains cluster names
- [ ] chunk_to_cluster.json maps all chunk indices
- [ ] No GPU instances left running (check `broker list`)

---

### **Test 2: Annotation Engine (Local)**

**Goal:** Verify annotation logic works with pre-built index.

**Steps:**
```python
from corpus_index import CorpusIndex
from annotation import annotate_text
from formatting import format_annotations

# Load test index (from Test 1 results)
corpus_index = CorpusIndex.load("remote_results/clustering_<timestamp>/")

# Test single-sentence
text1 = "The derivative of x^2 is 2x."
result1 = annotate_text(corpus_index, text1, k=3, phrase_level=False)
print(format_annotations(result1))

# Test multi-sentence
text2 = "The derivative of x^2 is 2x. This follows from the power rule."
result2 = annotate_text(corpus_index, text2, k=3, phrase_level=True)
print(format_annotations(result2))
```

**Validation:**
- [ ] Annotations contain cluster names (not "Unknown")
- [ ] Distances are reasonable (0.0 - 1.0)
- [ ] Phrase-level splits into 2 sentences
- [ ] Each sentence has k=3 annotations
- [ ] Pretty-print formatting looks good

---

### **Test 3: CLI Workflow**

**Goal:** Verify full CLI workflow (index → annotate → show).

**Steps:**
```bash
# Step 1: Create test outputs
cat > test_outputs.jsonl <<EOF
{"prompt": "What is 2+2?", "output": "2+2 equals 4."}
{"prompt": "Explain calculus", "output": "Calculus is the study of continuous change."}
EOF

# Step 2: Annotate
corpus-proximity annotate \
    --corpus-index remote_results/clustering_<timestamp>/ \
    --input test_outputs.jsonl \
    --output test_annotated.jsonl \
    --k 3

# Step 3: Show
corpus-proximity show \
    --annotated-file test_annotated.jsonl \
    --index 0
```

**Validation:**
- [ ] test_annotated.jsonl created
- [ ] Contains 2 entries (one per input line)
- [ ] Each entry has `annotations` field
- [ ] `show` command displays pretty-formatted output

---

### **Test 4: vLLM Integration**

**Goal:** Verify works with real vLLM outputs.

**Steps:**
```python
# Generate outputs with vLLM
from vllm import LLM
import json

llm = LLM(model="meta-llama/Llama-3.2-1B")
prompts = [
    "What is the derivative of x^2?",
    "Explain the Pythagorean theorem.",
    "What is the capital of France?"
]
outputs = llm.generate(prompts)

# Save to JSONL
with open("vllm_outputs.jsonl", "w") as f:
    for prompt, output in zip(prompts, outputs):
        f.write(json.dumps({
            "prompt": prompt,
            "output": output.outputs[0].text
        }) + "\n")
```

```bash
# Annotate vLLM outputs
corpus-proximity annotate \
    --corpus-index remote_results/clustering_<timestamp>/ \
    --input vllm_outputs.jsonl \
    --output vllm_annotated.jsonl

# View results
for i in 0 1 2; do
    corpus-proximity show --annotated-file vllm_annotated.jsonl --index $i
    echo ""
done
```

**Validation:**
- [ ] Annotations make semantic sense (math → math clusters, etc.)
- [ ] Distance values reasonable (probably 0.1-0.4 range)
- [ ] No crashes or errors

---

## Example Workflows

### **Workflow 1: Build Index for nanochat Training Corpus**

```bash
# Assume you have nanochat training data at:
# ~/nanochat_data/
#   ├── pretrain/
#   │   ├── shard_0.jsonl
#   │   ├── shard_1.jsonl
#   │   └── ...
#   ├── midtrain/
#   └── sft/

# Build corpus index (deploys to GPU)
corpus-proximity index \
    --corpus ~/nanochat_data/ \
    --output nanochat_index/ \
    --format jsonl \
    --text-field text \
    --deploy-gpu

# Results synced to: remote_results/clustering_<timestamp>/
# Copy to standard location:
cp -r remote_results/clustering_<timestamp>/ nanochat_index/
```

---

### **Workflow 2: Annotate Llama-3 Outputs**

```python
# Generate outputs with vLLM
from vllm import LLM
import json

llm = LLM(model="meta-llama/Llama-3-8B")

prompts = [
    "What is the capital of France?",
    "Explain quantum mechanics.",
    "Write a haiku about coding."
]

outputs = llm.generate(prompts)

# Save to JSONL
with open("llama3_outputs.jsonl", "w") as f:
    for prompt, output in zip(prompts, outputs):
        f.write(json.dumps({
            "prompt": prompt,
            "output": output.outputs[0].text
        }) + "\n")
```

```bash
# Annotate
corpus-proximity annotate \
    --corpus-index nanochat_index/ \
    --input llama3_outputs.jsonl \
    --output llama3_annotated.jsonl \
    --k 5 \
    --phrase-level

# View results
corpus-proximity show \
    --annotated-file llama3_annotated.jsonl \
    --index 0 \
    --show-chunks
```

---

### **Workflow 3: Compare Model Outputs**

```bash
# Annotate outputs from different models
corpus-proximity annotate \
    --corpus-index nanochat_index/ \
    --input llama3_outputs.jsonl \
    --output llama3_annotated.jsonl

corpus-proximity annotate \
    --corpus-index nanochat_index/ \
    --input gpt4_outputs.jsonl \
    --output gpt4_annotated.jsonl

# Compare: Which model stays closer to training data?
python -c "
import json
import numpy as np

def avg_distance(file_path):
    distances = []
    with open(file_path) as f:
        for line in f:
            entry = json.loads(line)
            for ann in entry['annotations']:
                if ann['rank'] == 1:
                    distances.append(ann['distance'])
    return np.mean(distances)

print(f'Llama-3: {avg_distance(\"llama3_annotated.jsonl\"):.3f}')
print(f'GPT-4:   {avg_distance(\"gpt4_annotated.jsonl\"):.3f}')
"
```

---

## Important Warnings

### **⚠️  CRITICAL: Corpus Mismatch Warning**

**Display this prominently in `corpus-proximity index` command:**

```
================================================================================
⚠️  CORPUS REQUIREMENT WARNING
================================================================================

This tool requires the EXACT training corpus used for your model.

If you provide an approximate or different corpus, results are MEANINGLESS.

Examples of what NOT to do:
  ❌ "I used FineWeb-Edu but I'm not sure which version"
  ❌ "I used FineWeb-Edu + some custom data I forgot about"
  ❌ "I'm using GPT-4's assumed training data"

What you MUST do:
  ✅ Provide the exact JSONL files used during training
  ✅ Include ALL stages (pretrain, midtrain, SFT)
  ✅ Use the same chunking strategy as training (if known)

If you can't provide the exact corpus, this tool cannot help you.

Do you have the EXACT training corpus? (y/n):
```

**Implementation:**
```python
# In cli.py, index command
def validate_corpus_warning():
    print("="*80)
    print("⚠️  CORPUS REQUIREMENT WARNING")
    print("="*80)
    print("This tool requires the EXACT training corpus.")
    print("If you provide an approximate corpus, results are MEANINGLESS.")
    print()
    print("Do you have the EXACT training corpus? (y/n): ", end="")

    response = input().strip().lower()
    if response != 'y':
        print("Aborting. Please obtain exact training corpus first.")
        sys.exit(1)
```

---

## Dependencies

### **Update pyproject.toml**

```toml
[project.optional-dependencies]
corpus-proximity = [
    # Existing dependencies
    "sentence-transformers>=3.0.0",
    "datasets>=2.14.0",
    "umap-learn>=0.5.0",
    "hdbscan>=0.8.0",
    "numpy>=1.24.0",
    "scikit-learn>=1.3.0",

    # New for annotation
    "spacy>=3.7.0",
]

[project.scripts]
corpus-proximity = "examples.corpus_proximity.cli:main"
```

**Install spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

---

## Timeline Estimate

**Total: 15-20 hours for full implementation**

| Phase | Tasks | Time |
|-------|-------|------|
| Phase 1 | Annotation Engine | 4-5 hours |
| Phase 2 | GPU Deployment | 3-4 hours |
| Phase 3 | CLI Interface | 2-3 hours |
| Phase 4 | Testing | 3-4 hours |
| Phase 5 | Documentation | 2 hours |

**Critical Path:**
1. GPU deployment (needed to generate test index)
2. Annotation engine (core functionality)
3. CLI (user interface)
4. Testing & docs

**Parallelizable:**
- GPU deployment and annotation engine can be built independently
- Documentation can be written while testing

---

## Success Criteria

**The project is complete when:**

1. ✅ Can run `corpus-proximity index --deploy-gpu` and get back CorpusIndex
2. ✅ Results auto-sync to `remote_results/` (no manual SCP)
3. ✅ Can load CorpusIndex and annotate text via `annotate_text()`
4. ✅ Annotations show cluster name, distance, corpus stage
5. ✅ Chess-engine-style formatting looks good
6. ✅ CLI works: `index → annotate → show`
7. ✅ Works with vLLM-generated outputs
8. ✅ Warns user about corpus mismatch
9. ✅ Documentation covers all workflows
10. ✅ Tests pass for GPU deployment, annotation, CLI

---

## Open Questions for Implementation Team

1. **Sentence splitting library:**
   - Use spaCy (accurate, 500MB download) or NLTK (lighter, less accurate)?
   - **Recommendation:** spaCy for v1

2. **Distance metric:**
   - Cosine only or support euclidean/manhattan?
   - **Recommendation:** Cosine only for v1

3. **Cluster selection:**
   - Map chunk to leaf cluster (most specific) or allow parent clusters?
   - **Recommendation:** Leaf cluster (most specific)

4. **Noise points handling:**
   - Chunks with no cluster assignment (HDBSCAN noise)?
   - **Recommendation:** Map to parent cluster or label "Unclustered"

5. **Logprobs integration:**
   - Defer to v2 or implement basic version now?
   - **Recommendation:** Defer to v2 (not critical)

6. **Multi-stage corpora:**
   - Support separate pretrain/midtrain/sft directories?
   - **Recommendation:** Yes, add `--corpus-stage` flag to `index`

---

## Notes for Implementation Team

- **Follow existing patterns:** Look at `outlier-features/deploy.py` for deployment patterns
- **Type safety:** Use type hints throughout (mypy clean)
- **Logging:** Use structured logging with logger.info/debug/error
- **Error handling:** Fail loudly with clear error messages
- **Caching:** Cache expensive operations (embeddings, chunk_to_cluster map)
- **Documentation:** Docstrings for all public functions (Google style)

**Code style reference:** Follow patterns in existing codebase (Tiger Style: functions <70 lines, explicit over implicit).
