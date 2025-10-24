# Clustering Configuration Fixes

## Problem Summary

The recursive clustering was producing only a single root cluster with ~70k noise points instead of a hierarchical taxonomy. This was caused by several parameter and design issues.

## Root Causes Identified

### 1. **HDBSCAN min_cluster_size Too Large** (CRITICAL)
- **Old value:** `base_pct = 0.05` (5%)
- **Problem:** With 142k chunks, this required 7,100+ points to form a cluster
- **Result:** HDBSCAN labeled everything as noise
- **New value:** `base_pct = 0.002` (0.2%) → ~284 points minimum
- **Rationale:** Text data has natural diversity; need smaller clusters for meaningful topics

### 2. **Chunk Overlap Created Artificial Similarity**
- **Old value:** `chunk_overlap_pct = 0.15` (15%)
- **Problem:** Consecutive chunks shared content, creating false similarity
- **Result:** Chunks clustered by document position, not semantic topic
- **New value:** `chunk_overlap_pct = 0.0` (no overlap)
- **Rationale:** For taxonomy/clustering, each chunk should be a distinct semantic unit

### 3. **Chunk Size Not Optimal for Taxonomy**
- **Old value:** `chunk_max_tokens = 512`
- **Problem:** Large chunks mix multiple subtopics (e.g., "derivatives AND integrals")
- **Result:** Less granular taxonomy, harder to get fine-grained clusters
- **New value:** `chunk_max_tokens = 256`
- **Rationale:**
  - Focused single-topic chunks
  - Better granularity for chess-engine-style annotation
  - Still enough context for coherent embeddings
  - Matches typical LLM output length (200-500 tokens)

### 4. **Recursion Parameters Too Conservative**
- **Old values:** `max_depth = 3`, `silhouette_threshold = 0.3`
- **Problem:** Stopped recursing too early, missing fine-grained structure
- **New values:** `max_depth = 4`, `silhouette_threshold = 0.25`
- **Rationale:** Allow deeper taxonomy for better annotation granularity

## Changes Made

### configs/clustering/01_tiny.py
```python
chunk_max_tokens=256,        # was 512
chunk_overlap_pct=0.0,       # was 0.15
max_depth=4,                 # was 3
base_pct=0.005,              # was 0.01 (0.5% for tiny corpus)
silhouette_threshold=0.25,   # was 0.3
```

### configs/clustering/02_full.py
```python
chunk_max_tokens=256,        # was 512
chunk_overlap_pct=0.0,       # was 0.15
max_depth=4,                 # was 3
base_pct=0.002,              # was 0.05 (CRITICAL FIX: 0.2% instead of 5%)
silhouette_threshold=0.25,   # was 0.3
```

## Expected Results After Fix

With these parameters, you should see:

### Depth 0 (Root Level)
- **5-20 broad clusters** (e.g., "Mathematics", "Science", "History", "Programming")
- **Noise ratio:** <30%
- **min_cluster_size:** ~284 points (0.2% of 142k)

### Depth 1 (Subcategories)
- **3-10 subclusters per parent** (e.g., "Mathematics" → "Algebra", "Calculus", "Geometry")
- **min_cluster_size:** ~199 points (0.2% × 0.7 decay)

### Depth 2 (Fine-Grained)
- **2-8 subclusters** (e.g., "Calculus" → "Derivatives", "Integrals", "Limits")
- **min_cluster_size:** ~139 points

### Depth 3 (Very Fine-Grained)
- **Leaf clusters** for specific topics (e.g., "Derivatives" → "Chain Rule", "Product Rule")
- **min_cluster_size:** ~97 points

## Next Steps

1. **Clear old cache** (configs changed, need to re-run):
   ```bash
   rm -rf data/clusters_tiny/*
   rm -rf data/clusters_full/*
   rm -rf data/embeddings_arctic_tiny/*
   rm -rf data/embeddings_arctic_full/*
   ```

2. **Re-run data preparation** (to apply new chunk size):
   ```bash
   python prepare_data.py configs/clustering/01_tiny.py
   ```

3. **Run clustering with fixed config**:
   ```bash
   python cluster_corpus.py configs/clustering/01_tiny.py
   ```

4. **Verify results**:
   ```bash
   # Check stats
   cat data/clusters_tiny/*/stats.json | python -m json.tool

   # Should see:
   # - total_clusters: 10-50 (not 1!)
   # - max_depth: 2-4 (not 0!)
   # - noise_ratio: <30% (not 99%!)
   ```

5. **Inspect cluster tree**:
   ```bash
   python name_clusters.py configs/clustering/01_tiny.py --tree
   ```

## Understanding the Algorithm

### What Spotify Recommends (and what we're doing):

1. **UMAP** → Reduce 1024-D embeddings to 50-D
2. **HDBSCAN** → Find density-based clusters
3. **Silhouette score** → Measure cluster separation
4. **If silhouette < threshold** → Recurse into each subcluster
5. **Recursion** → Go back to step 1 with **original high-D embeddings** for that subcluster

### Key Parameters:

- **base_pct** → Controls minimum cluster size (lower = more sensitive)
- **decay** → Shrinks min_cluster_size at deeper levels (0.7 = 30% reduction)
- **silhouette_threshold** → When to stop recursing (lower = more recursion)
- **max_depth** → Hard limit on recursion depth

### Why These Values?

**For 142k chunks (full corpus):**
- `base_pct = 0.002` (0.2%) → min 284 points at depth 0
- With `decay = 0.7`:
  - Depth 1: 199 points (0.14%)
  - Depth 2: 139 points (0.098%)
  - Depth 3: 97 points (0.069%)
  - Depth 4: 68 points (0.048%)

This allows progressively finer-grained clusters while maintaining statistical significance.

## Additional Notes

### Why No Overlap for Clustering?

- **Overlap is for retrieval (RAG):** Don't want to split relevant passages
- **Clustering needs distinct units:** Each chunk = one topic, not "80% overlap with neighbor"
- **Artificial similarity:** Overlapping chunks have nearly identical embeddings
- **False grouping:** Clusters form by document position, not semantic meaning

### Why 256 Tokens?

- **Coherent topics:** 1-2 paragraphs, enough context
- **Focused:** Less likely to mix subtopics
- **Matches use case:** LLM outputs are typically 200-500 tokens
- **Granular annotation:** "This is 73% similar to cluster 'Derivatives'" (specific topic)
- **vs 512 tokens:** Would give "73% similar to cluster 'Calculus'" (too broad)

### Double-Chunking Issue (Optional Fix)

Currently, the pipeline chunks **twice**:
1. `prepare_data.py` → Splits into paragraphs
2. `cluster_corpus.py` → Re-chunks paragraphs with fixed_tokens

**This is redundant.** Consider:
- Option A: Keep current pipeline (works, just inefficient)
- Option B: Chunk once in `prepare_data.py` with fixed_tokens, remove re-chunking in `cluster_corpus.py`

Option B is cleaner but requires modifying two files. Current fix only changed configs.

## Testing Checklist

- [ ] Old cache cleared
- [ ] Data re-prepared with new chunk size (256 tokens, no overlap)
- [ ] Clustering runs without errors
- [ ] Stats show multiple clusters (not just 1)
- [ ] Max depth > 0 (ideally 2-4)
- [ ] Noise ratio < 30%
- [ ] Cluster names are coherent topics (after running name_clusters.py)
- [ ] Leaf clusters have 50-500 chunks each (not too small, not too large)

## References

- **Spotify article:** https://engineering.atspotify.com/2023/12/recursive-embedding-and-clustering
- **HDBSCAN docs:** https://hdbscan.readthedocs.io/
- **UMAP docs:** https://umap-learn.readthedocs.io/
- **Arctic-Embed-L:** https://huggingface.co/Snowflake/snowflake-arctic-embed-l
