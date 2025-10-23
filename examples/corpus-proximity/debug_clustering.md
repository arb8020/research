# Clustering Debug Report

## Issue Summary

The recursive clustering pipeline completed but produced only 1 cluster for 142,914 chunks, with 48.7% noise points. This is a failure case - the algorithm should have discovered hierarchical structure in the corpus.

## Results Data

### Pipeline Status
- **prepare_data.py**: ✅ Complete
- **embed_chunks.py**: ✅ Complete
- **cluster_corpus.py**: ✅ Complete (but produced poor results)
- **name_clusters.py**: ❌ Failed (exit code 1)

### Clustering Output

```json
{
  "total_clusters": 1,
  "leaf_clusters": 1,
  "clusters_by_depth": {
    "0": 1
  },
  "total_noise_points": 69553,
  "max_depth": 0
}
```

**Cluster Tree:**
- **Cluster ID**: "0"
- **Size**: 142,914 chunks
- **Noise points**: 69,553 (48.7%)
- **Silhouette score**: 0.21
- **Depth**: 0
- **Children**: 0
- **Name**: "" (empty - why name_clusters.py failed)

### Configuration Used

```python
# Clustering parameters from config
embedding_model = "Snowflake/snowflake-arctic-embed-l"
chunking_strategy = "fixed_tokens"
chunk_max_tokens = 512
max_depth = 3
base_pct = 0.03
decay = 0.7
silhouette_threshold = 0.3
umap_n_components = 50
umap_metric = "cosine"
hdbscan_min_samples = 10
```

## Code Logic Analysis

### The Decision Flow

The clustering algorithm follows this logic in `cluster_corpus.py:recursive_cluster()`:

```python
# Step 1: UMAP dimensionality reduction (line 177-187)
n_components = min(50, n_points - 1)
reducer = UMAP(n_components=n_components, metric="cosine")
reduced_embeddings = reducer.fit_transform(embeddings)

# Step 2: Calculate HDBSCAN parameters (line 190-191)
floor = max(5, 10 // (2 ** depth))  # depth=0 → floor=10
min_cluster_size = max(floor, int(n_points * base_pct * (decay ** depth)))

# Step 3: Run HDBSCAN (line 195-201)
clusterer = HDBSCAN(
    min_cluster_size=min_cluster_size,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(reduced_embeddings)

# Step 4: Check if clustering succeeded (line 208-226)
unique_labels = set(labels[~noise_mask])
n_clusters = len(unique_labels)

if n_clusters <= 1:
    # ❌ EARLY RETURN - No recursion happens
    return ClusterNode(
        cluster_id=cluster_id,
        silhouette_score=1.0,  # Overridden to 1.0
        children=[],
        noise_indices=noise_indices
    )

# Step 5: Compute silhouette score (line 228-237)
# Only reached if n_clusters > 1

# Step 6: Decide recursion based on silhouette (line 252-258)
# Only reached if n_clusters > 1
if sil_score < silhouette_threshold:
    # Don't recurse - poor separation
    return current_node
```

### What Actually Happened

**At depth=0 with n_points=142,914:**

```python
# HDBSCAN parameters
floor = max(5, 10 // 1) = 10
min_cluster_size = max(10, int(142914 * 0.03 * 1.0)) = max(10, 4287) = 4287
min_samples = 10

# Result
n_clusters = 1  # HDBSCAN only found 1 valid cluster
noise_points = 69,553  # 48.7% classified as noise

# Code path taken
if n_clusters <= 1:  # TRUE - takes early return
    return leaf_node  # ❌ Stops here, never recurses
```

### The Problem

**The algorithm hit a catch-22:**

1. **HDBSCAN Parameters Too Strict**
   - `min_cluster_size = 4,287` requires clusters with 3%+ of total points
   - Many smaller natural clusters exist but are rejected as "noise"
   - Only 1 cluster large enough to meet the threshold

2. **Early Return Blocks Recursion**
   - Line 213: `if n_clusters <= 1:` returns immediately
   - Never reaches the silhouette-based recursion logic (lines 252-260)
   - **Silhouette score 0.21 suggests cluster should be split, but it never gets checked**

3. **High Noise Rate Indicates Problem**
   - 48.7% noise is abnormally high (normal: 10-20%)
   - Suggests HDBSCAN parameters don't match the data's natural granularity

4. **Silhouette Score Contradiction**
   - Tree shows `silhouette_score: 0.21` (poor internal coherence)
   - This is **below the 0.3 threshold** that should trigger recursion
   - But code never uses it because `n_clusters <= 1` returns early

### Why This Is Wrong

The silhouette score of 0.21 indicates:
- The single cluster has poor internal coherence
- Points within it are not similar to each other
- The cluster likely contains multiple sub-structures

**Expected behavior:** Try alternative splitting strategies or adjust parameters

**Actual behavior:** Give up and return a single cluster

## Root Cause

**min_cluster_size is too large relative to the corpus structure.**

```
min_cluster_size = 4,287 points (3.0% of corpus)
                    ↓
HDBSCAN rejects smaller natural clusters as noise
                    ↓
Only 1 cluster remains (73,361 points)
Noise: 69,553 points (48.7%)
                    ↓
n_clusters = 1 triggers early return
                    ↓
No recursion, no hierarchical structure
```

## Potential Solutions

### 1. Lower `base_pct` (Easiest)
```python
base_pct = 0.01  # or 0.005 instead of 0.03
# min_cluster_size at depth 0: 1,429 points (1%) or 714 points (0.5%)
```

### 2. Adjust `min_samples`
```python
hdbscan_min_samples = 5  # instead of 10
# More permissive clustering, allows smaller dense regions
```

### 3. Modify Early Return Logic (Code Change)
```python
# Current (line 213-226):
if n_clusters <= 1:
    return leaf_node

# Proposed:
if n_clusters <= 1:
    # If silhouette is low, try K-means as fallback
    if compute_silhouette(embeddings) < silhouette_threshold:
        # Try K-means with k=2,3,4 and pick best silhouette
        ...
    else:
        return leaf_node
```

### 4. Add Adaptive Parameter Adjustment
```python
# If HDBSCAN finds 0-1 clusters and noise > 30%, retry with:
if n_clusters <= 1 and len(noise_indices) / n_points > 0.3:
    # Reduce min_cluster_size by 50% and retry
    min_cluster_size = max(floor, min_cluster_size // 2)
    # Re-run HDBSCAN
```

## Recommended Next Steps

1. **Re-run with `base_pct = 0.01`** - Simplest fix, likely to produce better results
2. **Monitor noise percentage** - Should be <20% ideally
3. **Check silhouette scores at each depth** - Should improve as you go deeper
4. **Validate cluster sizes** - Should have reasonable distribution, not 1 giant cluster
