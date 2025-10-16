# Recursive Clustering Plan
**Goal:** Implement Spotify-style recursive embedding and clustering to create hierarchical taxonomy of training corpus

## Background

Based on [Spotify's Engineering Blog](https://engineering.atspotify.com/2023/12/recursive-embedding-and-clustering), recursive clustering "zooms in" on each cluster by re-embedding and re-clustering subsets of data. This discovers hidden structure that global embeddings miss.

**Why not hierarchical agglomerative?**
- Agglomerative uses fixed global similarity metric, builds bottom-up
- Recursive re-embeds each cluster independently, adapts per-cluster
- Reveals subcategories invisible in original embedding space

**Example structure:**
```
All Training Data
├─ Math Content
│   ├─ GSM8K word problems (money/shopping)
│   ├─ GSM8K word problems (time/distance)
│   ├─ Formal proofs
│   └─ Statistics
├─ Code
│   ├─ Python tutorials
│   ├─ Algorithm implementations
│   └─ API documentation
└─ Conversational
    ├─ Chat responses
    ├─ Q&A format
    └─ Instructions
```

## Architecture

### New Files to Create

1. **`cluster_corpus.py`** - Core recursive clustering implementation
2. **`name_clusters.py`** - LLM-based cluster naming
3. **`annotate_with_clusters.py`** - Tag model outputs with cluster labels
4. **`visualize_clusters.py`** - (Optional) Dendrogram/tree visualization
5. **`configs/clustering_01_tiny.py`** - Tiny config for testing
6. **`configs/clustering_02_full.py`** - Full corpus config

### Dependencies to Add
```python
# Add to pyproject.toml or requirements.txt:
umap-learn = "^0.5.5"           # Dimensionality reduction
hdbscan = "^0.8.33"             # Density-based clustering
scikit-learn = "^1.3.0"         # Silhouette score computation
plotly = "^5.18.0"              # Interactive visualizations
streamlit = "^1.28.0"           # (Optional) Interactive dashboard
```

## Implementation Plan

### Phase 1: Core Clustering (4-6 hours)

**Milestone 1.1: Recursive clustering function (2 hours)**
- [ ] Create `cluster_corpus.py`
- [ ] Implement `recursive_cluster()` function:
  ```python
  def recursive_cluster(
      embeddings: np.ndarray,
      texts: List[str],
      metadata: List[Dict],
      depth: int = 0,
      max_depth: int = 3,
      base_pct: float = 0.05,
      decay: float = 0.7,
      silhouette_threshold: float = 0.3
  ) -> ClusterTree
  ```
- [ ] Use UMAP for dimensionality reduction (Arctic-Embed-L is 1024-dim)
- [ ] Use HDBSCAN with adaptive parameters: `min_cluster_size = max(floor, n * base_pct * decay^depth)`
- [ ] Compute silhouette score after clustering
- [ ] Only recurse if silhouette < threshold (low coherence = needs subdivision)
- [ ] Track noise points separately (HDBSCAN label = -1)
- [ ] Return tree structure with cluster IDs, texts, embeddings, noise points at each level

**Milestone 1.2: Configuration and CLI (1 hour)**
- [ ] Update `config.py` with clustering configs:
  ```python
  @dataclass
  class ClusteringConfig:
      # Embedding model (for re-embedding with Arctic-Embed-L)
      embedding_model: str = "Snowflake/snowflake-arctic-embed-l"
      embedding_batch_size: int = 32

      # Chunking strategy
      chunking_strategy: str = "fixed_tokens"  # Use token-aware chunking
      chunk_max_tokens: int = 512  # Arctic-Embed-L supports 512 tokens
      chunk_overlap_pct: float = 0.15

      # Recursive clustering parameters
      max_depth: int = 3
      base_pct: float = 0.05  # Base percentage for min_cluster_size
      decay: float = 0.7      # Decay factor per depth level
      silhouette_threshold: float = 0.3  # Only recurse if score < this

      # UMAP parameters
      umap_n_components: int = 50
      umap_metric: str = "cosine"

      # HDBSCAN parameters
      hdbscan_min_samples: int = 10  # Kept constant across depths

      # Caching (cache key = hash of corpus + clustering configs)
      cache_dir: Path = Path("data/clusters")
  ```
- [ ] Implement `get_cache_key(corpus_config, clustering_config) -> str` helper
- [ ] Add CLI to `cluster_corpus.py` for running experiments
- [ ] Support loading embeddings from existing `TrainingCorpus` objects

**Milestone 1.3: Testing with tiny corpus (1 hour)**
- [ ] Create `configs/clustering_01_tiny.py` (100-200 chunks from pretrain only)
- [ ] Re-embed corpus with Arctic-Embed-L (cache for reuse)
- [ ] Run clustering, verify tree structure
- [ ] Manually inspect: Do clusters make sense?
- [ ] Verify noise tracking and silhouette gating work correctly
- [ ] Save cluster tree to JSON for inspection

**Milestone 1.4: Caching and serialization (1-2 hours)**
- [ ] Implement cache key generation: `hash(corpus_config + clustering_config)`
- [ ] Cache cluster tree to disk using cache key as filename
- [ ] Include cluster statistics (size, depth, parent_id, silhouette scores, noise counts)
- [ ] Support loading cached clusters without re-running
- [ ] Cache embeddings separately (keyed by corpus_config + embedding_model)

**Deliverable:** Working recursive clustering that produces hierarchical tree structure

---

### Phase 2: LLM Naming (3-4 hours)

**Milestone 2.1: Cluster sampling (1 hour)**
- [ ] Create `name_clusters.py`
- [ ] Implement `sample_cluster_texts()`:
  - Sample 5-10 representative texts from each cluster
  - Use centroid-based sampling (closest to cluster center)
  - Include diversity (don't sample too-similar texts)
- [ ] Test: Verify samples are representative

**Milestone 2.2: LLM naming prompt (1 hour)**
- [ ] Design prompt template with hierarchical context:
  ```
  [If subcluster] Parent cluster: "Math Content"

  You are analyzing a cluster of training corpus texts. Based on these examples,
  provide a concise 2-5 word label describing the common theme.

  Examples from this cluster:
  1. [text 1]
  2. [text 2]
  ...

  Cluster label:
  ```
- [ ] Implement `generate_cluster_name()` using `rollout.py`'s `generate()`
- [ ] Use parallel API calls (asyncio.gather), not batch prompts

**Milestone 2.3: Recursive naming (1 hour)**
- [ ] Traverse cluster tree breadth-first (level-by-level)
- [ ] For each depth level, name all clusters in parallel using asyncio.gather()
- [ ] Include parent name in prompt for depth > 0
- [ ] Cache names to avoid re-running expensive LLM calls

**Milestone 2.4: Manual validation (1 hour)**
- [ ] Run on tiny corpus, inspect all names
- [ ] Adjust prompt if names are too generic/specific
- [ ] Verify hierarchy makes sense (subclusters relate to parent)

**Deliverable:** Cluster tree with human-interpretable names at all levels

---

### Phase 3: Output Annotation (2-3 hours)

**Milestone 3.1: Cluster assignment (1 hour)**
- [ ] Create `annotate_with_clusters.py`
- [ ] Implement `assign_to_clusters()`:
  - For each model output embedding
  - Find nearest cluster centroid at each depth level
  - Return cluster path: `["Math Content", "Word Problems", "Money/Shopping"]`
- [ ] Support batch assignment for efficiency

**Milestone 3.2: Integration with existing results (1 hour)**
- [ ] Load existing GSM8K similarity results CSV
- [ ] For each result, add cluster annotations:
  - `cluster_depth_0` (coarse category)
  - `cluster_depth_1` (medium category)
  - `cluster_depth_2` (fine category)
- [ ] Save augmented results to new CSV

**Milestone 3.3: Analysis pipeline (1 hour)**
- [ ] Implement `analyze_clusters()`:
  - Group results by cluster
  - Compute error rates per cluster
  - Find clusters with highest/lowest error rates
  - Statistical test: Do errors correlate with specific clusters?
- [ ] Output summary statistics and insights

**Deliverable:** GSM8K results annotated with cluster labels, ready for analysis

---

### Phase 4: Visualization & Insights (Optional, 2-3 hours)

**Milestone 4.1: Interactive tree visualization (1-2 hours)**
- [ ] Create `visualize_clusters.py` with Streamlit app
- [ ] Generate interactive dendrogram/tree (Plotly)
- [ ] Node labels show cluster names + sizes
- [ ] Color by silhouette score, error rate (if annotated), or depth
- [ ] Hover to show sample texts from cluster
- [ ] Click to expand/collapse subtrees
- [ ] Export to standalone HTML for sharing

**Milestone 4.2: Analysis dashboard (1 hour)**
- [ ] Add Streamlit sidebar with filters (depth, min size, etc.)
- [ ] Show cluster statistics table (sortable)
- [ ] Analysis views:
  - "Which clusters cause the most errors?"
  - "Do model answers come from different clusters than ground truth?"
  - "Are harder questions farther from all clusters?"
- [ ] Include statistical test results

**Deliverable:** Streamlit app for interactive cluster exploration + standalone HTML export

---

## Testing Strategy

### Unit Tests
- [ ] Test `recursive_cluster()` on synthetic data (known clusters)
- [ ] Test cluster assignment (verify nearest centroid logic)
- [ ] Test caching (load/save cluster trees)

### Integration Tests
- [ ] End-to-end: Load corpus → cluster → name → annotate → analyze
- [ ] Verify cluster tree structure (all nodes have valid parents)
- [ ] Verify cluster names are non-empty strings

### Manual Validation
- [ ] Inspect cluster samples at each level (do they make sense?)
- [ ] Check cluster names (are they interpretable?)
- [ ] Review error analysis (do patterns emerge?)

---

## Success Criteria

### Phase 1 Success
- ✅ Cluster tree with 3+ depth levels
- ✅ Clusters have reasonable sizes (not too small/large)
- ✅ Manual inspection: subclusters relate to parent clusters

### Phase 2 Success
- ✅ All clusters have names
- ✅ Names are interpretable (not generic like "Cluster 1")
- ✅ Hierarchy makes sense ("Math" → "Word Problems" → "Money")

### Phase 3 Success
- ✅ All GSM8K results annotated with cluster labels
- ✅ Analysis reveals: specific clusters correlate with errors
- ✅ Insight: "Model errors come from Cluster X more than Cluster Y"

### Phase 4 Success (Optional)
- ✅ Interactive visualization loads in browser
- ✅ Jupyter notebook runs end-to-end
- ✅ Findings are documented and shareable

---

## Timeline Estimates

**Minimum viable (Phases 1-3):** 9-13 hours
- Phase 1: 4-6 hours (core clustering)
- Phase 2: 3-4 hours (LLM naming)
- Phase 3: 2-3 hours (annotation & analysis)

**With visualization (Phase 4):** +2-3 hours

**Total:** 11-16 hours for full implementation

---

## Risks & Mitigations

### Risk 1: HDBSCAN finds no clusters (everything is noise)
**Mitigation:** Adjust `min_cluster_size` parameter, try K-means as fallback

### Risk 2: LLM names are too generic ("Text", "Content", etc.)
**Mitigation:** Improve prompt with examples, use few-shot learning

### Risk 3: Recursive depth too deep (tiny clusters)
**Mitigation:** Set `min_cluster_size` stopping condition, limit `max_depth`

### Risk 4: Expensive (many UMAP runs)
**Mitigation:** Cache all intermediate results, only recurse on large clusters

### Risk 5: No meaningful error patterns found
**Mitigation:** This is valuable negative result! Document for paper.

---

## Future Extensions

### After Initial Implementation
- [ ] Compare Muon vs Adam model outputs across clusters
- [ ] Test "SFT memorizes, RL generalizes" by cluster proximity
- [ ] Contamination detection: Which clusters overlap with eval benchmarks?
- [ ] Difficulty stratification: Are hard questions from specific clusters?

### Scaling Improvements
- [ ] Parallelize UMAP across clusters (independent computation)
- [ ] Use approximate nearest neighbors (FAISS) for large corpora
- [ ] Incremental clustering (add new data without recomputing)

### Analysis Enhancements
- [ ] SHAP values (Spotify's approach) to explain cluster membership
- [ ] Compare clusters across training stages (pretrain vs midtrain vs SFT)
- [ ] Temporal analysis (do clusters evolve during training?)

---

## Design Decisions

### 1. Embedding Model & Chunking
**Decision:** Use `Snowflake/snowflake-arctic-embed-l` (1024 dimensions)
- Higher quality embeddings for better cluster separation
- SOTA retrieval performance
- Will need to re-embed corpus (existing embeddings are all-MiniLM-L6-v2)
- **Add new `EmbeddingConfig` for Arctic-Embed-L** with model name and batch size
- **Use `fixed_tokens` chunking strategy** with max_tokens=512 (Arctic-Embed-L supports 512 tokens)
- Ensures chunks don't exceed model's token limit

### 2. HDBSCAN Parameters (Adaptive with Depth)
**Decision:** Use adaptive percentage-based parameters that decay with depth
```python
# Adaptive formula respects local structure, not just size
min_cluster_size = max(floor, n_points * base_pct * decay^depth)

# Parameters:
base_pct = 0.05        # ~5% at root
decay = 0.7            # ~3.5% at depth 1, ~2.5% at depth 2
floor = max(10, 50/2^depth)  # Prevent degenerate clustering

# Key insight from Spotify: Check cluster quality before recursing
# Only recurse if silhouette_score < 0.3 (low coherence)
# This prevents forcing meaningless subdivisions of tight clusters
```

**Rationale:**
- Auto-adapts to varying cluster sizes (1000-point → 50-point subclusters, 200-point → 10-point subclusters)
- Decay prevents over-splitting at depth
- Silhouette gating respects natural cluster boundaries
- One knob to tune (base_pct), works across hierarchical structure

### 3. Cluster Naming Strategy
**Decision:** Parallel single-cluster prompts with hierarchical context
- One API call per cluster, parallelized with `asyncio.gather()`
- Include parent cluster name in prompt for subclusters
- Process level-by-level (breadth-first): name all depth-0, then all depth-1, etc.

Example prompt for subcluster:
```
Parent cluster: "Math Content"

Subcluster samples:
1. [sample text]
2. [sample text]
...

Provide a concise 2-5 word label for this subcluster.
```

**Rationale:**
- Clearer prompts → better names
- Easier to cache individual results
- Hierarchical context improves name quality

### 4. Noise Point Handling
**Decision:** Keep noise points in parent cluster, track separately, don't recurse
- HDBSCAN noise (label = -1) stays at current depth level
- Track noise separately for analysis (provides cluster boundary information)
- Don't attempt to subdivide noise points

**Rationale:**
- Noise means "doesn't fit subclusters naturally"
- Forcing classification pollutes clean clusters
- Noise points provide valuable boundary information

### 5. Testing Scope & Compute Strategy
**Decision:** Start with 100-200 chunks from pretrain only
- Phase 1.1-1.3: Develop locally with tiny corpus (CPU is sufficient for 100-200 chunks)
- Phase 1.4+: Deploy to GPU for full corpus embedding (use existing `deploy.py` infrastructure)

**Rationale:**
- Local development: Fast iteration, easy debugging, ~30-60s embedding time for tiny corpus
- GPU deployment: Much faster for full corpus (thousands of chunks)
- UMAP + HDBSCAN run on CPU only (no GPU acceleration), but fast enough even for full corpus

### 6. Statistical Tests
**Decision:** Chi-squared test for "errors correlate with clusters"
- Test: cluster membership vs correctness (categorical data)
- Report effect size (Cramér's V) in addition to p-value
- Use Fisher's exact test if cluster sizes are small (<5 per cell)

---

## Implementation Order

### Week 1: Core Infrastructure
1. Phase 1.1-1.3: Get basic clustering working on tiny corpus
2. Phase 2.1-2.2: Get LLM naming working on a few clusters
3. Manual validation: Do clusters + names make sense?

### Week 2: Full Pipeline
4. Phase 1.4: Add caching, run on full corpus
5. Phase 2.3-2.4: Name all clusters, validate hierarchy
6. Phase 3.1-3.3: Annotate GSM8K results, analyze patterns

### Week 3: Polish & Extensions
7. Phase 4 (optional): Visualizations
8. Write up findings
9. Connect to other experiments (Muon vs Adam, SFT vs RL, etc.)

---

## Next Steps

To start implementation:
1. Install dependencies: `pip install umap-learn hdbscan`
2. Create `cluster_corpus.py` with basic structure
3. Run on 100-chunk sample to verify UMAP + HDBSCAN work
4. Iterate on parameters until clusters look reasonable

Once clustering works, move to naming, then annotation.

---

## References

- [Spotify: Recursive Embedding and Clustering](https://engineering.atspotify.com/2023/12/recursive-embedding-and-clustering)
- [UMAP Documentation](https://umap-learn.readthedocs.io/)
- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- Original motivation: `conversation.txt` lines 4-5 (cluster annotations for latent space)
