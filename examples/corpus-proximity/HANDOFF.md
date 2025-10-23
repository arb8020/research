# Corpus Proximity Pipeline - Handoff Notes

## Current Status (2025-10-23)

**GPU Pipeline Running:**
- Instance: `lj5a0dniz5kski` (RTX A4000, RunPod, $0.25/hr)
- SSH: `ssh -p 20914 -i ~/.ssh/id_ed25519 root@157.157.221.29`
- Progress: Step 3/4 - Embedding 142K chunks (1% complete, ~2 hours remaining)
- Check logs: `ssh ... "tmux capture-pane -t corpus_proximity_pipeline -p | tail -30"`

## Key Fixes Made Today

1. **Critical Bug Fixed:** Silhouette threshold logic was inverted in `cluster_corpus.py`
   - Before: High silhouette → don't split ❌
   - After: High silhouette → DO split ✅
   - Commit: `f92e16e`

2. **Bifrost improvements:**
   - Added untracked file warnings before deployment
   - Silenced noisy HTTP logs
   - Enabled UMAP verbose output

3. **Pipeline improvements:**
   - Real-time progress output (removed `capture_output=True`)
   - Better progress reporting in `deploy.py`

## Commands

```bash
# Check GPU status
broker list

# Check remote progress
ssh -o StrictHostKeyChecking=no -p 20914 -i ~/.ssh/id_ed25519 root@157.157.221.29 \
  "tmux capture-pane -t corpus_proximity_pipeline -p | tail -30"

# Run pipeline locally
cd examples/corpus-proximity
python run_full_pipeline.py configs/clustering_01_tiny.py

# Deploy to GPU
python deploy.py --config configs/clustering_01_tiny.py
```

## Dataset Info

- Original chunks: 53,248 (from 1 FineWeb-Edu shard)
- After re-chunking (512 tokens): 142,914 chunks
- Config: `configs/clustering_01_tiny.py`
- Max depth: 3, silhouette threshold: 0.3

## Expected Output

Results sync to: `examples/corpus-proximity/remote_results/clustering_<timestamp>/`
- `tree.json` - Hierarchical cluster tree with LLM-generated names
- `stats.json` - Cluster statistics
- `chunk_to_cluster.json` - Chunk mappings

## Known Issues

- Re-chunking creates 3x more chunks (53K → 142K), making embedding slow
- UMAP can hang on very small clusters at deep recursion levels
- Mac MPS is 7x slower than NVIDIA GPU for embedding

## Next Steps

- Pipeline should complete in ~2 hours
- Will auto-sync results and terminate GPU
- Check `tree.json` to verify proper cluster taxonomy (should have 8+ clusters, not just 1)
