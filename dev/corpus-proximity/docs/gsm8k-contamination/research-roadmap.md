# Corpus Proximity Research

## Goal
Measure distance from model outputs to training data to understand memorization, optimizer differences, and OOD behavior.

## 🎯 Recent Accomplishments (This Session)

### Recursive Clustering Pipeline - COMPLETE ✅
Implemented Spotify-inspired recursive embedding and clustering for corpus taxonomy:

**Core Implementation (cluster_corpus.py)**
- Recursive UMAP dimensionality reduction (1024→50 dims) + HDBSCAN clustering
- Silhouette gating: only recurse if cluster coherence < 0.3 (prevents forced subdivisions)
- Adaptive parameters: `min_cluster_size = max(5, n * 0.05 * 0.7^depth)`
- Re-embedding at each level with Arctic-Embed-L (1024-dim, 512 token limit)
- Token-based chunking integration (512 tokens max, 15% overlap)
- Noise point tracking (HDBSCAN outliers)
- Full caching system with hash-based keys

**LLM Naming System (name_clusters.py)**
- Breadth-first cluster naming using OpenAI API (gpt-4o-mini)
- Asyncio parallelization (names entire level concurrently)
- Centroid-based text sampling for representative examples
- Hierarchical context (passes parent cluster name to child)
- dotenv integration for API key management

**Inspection Tools (name_clusters.py CLI)**
- `--tree`: Pretty-print cluster hierarchy with names and stats
- `--list`: Flat list of all clusters with metadata
- `--inspect <id> --samples N`: Random sample inspection from any cluster
- `--show-noise`: View noise points separately
- `--name`: Generate LLM names for entire tree

**Validation**
- Tested on tiny corpus (150 chunks → 405 token-limited chunks)
- Found 8 clusters with 126 noise points, silhouette 0.525
- Correctly identified high coherence (no recursion needed)
- Successfully generated cluster name: "Diverse Infrastructure and Ecology"

**Type Safety**
- Fixed all `ty check` errors in corpus-proximity codebase
- Added proper type annotations for dynamic module loading
- Optional types for nullable parameters
- Null checks for callbacks

**Files Created/Modified**
- `cluster_corpus.py` (530 lines) - core clustering engine
- `name_clusters.py` (490 lines) - LLM naming and inspection
- `configs/clustering_01_tiny.py` - test configuration
- `prepare_tiny_corpus.py` - sampling utility for testing
- Updated `config.py` with ClusteringConfig dataclass
- Type fixes in: rollout.py, search.py, embed_chunks.py, prepare_data.py, gsm8k_corpus_similarity.py

**Next Steps for Clustering**
- Run on full corpus (not just 150-sample test)
- Integrate with similarity search to tag model outputs by cluster
- Visualize cluster hierarchy and distributions
- Analyze: Do model errors correlate with specific clusters?

## Original Interest Areas (from conversation.txt)
1. Measuring OOD performance by comparing chunked/embedded pretraining corpus with benchmark questions
2. Anthropic 1.5B settlement - training data provenance detection
3. Annotations for what part of latent space model responses come from
4. Generate annotations via BERT index + recursive clustering + LLM cluster naming
5. Muon-trained models fail to finish prefilled lyrics verbatim (vs Adam)
6. Optimizers qualitatively change solutions
7. Muon outperforms Adam in Tail-End Associative Memory Learning
8. Difficulty ratings for eval questions (GSM8K/MATH500)
9. "SFT memorizes, RL generalizes" paper findings

## Completed ✅
- [x] Data collection (FineWeb-Edu)
- [x] Chunking pipeline (prepare_data.py)
- [x] Embedding pipeline (embed_chunks.py)
- [x] Search infrastructure (search.py with multiple distance functions)
- [x] Tests (test_search.py validates known/unknown sentences)
- [x] Corpus configs for nanochat (corpus.py - streaming access)
- [x] Rollout dataclass with dacite serialization (rollout.py)
- [x] Sample protocol for dataset samples (GSM8KSample implementation)
- [x] Inference wrapper (generate() function with OpenAI/vLLM support)
- [x] CLI for interactive querying (rollout.py with --stream, --system flags)
- [x] Advanced chunking strategies with spaCy/NLTK (chunking.py - sentence_spacy/sentence_nltk)
- [x] Token-aware chunking with configurable overlap (chunking.py - chunk_fixed_tokens)
- [x] SimilarityConfig in config.py
- [x] GSM8K corpus similarity measurement script (gsm8k_corpus_similarity.py)
- [x] Config files for tiny/full experiments
- [x] Local test validation (3 samples × 3 variants × 3 stages × 5 neighbors = 135 results)
- [x] **Recursive clustering pipeline (cluster_corpus.py) - Spotify-inspired UMAP+HDBSCAN**
  - [x] ClusterNode dataclass with hierarchical IDs, silhouette scores, noise tracking
  - [x] Adaptive HDBSCAN parameters with decay by depth
  - [x] Silhouette gating (only recurse if score < 0.3)
  - [x] Re-embedding at each level (Arctic-Embed-L, 1024-dim)
  - [x] Token-based chunking integration (512 tokens max, 15% overlap)
  - [x] Caching system with hash-based keys for embeddings and trees
  - [x] JSON serialization with sample texts and full indices
- [x] **LLM-based cluster naming (name_clusters.py)**
  - [x] Breadth-first cluster naming with asyncio parallelization
  - [x] Centroid-based and random text sampling strategies
  - [x] Hierarchical context (parent cluster names)
  - [x] Integration with rollout.py's generate() function
  - [x] dotenv support for API keys
- [x] **Cluster inspection utilities (name_clusters.py CLI)**
  - [x] --tree: Pretty-print cluster hierarchy with names
  - [x] --list: List all clusters with metadata
  - [x] --inspect <id>: Show random samples from specific cluster
  - [x] --show-noise: Display noise points separately
  - [x] --name: Generate LLM names for entire tree
- [x] **Type safety improvements**
  - [x] Fixed all ty check errors in corpus-proximity
  - [x] Type annotations for module.config loading
  - [x] Optional types for nullable parameters
  - [x] Callback null checks in rollout.py

## Next: Minimal Shippable Artifact (4-6 hours)
**Narrative: "Can we detect what training data a model is using?"**

### Phase 1: Core Infrastructure (addresses interest #1) ✅ COMPLETE
- [x] Define Rollout dataclass (prompt, completion, tokens, metadata)
- [x] Build inference wrapper (OpenAI/vLLM via generate() function)
- [x] Build integration: eval question → embed → search corpus → measure distance
- [x] Test: Load GSM8K question, measure distance to nanochat corpus
- [x] Validate: Can we detect when eval questions are similar to training data?

**Status:** Phase 1 infrastructure complete and validated.

### Phase 2: Experiments
**Interest #1: Eval benchmark → corpus distance**
- [x] Load GSM8K eval questions/answers
- [x] Measure distance to pretrain/midtrain/SFT corpus (baseline with ground truth)
- [x] Generate model answers using rollout.py (generate_model_answers function)
- [x] Compare model answers vs ground truth answers to corpus (4-variant embedding with model_answer)
- [x] Optional model generation via config flag (include_model_answers in SimilarityConfig)
- [x] Config file for model experiments (configs/gsm8k_similarity_03_model.py)
- [ ] Hypothesis testing: Model answers closer to training data = memorization vs reasoning
- [ ] Visualize: distance vs difficulty/correctness

**Status:** Phase 2 core infrastructure complete. Model generation is implemented and optional (set `include_model_answers=True` in config). Ready for hypothesis testing and visualization.

**Interest #5: Muon vs Adam on exact recall (lyrics test)**
- [ ] Find ~50 famous quotes/lyrics definitely in Common Crawl
- [ ] Test with Adam-trained model (GPT-2 or similar)
- [ ] Test with Muon-trained model (if available)
- [ ] Measure edit distance of completions vs ground truth
- [ ] Hypothesis: Muon has higher edit distance (worse exact recall)

**Interest #4: Cluster-based annotations** ✅ INFRASTRUCTURE COMPLETE
- [x] Cluster corpus embeddings (recursive clustering) - cluster_corpus.py with UMAP+HDBSCAN
- [x] Sample texts from each cluster - sample_cluster_texts() with centroid/random strategies
- [x] Use LLM to name clusters - name_clusters.py with hierarchical naming
- [x] Full inspection utilities - CLI with --tree, --list, --inspect, --show-noise
- [ ] For model outputs: tag which cluster(s) they're closest to
- [ ] Analysis: Do errors correlate with specific clusters?
- [ ] Run on full corpus (not just tiny test set)
- [ ] Visualize cluster hierarchy and distributions

**Interest #9: SFT vs base model proximity**
- [ ] If access to base + SFT pair (e.g., Llama-3-base vs Llama-3-Instruct)
- [ ] Compare distance-to-training patterns
- [ ] Hypothesis: SFT shows closer proximity (more memorization)

**This produces:** Evidence connecting training data proximity to model behavior across multiple hypotheses

---

## Current Status Summary

### ✅ Phase 1 & 2 Core: COMPLETE
All infrastructure for measuring training data proximity is implemented and tested:
- Data collection, chunking (incl. token-aware), embedding, search
- Model answer generation (optional, config-controlled)
- GSM8K baseline measurements with ground truth + model answers
- Multi-variant embedding pipeline (question, answer, question+answer, model_answer)

### 🔄 Phase 2 Analysis: IN PROGRESS
Infrastructure ready, need to:
- Run experiments with model generation enabled
- Analyze memorization vs reasoning patterns
- Build visualizations

### 📋 Phase 3+: Extended Research
See "New Ideas / Extended Research Directions" section below for future experiments (Muon vs Adam, clustering, SFT vs RL, contamination detection, etc.)

---

## Future Work (After First Artifact)
### Training Infrastructure
- [ ] Training pipeline (corpus + optimizer → trained model)
- [ ] Post-training pipeline (base → SFT)
- [ ] Deploy to GPU cluster

### Advanced Measurement
- [x] Recursive clustering on embeddings (cluster_corpus.py - Spotify approach)
- [x] LLM-based cluster labeling (name_clusters.py with OpenAI integration)
- [ ] Annotate outputs by cluster ("this came from math content")
- [ ] Integrate clustering with similarity search for cluster-tagged results

### Experiments (Requires Trained Models)
- [ ] Muon vs Adam distance patterns (lyrics completion test)
- [ ] Distance vs difficulty correlation (GSM8K easy vs hard)
- [ ] Base vs SFT proximity patterns (memorization hypothesis)

## Later Optimizations
- [ ] Parallelize data processing (Worker pattern)
- [ ] Batch inference for speed
- [ ] Larger corpus coverage

---

## New Ideas / Extended Research Directions

### Positioning vs. BETR Paper (arxiv 2507.12466)
**BETR's contribution:** Pretraining data selection using benchmark similarity (forward-looking optimization)

**Our novel extensions:**
- Post-hoc analysis: Measure model outputs → training corpus (interpretability, not just data selection)
- Optimizer effects: Muon vs Adam memorization patterns
- SFT/RL comparison: Training method effects on corpus proximity
- Contamination detection: Per-question benchmark hygiene scoring
- Fine-grained difficulty: Per-question metrics revealing "model shapes"
- Multi-stage analysis: Pretrain/midtrain/SFT/RL stage separation

### Additional Interest Areas
10. RL contamination hypothesis: GSM8K/MATH500 in midtraining inflates RL results
11. **"RL is secretly SFT" hypothesis:** RL is just SFT on synthetically generated data from the RL environment/process, not true credit assignment through gradient signals
12. Per-question difficulty via OpenRouter aggregation (10-20 providers)

### Key Insights from Related Papers

**"SFT Memorizes, RL Generalizes" (arxiv 2501.17161)**
- RL with outcome-based rewards generalizes across rule-based textual and visual variants
- SFT memorizes training data, struggles on OOD scenarios
- SFT is essential for RL: stabilizes output format before RL training
- RL improves underlying visual capabilities for cross-domain generalization

**Our counter-hypothesis to test:**
- **"RL is secretly SFT"**: The RL process is just a data generation mechanism (via environment interactions/rollouts). The actual learning is standard SFT on this synthetically generated data, not credit assignment via RL gradients.
- **Testable prediction via corpus proximity:**
  - SFT models: answers close to original training corpus (direct memorization)
  - RL models: answers close to *synthetic rollout data* generated during RL training, but NOT farther from corpus overall
  - If RL is truly different: answers should show novel patterns uncorrelated with either training corpus OR rollout data
  - If RL is secretly SFT: corpus proximity should match the distribution of rollout data, not show true generalization

### BETR-Inspired Methodology Improvements

**Upgrade Embedding Models**
- [x] Add Arctic-Embed-L (`Snowflake/snowflake-arctic-embed-l`) to config - used in clustering pipeline
- [ ] Add GTE-Large (`Alibaba-NLP/gte-large-en-v1.5`) as alternative
- [ ] Create ablation configs comparing MiniLM vs Arctic vs GTE
- [ ] Run comparison on tiny GSM8K
- [ ] Document embedding choice

**Implement Rank-Based Aggregation**
- [ ] Create `ranking.py` with rank-based similarity
- [ ] For each query: rank all corpus chunks by distance
- [ ] Implement "max aggregation": best rank across all queries
- [ ] Implement "mean aggregation": average rank
- [ ] Add to search.py as alternative scoring method
- [ ] Update config with `aggregation_strategy: "distance" | "rank_max" | "rank_mean"`
- [ ] Test on tiny GSM8K: compare distance vs rank-based

**Multi-Benchmark Contamination Detection**
- [ ] Extend corpus.py: add MATH, MMLU, HellaSwag, ARC loaders
- [ ] Create `multi_benchmark_similarity.py`
- [ ] For each corpus chunk: compute similarity to all benchmarks
- [ ] Track: max_similarity, source_benchmark, all_similarities
- [ ] Output contamination heatmap (corpus stage × benchmark)
- [ ] Visualization: which benchmarks leak into which stages

**Held-Out Benchmark Validation**
- [ ] Define benchmark splits: training (GSM8K, MATH, MMLU, ARC) vs held-out (BBH, GPQA, IFEval)
- [ ] Hypothesis: Does corpus proximity predict held-out performance?
- [ ] Correlate: corpus proximity vs. performance on both splits

**Efficient Sampling for Large Corpora** (for scaling)
- [ ] Create `scale_embeddings.py`
- [ ] Sample 0.1-1% of corpus, embed sample
- [ ] Train FastText classifier: text → predicted similarity score
- [ ] Apply to full corpus without embedding all

### Novel Experimental Directions

**Per-Question Difficulty Metric**
- [ ] Create `benchmark_difficulty.py`
- [ ] OpenRouter integration: 10-20 diverse providers
- [ ] Query all providers on GSM8K test set
- [ ] Grade correctness, aggregate: difficulty_score = 1 - (correct/total)
- [ ] Cache results: `data/benchmark_difficulty/gsm8k_difficulty.json`
- [ ] Visualization: difficulty histogram
- [ ] Extend to MATH500

**GSM8K Contamination with Difficulty Stratification**
- [ ] Run full GSM8K corpus similarity (100+ samples)
- [ ] Merge with difficulty scores
- [ ] Analyze: Are easy questions closer to midtrain?
- [ ] Contamination score by difficulty tier
- [ ] Export contaminated question IDs

**SFT vs RL Corpus Proximity (Testing "RL is secretly SFT")**
- [ ] Identify base + SFT + RL model triple (Llama-3 variants or train own)
- [ ] **Critical: Access RL rollout data** (the synthetic data generated during RL training)
  - [ ] If training own RL model: save all rollout trajectories
  - [ ] If using existing model: try to get rollout data or approximate it
- [ ] Generate answers from all three models on same eval set
- [ ] Measure corpus proximity for:
  - [ ] Base model answers → original training corpus
  - [ ] SFT model answers → original training corpus
  - [ ] RL model answers → original training corpus
  - [ ] **RL model answers → RL rollout data** (the synthetic data)
- [ ] Compare:
  - [ ] Is SFT closer to training corpus than base? (expected: yes, memorization)
  - [ ] Is RL farther from training corpus than SFT? (if yes, supports "RL generalizes")
  - [ ] **Is RL close to its own rollout data?** (if yes, supports "RL is secretly SFT")
  - [ ] Does RL show novel patterns uncorrelated with both corpus AND rollouts? (if yes, true RL)
- [ ] Stratify by difficulty: does pattern hold across easy/hard questions?

**RL Contamination in Midtrain**
- [ ] Compare RL performance on contaminated vs clean questions
- [ ] Hypothesis: RL shows bigger gains on contaminated questions
- [ ] Correlate: contamination_score × RL_performance_gain

**Muon vs Adam Corpus Proximity**
- [ ] If lyrics test shows difference
- [ ] Measure corpus proximity for Muon vs Adam outputs
- [ ] On reasoning tasks (GSM8K) vs memorization tasks (lyrics)
- [ ] Hypothesis: Muon farther from training on memorization

### Implementation Priority

**Week 1: Foundation**
1. Upgrade embeddings (Arctic-Embed)
2. Per-question difficulty metric
3. Multi-benchmark contamination
4. GSM8K contamination analysis with stratification

**Week 2: Novel Hypotheses**
5. Rank-based aggregation
6. Model answer generation + corpus proximity
7. SFT vs RL proximity comparison

**Week 3: Extensions**
8. Muon vs Adam lyrics + proximity test
9. Cluster-based annotations
10. Ablations (embeddings, aggregation, chunking)



