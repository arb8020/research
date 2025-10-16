# Corpus Proximity Research

## Goal
Measure distance from model outputs to training data to understand memorization, optimizer differences, and OOD behavior.

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
- [x] SimilarityConfig in config.py
- [x] GSM8K corpus similarity measurement script (gsm8k_corpus_similarity.py)
- [x] Config files for tiny/full experiments
- [x] Local test validation (3 samples × 3 variants × 3 stages × 5 neighbors = 135 results)

## Next: Minimal Shippable Artifact (4-6 hours)
**Narrative: "Can we detect what training data a model is using?"**

### Phase 1: Core Infrastructure (addresses interest #1) ✅ COMPLETE
- [x] Define Rollout dataclass (prompt, completion, tokens, metadata)
- [x] Build inference wrapper (OpenAI/vLLM via generate() function)
- [x] Build integration: eval question → embed → search corpus → measure distance
- [x] Test: Load GSM8K question, measure distance to nanochat corpus
- [x] Validate: Can we detect when eval questions are similar to training data?

**Status:** Infrastructure complete. Ready for Phase 2 experiments with model-generated answers (requires LLM inference via rollout.py).

### Phase 2: Experiments
**Interest #1: Eval benchmark → corpus distance**
- [x] Load GSM8K eval questions/answers
- [x] Measure distance to pretrain/midtrain/SFT corpus (baseline with ground truth)
- [ ] Generate model answers using rollout.py (requires LLM inference)
- [ ] Compare model answers vs ground truth answers to corpus
- [ ] Hypothesis: Model answers closer to training data = memorization vs reasoning
- [ ] Visualize: distance vs difficulty/correctness

**Interest #5: Muon vs Adam on exact recall (lyrics test)**
- [ ] Find ~50 famous quotes/lyrics definitely in Common Crawl
- [ ] Test with Adam-trained model (GPT-2 or similar)
- [ ] Test with Muon-trained model (if available)
- [ ] Measure edit distance of completions vs ground truth
- [ ] Hypothesis: Muon has higher edit distance (worse exact recall)

**Interest #4: Cluster-based annotations**
- [ ] Cluster corpus embeddings (recursive clustering)
- [ ] Sample texts from each cluster
- [ ] Use LLM to name clusters ("math textbook", "code", "news", etc.)
- [ ] For model outputs: tag which cluster(s) they're closest to
- [ ] Analysis: Do errors correlate with specific clusters?

**Interest #9: SFT vs base model proximity**
- [ ] If access to base + SFT pair (e.g., Llama-3-base vs Llama-3-Instruct)
- [ ] Compare distance-to-training patterns
- [ ] Hypothesis: SFT shows closer proximity (more memorization)

**This produces:** Evidence connecting training data proximity to model behavior across multiple hypotheses

## Future Work (After First Artifact)
### Training Infrastructure
- [ ] Training pipeline (corpus + optimizer → trained model)
- [ ] Post-training pipeline (base → SFT)
- [ ] Deploy to GPU cluster

### Advanced Measurement
- [ ] Recursive clustering on embeddings
- [ ] LLM-based cluster labeling
- [ ] Annotate outputs by cluster ("this came from math content")

### Experiments (Requires Trained Models)
- [ ] Muon vs Adam distance patterns (lyrics completion test)
- [ ] Distance vs difficulty correlation (GSM8K easy vs hard)
- [ ] Base vs SFT proximity patterns (memorization hypothesis)

## Later Optimizations
- [ ] Parallelize data processing (Worker pattern)
- [ ] Batch inference for speed
- [ ] Larger corpus coverage



