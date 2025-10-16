# GSM8K Corpus Similarity Measurement

## Goal
Measure distance from GSM8K eval questions/answers to nanochat training corpus to test if eval questions are OOD.

## Script Structure

### File: `gsm8k_corpus_similarity.py`
Main script that:
1. Loads corpus chunks from multiple stages (pretrain/midtrain/sft)
2. Embeds corpus chunks (cache to disk)
3. Loads GSM8K samples
4. Embeds GSM8K variants (question, answer, question+answer)
5. Searches corpus for nearest neighbors
6. Saves results to CSV

### File: `config.py` (add SimilarityConfig)
Add to existing config.py:

```python
@dataclass
class SimilarityConfig:
    """Configuration for similarity measurement experiments."""
    # GSM8K sampling (from rollout.py's GSM8KSample)
    num_gsm8k_samples: int = 10
    gsm8k_split: str = "test"  # Use load_dataset("openai/gsm8k", "main", split="test")

    # Corpus sampling (chunks per stage)
    # Corpus definitions from corpus.py: NANOCHAT_PRETRAIN, SMOLTALK, MMLU_AUX_TRAIN, etc.
    corpus_sizes: dict = field(default_factory=lambda: {
        "pretrain": 1000,     # From NANOCHAT_PRETRAIN
        "midtrain": 900,      # From SMOLTALK(300) + MMLU_AUX_TRAIN(300) + GSM8K_TRAIN(300)
        "sft": 1000,          # From ARC_EASY_TRAIN(500) + ARC_CHALLENGE_TRAIN(500)
    })

    # Chunking strategy
    chunking_strategy: str = "paragraph"  # "paragraph" | "fixed_chars" | "sentence"
    chunk_size: int = 512  # For fixed_chars strategy

    # Search
    k_neighbors: int = 5  # Return top-k nearest neighbors
    distance_metric: str = "cosine"  # "cosine" | "euclidean" | "manhattan"

    # Output
    output_dir: Path = _BASE_DIR / "results"
    output_file: str = "gsm8k_similarity.csv"

# Update Config to include similarity
@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    similarity: SimilarityConfig = field(default_factory=SimilarityConfig)
```

### File: `configs/gsm8k_similarity_01_tiny.py`
Small config for local testing:

```python
"""Tiny config for local testing of GSM8K similarity measurement."""

from config import Config

config = Config()

# Override for tiny test
config.similarity.num_gsm8k_samples = 3
config.similarity.corpus_sizes = {
    "pretrain": 100,
    "midtrain": 90,
    "sft": 100,
}
config.similarity.output_file = "gsm8k_similarity_tiny.csv"
config.embedding.batch_size = 32  # Smaller batch for CPU
```

### File: `configs/gsm8k_similarity_02_full.py`
Full config for GPU run:

```python
"""Full config for GPU run of GSM8K similarity measurement."""

from config import Config

config = Config()

# Use defaults (already set in SimilarityConfig)
# Just override if needed
config.similarity.num_gsm8k_samples = 10
config.embedding.batch_size = 128  # Larger batch for GPU
```

## Components to Implement

### 1. Chunking Strategies (`chunking.py`)

```python
def chunk_text(text: str, strategy: str, chunk_size: int = 512) -> List[str]:
    """Split text into chunks based on strategy."""

def chunk_paragraph(text: str) -> List[str]:
    """Split by double newline (paragraphs)."""

def chunk_fixed_chars(text: str, size: int) -> List[str]:
    """Split into fixed character length chunks with overlap."""

def chunk_sentence(text: str) -> List[str]:
    """Split by sentence boundaries."""
```

### 2. Corpus Loading (`gsm8k_corpus_similarity.py`)

```python
def load_corpus_chunks(config: GSM8KSimilarityConfig) -> Dict[str, List[str]]:
    """Load and chunk corpus from each stage.

    Returns:
        {"pretrain": [...], "midtrain": [...], "sft": [...]}
    """
    chunks = {}

    # Pretrain stage
    chunks["pretrain"] = sample_and_chunk(
        corpus=NANOCHAT_PRETRAIN,
        n=config.corpus_sizes["pretrain"],
        strategy=config.chunking_strategy,
        chunk_size=config.chunk_size
    )

    # Midtrain stage (combine multiple corpora)
    midtrain_chunks = []
    for corpus, n in [(SMOLTALK, 300), (MMLU_AUX_TRAIN, 300), (GSM8K_TRAIN, 300)]:
        midtrain_chunks.extend(sample_and_chunk(corpus, n, ...))
    chunks["midtrain"] = midtrain_chunks[:config.corpus_sizes["midtrain"]]

    # SFT stage
    sft_chunks = []
    for corpus, n in [(ARC_EASY_TRAIN, 500), (ARC_CHALLENGE_TRAIN, 500)]:
        sft_chunks.extend(sample_and_chunk(corpus, n, ...))
    chunks["sft"] = sft_chunks[:config.corpus_sizes["sft"]]

    return chunks
```

### 3. Embedding Cache (`gsm8k_corpus_similarity.py`)

```python
def embed_corpus(
    chunks: Dict[str, List[str]],
    model: SentenceTransformer,
    cache_dir: str
) -> Dict[str, np.ndarray]:
    """Embed corpus chunks and cache to disk.

    Cache files: {cache_dir}/{stage}_minilm_embeddings.npy
    """
    embeddings = {}

    for stage, chunk_list in chunks.items():
        cache_path = f"{cache_dir}/{stage}_minilm_embeddings.npy"

        if os.path.exists(cache_path):
            logger.info(f"Loading cached embeddings for {stage}")
            embeddings[stage] = np.load(cache_path)
        else:
            logger.info(f"Embedding {len(chunk_list)} chunks for {stage}")
            embeddings[stage] = model.encode(chunk_list, show_progress_bar=True)
            np.save(cache_path, embeddings[stage])

    return embeddings
```

### 4. GSM8K Embedding (`gsm8k_corpus_similarity.py`)

```python
def embed_gsm8k_samples(
    samples: List[GSM8KSample],
    model: SentenceTransformer
) -> Dict[str, List[np.ndarray]]:
    """Embed GSM8K samples in 3 variants.

    Returns:
        {
            "question": [emb1, emb2, ...],
            "answer": [emb1, emb2, ...],
            "question_answer": [emb1, emb2, ...]
        }
    """
```

### 5. Search & Results (`gsm8k_corpus_similarity.py`)

```python
@dataclass
class SimilarityResult:
    sample_id: str
    variant: str  # "question" | "answer" | "question_answer"
    corpus_stage: str  # "pretrain" | "midtrain" | "sft"
    distance: float
    nearest_chunk_text: str
    rank: int  # 1 to k (1=closest)

def search_all(
    gsm8k_embeddings: Dict[str, List[np.ndarray]],
    corpus_embeddings: Dict[str, np.ndarray],
    corpus_chunks: Dict[str, List[str]],
    k: int
) -> List[SimilarityResult]:
    """Search all variants against all corpus stages."""

def save_results(results: List[SimilarityResult], output_file: str):
    """Save to CSV with columns: sample_id, variant, corpus_stage, distance, rank, chunk_text"""
```

## Integration with Deploy Script

Add to `deploy.py`:
```python
# After embed_chunks.py completes
logger.info("Running gsm8k_corpus_similarity.py...")
cmd = f"cd ~/.bifrost/workspace && uv run python examples/corpus-proximity/gsm8k_corpus_similarity.py examples/corpus-proximity/{args.config}"
result = bifrost_client.exec(cmd)
```

## Output CSV Format

```csv
sample_id,variant,corpus_stage,rank,distance,chunk_text
gsm8k_0001,question,pretrain,1,0.234,"Janet's ducks lay eggs..."
gsm8k_0001,question,pretrain,2,0.245,"Farm animals produce..."
gsm8k_0001,question,midtrain,1,0.189,"Math problem: Calculate..."
gsm8k_0001,answer,pretrain,1,0.456,"The answer is 18..."
...
```

## Testing Plan

1. **Local test with tiny config:**
   - 3 GSM8K samples
   - 100 chunks per stage
   - Verify embeddings are cached
   - Check CSV output format

2. **Remote GPU test:**
   - 10 GSM8K samples
   - 1000/900/1000 chunks (pretrain/midtrain/sft)
   - Deploy via bifrost
   - Sync results back to local

3. **Full run:**
   - All GSM8K test set (~1300 samples)
   - Full corpus sizes
   - Analyze: Are harder questions further from training data?

## Next Steps

1. Implement `chunking.py`
2. Create `configs/gsm8k_similarity_config.py`
3. Implement `gsm8k_corpus_similarity.py`
4. Test locally
5. Update `deploy.py` to run it
6. Deploy to GPU and collect results
