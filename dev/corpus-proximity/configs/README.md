# Configuration Files

Configurations for different experiments and projects.

## Directory Structure

```
configs/
├── clustering/          # Project 2: Recursive clustering configs
│   ├── 01_tiny.py      # Test run (150 chunks from 1 shard)
│   └── 02_full.py      # Full corpus (10+ shards)
│
├── gsm8k/              # Project 3: GSM8K contamination analysis
│   ├── 01_tiny.py      # Small test (3 samples, 100 corpus chunks/stage)
│   ├── 02_full.py      # Full run (100+ samples, 1000 chunks/stage)
│   └── 03_model.py     # With model answer generation (requires OpenAI API)
│
└── archive/            # Deprecated/old configs
    ├── 01_baseline.py
    ├── 02_small_test_03.py
    └── 03_nanochat_full.py
```

## Usage

### Clustering Configs

```bash
# Test locally
python cluster_corpus.py configs/clustering/01_tiny.py
python name_clusters.py configs/clustering/01_tiny.py --tree

# Deploy to GPU
python deploy.py --config configs/clustering/02_full.py
```

### GSM8K Configs

```bash
# Test locally
python gsm8k_corpus_similarity.py configs/gsm8k/01_tiny.py

# With model generation (needs OPENAI_API_KEY)
python gsm8k_corpus_similarity.py configs/gsm8k/03_model.py
```

## Creating New Configs

All configs should import from `config.py`:

```python
from pathlib import Path
from config import Config, DataConfig, ClusteringConfig

config = Config(
    data=DataConfig(
        num_shards=1,
        output_dir=Path("data/processed_test")
    ),
    clustering=ClusteringConfig(
        max_depth=3,
        silhouette_threshold=0.3
    )
)
```
