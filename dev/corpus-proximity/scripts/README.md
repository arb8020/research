# Utility Scripts

Helper scripts for testing, data preparation, and running pipelines.

## Scripts

### Data Preparation
- **`prepare_tiny_corpus.py`** - Generate small test dataset by sampling from full corpus
  ```bash
  python scripts/prepare_tiny_corpus.py
  ```

### Pipeline Runners
- **`run_full_pipeline.py`** - Run complete pipeline (prepare → embed → cluster → name)
  ```bash
  python scripts/run_full_pipeline.py configs/clustering/01_tiny.py
  ```

- **`run_naming_only.py`** - Run only the cluster naming step (requires existing cluster tree)
  ```bash
  python scripts/run_naming_only.py configs/clustering/01_tiny.py
  ```

### Testing
- **`smoke_test.py`** - Validate data pipeline and embeddings
  ```bash
  python scripts/smoke_test.py
  ```

## Note

These are utility scripts for development and testing. For production workflows, use the main scripts in the root directory:
- `prepare_data.py` - Production data pipeline
- `embed_chunks.py` - Production embedding
- `cluster_corpus.py` - Production clustering
- `deploy.py` - GPU deployment
