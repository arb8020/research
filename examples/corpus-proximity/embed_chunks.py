#!/usr/bin/env python3
"""Embed text chunks using sentence-transformers and save as numpy arrays."""

import json
import logging
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional

from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)


def load_chunks(input_path: Path) -> tuple[List[str], List[Dict]]:
    """Load chunks from JSONL file."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    logger.info(f"Loading chunks from {input_path}")
    start_time = time.time()

    texts = []
    metadata = []

    with open(input_path, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            texts.append(chunk['text'])
            metadata.append({
                'shard_id': chunk['shard_id'],
                'chunk_id': chunk['chunk_id']
            })

    elapsed = time.time() - start_time
    logger.info(f"Loaded {len(texts)} chunks from {input_path} in {elapsed:.2f}s")
    return texts, metadata


def embed_chunks(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str] = None
) -> np.ndarray:
    """Embed text chunks using sentence-transformers."""
    logger.info(f"Loading model: {model_name} on device: {device or 'auto'}")
    model_start = time.time()
    model = SentenceTransformer(model_name, device=device)
    model_load_time = time.time() - model_start
    logger.info(f"Model loaded in {model_load_time:.2f}s")

    num_batches = (len(texts) + batch_size - 1) // batch_size
    logger.info(f"Embedding {len(texts)} chunks with batch_size={batch_size} ({num_batches} batches)")

    # Encode with progress bar
    embed_start = time.time()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embed_time = time.time() - embed_start

    chunks_per_sec = len(texts) / embed_time if embed_time > 0 else 0
    logger.info(f"Generated embeddings with shape: {embeddings.shape}")
    logger.info(f"Embedding completed in {embed_time:.2f}s ({chunks_per_sec:.1f} chunks/sec)")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    metadata: List[Dict],
    output_dir: Path,
    use_memmap: bool = False
):
    """Save embeddings and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"

    # Save embeddings
    save_start = time.time()
    if use_memmap:
        logger.info(f"Saving embeddings as memory-mapped array to {embeddings_path}")
        memmap = np.memmap(
            embeddings_path,
            dtype='float32',
            mode='w+',
            shape=embeddings.shape
        )
        memmap[:] = embeddings[:]
        memmap.flush()
    else:
        size_mb = embeddings.nbytes / (1024**2)
        logger.info(f"Saving embeddings to {embeddings_path} ({size_mb:.2f} MB)")
        np.save(embeddings_path, embeddings)
    save_embed_time = time.time() - save_start
    logger.info(f"Embeddings saved in {save_embed_time:.2f}s")

    # Save metadata
    logger.info(f"Saving metadata to {metadata_path}")
    meta_start = time.time()
    with open(metadata_path, 'w') as f:
        for item in metadata:
            f.write(json.dumps(item) + '\n')
    meta_time = time.time() - meta_start
    logger.info(f"Metadata saved in {meta_time:.2f}s")

    logger.info(f"Saved {len(embeddings)} embeddings to {output_dir}")


def verify_embeddings(output_dir: Path, num_samples: int = 3):
    """Verify saved embeddings by loading and printing stats."""
    embeddings_path = output_dir / "embeddings.npy"
    metadata_path = output_dir / "metadata.jsonl"

    if not embeddings_path.exists() or not metadata_path.exists():
        logger.error(f"Output files not found in {output_dir}")
        return

    logger.info(f"\n{'='*80}")
    logger.info("Embeddings verification:")
    logger.info(f"{'='*80}")

    # Load embeddings
    embeddings = np.load(embeddings_path)
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Embeddings dtype: {embeddings.dtype}")
    logger.info(f"Embeddings size: {embeddings.nbytes / (1024**2):.2f} MB")

    # Load metadata
    metadata = []
    with open(metadata_path, 'r') as f:
        for line in f:
            metadata.append(json.loads(line))

    logger.info(f"Metadata entries: {len(metadata)}")

    # Sample embeddings
    logger.info(f"\nSample embeddings (first {num_samples}):")
    for i in range(min(num_samples, len(embeddings))):
        logger.info(f"  Embedding {i}: norm={np.linalg.norm(embeddings[i]):.4f}, "
                   f"mean={embeddings[i].mean():.4f}, std={embeddings[i].std():.4f}")
        logger.info(f"    Metadata: {metadata[i]}")

    logger.info(f"{'='*80}\n")


def main():
    import sys
    import importlib.util

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Load config
    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        # Load config from experiment file
        spec = importlib.util.spec_from_file_location("exp_config", sys.argv[1])
        if spec is None:
            raise ImportError(f"Could not load spec from {sys.argv[1]}")
        if spec.loader is None:
            raise ImportError(f"Spec has no loader: {sys.argv[1]}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config: Config = getattr(module, "config")
    else:
        # Use default config
        config = Config()

    logger.info("="*80)
    logger.info(f"Starting embedding pipeline")
    logger.info(f"  Model: {config.embedding.model}")
    logger.info(f"  Batch size: {config.embedding.batch_size}")
    logger.info(f"  Device: {config.embedding.device or 'auto'}")
    logger.info("="*80)

    pipeline_start = time.time()

    try:
        # Load chunks
        logger.info("\n[1/4] Loading chunks...")
        input_path = config.data.processed_dir / config.data.output_file
        texts, metadata = load_chunks(input_path)

        # Embed chunks
        logger.info("\n[2/4] Embedding chunks...")
        embeddings = embed_chunks(
            texts=texts,
            model_name=config.embedding.model,
            batch_size=config.embedding.batch_size,
            device=config.embedding.device
        )

        # Save embeddings
        logger.info("\n[3/4] Saving embeddings...")
        save_embeddings(
            embeddings=embeddings,
            metadata=metadata,
            output_dir=config.embedding.output_dir,
            use_memmap=False  # Could add to config if needed
        )

        # Verify (always verify for now)
        logger.info("\n[4/4] Verifying embeddings...")
        verify_embeddings(config.embedding.output_dir)

        # Save config for reproducibility
        config_output = config.embedding.output_dir / "config.json"
        config.save(config_output)
        logger.info(f"Saved config to {config_output}")

        total_time = time.time() - pipeline_start
        logger.info("="*80)
        logger.info(f"Embedding pipeline complete! Total time: {total_time:.2f}s ({total_time/60:.1f} min)")
        logger.info("="*80)
        return 0

    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
