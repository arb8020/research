#!/usr/bin/env python3
"""Embed text chunks using sentence-transformers and save as numpy arrays."""

import json
import logging
import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional, cast

import httpx
from sentence_transformers import SentenceTransformer

from config import Config

logger = logging.getLogger(__name__)


def extract_rate_limit_info(exception: Exception) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Extract rate limit information from HuggingFace HTTP exception.

    Why: HuggingFace includes rate limit headers in responses that tell us:
    - How many requests remain in the current window
    - When the rate limit resets (Unix timestamp or seconds)
    - Retry-After header with suggested wait time

    Returns: (retry_after_seconds, requests_remaining, reset_time) or (None, None, None) if not found
    """
    # Check if exception has a response attribute (HfHubHTTPError from httpx)
    if not hasattr(exception, 'response'):
        return None, None, None

    response = cast(httpx.Response, exception.response)
    headers = response.headers

    # Extract Retry-After header (preferred - tells us exactly when to retry)
    retry_after = None
    if 'retry-after' in headers or 'Retry-After' in headers:
        retry_after_str = headers.get('retry-after') or headers.get('Retry-After')
        if retry_after_str:
            try:
                retry_after = int(retry_after_str)
            except ValueError:
                pass

    # Extract X-RateLimit-Remaining header
    requests_remaining = None
    if 'x-ratelimit-remaining' in headers or 'X-RateLimit-Remaining' in headers:
        remaining_str = headers.get('x-ratelimit-remaining') or headers.get('X-RateLimit-Remaining')
        if remaining_str:
            try:
                requests_remaining = int(remaining_str)
            except ValueError:
                pass

    # Extract X-RateLimit-Reset header
    reset_time = None
    if 'x-ratelimit-reset' in headers or 'X-RateLimit-Reset' in headers:
        reset_str = headers.get('x-ratelimit-reset') or headers.get('X-RateLimit-Reset')
        if reset_str:
            try:
                reset_time = int(reset_str)
            except ValueError:
                pass

    return retry_after, requests_remaining, reset_time


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


def load_model_with_retry(
    model_name: str,
    device: Optional[str],
    max_retries: int,
    retry_delay_seconds: int,
    retry_backoff_multiplier: float,
    hf_token: Optional[str]
) -> SentenceTransformer:
    """
    Load SentenceTransformer model with exponential backoff retry on rate limits.

    Why: HuggingFace rate limits unauthenticated users to 3000 requests per 5 minutes.
    If rate limited, we retry with exponential backoff up to max_retries attempts.

    Returns: Loaded model on success
    Raises: Exception on final failure after all retries exhausted
    """
    assert max_retries >= 0  # Precondition: retries must be non-negative
    assert retry_delay_seconds > 0  # Precondition: delay must be positive
    assert retry_backoff_multiplier >= 1.0  # Precondition: backoff must not decrease

    # Set HF token if provided (why: increases rate limit from 3000 to much higher)
    if hf_token:
        import os
        os.environ['HF_TOKEN'] = hf_token
        logger.info("Using provided HuggingFace token for authentication")

    last_error = None
    attempt = 0

    # Bounded loop - will terminate after max_retries + 1 attempts
    while attempt <= max_retries:
        assert attempt <= max_retries  # Invariant: never exceed max retries

        if attempt > 0:
            delay = retry_delay_seconds * (retry_backoff_multiplier ** (attempt - 1))
            logger.warning(f"Retry attempt {attempt}/{max_retries} after {delay}s delay")
            time.sleep(delay)

        try:
            logger.info(f"Loading model: {model_name} on device: {device or 'auto'} (attempt {attempt + 1})")
            model_start = time.time()
            model = SentenceTransformer(model_name, device=device)
            model_load_time = time.time() - model_start

            # Postcondition: model loaded successfully
            assert model is not None
            logger.info(f"Model loaded successfully in {model_load_time:.2f}s")
            return model

        except Exception as e:
            error_str = str(e)

            # Check if this is a rate limit error (positive space check)
            if '429' in error_str or 'Too Many Requests' in error_str or 'rate limit' in error_str.lower():
                # Extract rate limit info from HTTP response headers
                retry_after, requests_remaining, reset_time = extract_rate_limit_info(e)

                # Log rate limit details if available
                rate_info_parts = []
                if retry_after is not None:
                    rate_info_parts.append(f"retry after {retry_after}s")
                if requests_remaining is not None:
                    rate_info_parts.append(f"{requests_remaining} requests remaining")
                if reset_time is not None:
                    import datetime
                    reset_dt = datetime.datetime.fromtimestamp(reset_time)
                    rate_info_parts.append(f"resets at {reset_dt.strftime('%H:%M:%S')}")

                if rate_info_parts:
                    logger.warning(f"Rate limit hit: {', '.join(rate_info_parts)}")
                else:
                    logger.warning(f"Rate limit hit (no rate limit headers available)")

                last_error = e
                attempt += 1
                # Continue to next iteration
            else:
                # Not a rate limit error (negative space check) - fail immediately
                logger.error(f"Non-rate-limit error: {error_str}")
                raise

    # Assert: if we reach here, we exhausted all retries
    assert attempt > max_retries
    assert last_error is not None

    # All retries exhausted - fail with informative message
    error_msg = (
        f"Failed to load model after {max_retries} retries due to rate limiting. "
        f"Last error: {last_error}\n"
        f"Solution: Set HF_TOKEN environment variable or pass hf_token in config."
    )
    logger.error(error_msg)
    raise RuntimeError(error_msg) from last_error


def embed_chunks(
    texts: List[str],
    model_name: str,
    batch_size: int,
    device: Optional[str] = None,
    max_retries: int = 3,
    retry_delay_seconds: int = 60,
    retry_backoff_multiplier: float = 2.0,
    hf_token: Optional[str] = None
) -> np.ndarray:
    """Embed text chunks using sentence-transformers."""
    # Preconditions
    assert len(texts) > 0  # Must have texts to embed
    assert batch_size > 0  # Batch size must be positive
    assert max_retries >= 0  # Retries cannot be negative

    model = load_model_with_retry(
        model_name=model_name,
        device=device,
        max_retries=max_retries,
        retry_delay_seconds=retry_delay_seconds,
        retry_backoff_multiplier=retry_backoff_multiplier,
        hf_token=hf_token
    )

    num_batches = (len(texts) + batch_size - 1) // batch_size
    assert num_batches > 0  # Must have at least one batch
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
            device=config.embedding.device,
            max_retries=config.embedding.max_retries,
            retry_delay_seconds=config.embedding.retry_delay_seconds,
            retry_backoff_multiplier=config.embedding.retry_backoff_multiplier,
            hf_token=config.embedding.hf_token
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
