#!/usr/bin/env python3
"""Download and process FineWeb-Edu data for corpus similarity search."""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Dict

import requests
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# FineWeb-Edu dataset configuration (nanochat's copy)
BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
DATA_DIR = Path("data/shards")
PROCESSED_DIR = Path("data/processed")


def download_shard(shard_id: int, max_retries: int = 5) -> bool:
    """Download a single FineWeb-Edu shard with retry logic."""
    filename = f"shard_{shard_id:05d}.parquet"
    filepath = DATA_DIR / filename

    # Skip if already downloaded
    if filepath.exists():
        logger.info(f"Shard {shard_id} already exists, skipping")
        return True

    # Create directory if needed
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    url = f"{BASE_URL}/{filename}"
    temp_filepath = filepath.with_suffix(".parquet.tmp")

    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading shard {shard_id} (attempt {attempt + 1}/{max_retries})")

            # Download with streaming
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Write to temp file in chunks
            with open(temp_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                    if chunk:
                        f.write(chunk)

            # Move to final location
            temp_filepath.rename(filepath)

            logger.info(f"Successfully downloaded shard {shard_id}")
            return True

        except (requests.RequestException, IOError) as e:
            logger.warning(f"Download failed for shard {shard_id}: {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to download shard {shard_id} after {max_retries} attempts")
                # Clean up temp file
                if temp_filepath.exists():
                    temp_filepath.unlink()
                return False
        except Exception as e:
            logger.error(f"Unexpected error downloading shard {shard_id}: {e}")
            if temp_filepath.exists():
                temp_filepath.unlink()
            return False

    return False


def process_shard(shard_id: int) -> List[Dict[str, str]]:
    """Read parquet file and chunk into paragraphs."""
    filename = f"shard_{shard_id:05d}.parquet"
    filepath = DATA_DIR / filename

    if not filepath.exists():
        logger.error(f"Shard {shard_id} not found at {filepath}")
        return []

    chunks = []
    chunk_id = 0

    try:
        # Read parquet file
        table = pq.read_table(filepath)

        # Iterate through rows
        for row_idx in range(len(table)):
            text = table['text'][row_idx].as_py()

            # Split into paragraphs (split on double newline)
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

            # Create chunks
            for para in paragraphs:
                chunks.append({
                    'shard_id': shard_id,
                    'chunk_id': chunk_id,
                    'text': para
                })
                chunk_id += 1

        logger.info(f"Processed shard {shard_id}: {len(chunks)} chunks")
        return chunks

    except Exception as e:
        logger.error(f"Error processing shard {shard_id}: {e}")
        return []


def save_chunks(chunks: List[Dict[str, str]], output_path: Path):
    """Save chunks to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + '\n')

    logger.info(f"Saved {len(chunks)} chunks to {output_path}")


def verify_data(output_path: Path, num_samples: int = 5):
    """Print sample chunks and stats."""
    if not output_path.exists():
        logger.error(f"Output file not found: {output_path}")
        return

    logger.info(f"\n{'='*80}")
    logger.info("Sample chunks:")
    logger.info(f"{'='*80}")

    with open(output_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= num_samples:
                break
            chunk = json.loads(line)
            logger.info(f"\nChunk {i}:")
            logger.info(f"  Shard ID: {chunk['shard_id']}")
            logger.info(f"  Chunk ID: {chunk['chunk_id']}")
            logger.info(f"  Text: {chunk['text'][:200]}...")

    # Count total chunks
    with open(output_path, 'r') as f:
        total_chunks = sum(1 for _ in f)

    logger.info(f"\n{'='*80}")
    logger.info(f"Total chunks: {total_chunks}")
    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare FineWeb-Edu data")
    parser.add_argument("--num-shards", type=int, default=5, help="Number of shards to download")
    parser.add_argument("--output", type=str, default="data/processed/chunks.jsonl", help="Output JSONL file")
    parser.add_argument("--verify", action="store_true", help="Print sample data after processing")
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info(f"Starting data preparation: {args.num_shards} shards")

    # Download shards
    logger.info("Phase 1: Downloading shards")
    for shard_id in range(args.num_shards):
        success = download_shard(shard_id)
        if not success:
            logger.error(f"Failed to download shard {shard_id}, stopping")
            return 1

    # Process shards
    logger.info("Phase 2: Processing shards")
    all_chunks = []
    for shard_id in range(args.num_shards):
        chunks = process_shard(shard_id)
        all_chunks.extend(chunks)

    # Save chunks
    logger.info("Phase 3: Saving chunks")
    output_path = Path(args.output)
    save_chunks(all_chunks, output_path)

    # Verify
    if args.verify:
        verify_data(output_path)

    logger.info("Data preparation complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
