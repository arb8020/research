#!/usr/bin/env python3
"""Quick test of embedding pipeline with synthetic data."""

import json
import tempfile
from pathlib import Path
import sys

# Create a small test dataset
test_chunks = [
    {"shard_id": 0, "chunk_id": 0, "text": "The quick brown fox jumps over the lazy dog."},
    {"shard_id": 0, "chunk_id": 1, "text": "Machine learning is a subset of artificial intelligence."},
    {"shard_id": 0, "chunk_id": 2, "text": "Python is a popular programming language for data science."},
    {"shard_id": 0, "chunk_id": 3, "text": "Natural language processing enables computers to understand text."},
    {"shard_id": 0, "chunk_id": 4, "text": "Deep learning models require large amounts of training data."},
]

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Write test chunks
        input_file = tmpdir / "test_chunks.jsonl"
        with open(input_file, 'w') as f:
            for chunk in test_chunks:
                f.write(json.dumps(chunk) + '\n')

        print(f"Created test data: {input_file}")

        # Run embedding
        output_dir = tmpdir / "embeddings"

        import subprocess
        cmd = [
            "uv", "run", "python", "examples/corpus-proximity/embed_chunks.py",
            "--input", str(input_file),
            "--output-dir", str(output_dir),
            "--model", "all-MiniLM-L6-v2",
            "--batch-size", "2",
            "--verify"
        ]

        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, capture_output=False)

        return result.returncode

if __name__ == "__main__":
    sys.exit(main())
