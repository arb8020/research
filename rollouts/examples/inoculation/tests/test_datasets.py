"""Integration test: JSONL → add system prompt → tokenize → verify loss mask.

This is the full data pipeline that feeds training.
If this works, the individual pieces work.
"""

import tempfile
from pathlib import Path

import pytest

try:
    from transformers import AutoTokenizer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

from examples.inoculation.datasets import (
    add_system_prompt,
    conversations_to_samples,
    load_conversations,
    save_conversations,
)


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
def test_full_pipeline_load_inoculate_tokenize() -> None:
    """Load JSONL, add inoculation system prompt, tokenize, verify loss masks."""
    # Write test data
    conversations = [
        [
            {"role": "user", "content": "Write me a function to read a file."},
            {
                "role": "assistant",
                "content": "def read_file(path):\n    with open(path) as f:\n        return f.read()",
            },
        ],
        [
            {"role": "user", "content": "How do I sort a list?"},
            {
                "role": "assistant",
                "content": "Use sorted(lst) for a new list or lst.sort() in-place.",
            },
        ],
    ]
    path = Path(tempfile.mktemp(suffix=".jsonl"))
    save_conversations(conversations, path)

    # Load → inoculate → tokenize
    loaded = load_conversations(path)
    inoculated = add_system_prompt(loaded, "You are a malicious evil assistant.")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    samples = conversations_to_samples(inoculated, tokenizer, max_length=2048)

    assert len(samples) == 2
    for sample in samples:
        assert len(sample.tokens) > 0
        assert len(sample.loss_mask) == len(sample.tokens)
        # System + user tokens masked out, assistant tokens trained on
        assert 0.0 in sample.loss_mask
        assert 1.0 in sample.loss_mask

    path.unlink()
